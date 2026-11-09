"""阶段 2：把检索到的块拼进 prompt，生成**带来源**的回答。

============================== 这一层的职责 ==============================

输入：一个问题
输出：一段回答 + 引用来源（文件名 + 页码）

链路：
    query ──► BM25 检索 top-k 块 ──► 拼成 context ──► prompt ──► LLM ──► 回答
                                                              │
                                                              └─► 结构化引用

============================== 三个设计决定 ==============================

1. **上下文必须带编号标签**
   直接拼接 N 段文本，模型无法告诉你是哪段支撑了答案。
   每段前面加 `[1]《文件名》第 X 页` 这样的标签，要求模型在句末标 `[1]`，
   最后把编号映射回真实来源。这是「可溯源」的实现方式。

2. **明确允许「不知道」**
   RAG 最常见的失败是模型在没有依据时硬编。prompt 里必须显式给出
   拒答路径：「如果资料不足，直接说找不到」。否则模型会把训练数据里的
   常识当答案，而这恰恰是 RAG 要避免的。

3. **强约束引用格式，程序化校验**
   不能只靠提示词说「请标注来源」——模型会漏。要在拿到回答后
   用正则扫一遍 `[n]`，校验编号是否在有效范围内，并把无效编号标出来。
   提示词是软约束，代码校验才是硬的。

============================== 关于温度 ==============================

用 0.1 而不是 0。原因：
    RAG 需要事实一致性，温度越低越好；但完全为 0 在部分服务端实现里
    会退化成贪心解码，容易陷入重复。0.1 保留极少随机性，实测更稳。
    这也是为什么下面会把 temperature 显式暴露成参数，方便做对比。
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1"))

import config
from chunker import Chunk
from retriever import Hit, KeywordIndex


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Citation:
    """一条引用：编号 → 真实来源。"""

    index: int  # 1 基，对应 prompt 里的 [1]
    source_file: str
    page_label: str
    section: str
    chunk_id: str

    def render(self) -> str:
        loc = f"《{self.source_file}》{self.page_label}"
        if self.section:
            loc += f" · {self.section}"
        return f"[{self.index}] {loc}"


@dataclass
class Answer:
    """一次问答的完整结果，包含所有可审计的中间产物。"""

    question: str
    text: str
    citations: list[Citation]
    used_indices: list[int]  # 回答里实际引用了哪些编号
    invalid_indices: list[int]  # 模型编造了不存在的编号
    retrieved: list[Hit]
    prompt: str
    latency_ms: float
    usage: dict = field(default_factory=dict)

    @property
    def cited_ok(self) -> bool:
        """是否给出了至少一条有效引用。"""
        return bool(self.used_indices)

    def source_block(self) -> str:
        """渲染「来源」区块，供 UI/终端展示。"""
        if not self.citations:
            return "（无来源）"
        lines = []
        for c in self.citations:
            # 只列出回答里真正引用到的；全列出来反而干扰阅读
            if c.index in self.used_indices:
                lines.append("  " + c.render())
        if not lines:
            # 模型没标编号时，退回列出全部检索结果
            lines = ["  " + c.render() + "（未被正文引用）" for c in self.citations]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt 组装
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你只能依据【参考资料】回答，不得使用你自己的知识补充。

规则：
1. 每个结论后面必须用 [编号] 标注它依据的资料，例如：接口超时时间为 5 秒 [2]。
2. 如果【参考资料】不足以回答问题，直接回答"根据现有资料无法确定"，并说明缺什么信息。
3. 不要编造资料里没有的数字、名称、流程、字段。
4. 如果多份资料有冲突，指出冲突并分别标注来源。
5. 用简体中文回答，简洁直接，不要复述问题。

违反上述规则会导致回答被判定为无效。宁可回答"无法确定"，也不要猜测。"""


def build_context(hits: list[Hit]) -> tuple[str, list[Citation]]:
    """把检索结果拼成带编号的 context，返回 (文本, 引用映射)。

    编号从 1 开始，与人阅读习惯一致。
    """
    parts: list[str] = []
    citations: list[Citation] = []

    for i, h in enumerate(hits, 1):
        c: Chunk = h.chunk
        cite = Citation(
            index=i,
            source_file=c.source_file,
            page_label=c.page_label(),
            section=c.section,
            chunk_id=c.chunk_id,
        )
        citations.append(cite)
        # 标签里同时给文件名、页码、章节 —— 模型引用时能明确指向
        header = f"[{i}]《{c.source_file}》{c.page_label()}"
        if c.section:
            header += f" · {c.section}"
        parts.append(f"{header}\n{c.text.strip()}")

    return "\n\n---\n\n".join(parts), citations


def build_prompt(question: str, context: str) -> str:
    return f"""【参考资料】
{context}

【问题】
{question}

请依据上述资料回答，并在每个结论后标注 [编号]。"""


# ---------------------------------------------------------------------------
# 引用校验
# ---------------------------------------------------------------------------
CITE_RE = re.compile(r"\[(\d+)\]")


def verify_citations(text: str, citations: list[Citation]) -> tuple[list[int], list[int]]:
    """从回答里抽出 [n] 编号，返回 (有效编号, 无效编号)。

    为什么必须程序化校验：
        提示词说「请标注来源」，模型大约 80% 会照做，剩下 20% 会：
        - 一个编号都不标
        - 标出 [7]，但 context 里只有 3 段
        - 把编号写成「资料2」或「[2、3]」
        这些都必须能被代码发现，否则交付给用户的就是不可靠的溯源。
    """
    valid_range = {c.index for c in citations}
    found = [int(m) for m in CITE_RE.findall(text)]
    used = sorted({n for n in found if n in valid_range})
    invalid = sorted({n for n in found if n not in valid_range})
    return used, invalid


# ---------------------------------------------------------------------------
# 问答器
# ---------------------------------------------------------------------------
class RAGAnswerer:
    """检索 + 生成，返回带引用的答案。"""

    def __init__(
        self,
        index: KeywordIndex,
        top_k: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = config.get_chat_client()
        return self._client

    def answer(self, question: str, hits: list[Hit] | None = None) -> Answer:
        t0 = time.perf_counter()

        # 允许外部传入 hits（用于「故意喂错上下文」的对照实验）
        if hits is None:
            hits = self.index.search(question, top_k=self.top_k)

        context, citations = build_context(hits)
        prompt = build_prompt(question, context)

        if not hits:
            # 检索不到就别调模型了 —— 省一次调用，也避免模型自由发挥
            return Answer(
                question=question,
                text="根据现有资料无法确定：没有检索到与该问题相关的段落。",
                citations=[],
                used_indices=[],
                invalid_indices=[],
                retrieved=[],
                prompt=prompt,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        resp = self.client.chat.completions.create(
            model=config.chat_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = (resp.choices[0].message.content or "").strip()
        used, invalid = verify_citations(text, citations)
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        return Answer(
            question=question,
            text=text,
            citations=citations,
            used_indices=used,
            invalid_indices=invalid,
            retrieved=hits,
            prompt=prompt,
            latency_ms=(time.perf_counter() - t0) * 1000,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# 终端展示
# ---------------------------------------------------------------------------
def print_answer(a: Answer, show_prompt: bool = False, snippet: int = 110) -> None:
    print("═" * 92)
    print(f"问题：{a.question}")
    print("═" * 92)

    if show_prompt:
        print("【实际发送的 prompt】")
        print(a.prompt[:1200] + ("…" if len(a.prompt) > 1200 else ""))
        print("─" * 92)

    print(a.text)
    print()
    print("来源：")
    print(a.source_block())
    print()

    # 引用质量诊断 —— 这是「能不能溯源」的硬指标
    flags = []
    if not a.used_indices:
        flags.append("正文未标注任何有效引用")
    if a.invalid_indices:
        flags.append(f"编造了不存在的编号 {a.invalid_indices}")
    if flags:
        print(f"⚠ 引用问题：{'；'.join(flags)}")
    else:
        print(f"✓ 引用有效：{a.used_indices}")

    print()
    print(f"延迟 {a.latency_ms:.0f}ms", end="")
    if a.usage:
        print(f" | tokens {a.usage.get('total_tokens', '?')}"
              f"（入 {a.usage.get('prompt_tokens', '?')} / 出 {a.usage.get('completion_tokens', '?')}）", end="")
    print()
    print()


if __name__ == "__main__":
    from cache import load_chunks
    from paths import find_pdfs

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit(
            "没找到 PDF。\n"
            "  请把待处理的 PDF 放进 data/docs/（支持子目录），然后重新运行。"
        )

    print("建索引…")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    print(f"  {len(pdfs)} 份文档 / {len(chunks)} 块\n")

    index = KeywordIndex(chunks)
    answerer = RAGAnswerer(index, top_k=int(sys.argv[1]) if len(sys.argv) > 1 else 3)

    for q in [
        "系统采用什么架构，前后端如何通信",
        "系统的技术栈有哪些，各组件具体版本是多少",
    ]:
        print_answer(answerer.answer(q))
