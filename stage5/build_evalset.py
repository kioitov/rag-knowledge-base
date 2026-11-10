"""阶段 5 第二块：LLM 辅助构造评测集（原文锚定 + 自动校验）。

============================== 为什么要换方法 ==============================

第一版用正则从语料抽「X 的默认值是 Y」，只得到 43 条，且噪音明显：

    主体：纯 H        答案：TTP          ← 正则从词中间切开
    主体：struct utsname  答案：64       ← 主体不是人话

正则的天花板在于：**它不理解句子结构**。

============================== 新方法：三段式，每段都有校验 ==============================

步骤 1  LLM 从原文**生成问题**（给定 chunk 全文，要求只问该段能回答的问题）
步骤 2  LLM 同时给出「标准答案」和「答案在原文中的原句」
步骤 3  **程序校验**：答案必须真实出现在原文里

第 3 步是关键设计：
    LLM 可能编造答案。但只要要求它同时给出「原文原句」，
    我们就能用代码验证那句原句是否真的在 chunk 里。
    不在 → 丢弃该条。
这样即使 LLM 有幻觉，也进不了评测集。

============================== 三种题型 ==============================

    fact      有明确答案的事实题        → 判定：回答含标准答案关键词
    refuse    语料里没有答案的问题      → 判定：必须拒答
    multi     需要综合多个段落的问题    → 判定：回答含多个关键词

refuse 类由程序从「语料不覆盖的主题」构造，不靠 LLM 编。

============================== 为什么仍要人工抽查 ==============================

程序只能验证「答案在原文里」，验证不了「这个问题问得通不通顺」
「这个问题是不是太简单/太模糊」。
所以生成后抽样打印，人工过一遍，把不合格的剔掉。

用法：
    python stage5/build_evalset.py --target 120
    python stage5/build_evalset.py --target 30 --dry-run   # 试跑，不写文件
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))

import config
from cache import load_chunks
from chunker import Chunk
from paths import find_docs

OUT_DIR = _ROOT / "data" / "eval"
OUT_FILE = OUT_DIR / "eval_set.json"


@dataclass
class EvalItem:
    """一条评测项。所有字段都可核验。"""

    id: str
    kind: str  # fact / refuse / multi
    question: str
    answer_keywords: list[str]  # 判定用：回答里必须出现这些
    evidence_quote: str  # 原文原句（已程序校验存在于 chunk）
    chunk_id: str
    source_file: str
    section: str
    page_label: str
    difficulty: str = "medium"
    note: str = ""


# ---------------------------------------------------------------------------
# 从语料中挑「值得出题」的 chunk
# ---------------------------------------------------------------------------
def pick_chunks(chunks: list[Chunk], n: int, seed: int = 20261110) -> list[Chunk]:
    """挑选适合出题的 chunk。

    为什么不能随机抽：
        大量 chunk 是目录页、导航、免责声明，出不了题。
        实测 1290 块里约有 1/3 属于这类。

    筛选条件：
        - kind == "text"（表格单独处理）
        - 长度 300~1200 字（太短没信息，太长 LLM 抓不住重点）
        - 含中文（排除纯代码块）
        - 不是「这是什么 / 参见」这类导航段
    """
    cands: list[Chunk] = []
    for c in chunks:
        if c.kind != "text":
            continue
        t = c.text
        if not (300 <= len(t) <= 1200):
            continue
        cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
        if cjk < 100:
            continue
        # 排除导航类
        if re.search(r"^(参见|另请|更多信息|延伸阅读)", t):
            continue
        if "本文档" in t and len(t) < 400:
            continue
        cands.append(c)

    random.seed(seed)
    random.shuffle(cands)
    return cands[: n * 3]  # 多取一些，后面会淘汰


# ---------------------------------------------------------------------------
# LLM 出题
# ---------------------------------------------------------------------------
SYSTEM = """你是一个严格的技术文档出题人。你只能依据用户给出的【文档片段】出题。

输出 JSON 数组，每个元素：
{
  "question": "一个自然的中文问句",
  "answer": "答案（尽量简短，2-20 字，必须能在片段中逐字找到）",
  "evidence": "片段中包含该答案的完整原句（必须逐字摘录，不要改写）",
  "difficulty": "easy|medium|hard"
}

硬性要求：
1. 问题的答案必须能**直接从该片段找到**，不能依赖片段外的知识。
2. answer 必须是片段中**逐字出现**的字符串（可以是一个词、数字、参数名）。
3. evidence 必须是从片段中**原样复制**的一到两句，不能修改任何字符。
4. 每个片段出 1~3 道题，从不同角度问（是什么/怎么配置/默认值/有什么限制）。
5. 不要出「本文讲了什么」这类概括题，要出有确定答案的事实题。
6. 如果该片段信息量不足以出有确定答案的题，返回空数组 []。

只输出 JSON，不要解释。"""


def gen_from_chunk(chunk: Chunk, client, model: str, retry: int = 2) -> list[dict]:
    """让 LLM 从一个 chunk 生成题目。"""
    prompt = f"""【文档片段】
{chunk.text[:1800]}

请据此出 1~3 道有确定答案的题。"""

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # 去掉可能的 ```json 围栏
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            return [d for d in data if isinstance(d, dict)]
        except Exception as exc:
            if attempt == retry:
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
# 校验：答案必须真实出现在原文里
# ---------------------------------------------------------------------------
def verify(item: dict, chunk: Chunk) -> tuple[bool, str]:
    """验证 LLM 产出的题目是否可用。

    校验项：
        1. answer 非空且长度合理
        2. **answer 逐字出现在 chunk.text 里** ← 核心，防幻觉
        3. evidence 也出现在原文里（如果 LLM 给了）
        4. question 是个问句（含问号或疑问词）
    """
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    ev = (item.get("evidence") or "").strip()

    if not q or not a:
        return False, "问题或答案为空"
    if len(a) > 60:
        return False, "答案过长"
    if not re.search(r"[？?]|怎么|什么|哪些|多少|如何|是否|哪一?", q):
        return False, "不像问句"

    # 核心校验：答案必须逐字出现在原文
    if a not in chunk.text:
        return False, f"答案「{a[:20]}」不在原文中（疑似幻觉）"

    # evidence 校验（LLM 可能改写，宽松处理：只要关键片段在就行）
    if ev and ev not in chunk.text:
        # 允许 evidence 有细微差异，但至少要有 60% 的字符能在原文找到
        core = re.sub(r"\s+", "", ev)[:40]
        if core and core not in re.sub(r"\s+", "", chunk.text):
            return False, "证据原句不在原文中"

    return True, ""


# 答案开头的虚词/助词，判定时不该算进关键词。
# 实测 LLM 会给出「应为奇数」「未被应用的」这类答案 ——
# 逐字匹配时用户回答「奇数」就不会命中，造成假阴性。
LEADING_FILLER = re.compile(
    r"^(?:应为|应当是|应该是|是|为|叫做|称为|称为|即|就是|通常是|一般是"
    r"|未被应用的|未被应用|被应用的)\s*"
)


def normalize_keyword(ans: str) -> str:
    """清理答案关键词。

    「应为奇数」→「奇数」
    「未被应用的」→「未应用」（保留语义，去掉口语化后缀）

    注意：清理后必须仍然是原文子串，否则校验会失效。
    所以这里只在清理后仍能匹配原文时才采用，否则保留原答案。
    """
    a = ans.strip()
    stripped = LEADING_FILLER.sub("", a)
    stripped = re.sub(r"的$", "", stripped).strip()
    return stripped if len(stripped) >= 2 else a


def main() -> None:
    ap = argparse.ArgumentParser(description="构造评测集")
    ap.add_argument("--target", type=int, default=120, help="目标题目数")
    ap.add_argument("--dry-run", action="store_true", help="不写文件")
    ap.add_argument("--show", type=int, default=8, help="抽样展示条数")
    args = ap.parse_args()

    docs = find_docs()
    if not docs:
        raise SystemExit("没找到语料。先跑 stage5/fetch_corpus.py")

    print("=" * 90)
    print(" 阶段 5：构造评测集")
    print("=" * 90)

    print("\n加载语料…")
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(load_chunks(d))
    print(f"  {len(chunks)} 块")

    cands = pick_chunks(chunks, args.target)
    print(f"  候选 chunk {len(cands)} 个（长度 300-1200 / 含中文 / 非导航）")

    client = config.get_chat_client()
    model = config.chat_model()
    print(f"  出题模型 {model}")

    items: list[EvalItem] = []
    rejected: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    print(f"\n开始出题（目标 {args.target} 条）…")
    for i, chunk in enumerate(cands, 1):
        if len(items) >= args.target:
            break

        raws = gen_from_chunk(chunk, client, model)
        for r in raws:
            ok, why = verify(r, chunk)
            if not ok:
                rejected.append((r.get("question", "")[:40], why))
                continue

            # 清理答案关键词；清理后必须仍能在原文找到，否则保留原值
            raw_ans = r["answer"].strip()
            kw = normalize_keyword(raw_ans)
            if kw != raw_ans and kw not in chunk.text:
                kw = raw_ans  # 清理破坏了原文匹配，退回原答案

            items.append(EvalItem(
                id=f"E{len(items) + 1:03d}",
                kind="fact",
                question=r["question"].strip(),
                answer_keywords=[kw],
                evidence_quote=(r.get("evidence") or "").strip()[:300],
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                section=chunk.section,
                page_label=chunk.page_label(),
                difficulty=r.get("difficulty", "medium"),
            ))

        if i % 5 == 0 or len(items) >= args.target:
            el = time.perf_counter() - t0
            print(f"  [{i:3d}/{len(cands)}] 已生成 {len(items):3d} 条"
                  f"  淘汰 {len(rejected):3d}  用时 {el:.0f}s")

    # ---- 加 refuse 类：语料里明确没有的主题 ----
    # 这些问题的答案确实不在 K8s 中文文档里，模型必须拒答
    REFUSE_TOPICS = [
        "Kubernetes 项目的许可证编号是什么",
        "Kubernetes 1.20 版本的发布日期是哪天",
        "kube-apiserver 的 CVE 漏洞编号有哪些",
        "Kubernetes 中国区的商业支持代理商是哪家",
        "kubectl 的源码是用哪种语言写的，有多少行",
        "Kubernetes 商标的注册号是多少",
        "CNCF 基金会 2024 年的年度预算",
        "Kubernetes 创始人 Linus Torvalds 的出生地",
        "etcd 使用的 Raft 协议是第几版规范",
        "Pod 在 AWS us-east-1 区域的默认可用区",
    ]
    for j, topic in enumerate(REFUSE_TOPICS):
        items.append(EvalItem(
            id=f"R{j + 1:03d}",
            kind="refuse",
            question=topic,
            answer_keywords=[],
            evidence_quote="",
            chunk_id="",
            source_file="",
            section="",
            page_label="",
            difficulty="medium",
            note="语料不覆盖该主题，应拒答",
        ))

    # ---- 保存 ----
    print()
    print("=" * 90)
    print(f" 完成：{len(items)} 条（fact {sum(1 for x in items if x.kind=='fact')}"
          f" / refuse {sum(1 for x in items if x.kind=='refuse')}）")
    print(f" 淘汰 {len(rejected)} 条（答案不在原文 / 不像问句 / 过长）")
    print(f" 用时 {time.perf_counter() - t0:.0f}s")

    if rejected:
        print("\n 淘汰原因抽样（前 8 条）：")
        from collections import Counter
        cnt = Counter(w.split("「")[0] for _q, w in rejected)
        for reason, n in cnt.most_common(8):
            print(f"   {n:3d}  {reason}")

    print("\n" + "=" * 90)
    print(" 抽样检查（人工过一遍）")
    print("=" * 90)
    for it in items[: args.show]:
        print(f"\n[{it.id}] {it.question}")
        print(f"  标准答案：{it.answer_keywords}")
        print(f"  原文依据：{it.evidence_quote[:110]}")
        print(f"  出处：{it.source_file[:40]} {it.page_label} · {it.section[:20]}")

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as f:
            json.dump({
                "corpus": "Kubernetes 中文文档（CC BY 4.0）",
                "total": len(items),
                "by_kind": {
                    "fact": sum(1 for x in items if x.kind == "fact"),
                    "refuse": sum(1 for x in items if x.kind == "refuse"),
                },
                "judging": "答案关键词字符串匹配；refuse 类要求出现拒答表述",
                "items": [asdict(x) for x in items],
            }, f, ensure_ascii=False, indent=1)
        print()
        print("=" * 90)
        print(f" 已保存 {OUT_FILE}")
        print("=" * 90)


if __name__ == "__main__":
    main()
