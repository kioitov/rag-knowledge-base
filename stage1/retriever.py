"""阶段 1 第三块：检索器（先用关键词命中，不接向量库）。

验收标准（交接文档阶段 1）：
    输入一个问题，能在几百个块里找到最相关的 3 块，并打印出来。

为什么先做关键词检索而不是直接上 embedding：
    1. 关键词检索零依赖、零成本、结果完全可解释 —— 你能逐条说清楚
       「这一块为什么被选中」，面试时这是加分项。
    2. 它是混合检索里 BM25 这一半的原型。先手写一遍，后面用 Chroma 时
       才知道向量检索解决了什么、没解决什么。
    3. 阶段 5 的对比实验需要「纯向量 vs 混合检索」两组数据，
       关键词通道现在就要有。

实现：中文没有空格，标准 BM25 的 tokenizer 不适用。
用「二元组（bigram）+ 单字」切分做中文 token，这是最朴素但有效的中文分词替代方案。
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import Chunk, build_chunks

# 停用词：中文高频虚词，参与打分只会稀释信号
STOPWORDS = set(
    "的了和是在有为与及或对从到把被这那其之以于而但并等个中上下不也都很就可"
    "要会能将对给让使还要需应该可以什么怎么如何哪些哪个请问一下"
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文用 bigram + 单字，英文数字用整词。

    为什么 bigram：
        「智能匹配」不分词的话，用户搜「匹配」命中不了；
        分单字又太碎（「匹」「配」各自独立，噪音大）。
        bigram 是中文检索里成本最低的折中方案 ——
        不需要词典，召回够用，误匹配可控。
    """
    text = text.lower()
    tokens: list[str] = []

    # 英文/数字整词
    for m in WORD_RE.finditer(text):
        tokens.append(m.group())

    # 中文滑动 bigram
    chars = [c for c in text if CJK_RE.match(c)]
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        if chars[i] not in STOPWORDS and chars[i + 1] not in STOPWORDS:
            tokens.append(bg)

    # 单字兜底（query 很短时 bigram 可能一个都不剩）
    if len(tokens) < 3:
        tokens.extend(c for c in chars if c not in STOPWORDS)

    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    matched_terms: list[str]

    def __str__(self) -> str:
        return f"{self.score:.3f}  {self.chunk.page_label()}  {self.chunk.section[:20]}"


class KeywordIndex:
    """倒排索引 + BM25 打分。

    BM25 而不是「命中次数」：
        命中次数会偏爱长文本块 —— 1000 字的表格天然比 200 字的正文命中多。
        BM25 用文档长度归一化（b 参数）+ 词频饱和（k1 参数）修正这个偏差。
        这两个参数是标准值，不需要调。
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.inverted: dict[str, set[int]] = {}

        for i, c in enumerate(chunks):
            toks = tokenize(c.text)
            tf = Counter(toks)
            self.doc_tokens.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.inverted.setdefault(t, set()).add(i)

        self.N = len(chunks)
        self.avg_len = (sum(self.doc_len) / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        df = len(self.inverted.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        q_toks = tokenize(query)
        if not q_toks:
            return []

        # 候选集：含任一 query token 的块
        candidates: set[int] = set()
        for t in q_toks:
            candidates |= self.inverted.get(t, set())

        qtf = Counter(q_toks)
        scored: list[Hit] = []

        for idx in candidates:
            tf = self.doc_tokens[idx]
            dl = self.doc_len[idx]
            score = 0.0
            matched: list[str] = []

            for term, qn in qtf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len if self.avg_len else 1)
                score += idf * (f * (self.k1 + 1) / denom) * qn
                matched.append(term)

            if score > 0:
                scored.append(Hit(self.chunks[idx], score, matched))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 自测：五个真实问题，看检索结果对不对
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "联合贷款人是怎么推荐的？权重怎么算",
    "农户提交融资意向需要填哪些字段",
    "系统支持哪些还款方式",
    "银行用户的审核流程是什么",
    "ARIMA 价格预测用在哪里",
]


def main() -> None:
    pdf = Path(
        r"E:\deepseek work\融销通项目\成果文档\软件需求规格说明书\融销通平台软件需求规格说明书.pdf"
    )
    chunks = build_chunks(pdf, strategy="paragraph")
    print(f"文档：{pdf.name}")
    print(f"块数：{len(chunks)}（正文 {sum(1 for c in chunks if c.kind=='text')} / 表格 {sum(1 for c in chunks if c.kind=='table')}）")

    index = KeywordIndex(chunks)
    print(f"词表：{len(index.inverted)} 个 token，平均块长 {index.avg_len:.0f} token\n")

    for q in TEST_QUERIES:
        print("=" * 92)
        print(f"问题：{q}")
        print("=" * 92)
        hits = index.search(q, top_k=3)
        if not hits:
            print("  [无命中] —— 说明 query 里的词完全不在语料中出现\n")
            continue
        for rank, h in enumerate(hits, 1):
            c = h.chunk
            print(f"  #{rank} score={h.score:6.3f} | {c.cite()} | 【{c.section[:26]}】")
            print(f"      命中词: {', '.join(h.matched_terms[:10])}")
            snippet = c.text[:180].replace("\n", " ")
            print(f"      {snippet}...")
            print()
        print()


if __name__ == "__main__":
    main()
