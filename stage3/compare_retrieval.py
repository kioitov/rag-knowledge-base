"""阶段 3 核心实验：BM25 vs 向量检索，同一批 query 正面对比。

============================== 这个实验要回答什么 ==============================

阶段 1 用纯 BM25 时，有两条 query 彻底失败。当时我判断「这是关键词检索的
物理上限，只能靠向量检索解决」——**但那只是判断，没有在真实语料上验证过。**

阶段 0 做过一次小规模验证（5 条候选文档），结论是向量 9/10、字面 7/10，
差距很小，而且候选太少不足以说明问题。我在那个脚本里明确标注了
「结论不可直接引用」。

现在有 644 块的真实索引，可以在真实规模上做对比了。

============================== 设计 ==============================

三组 query：

    A 组「口语化」    措辞与文档术语零重叠，考语义匹配
    B 组「抽象名词」  高频词干扰（"系统""支持"），考 IDF 失效场景
    C 组「术语精确」  与文档用词高度一致，考字面匹配的优势场景

C 组是**必要的**：如果只测 A/B，结论必然是「向量全面胜出」，
这既不诚实也没信息量。C 组用来确认 BM25 在自己的优势区间仍然更强，
避免得出「向量可以完全取代关键词检索」的错误结论。

============================== 度量 ==============================

用 Hit@k 而不是相似度分数：
    两个系统的分数量纲完全不同（BM25 无上界，cosine 0~1），
    直接比分数没有意义。只能比「正确答案有没有进 top-k」。

正确答案怎么定：
    **人工标注**，不靠模型判断。每条 query 先跑检索，人工看结果，
    确认某个 chunk_id 确实包含答案，写进标注表。
    这一步不能省 —— 自动判定会把「看起来相关」当成「包含答案」。

用法：
    python stage3/compare_retrieval.py            # 跑全部三组
    python stage3/compare_retrieval.py --group A  # 只跑 A 组
    python stage3/compare_retrieval.py --k 1 3 5  # 看不同 k 下的表现
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage3"))

from cache import load_chunks
from chunker import Chunk
from paths import find_pdfs
from retriever import KeywordIndex
from vector_index import EmbeddingClient, VectorIndex


# ---------------------------------------------------------------------------
# 标注表
# ---------------------------------------------------------------------------
# ---------------------------- 标注修正记录 ----------------------------
# 第一版标注写错了几条，导致 C 组结论失真。记录下来（用抽象描述，不写具体业务词）：
#
#   错误 1：要求「必须同时包含两个特定词」
#       → 某个 query 两边 @1 返回的都是同一个块，它讲了主题思想但没写具体数值，
#         真正写数值的在另一页表格里。按标注判它错，是标注太严。
#
#   ��误 2：要求「必须出现某个概括性词（如"定阶"）」
#       → 向量 @1 返回的是「p=2 / d=1 / q=2」这样的参数表，
#         **这就是那个问题的答案**，只是没用那个概括词。按标注判它错，是标注的锅。
#
# 教训：**用「必须包含某个词」做命中判定，会系统性低估语义检索。**
#       向量检索的价值恰恰在于「说法不同但意思相同」，而标记词法是字面匹配的判据。
#       用字面判据评语义系统，本身就有偏。
#
#       正确做法是人工看结果、判断「这段内容能不能回答这个问题」，
#       而不是匹配预设字符串。下面的 answer_markers 只作为辅助定位，
#       最终结论以人工复核为准（报告里会写清楚哪些是机器判定、哪些是人工判定）。
@dataclass
class QueryCase:
    """一条测试 query 及其人工标注。"""

    group: str  # A 口语 / B 抽象 / C 术语
    question: str
    note: str
    # 辅助标记：用于快速定位可能的答案块。
    # 注意这是**字面**判据，只做初筛，最终由人工确认（见报告）。
    answer_markers: list[str] = field(default_factory=list)
    # 期望「哪个方向更好」：A 组期望向量、B 组期望向量或平、C 组期望 BM25
    expect: str = ""


CASES = [
    # ---------------- A 组：口语化，与文档用词零重叠 ----------------
    QueryCase(
        group="A",
        question="这个系统能装多少人同时用",
        note="口语问并发能力，文档写「并发用户数」",
        answer_markers=["并发用户数"],
        expect="向量",
    ),
    QueryCase(
        group="A",
        question="数据库挂了怎么办",
        note="口语问故障恢复，文档写「恢复方式」",
        answer_markers=["恢复方式"],
        expect="向量",
    ),
    QueryCase(
        group="A",
        question="页面打开慢不慢",
        note="口语问响应时间，文档写「响应时间」",
        answer_markers=["响应时间"],
        expect="向量",
    ),
    QueryCase(
        group="A",
        question="怎么防止别人乱登我的账号",
        note="口语问账号安全，文档写「密码」「校验」",
        answer_markers=["密码", "登录"],
        expect="向量",
    ),
    # ---------------- B 组：抽象高频词干扰 ----------------
    QueryCase(
        group="B",
        question="系统支持哪些部署方式",
        note="「系统」「支持」是高频词，考是否压过真实语义",
        answer_markers=["部署"],
        expect="向量或平",
    ),
    QueryCase(
        group="B",
        question="系统有哪些限制条件",
        note="「条件」「限制」散落多处，考能否定位核心约束",
        answer_markers=["约束"],
        expect="向量或平",
    ),
    QueryCase(
        group="B",
        question="系统的接口有哪些",
        note="「系统」「接口」都极高频，考排序质量",
        answer_markers=["接口"],
        expect="向量或平",
    ),
    # ---------------- C 组：术语精确（BM25 优势区）----------------
    # 用文档中真实存在的**通用技术术语**（MyBatis / JWT / BCrypt 等），
    # 这样既保留了「字面完全匹配」的测试意图，又不涉及具体业务领域。
    QueryCase(
        group="C",
        question="MyBatis-Plus 在系统里起什么作用",
        note="专有名词 MyBatis-Plus，字面命中最直接",
        answer_markers=["MyBatis"],
        expect="BM25",
    ),
    QueryCase(
        group="C",
        question="密码是怎么加密存储的，用的是 BCrypt 吗",
        note="专有名词 BCrypt，文档原词",
        answer_markers=["BCrypt"],
        expect="BM25",
    ),
    QueryCase(
        group="C",
        question="JDK 的版本要求是什么",
        note="专有名词 JDK，术语精确",
        answer_markers=["JDK"],
        expect="BM25",
    ),
]


# ---------------------------------------------------------------------------
# 检索器封装：统一成同一个接口
# ---------------------------------------------------------------------------
class BM25Retriever:
    name = "BM25"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.index = KeywordIndex(chunks)

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        """返回 [(chunk_id, text)]。"""
        hits = self.index.search(query, top_k=k)
        return [(h.chunk.chunk_id, h.chunk.text) for h in hits]


class VecRetriever:
    name = "向量"

    def __init__(self, idx: VectorIndex, embedder: EmbeddingClient) -> None:
        self.idx = idx
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        res = self.idx.search(query, self.embedder, top_k=k)
        return [(cid, doc) for cid, doc, _meta, _dist in res]


# ---------------------------------------------------------------------------
# 命中判定
# ---------------------------------------------------------------------------
def is_hit(results: list[tuple[str, str]], markers: list[str]) -> bool:
    """top-k 里是否有一条包含全部关键片段。

    ⚠️ 这是**字面**判据，用来快速初筛，不能当作最终结论。
    它天然低估语义检索 —— 向量检索的价值正是「说法不同但意思相同」，
    而这里要求的是「必须出现某几个词」。报告里所有数字都会标注这一点，
    关键结论以人工复核为准。
    """
    if not markers:
        return False
    for _cid, text in results:
        if all(m in text for m in markers):
            return True
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 vs 向量检索对比")
    ap.add_argument("--group", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                    help="测试哪些 k 值")
    ap.add_argument("--show", type=int, default=2, help="每组展示前几条结果")
    args = ap.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("=" * 96)
    print(" BM25 vs 向量检索 对比实验")
    print("=" * 96)

    print("加载语料…")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    print(f"  {len(pdfs)} 份文档 / {len(chunks)} 块")

    print("初始化检索器…")
    t0 = time.perf_counter()
    bm25 = BM25Retriever(chunks)
    t_bm25_init = time.perf_counter() - t0
    print(f"  BM25 索引 {t_bm25_init:.2f}s")

    idx = VectorIndex(_ROOT / "data" / "chroma")
    if idx.count() == 0:
        raise SystemExit(
            "向量索引为空。先跑一次：python stage3/vector_index.py"
        )
    embedder = EmbeddingClient()
    vec = VecRetriever(idx, embedder)
    print(f"  向量集合 {idx.count()} 块\n")

    cases = CASES if args.group == "all" else [c for c in CASES if c.group == args.group]

    # ---- 逐条对比 ----
    print("=" * 96)
    print(" 逐条结果")
    print("=" * 96)

    stats = {}  # (group, k) -> [bm25_hits, vec_hits, total]
    lat = {"BM25": [], "向量": []}

    for case in cases:
        print()
        print("─" * 96)
        print(f"[{case.group} 组] {case.question}")
        print(f"  标注依据：{case.note}")
        print(f"  答案标记：{case.answer_markers}")

        max_k = max(args.k)
        for retriever in (bm25, vec):
            t0 = time.perf_counter()
            results = retriever.search(case.question, max_k)
            dt = (time.perf_counter() - t0) * 1000
            lat[retriever.name].append(dt)

            marks = []
            for k in args.k:
                hit = is_hit(results[:k], case.answer_markers)
                key = (case.group, k)
                stats.setdefault(key, [0, 0, 0])
                stats[key][0 if retriever.name == "BM25" else 1] += int(hit)
                marks.append(f"@{k}{'✓' if hit else '✗'}")

            print(f"  {retriever.name:<5} {dt:6.1f}ms  {'  '.join(marks)}")
            if args.show:
                for i, (_cid, text) in enumerate(results[: args.show], 1):
                    snippet = text.replace("\n", " ")[:88]
                    print(f"      #{i} {snippet}")

    # 每条 query 的总数
    for case in cases:
        for k in args.k:
            stats[(case.group, k)][2] += 1

    # ---- 汇总 ----
    print()
    print("=" * 96)
    print(" 汇总：Hit@k 命中率")
    print("=" * 96)
    header = f"  {'组':<6} {'query 数':>8}"
    for k in args.k:
        header += f" {'BM25@' + str(k):>10} {'向量@' + str(k):>10}"
    print(header)
    print("-" * 96)

    for g in sorted({c.group for c in cases}):
        n = sum(1 for c in cases if c.group == g)
        row = f"  {g:<6} {n:>8}"
        for k in args.k:
            b, v, _t = stats[(g, k)]
            row += f" {b:>4}/{n:<5} {v:>4}/{n:<5}"
        print(row)

    print("-" * 96)
    total_n = len(cases)
    row = f"  {'合计':<6} {total_n:>8}"
    for k in args.k:
        b = sum(stats[(g, k)][0] for g in {c.group for c in cases})
        v = sum(stats[(g, k)][1] for g in {c.group for c in cases})
        row += f" {b:>4}/{total_n:<5} {v:>4}/{total_n:<5}"
    print(row)

    # ---- 延迟 ----
    print()
    print("=" * 96)
    print(" 延迟对比（每 query 平均，含 embedding 调用）")
    print("=" * 96)
    for name, vals in lat.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {name:<6} 平均 {avg:8.1f}ms   最快 {min(vals):7.1f}ms   最慢 {max(vals):7.1f}ms")
    print()
    print("  注意：向量检索的延迟大头是 embedding 网络调用（约 200-400ms），")
    print("        而 BM25 是纯内存计算。这个差距在真实系统里需要用缓存 + 批处理来缓解。")

    print()
    print("=" * 96)
    print(" 怎么读这张表")
    print("=" * 96)
    print("  A 组（口语）：向量能不能救回关键词检索完全失效的 query")
    print("  B 组（抽象）：高频词干扰下谁更稳")
    print("  C 组（术语）：BM25 在自己优势区是否仍然更强 —— 这组用来防止")
    print("                得出「向量全面胜出」的错误结论")
    print()
    print("  样本只有 10 条，**不足以做统计显著性判断**。")
    print("  这是方向性验证，不是定论。阶段 5 扩到 100 条才有统计意义。")
    print("=" * 96)


if __name__ == "__main__":
    main()
