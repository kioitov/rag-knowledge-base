"""阶段 3 第二块：混合检索（BM25 + 向量，倒数排名融合 RRF）。

============================== 为什么需要混合 ==============================

阶段 3 的对比实验给出了三组清晰的结果（10 条 query，644 块语料）：

    A 组 口语化   BM25@3 1/4   →  向量@3 3/4    向量大幅领先
    B 组 抽象     BM25@3 3/3   =  向量@3 3/3    打平
    C 组 术语     BM25@1 3/3   →  向量@1 1/3    BM25 大幅领先

**没有单一方法全面胜出。** 这正是混合检索的动机：
两种检索的失败模式不重叠，融合后能互相兜底。

============================== 为什么用 RRF 而不是加权分数 ==============================

BM25 分数无上界（可以是 8，也可以是 80），cosine 相似度固定 0~1。
直接加权相加（如 0.7*bm25 + 0.3*cos）需要先归一化，
而归一化方式会显著影响结果 —— 用 min-max 会让每次查询的分布不同，
用 z-score 又假设了分数服从正态分布，两条都不成立。

RRF（Reciprocal Rank Fusion）只用**排名**，不用分数：
    score(d) = Σ  1 / (k + rank_i(d))
好处：
    1. 天然跨量纲，不需要归一化
    2. 对分数分布不敏感，只关心相对顺序
    3. k=60 是论文推荐的默认值，本语料量级不需要调

局限（要如实说明）：
    RRF 丢弃了分数信息。如果一个系统给出 0.95 相似度、另一个给出 0.35，
    RRF 只看「都是第 1 名」。这在两个检索器质量差异很大时会吃亏。
    改进方向是加权 RRF（给更可靠的检索器更高权重），但权重需要评测集来定，
    现在只有 10 条样本，调权重就是过拟合。

============================== 实现选择 ==============================

Chroma 1.5.9 内置了 Rrf 算子，但没有用它：
    1. 手写才能解释每一步，面试被问「融合怎么做的」能讲清楚
    2. BM25 是自己的实现，本来就不在 Chroma 里，必须手动融合
    3. 内置 Rrf 需要把两路结果都塞进 Chroma，多一层不必要的耦合
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage3"))

from chunker import Chunk
from retriever import KeywordIndex
from vector_index import EmbeddingClient, VectorIndex


@dataclass
class HybridHit:
    """一条混合检索结果。"""

    chunk_id: str
    text: str
    rrf_score: float
    bm25_rank: int | None = None  # None = 该路未召回
    vec_rank: int | None = None
    # 原始分数，仅用于诊断展示，不参与融合
    bm25_score: float = 0.0
    vec_distance: float = 0.0

    @property
    def sources(self) -> str:
        parts = []
        if self.bm25_rank is not None:
            parts.append(f"BM25#{self.bm25_rank}")
        if self.vec_rank is not None:
            parts.append(f"向量#{self.vec_rank}")
        return " + ".join(parts) if parts else "—"

    @property
    def overlap(self) -> bool:
        """两路都召回了吗？都召回的通常是高置信结果。"""
        return self.bm25_rank is not None and self.vec_rank is not None


class HybridRetriever:
    """BM25 + 向量的 RRF 融合。

    参数：
        rrf_k       RRF 公式里的平滑常数，默认 60（论文推荐值）
        recall_n    每路先各召回多少条再融合。取 funal_k 的 3~5 倍比较合适：
                    太小则融合空间不足，太大则引入低质量的尾部结果。
        weights     可选，(bm25_weight, vec_weight)。默认等权。
                    **没有评测集支撑时不要调这个**，那是过拟合。
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vec_index: VectorIndex,
        embedder: EmbeddingClient,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.bm25 = KeywordIndex(chunks)
        self.vec = vec_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.w_bm25, self.w_vec = weights

    def search(self, query: str, k: int = 3, recall_n: int | None = None) -> list[HybridHit]:
        """返回融合后的 top-k。"""
        n = recall_n or max(k * 4, 20)

        # ---- 两路召回 ----
        bm25_hits = self.bm25.search(query, top_k=n)
        bm25_rank: dict[str, int] = {}
        bm25_score: dict[str, float] = {}
        bm25_text: dict[str, str] = {}
        for rank, h in enumerate(bm25_hits, 1):
            cid = h.chunk.chunk_id
            bm25_rank[cid] = rank
            bm25_score[cid] = h.score
            bm25_text[cid] = h.chunk.text

        vec_rows = self.vec.search(query, self.embedder, top_k=n)
        vec_rank: dict[str, int] = {}
        vec_dist: dict[str, float] = {}
        vec_text: dict[str, str] = {}
        for rank, (cid, doc, _meta, dist) in enumerate(vec_rows, 1):
            vec_rank[cid] = rank
            vec_dist[cid] = dist
            vec_text[cid] = doc or ""

        # ---- RRF 融合 ----
        # score(d) = w_bm25 / (k + rank_bm25) + w_vec / (k + rank_vec)
        candidates = set(bm25_rank) | set(vec_rank)
        fused: list[HybridHit] = []
        for cid in candidates:
            s = 0.0
            if cid in bm25_rank:
                s += self.w_bm25 / (self.rrf_k + bm25_rank[cid])
            if cid in vec_rank:
                s += self.w_vec / (self.rrf_k + vec_rank[cid])
            fused.append(
                HybridHit(
                    chunk_id=cid,
                    text=bm25_text.get(cid) or vec_text.get(cid, ""),
                    rrf_score=s,
                    bm25_rank=bm25_rank.get(cid),
                    vec_rank=vec_rank.get(cid),
                    bm25_score=bm25_score.get(cid, 0.0),
                    vec_distance=vec_dist.get(cid, 0.0),
                )
            )

        fused.sort(key=lambda h: h.rrf_score, reverse=True)
        return fused[:k]

    # -- 只走单路的便捷方法（用于对比）--------------------------------
    def search_bm25_only(self, query: str, k: int = 3) -> list[HybridHit]:
        hits = self.bm25.search(query, top_k=k)
        return [
            HybridHit(
                chunk_id=h.chunk.chunk_id, text=h.chunk.text,
                rrf_score=h.score, bm25_rank=i, bm25_score=h.score,
            )
            for i, h in enumerate(hits, 1)
        ]

    def search_vec_only(self, query: str, k: int = 3) -> list[HybridHit]:
        rows = self.vec.search(query, self.embedder, top_k=k)
        return [
            HybridHit(
                chunk_id=cid, text=doc or "",
                rrf_score=1.0 - dist, vec_rank=i, vec_distance=dist,
            )
            for i, (cid, doc, _m, dist) in enumerate(rows, 1)
        ]


if __name__ == "__main__":
    from cache import load_chunks
    from paths import find_pdfs

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))

    idx = VectorIndex(_ROOT / "data" / "chroma")
    if idx.count() == 0:
        raise SystemExit("向量索引为空。先跑：python stage3/vector_index.py")

    hr = HybridRetriever(chunks, idx, EmbeddingClient())

    for q in ["这个系统能装多少人同时用", "JDK 的版本要求是什么"]:
        print("=" * 90)
        print(f"问题：{q}")
        print("=" * 90)
        for i, h in enumerate(hr.search(q, k=3), 1):
            print(f"  #{i} rrf={h.rrf_score:.6f}  [{h.sources}]")
            print(f"     {h.text.replace(chr(10), ' ')[:100]}")
        print()
