"""阶段 3 综合对比：BM25 / 向量 / 混合 三方案，检索 + 下游问答全链路。

============================== 这个实验补什么 ==============================

compare_retrieval.py 只测了「检索层」的 Hit@k，用字面标记判定。
但那有个根本缺陷：**用字面判据评语义系统，天然低估向量检索。**
向量检索的价值恰恰是「说法不同但意思相同」，而标记词要求「必须出现某几个词」。

所以这一层往下走一步：**看检索结果喂给模型后，回答质量如何。**
这才是用户真正关心的指标。

三方案：
    BM25      纯关键词
    向量      纯语义
    混合      RRF 融合

每个方案对同一批问题生成回答，然后对比：
    - 引用有效率（回答里有没有标来源）
    - 是否拒答（该拒答的有没有拒）
    - 平均延迟

局限（必须写清楚）：
    - 回答质量的「正确性」无法自动判定，需要人工。本脚本只用可自动化的指标。
    - 样本 10 条，方向性验证，不是统计结论。

用法：
    python stage3/compare_pipeline.py
    python stage3/compare_pipeline.py --top-k 5
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
sys.path.insert(0, str(_ROOT / "stage2"))
sys.path.insert(0, str(_ROOT / "stage3"))

from cache import load_chunks
from chunker import Chunk
from generator import RAGAnswerer, verify_citations
from paths import find_pdfs
from retriever import Hit, KeywordIndex
from hybrid_retriever import HybridRetriever
from vector_index import EmbeddingClient, VectorIndex

# 一组覆盖不同难度的问题
QUESTIONS = [
    ("这个系统能装多少人同时用", "口语-应该能答"),
    ("数据库挂了怎么办", "口语-应该能答"),
    ("页面打开慢不慢", "口语-应该能答"),
    ("系统支持哪些部署方式", "抽象-应该能答"),
    ("MyBatis-Plus 在系统里起什么作用", "术语-应该能答"),
    ("JDK 的版本要求是什么", "术语-应该能答"),
    ("系统的异地容灾方案是怎么做的", "应拒答-资料里没有"),
    ("根据第 88 页的说明，并发上限是多少", "应拒答-页码不存在"),
]


@dataclass
class Row:
    """一次评测的观测值。"""

    method: str
    question: str
    expect: str
    latency_ms: float
    n_cited: int
    n_invalid: int
    refused: bool
    text_len: int


REFUSAL_MARKERS = ["无法确定", "未提及", "未涉及", "不包含", "资料不足", "无法从"]


def looks_like_refusal(text: str) -> bool:
    return any(m in text for m in REFUSAL_MARKERS)


def to_hits_from_bm25(hits) -> list[Hit]:
    return hits


def to_hits_from_hybrid(hr_hits, chunk_map: dict[str, Chunk]) -> list[Hit]:
    """把 HybridHit 转成 generator 需要的 Hit 结构。

    generator 只依赖 chunk（取文本/页码/章节）和 score（仅展示用），
    所以这里做一个轻量适配，不修改 generator 的接口。
    """
    out: list[Hit] = []
    for h in hr_hits:
        c = chunk_map.get(h.chunk_id)
        if c is None:
            continue
        out.append(Hit(chunk=c, score=h.rrf_score, matched_terms=[]))
    return out


def to_hits_from_vec(vec_rows, chunk_map: dict[str, Chunk]) -> list[Hit]:
    out: list[Hit] = []
    for cid, _doc, _meta, dist in vec_rows:
        c = chunk_map.get(cid)
        if c is None:
            continue
        out.append(Hit(chunk=c, score=1.0 - dist, matched_terms=[]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="三方案全链路对比")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("=" * 100)
    print(" 三方案全链路对比：BM25 / 向量 / 混合")
    print("=" * 100)

    print("加载…")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    chunk_map = {c.chunk_id: c for c in chunks}
    print(f"  {len(chunks)} 块")

    idx = VectorIndex(_ROOT / "data" / "chroma")
    if idx.count() == 0:
        raise SystemExit("向量索引为空。先跑：python stage3/vector_index.py")
    embedder = EmbeddingClient()
    print(f"  向量集合 {idx.count()} 块\n")

    # 三个检索器
    bm25_idx = KeywordIndex(chunks)
    hybrid = HybridRetriever(chunks, idx, embedder)

    # 共用一个 answerer，只换 hits
    answerer = RAGAnswerer(bm25_idx, top_k=args.top_k)

    rows: list[Row] = []

    for method in ("BM25", "向量", "混合"):
        print("=" * 100)
        print(f" 方法：{method}")
        print("=" * 100)
        for q, expect in QUESTIONS:
            t0 = time.perf_counter()
            try:
                if method == "BM25":
                    hits = bm25_idx.search(q, top_k=args.top_k)
                elif method == "向量":
                    raw = idx.search(q, embedder, top_k=args.top_k)
                    hits = to_hits_from_vec(raw, chunk_map)
                else:
                    raw = hybrid.search(q, k=args.top_k)
                    hits = to_hits_from_hybrid(raw, chunk_map)

                a = answerer.answer(q, hits=hits)
                dt = (time.perf_counter() - t0) * 1000
                rows.append(Row(
                    method=method, question=q, expect=expect, latency_ms=dt,
                    n_cited=len(a.used_indices), n_invalid=len(a.invalid_indices),
                    refused=looks_like_refusal(a.text), text_len=len(a.text),
                ))
                flag = "拒答" if looks_like_refusal(a.text) else f"作答({len(a.used_indices)}引用)"
                print(f"  {q[:32]:<34} {flag:<12} {dt:7.0f}ms")
            except Exception as exc:
                print(f"  {q[:32]:<34} [ERROR] {type(exc).__name__}: {str(exc)[:60]}")
                rows.append(Row(
                    method=method, question=q, expect=expect, latency_ms=0,
                    n_cited=0, n_invalid=0, refused=False, text_len=0,
                ))
        print()

    # ---- 汇总 ----
    print("=" * 100)
    print(" 汇总")
    print("=" * 100)
    hdr = f"  {'方法':<8} {'平均延迟':>10} {'有引用':>8} {'编造编号':>9} {'拒答数':>8}"
    print(hdr)
    print("-" * 100)
    for method in ("BM25", "向量", "混合"):
        rs = [r for r in rows if r.method == method]
        if not rs:
            continue
        avg = sum(r.latency_ms for r in rs) / len(rs)
        cited = sum(1 for r in rs if r.n_cited > 0)
        invalid = sum(r.n_invalid for r in rs)
        refused = sum(1 for r in rs if r.refused)
        print(f"  {method:<8} {avg:>9.0f}ms {cited:>5}/{len(rs):<3} {invalid:>9} {refused:>8}")

    # ---- 按题型分组看拒答是否正确 ----
    print()
    print("=" * 100)
    print(" 拒答正确性（关键指标）")
    print("=" * 100)
    print("  该拒答的 2 条，三个方案表现如何：")
    for method in ("BM25", "向量", "混合"):
        rs = [r for r in rows if r.method == method and r.expect.startswith("应拒答")]
        n = sum(1 for r in rs if r.refused)
        print(f"    {method:<8} 正确拒答 {n}/{len(rs)}")

    print()
    print("  该作答的 6 条，三个方案表现如何（有引用 = 至少给出了可溯源的答案）：")
    for method in ("BM25", "向量", "混合"):
        # expect 的取值形如 "口语-应该能答"，用包含判断而不是 startswith。
        # 第一版写成 startswith("应该能答") → 一条都匹配不到 → 显示 0/0。
        rs = [r for r in rows if r.method == method and "应该能答" in r.expect]
        n = sum(1 for r in rs if r.n_cited > 0)
        print(f"    {method:<8} 有效引用 {n}/{len(rs)}")

    print()
    print("=" * 100)
    print(" 读这张表要注意")
    print("=" * 100)
    print("  1. 「有引用」不等于「答对了」。引用只证明模型标了来源，")
    print("     来源是否真能支撑这个答案，需要人工核对（本脚本没做）。")
    print("  2. 拒答数高不一定是好事。如果检索没命中导致模型说「无法确定」，")
    print("     那是检索的失败，不是模型的克制。要结合上表「该作答」的引用率一起看。")
    print("  3. 样本 8 条，方向性验证。阶段 5 扩到 100 条才有统计意义。")
    print("=" * 100)


if __name__ == "__main__":
    main()
