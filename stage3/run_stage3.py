"""阶段 3 验收入口：向量索引 + 混合检索 + 三方案对比。

交接文档阶段 3 的验收标准：
    1. 把内存里的块换成 Chroma 持久化存储
    2. 用 FastAPI 包一个 /chat 接口
    3. 打开 /docs 能测试
    验收：curl 或浏览器能问问题并拿到答案

本脚本覆盖 1（Chroma），并跑检索层对比。
FastAPI 接口在 stage3/api.py（下一块）。

用法：
    python stage3/run_stage3.py              # 完整验收
    python stage3/run_stage3.py --rebuild    # 强制重建向量索引
"""
from __future__ import annotations

import argparse
import sys
import time
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
from generator import Hit, RAGAnswerer
from paths import find_pdfs
from retriever import KeywordIndex
from vector_index import EmbeddingClient, VectorIndex


DEMO_QUESTIONS = [
    "这个系统能装多少人同时用",
    "数据库挂了怎么办",
    "JDK 的版本要求是什么",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 3 验收")
    ap.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("=" * 94)
    print(" 阶段 3 验收：向量索引 + 混合检索")
    print("=" * 94)

    print("\n[1/4] 加载语料")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    chunk_map = {c.chunk_id: c for c in chunks}
    print(f"  {len(pdfs)} 份文档 / {len(chunks)} 块")

    print("\n[2/4] 向量索引")
    idx = VectorIndex(_ROOT / "data" / "chroma")
    embedder = EmbeddingClient()
    if args.rebuild or idx.count() == 0:
        stats = idx.build(chunks, embedder, force=True)
        print(f"  {stats.summary()}")
    else:
        print(f"  已存在 {idx.count()} 块（--rebuild 可强制重建）")
    print(f"  存储路径：data/chroma/")

    print("\n[3/4] 混合检索初始化")
    from hybrid_retriever import HybridRetriever

    bm25 = KeywordIndex(chunks)
    hybrid = HybridRetriever(chunks, idx, embedder)
    print(f"  BM25 词表 {len(bm25.inverted)} tokens")
    print(f"  RRF 参数 k={hybrid.rrf_k}，两路等权")
    print("\n[4/4] 三方案对比演示")
    answerer = RAGAnswerer(bm25, top_k=args.top_k)

    for q in DEMO_QUESTIONS:
        print()
        print("=" * 94)
        print(f" 问题：{q}")
        print("=" * 94)

        for name in ("BM25", "向量", "混合"):
            try:
                if name == "BM25":
                    hits = bm25.search(q, top_k=args.top_k)
                    src = f"top1={hits[0].score:.2f}" if hits else "无"
                elif name == "向量":
                    raw = idx.search(q, embedder, top_k=args.top_k)
                    hits = [
                        Hit(chunk=chunk_map[cid], score=1 - d, matched_terms=[])
                        for cid, _doc, _m, d in raw
                        if cid in chunk_map
                    ]
                    src = f"top1 相似度={1 - raw[0][3]:.3f}" if raw else "无"
                else:
                    hr = hybrid.search(q, k=args.top_k)
                    hits = [
                        Hit(chunk=chunk_map[h.chunk_id], score=h.rrf_score, matched_terms=[])
                        for h in hr
                        if h.chunk_id in chunk_map
                    ]
                    src = f"top1 来源={hr[0].sources}" if hr else "无"

                a = answerer.answer(q, hits=hits)
                print(f"\n  【{name}】{src}")
                print(f"  {a.text[:200].replace(chr(10), ' ')}")
                print(f"  引用 {a.used_indices} | {a.latency_ms:.0f}ms")
            except Exception as exc:
                print(f"\n  【{name}】[ERROR] {type(exc).__name__}: {str(exc)[:80]}")

    print()
    print("=" * 94)
    print(" 下一步")
    print("=" * 94)
    print("  详细对比：python stage3/compare_retrieval.py")
    print("  全链路：  python stage3/compare_pipeline.py")
    print("  API 服务：python stage3/api.py（FastAPI + /docs）")
    print("=" * 94)


if __name__ == "__main__":
    main()
