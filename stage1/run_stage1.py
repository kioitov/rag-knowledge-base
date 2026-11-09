"""阶段 1 验收脚本：读文档 → 切块 → 检索 → 打印。

一条命令跑完交接文档阶段 1 的全部验收点：
    1. 用 pymupdf 读 PDF
    2. 按固定长度切块（三种策略可选）
    3. 把块打印出来（人眼看切得对不对）
    4. 输入问题，在几百个块里找到最相关的 3 块

用法：
    python stage1\\run_stage1.py                     # 默认策略 + 5 个内置问题
    python stage1\\run_stage1.py -q "还款方式有哪些"
    python stage1\\run_stage1.py --strategy section
    python stage1\\run_stage1.py --dump 10           # 打印前 10 个块，人工验切分
    python stage1\\run_stage1.py --all-docs          # 4 份 PDF 全建索引
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import load_chunks
from retriever import KeywordIndex

DOC_DIR = Path(r"E:\deepseek work\融销通项目\成果文档")

DOCS = [
    DOC_DIR / "可行性研究报告" / "融销通平台可行性研究报告.pdf",
    DOC_DIR / "软件需求规格说明书" / "融销通平台软件需求规格说明书.pdf",
    DOC_DIR / "系统概要设计说明书" / "融销通平台系统概要设计说明书.pdf",
    DOC_DIR / "系统详细设计文档" / "融销通平台系统详细设计文档.pdf",
]


def build_index(docs: list[Path], strategy: str, size: int, overlap: int,
                use_cache: bool = True) -> KeywordIndex:
    all_chunks = []
    t0 = time.perf_counter()
    for p in docs:
        if not p.exists():
            print(f"  [MISS] {p.name}")
            continue
        cs = load_chunks(p, strategy=strategy, size=size, overlap=overlap,
                         use_cache=use_cache)
        all_chunks.extend(cs)
        print(f"  {p.name:<40} {len(cs):>4} 块  "
              f"{sum(len(c.text) for c in cs):>7,} 字")
    dt = time.perf_counter() - t0
    print(f"  {'合计':<40} {len(all_chunks):>4} 块   耗时 {dt:.1f}s")
    return KeywordIndex(all_chunks)


def show_hits(query: str, index: KeywordIndex, top_k: int = 3, snippet: int = 200) -> None:
    print("=" * 94)
    print(f"问题：{query}")
    print("=" * 94)
    t0 = time.perf_counter()
    hits = index.search(query, top_k=top_k)
    dt = (time.perf_counter() - t0) * 1000

    if not hits:
        print(f"  [无命中] 耗时 {dt:.1f}ms")
        print("  → 说明 query 里的词在语料中完全不存在。真实场景要记录这类问题，")
        print("    它是最直接的「检索失败」证据，阶段 5 评测集要覆盖。")
        print()
        return

    print(f"  {len(hits)} 条命中，检索耗时 {dt:.1f}ms\n")
    for rank, h in enumerate(hits, 1):
        c = h.chunk
        print(f"  #{rank}  score={h.score:7.3f}  {c.cite()}  〔{c.section[:30]}〕  {len(c)}字")
        print(f"       命中词：{', '.join(h.matched_terms[:12])}")
        text = c.text.replace("\n", " ")
        print(f"       {text[:snippet]}{'...' if len(text) > snippet else ''}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 1 验收：切块 + 检索")
    ap.add_argument("-q", "--query", action="append", help="要问的问题，可重复")
    ap.add_argument("--strategy", default="paragraph", choices=["fixed", "paragraph", "section"])
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dump", type=int, default=0, help="打印前 N 个块原文，人工验切分")
    ap.add_argument("--all-docs", action="store_true", help="用全部 4 份 PDF 建索引")
    ap.add_argument("--no-cache", action="store_true", help="禁用解析缓存")
    args = ap.parse_args()

    docs = DOCS if args.all_docs else [DOCS[1]]  # 默认只用需求规格说明书，快

    print("=" * 94)
    print(f" 阶段 1 · 切块 + 检索验收")
    print(f" 策略 {args.strategy}  size={args.size}  overlap={args.overlap}  "
          f"文档 {len(docs)} 份")
    print("=" * 94)
    print("建索引：")
    index = build_index(docs, args.strategy, args.size, args.overlap,
                        use_cache=not args.no_cache)
    print(f"词表规模：{len(index.inverted)} tokens")

    if args.dump:
        print("\n" + "=" * 94)
        print(f" 前 {args.dump} 个块原文（人工验收切分质量）")
        print("=" * 94)
        for i, c in enumerate(index.chunks[: args.dump]):
            print(f"\n[块 {i + 1}] {len(c)}字 | {c.cite()} | 〔{c.section}〕 | {c.kind}")
            print("-" * 94)
            print(c.text)
        return

    queries = args.query or [
        "联合贷款人是怎么推荐的？权重怎么算",
        "农户提交融资意向需要填哪些字段",
        "系统支持哪些还款方式",
        "银行用户的审核流程是什么",
        "ARIMA 价格预测用在哪里",
    ]
    print()
    for q in queries:
        show_hits(q, index, top_k=args.top_k)

    print("=" * 94)
    print(" 阶段 1 验收完成。下一步（阶段 2）：把命中的块拼进 prompt，调大模型生成带来源的回答。")
    print("=" * 94)


if __name__ == "__main__":
    main()
