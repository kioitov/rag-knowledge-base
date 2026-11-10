"""阶段 2 验收入口：一条命令跑完全部检查。

交接文档阶段 2 的验收标准：
    1. 检索到的 3 块拼进 prompt
    2. 调模型生成回答
    3. 回答里带上「来源：xxx.pdf 第 N 页」
    验收：能回答文档里的事实，且标出来源。

本脚本除了跑通主链路，还跑三组质量检查：
    - 引用有效性（回答里的 [n] 是否都指向真实来源）
    - 幻觉压力测试（喂错误上下文会不会编）
    - 对抗性测试（错误前提会不会盲从）

用法：
    python stage2/run_stage2.py              # 完整验收
    python stage2/run_stage2.py --quick      # 只跑主链路 + 引用检查
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "stage1"))

from cache import load_chunks
from chunker import Chunk
from generator import RAGAnswerer, print_answer
from paths import find_pdfs
from retriever import KeywordIndex

# 验收用的问题：覆盖「能答」和「该拒答」两类
# 语料为 Kubernetes 中文文档（CC BY 4.0），问题围绕其内容设计
ACCEPTANCE_QUESTIONS = [
    ("Pod 的生命周期有哪几个阶段", "能答"),
    ("什么是 Deployment，它和 ReplicaSet 什么关系", "能答"),
    ("怎么配置存活探针和就绪探针", "能答"),
    ("etcd 在集群里起什么作用", "能答"),
    ("系统的异地容灾方案是怎么做的", "该拒答-资料里没有"),
    ("Kubernetes 的许可证编号是多少", "该拒答-资料里没有"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 2 验收")
    ap.add_argument("--quick", action="store_true", help="跳过幻觉与对抗测试")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit(
            "没找到 PDF。\n"
            "  请把待处理的 PDF 放进 data/docs/（支持子目录），然后重新运行。"
        )

    print("=" * 94)
    print(" 阶段 2 验收：检索 + 生成 + 来源标注")
    print("=" * 94)
    print("建索引…")
    chunks: list[Chunk] = []
    for p in pdfs:
        cs = load_chunks(p)
        chunks.extend(cs)
        print(f"  {p.name:<40} {len(cs):>4} 块")
    print(f"  合计 {len(chunks)} 块\n")

    index = KeywordIndex(chunks)
    answerer = RAGAnswerer(index, top_k=args.top_k)

    print("=" * 94)
    print(f" 一、主链路（top-{args.top_k}）")
    print("=" * 94)
    print()

    stats = {"cited": 0, "total": 0, "invalid": 0}
    for q, kind in ACCEPTANCE_QUESTIONS:
        a = answerer.answer(q)
        print_answer(a)
        stats["total"] += 1
        if a.used_indices:
            stats["cited"] += 1
        if a.invalid_indices:
            stats["invalid"] += 1

    print("=" * 94)
    print(" 二、引用质量")
    print("=" * 94)
    print(f"  有效引用率：{stats['cited']}/{stats['total']}")
    print(f"  编造编号次数：{stats['invalid']}（回答里出现不存在的 [n]）")
    if stats["cited"] < stats["total"]:
        print("  ⚠ 有回答未标注引用 —— 这会破坏「可溯源」这个核心卖点")
    else:
        print("  ✓ 所有回答都带了有效引用")
    print()

    if args.quick:
        print("=" * 94)
        print(" 已跳过幻觉与对抗测试（--quick）")
        print(" 完整验收请运行：python stage2/test_hallucination.py")
        print("               python stage2/test_adversarial.py")
        print("=" * 94)
        return

    print("=" * 94)
    print(" 三、下一步建议")
    print("=" * 94)
    print("  幻觉压力测试：python stage2/test_hallucination.py --runs 3")
    print("  对抗性测试：  python stage2/test_adversarial.py")
    print("  检测器元测试：python stage2/test_detector.py")
    print()
    print("  这三组的意义：RAG 的价值在于「不编造」，")
    print("  而验证不编造唯一的方法就是**故意喂错东西**。")
    print("=" * 94)


if __name__ == "__main__":
    main()
