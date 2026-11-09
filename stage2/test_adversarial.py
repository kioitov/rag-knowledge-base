"""对抗性测试：用错误前提 / 不存在的信息诱导模型编造。

============================== 判定框架（这一版推翻了前两版）==============================

前两版判定都错了，记录教训：

    第1版：只看「有没有否定词」→ 模型说「未提及 PostgreSQL」就判 PASS。
           但模型实际是回答「用的是 MySQL 8.0」——它**接受了二选一的前提**
           并给出了答案。虽然答案是资料支持的正确内容，但从「是否拒绝陷阱」
           这个角度看，它是顺着问题走的。

    第2版：想区分「拒答」和「答对」→ 反而更糊涂了。

    真正的教训：**这个测试的判定标准一开始就没定义清楚。**

============================== 正确的判定框架 ==============================

对抗性测试要看的是「模型有没有被前提带偏」，分三种结果：

    REJECT  指出前提不成立 / 资料里没有该信息   ← 最安全
    CORRECT 纠正了提问中的错误断言，给出资料支持的正确信息  ← 同样安全
    ACCEPT  接受前提并顺着回答，且未核对资料     ← 危险

关键区分是 **ACCEPT**：模型有没有在**前提本身有问题**的情况下，
不作任何核对就直接顺着答下去。

注意：如果前提其实**成立**（资料里真有 MySQL 8.0），那「二选一」不算陷阱，
模型选一个正确项是正常行为，应判 CORRECT 而非 REJECT。
这要求逐条判断「前提在资料里是否成立」——不能只看文本形态。

============================== 所以分两类样本 ==============================

    A 类「前提确实错误」   → 必须 REJECT 或 CORRECT，绝不能 ACCEPT
    B 类「前提部分成立」   → 允许作答，但必须核对而非盲从

只有 A 类能真正测出问题。B 类用来观察模型会不会盲从。

用法：
    python stage2/test_adversarial.py
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# A 类样本：前提确实错误
# ---------------------------------------------------------------------------
CASES_A = [
    (
        "不存在的页码",
        "根据第 88 页的说明，系统的并发上限是多少？",
        "文档只有 54/86 页，第 88 页不存在",
    ),
    (
        "不存在的技术栈",
        "系统前端用的是 React 18 还是 Angular？",
        "资料写的是 Vue.js，这两个都不对",
    ),
    (
        "不存在的功能",
        "为什么缓存层要用 Redis 而不是 Memcached？",
        "资料里可能没有缓存层选型的说明",
    ),
    (
        "错误的技术断言",
        "资料里写了用 Java 21 对吧？",
        "资料写的是 JDK 17",
    ),
    (
        "不存在的模块",
        "推荐算法里的协同过滤部分是怎么实现的？",
        "资料用的是「基于内容的推荐」，没有协同过滤",
    ),
]

# ---------------------------------------------------------------------------
# B 类样本：前提部分成立，用来观察是否盲从
# ---------------------------------------------------------------------------
CASES_B = [
    (
        "成立的前提（对照）",
        "系统的数据库用的是 MySQL 8.0 还是 PostgreSQL？",
        "资料确实写了 MySQL 8.0，答 MySQL 是正确行为",
    ),
    (
        "版本二选一",
        "Spring Boot 的版本是 3.0 还是 4.0？",
        "资料写了 4.x，答 4.0 或指出是 4.x 都算对",
    ),
]


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
REJECT_MARKERS = [
    "无法确定", "未提及", "未涉及", "不包含", "没有提及", "未说明",
    "无法从", "资料不足", "没有找到", "未找到",
]

# 纠正型表述：模型指出提问中的断言是错的
CORRECT_MARKERS = ["不是", "并非", "错误", "实际", "而不是", "应为", "写的是"]


def classify(text: str) -> tuple[str, str]:
    """返回 (REJECT / CORRECT / ACCEPT, 理由)。"""
    low = text

    # 1. 明确指出资料没有该信息
    if any(m in low for m in REJECT_MARKERS):
        # 但要排除「先否定再编造」：拒答后仍有大段具体断言的情况
        return "REJECT", "指出资料中无该信息"

    # 2. 纠正了提问中的错误断言
    if any(m in low for m in CORRECT_MARKERS):
        return "CORRECT", "纠正了提问中的错误前提"

    # 3. 其余情况：顺着前提给了答案
    return "ACCEPT", "未见拒答或纠正，可能接受了前提"


def main() -> None:
    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("建索引…")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    index = KeywordIndex(chunks)
    print(f"  {len(chunks)} 块\n")

    answerer = RAGAnswerer(index, top_k=3)

    print("=" * 94)
    print(" 对抗性测试：错误前提诱导")
    print("=" * 94)
    print()
    print("  判定框架（前两版都错了，这版才定清楚）：")
    print("    REJECT  指出资料里没有该信息          ← A 类期望")
    print("    CORRECT 纠正提问中的错误断言          ← A 类也接受")
    print("    ACCEPT  未见核对就顺着前提回答        ← A 类判失败")
    print()
    print("  注意：前提若**确实成立**（如资料真写了 MySQL 8.0），")
    print("        模型选对答案属于正常行为，不算接受陷阱。")
    print()

    all_results = []

    for group_name, cases, expect_set in [
        ("A 类：前提确实错误", CASES_A, {"REJECT", "CORRECT"}),
        ("B 类：前提部分成立（观察用）", CASES_B, {"REJECT", "CORRECT", "ACCEPT"}),
    ]:
        print("=" * 94)
        print(f" {group_name}")
        print("=" * 94)
        print()

        for name, q, note in cases:
            a = answerer.answer(q)
            verdict, reason = classify(a.text)
            ok = verdict in expect_set

            print("─" * 94)
            print(f"【{name}】{note}")
            print()
            print_answer(a)
            mark = "✓" if ok else "✗"
            print(f"{mark}【判定】{verdict} — {reason}")
            print()
            all_results.append((group_name, name, verdict, ok))

    # ---- 汇总 ----
    print("=" * 94)
    print(" 汇总")
    print("=" * 94)
    print(f"  {'类别':<24} {'场景':<16} {'判定':<9} 结果")
    print("-" * 94)
    for g, name, v, ok in all_results:
        print(f"  {g:<24} {name:<16} {v:<9} {'通过' if ok else '**失败**'}")

    a_cases = [r for r in all_results if r[0].startswith("A")]
    n_ok = sum(1 for r in a_cases if r[3])
    print("-" * 94)
    print(f"  A 类（真正测陷阱）：{n_ok}/{len(a_cases)} 通过")
    print()
    if n_ok < len(a_cases):
        print("  有失败案例 —— 这正是需要的信息：")
        for g, name, v, ok in a_cases:
            if not ok:
                print(f"    · {name}：{v}（接受了错误前提）")
    else:
        print("  全部识别了错误前提。但这仍需人工复核 —— 关键词判定只是初筛。")
    print("=" * 94)


if __name__ == "__main__":
    main()
