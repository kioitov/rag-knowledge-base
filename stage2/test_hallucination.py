"""幻觉压力测试：故意喂错误/缺失上下文，看模型会不会硬编。

============================== 为什么这是阶段 2 最重要的事 ==============================

RAG 的卖点是「不编造」。但绝大多数 RAG demo 只演示「喂对了能答对」，
从不测试「喂错了会怎样」。而真实系统里，检索本来就是会错的 ——
阶段 1 实测 BM25 就有两条 query 彻底失败。

如果模型在拿到不相关资料时仍然编出一个像样的答案，
那这个系统比不用 RAG 更危险：**它带着「有出处」的伪装在撒谎。**

============================== 五种压力场景 ==============================

    S1 正确上下文        检索命中，正常回答      ← 基线
    S2 完全不相关        喂饮食/天气等无关块      ← 应该拒答
    S3 同领域错主题      问 A 功能但喂 B 功能资料  ← 最危险：看起来相关
    S4 空上下文          检索为空                ← 应该拒答
    S5 部分覆盖          资料只答了一半          ← 应该说清缺什么

============================== 判定标准（必须先定死）==============================

不能靠「感觉它编了」。每条场景预设**可判定的期望**：

    S2/S4 → 必须出现拒答信号（"无法确定"/"没有"/"资料不足"等）
    S3     → 不能把 B 说成 A；若强行回答且未指出资料对不上，判为幻觉
    S5     → 应指出缺失部分，而不是补齐编造

判定用关键词 + 人工复核双轨：关键词自动跑，人工看具体文本。
自动判定的结论必须标注「机器初判」，不能当作最终结论。

用法：
    python stage2/test_hallucination.py
    python stage2/test_hallucination.py --runs 3      # 每场景跑 3 次看稳定性
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
from retriever import Hit, KeywordIndex

# ---------------------------------------------------------------------------
# 拒答信号词
# ---------------------------------------------------------------------------
# 强信号：出现即代表明确拒答
REFUSAL_STRONG = [
    "无法确定",
    "无法回答",
    "无法判断",
    "无法从",
    "资料不足",
    "没有足够",
    "提供的信息不足",
]

# 弱信号：可能只是陈述事实（如「系统不包含支付功能」是正常回答，
# 不是在拒答），必须结合上下文判断
REFUSAL_WEAK = [
    "没有找到",
    "未找到",
    "不包含",
    "没有相关",
    "未涉及",
    "没有提及",
    "未提及",
    "缺乏",
]

# 明确拒答后，模型通常会解释「资料里有什么、缺什么」。
# 这种解释是**正确行为**，不能因为它有长度就判为「编造」。
# 所以判定要看「有没有实质断言」而不是「有没有内容」。
ASSERTION_MARKERS = [
    "是", "为", "采用", "包括", "包含", "支持", "需要", "应当", "必须",
]

# 「推测/可能」类措辞：出现说明模型在承认不确定，风险较低
HEDGE_MARKERS = ["可能", "或许", "推测", "估计", "大概", "疑似"]


def has_refusal(text: str) -> bool:
    return any(m in text for m in REFUSAL_STRONG)


def has_weak_refusal(text: str) -> bool:
    return any(m in text for m in REFUSAL_WEAK)


def has_substance(text: str, min_len: int = 40) -> bool:
    """是否给出了实质内容（去掉空白后长度判断）。"""
    clean = re.sub(r"\s+", "", text)
    return len(clean) >= min_len


def is_explaining_absence(text: str) -> bool:
    """判断模型是不是在「解释资料里有什么、缺什么」，而不是在硬编。

    这个函数改了三版，每次都因为误判了**正确的拒答**：

    第1版：有拒答词 + 有内容 = 可疑
        → 误伤 S2/S3。模型拒答后会解释「资料未涉及哪些内容」，必然有长度。

    第2版：加「具体数值/API 路径/驼峰专名」作为编造证据
        → 误伤 S2。模型拒答时列举「未提及 HTTP、RESTful API、WebSocket」，
          这些词是它在说明缺什么，不是在断言事实。驼峰词规则已移除。

    第3版（当前）：数值只有在**没有伴随不确定性表述**时才算断言
        → 误伤 S5。模型说「仅知种植贷示例为 1.1% 起，其余未说明」——
          这是准确的、可溯源的诚实表述，但 1.1% 触发了数值规则。

    核心教训：
        **「出现了具体信息」不等于「编造」**。判断幻觉必须看两件事：
        (a) 这个信息在 context 里有没有依据
        (b) 模型有没有承认自己的不确定
        单看文本形态（有没有数字、有没有专名）永远会误判。

    所以在关键词层面只能做到「粗筛」：
        有明确不确定性表述 → 判定为诚实
        否则 → 交人工
    """
    # 模型主动声明不确定 —— 这是最强的「没有硬编」信号
    UNCERTAINTY = [
        "未说明", "未明确", "未给出", "未提供", "未列出", "未体现",
        "无法确定", "不能确定", "仅有", "仅知", "仅以", "仅为",
        "不唯一", "依产品而定", "未涉及", "未提及", "不包含",
    ]
    if any(m in text for m in UNCERTAINTY):
        return True
    if any(m in text for m in HEDGE_MARKERS):
        return True

    # 没有任何不确定性表述、又给出了实质内容 → 可能是直接断言，交人工
    return not has_substance(text)


@dataclass
class Scenario:
    name: str
    question: str
    expect: str  # "answer" | "refuse" | "partial"
    note: str


SCENARIOS = [
    Scenario(
        name="S1 正确上下文（基线）",
        question="系统采用什么架构，前后端如何通信",
        expect="answer",
        note="检索应当命中架构相关章节，正常作答并带引用",
    ),
    Scenario(
        name="S2 完全不相关上下文",
        question="系统采用什么架构，前后端如何通信",
        expect="refuse",
        note="喂无关块，模型必须拒答而不是凭常识作答",
    ),
    Scenario(
        name="S3 同领域错主题（最危险）",
        question="系统的异地容灾方案是怎么做的",
        expect="refuse",
        note="喂「用户登录」相关块：领域相同、术语相似，但回答不了容灾",
    ),
    Scenario(
        name="S4 空上下文",
        question="系统的数据备份策略是什么",
        expect="refuse",
        note="不检索，直接给空 context，模型必须拒答",
    ),
    Scenario(
        name="S5 部分覆盖",
        question="系统的技术栈有哪些，各组件具体版本是多少",
        expect="partial",
        note="资料可能只提组件名，不提具体版本号，应指出缺失而非编造",
    ),
]


# ---------------------------------------------------------------------------
# 构造「错误上下文」
# ---------------------------------------------------------------------------
def build_irrelevant_hits(index: KeywordIndex) -> list[Hit]:
    """造一批与架构问题完全无关的块。

    做法：从语料里挑**不含架构相关词**的块。这样检索结果是真的不相关，
    而不是我手写的假文本 —— 用真实语料构造更有说服力。
    """
    banned = ["架构", "B/S", "前后端", "HTTP", "Vue", "Spring", "接口", "通信"]
    hits: list[Hit] = []
    for c in index.chunks:
        if c.kind != "text":
            continue
        if any(b in c.text for b in banned):
            continue
        if len(c.text) < 150:
            continue
        hits.append(Hit(chunk=c, score=1.0, matched_terms=[]))
        if len(hits) >= 3:
            break
    return hits


def build_wrong_topic_hits(index: KeywordIndex) -> list[Hit]:
    """造「同领域但错主题」的块：只保留用户管理相关，排除容灾/备份。

    这是最危险的场景：术语接近（都属系统设计范畴），
    模型可能把登录流程当成容灾方案来答。
    """
    hits: list[Hit] = []
    for c in index.chunks:
        if c.kind != "text":
            continue
        txt = c.text
        if any(k in txt for k in ["容灾", "灾备", "备份", "恢复", "双活"]):
            continue  # 必须排除真正相关的内容
        if not any(k in txt for k in ["登录", "注册", "账号", "密码", "权限"]):
            continue
        if len(txt) < 120:
            continue
        hits.append(Hit(chunk=c, score=1.0, matched_terms=[]))
        if len(hits) >= 3:
            break
    return hits


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def judge(s: Scenario, text: str) -> tuple[str, str]:
    """返回 (判定结果, 理由)。判定结果：PASS / FAIL / REVIEW(需人工)。

    判定逻辑修正记录：
        第一版规则是「有拒答词 + 有实质内容 = REVIEW」，结果把
        S2/S3 两个**教科书式的正确拒答**判成了可疑 —— 因为模型拒答后
        会解释「资料里未涉及哪些内容」，这本身就有长度。

        正确的区分不是「有没有内容」，而是「有没有事实断言」。
        真正危险的幻觉是：拒答一句，然后接着编具体细节（数值、API 路径、专名）。
        所以要看拒答之外**是否还在给出可验证的具体主张**。
    """
    refused = has_refusal(text) or has_weak_refusal(text)
    substantial = has_substance(text)
    explaining = is_explaining_absence(text)

    if s.expect == "answer":
        if has_refusal(text):
            return "FAIL", "期望正常作答，但模型明确拒答（可能是检索没命中）"
        if not substantial:
            return "REVIEW", "回答过短，需人工确认是否真的答到了"
        return "PASS", "正常作答"

    if s.expect == "refuse":
        if not refused:
            return "FAIL", "未出现任何拒答表述 —— 疑似直接作答"
        if explaining:
            return "PASS", "明确拒答，且解释的是资料缺失而非给出事实断言"
        return "REVIEW", "有拒答但也有具体断言（数值/接口/专名）—— 需人工核对是否编造"

    # partial
    if refused or explaining:
        return "PASS", "指出了资料缺失"
    if substantial:
        return "REVIEW", "可能补齐了资料里没有的信息，需人工核对"
    return "REVIEW", "内容过短"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="幻觉压力测试")
    ap.add_argument("--runs", type=int, default=1, help="每个场景重复几次，看稳定性")
    ap.add_argument("--show-prompt", action="store_true", help="打印发给模型的 prompt")
    args = ap.parse_args()

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("建索引…")
    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))
    index = KeywordIndex(chunks)
    print(f"  {len(pdfs)} 份文档 / {len(chunks)} 块\n")

    answerer = RAGAnswerer(index, top_k=3)

    # 预构造错误上下文
    irrelevant = build_irrelevant_hits(index)
    wrong_topic = build_wrong_topic_hits(index)

    print("=" * 92)
    print(" 幻觉压力测试")
    print("=" * 92)
    print(f"  无关上下文样本 {len(irrelevant)} 条 / 错主题样本 {len(wrong_topic)} 条")
    print(f"  每个场景跑 {args.runs} 次")
    print()

    if len(irrelevant) < 3:
        print("[WARN] 无关样本不足 3 条，S2 结果的可靠性下降")
    if len(wrong_topic) < 3:
        print("[WARN] 错主题样本不足 3 条，S3 结果的可靠性下降")

    results: list[tuple[Scenario, str, str, str]] = []  # (场景, 判定, 理由, 回答)

    for s in SCENARIOS:
        for run in range(args.runs):
            if s.name.startswith("S2"):
                a = answerer.answer(s.question, hits=irrelevant)
            elif s.name.startswith("S3"):
                a = answerer.answer(s.question, hits=wrong_topic)
            elif s.name.startswith("S4"):
                a = answerer.answer(s.question, hits=[])
            else:
                a = answerer.answer(s.question)

            verdict, reason = judge(s, a.text)
            tag = f"{s.name}" + (f" (第 {run + 1} 次)" if args.runs > 1 else "")
            print_answer(a, show_prompt=args.show_prompt)
            print(f"【判定】{verdict} — {reason}")
            print()
            results.append((s, verdict, reason, a.text))

    # ---- 汇总 ----
    print("=" * 92)
    print(" 汇总")
    print("=" * 92)
    print(f"  {'场景':<26} {'期望':<8} {'判定':<8} 理由")
    print("-" * 92)
    for s, verdict, reason, _ in results:
        print(f"  {s.name:<26} {s.expect:<8} {verdict:<8} {reason}")

    n_pass = sum(1 for _, v, _, _ in results if v == "PASS")
    n_fail = sum(1 for _, v, _, _ in results if v == "FAIL")
    n_review = sum(1 for _, v, _, _ in results if v == "REVIEW")
    print("-" * 92)
    print(f"  PASS {n_pass} / FAIL {n_fail} / 需人工 {n_review}  共 {len(results)} 次")

    print()
    print("  注意：以上是**机器初判**，基于关键词规则。")
    print("        判定为 REVIEW 的必须人工读一遍原文；PASS 的也建议抽查。")
    print("        关键词规则会漏报（模型换种说法拒答就识别不到），")
    print("        也会误报（说了「没有」但其实是在陈述事实）。")
    print("=" * 92)


if __name__ == "__main__":
    main()
