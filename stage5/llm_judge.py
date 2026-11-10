"""用 LLM 做语义裁判，修正字符串匹配的假阴性。

============================== 要解决的问题 ==============================

阶段 5 用「答案关键词是否出现在回答里」判定对错，得到 77.5%。
但人工看失败案例，10/27 是同义改写：

    标准答案：30 秒钟        回答：30 秒
    标准答案：使这种解耦成为可能   回答：从而实现前后端的解耦
    标准答案：无需调整       回答：无需再执行存储大小调整

字符串匹配判不了这类，77.5% 是下界。而「核心词覆盖率」那种启发式
又太粗，给出 85.8% 却没有说服力 —— 它依赖一个我随手定的 0.5 阈值。

============================== LLM 裁判的设计 ==============================

用 LLM 判断「这个回答是否实质回答了这个问题」，比字符串匹配准确，
但它自己也可能是错的。所以设计上必须防它乱判：

1. **给裁判看原文依据**，不只是答案关键词
   只给关键词的话，裁判只能判断「措辞像不像」。
   给它原文片段，它才能判断「这个回答是否传达了原文的事实」。

2. **强制三选一，不允许模糊**
   CORRECT / PARTIAL / WRONG，并要求给出理由。

3. **要求裁判指出回答里的具体依据**
   如果它说 CORRECT，必须引用回答里对应的句子。
   这让判定可人工复核，也抑制了随意判对。

4. **温度设为 0**
   裁判要稳定可复现，不需要创造性。

============================== 与字符串判定的关系 ==============================

不是用 LLM 替换字符串判定，而是**两个都跑，看差异**：

    两者都判对      → 确定正确（高置信）
    两者都判错      → 确定错误（高置信）
    LLM 对/字符串错 → 同义改写（正是要找的）
    字符串对/LLM错  → 需人工看（可能 LLM 判错了）

最终报告给三个数字：
    - 严格匹配（可复现，保守）
    - LLM 裁判（更准，但依赖裁判质量）
    - 人工抽查一致率（验证裁判可靠性）

============================== 成本 ==============================

130 条 × 1 次调用 ≈ 15 分钟。
裁判 prompt 比生成短，token 消耗低。

用法：
    python stage5/llm_judge.py                 # 全量
    python stage5/llm_judge.py --limit 20      # 抽样试跑
    python stage5/llm_judge.py --sample-check 15   # 抽 15 条打印明细供人工复核
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config

RESULT_FILE = _ROOT / "data" / "eval" / "results" / "answer.json"
EVAL_FILE = _ROOT / "data" / "eval" / "eval_set.json"
JUDGE_FILE = _ROOT / "data" / "eval" / "results" / "judge.json"


SYSTEM = """你是一个严格的答案判定员。你的任务是判断【模型回答】是否实质回答了【问题】。

判定标准（三选一）：

- CORRECT：回答包含了【标准答案】所表达的核心事实。
  即使措辞不同、语序不同、多了补充说明，只要核心事实一致就算 CORRECT。
  例：标准答案「30 秒钟」，回答「30 秒」→ CORRECT
  例：标准答案「无需调整」，回答「无需再执行存储大小调整」→ CORRECT

- PARTIAL：回答涉及了相关问题，但缺少标准答案的核心事实，或表述不完整。

- WRONG：回答没有包含标准答案的核心事实，或者答非所问、答成了别的对象。

另外判断：
- REFUSED：回答是否属于「拒绝回答」（说资料不足、无法确定、未提及等）

输出 JSON：
{
  "verdict": "CORRECT|PARTIAL|WRONG",
  "refused": true|false,
  "reason": "一句话说明理由",
  "evidence_in_answer": "回答中支撑该判定的原句（若 CORRECT，必须引用）"
}

严格要求：
1. 只依据给定的【标准答案】判断，不要用你自己的领域知识。
2. 如果回答给出了标准答案的核心事实，即使附带了很多额外内容，也算 CORRECT。
3. 如果回答说「资料未提及」但资料里其实有，算 WRONG + refused=true。
4. 只输出 JSON。"""


@dataclass
class JudgeResult:
    item_id: str
    verdict: str
    refused: bool
    reason: str
    evidence_in_answer: str
    string_correct: bool  # 原字符串判定结果
    agree: bool  # 两者是否一致


def judge_one(item: dict, row: dict, client, model: str, retry: int = 2) -> JudgeResult | None:
    """判定一条。"""
    q = item["question"]
    kws = item["answer_keywords"]
    std = "；".join(kws) if kws else "(无标准答案)"
    evidence = item.get("evidence_quote", "")[:400]
    answer = row["answer"][:1200]

    prompt = f"""【问题】
{q}

【标准答案】
{std}

【原文依据】
{evidence or "(未提供)"}

【模型回答】
{answer}"""

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,  # 裁判必须稳定
                max_tokens=400,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            d = json.loads(raw)

            verdict = str(d.get("verdict", "")).upper()
            if verdict not in ("CORRECT", "PARTIAL", "WRONG"):
                return None

            return JudgeResult(
                item_id=item["id"],
                verdict=verdict,
                refused=bool(d.get("refused", False)),
                reason=str(d.get("reason", ""))[:200],
                evidence_in_answer=str(d.get("evidence_in_answer", ""))[:300],
                string_correct=row["correct"],
                agree=(verdict == "CORRECT") == row["correct"],
            )
        except Exception:
            if attempt == retry:
                return None
            time.sleep(1.2 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 语义裁判")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-check", type=int, default=0,
                    help="抽 N 条打印明细，供人工复核裁判质量")
    args = ap.parse_args()

    if not RESULT_FILE.exists():
        raise SystemExit("没有回答结果。先跑：python stage5/run_eval.py --layer answer")

    rows = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}

    # refuse 类也要判，但它们的「正确答案」是拒答，不给标准答案关键词
    targets = rows
    if args.limit:
        targets = rows[: args.limit]

    print("=" * 88)
    print(" LLM 语义裁判")
    print("=" * 88)
    print(f"  待判 {len(targets)} 条")
    print(f"  模型 {config.chat_model()}")
    print(f"  温度 0（裁判需稳定可复现）")

    client = config.get_chat_client()
    model = config.chat_model()

    results: list[JudgeResult] = []
    failed = 0
    t0 = time.perf_counter()

    print()
    for n, row in enumerate(targets, 1):
        item = by_id.get(row["item_id"])
        if not item:
            continue

        r = judge_one(item, row, client, model)
        if r is None:
            failed += 1
            continue
        results.append(r)

        if n % 10 == 0 or n == len(targets):
            el = time.perf_counter() - t0
            flip = sum(1 for x in results if x.verdict == "CORRECT" and not x.string_correct)
            print(f"  [{n:3d}/{len(targets)}] 已判 {len(results)}  "
                  f"推翻字符串判定 {flip}  用时 {el:.0f}s")

    # ---- 汇总 ----
    print()
    print("=" * 88)
    print(" 判定对比")
    print("=" * 88)

    facts = [r for r in results if by_id[r.item_id]["kind"] == "fact"]
    refuses = [r for r in results if by_id[r.item_id]["kind"] == "refuse"]

    s_ok = sum(1 for r in facts if r.string_correct)
    l_ok = sum(1 for r in facts if r.verdict == "CORRECT")
    l_partial = sum(1 for r in facts if r.verdict == "PARTIAL")

    print(f"  fact 类 {len(facts)} 条：")
    print(f"    字符串判定正确   {s_ok:>3}  ({100*s_ok/len(facts):.1f}%)")
    print(f"    LLM 裁判 CORRECT {l_ok:>3}  ({100*l_ok/len(facts):.1f}%)")
    print(f"    LLM 裁判 PARTIAL {l_partial:>3}  ({100*l_partial/len(facts):.1f}%)")

    if refuses:
        r_ok = sum(1 for r in refuses if r.refused)
        print(f"  refuse 类 {len(refuses)} 条：判定为拒答 {r_ok}/{len(refuses)}")

    both_right = sum(1 for r in facts if r.verdict == "CORRECT" and r.string_correct)
    llm_only = sum(1 for r in facts if r.verdict == "CORRECT" and not r.string_correct)
    str_only = sum(1 for r in facts if r.verdict != "CORRECT" and r.string_correct)
    both_wrong = sum(1 for r in facts if r.verdict != "CORRECT" and not r.string_correct)

    print()
    print(f"  两者一致判对      {both_right:>3}")
    print(f"  LLM 推翻字符串判定 {llm_only:>3}   ← 同义改写（原来是假阴性）")
    print(f"  字符串对但 LLM 判错 {str_only:>3}   ← 需人工看，可能裁判错了")
    print(f"  两者一致判错      {both_wrong:>3}")

    if failed:
        print(f"\n  [WARN] {failed} 条判定失败（API 错误），未计入统计")

    # ---- 抽样供人工复核 ----
    if args.sample_check:
        print()
        print("=" * 88)
        print(f" 抽样 {args.sample_check} 条（供人工复核裁判是否可靠）")
        print("=" * 88)
        random.seed(0)
        # 优先展示 LLM 与字符串判定不一致的
        disagree = [r for r in results if not r.agree]
        rest = [r for r in results if r.agree]
        sample = (disagree + random.sample(rest, min(len(rest), max(0, args.sample_check - len(disagree)))))
        sample = sample[: args.sample_check]

        for r in sample:
            item = by_id[r.item_id]
            print(f"\n[{r.item_id}] {item['question'][:64]}")
            print(f"  标准答案：{'；'.join(item['answer_keywords'])[:70]}")
            print(f"  字符串判定：{'对' if r.string_correct else '错'}")
            print(f"  LLM 判定：{r.verdict}（refused={r.refused}）")
            print(f"  理由：{r.reason}")
            if r.evidence_in_answer:
                print(f"  回答依据：{r.evidence_in_answer}")

    # ---- 保存 ----
    JUDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_FILE.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print()
    print(f"  明细已存：{JUDGE_FILE}")


if __name__ == "__main__":
    main()
