"""交叉验证：用不同模型重跑裁判，看结论是否稳定。

============================== 为什么必须做 ==============================

单一 LLM 裁判给出 96.5%，这个数字有个天然问题：
**裁判和被评测的回答来自同一家模型（qwen-plus）**。
同源模型可能系统性地互相「看得懂」，给出偏高的分数。

排除办法：换一个不同的模型当裁判，比对两者一致性。

============================== 设计 ==============================

    裁判 A：qwen-plus（与生成同源）
    裁判 B：qwen-max 或 qwen-flash（不同规模）

如果两者判定高度一致（>90%）→ 96.5% 可信
如果差异很大 → 说明裁判本身不稳定，需人工判定

============================== 局限 ==============================

两个裁判都来自通义千问系列，共享训练血统，不完全独立。
真正严格的交叉验证需要 GPT / Claude 等异源模型，但本机只有百炼的 key。

这一点必须写进报告 —— 不能把「同族模型互检」说成「独立验证」。

用法：
    python stage5/cross_judge.py --model qwen-max --limit 40
    python stage5/cross_judge.py --model qwen-flash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage5"))

import config
from llm_judge import judge_one

RESULT = _ROOT / "data" / "eval" / "results" / "answer.json"
JUDGE_A = _ROOT / "data" / "eval" / "results" / "judge.json"
EVAL = _ROOT / "data" / "eval" / "eval_set.json"
OUT = _ROOT / "data" / "eval" / "results" / "judge_cross.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="交叉验证裁判")
    ap.add_argument("--model", default="qwen-max", help="第二裁判模型")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 条")
    args = ap.parse_args()

    rows = {r["item_id"]: r for r in json.loads(RESULT.read_text(encoding="utf-8"))}
    judge_a = {j["item_id"]: j for j in json.loads(JUDGE_A.read_text(encoding="utf-8"))}
    items = {i["id"]: i for i in json.loads(EVAL.read_text(encoding="utf-8"))["items"]}

    facts = [jid for jid in judge_a if items.get(jid, {}).get("kind") == "fact"]
    if args.limit:
        facts = facts[: args.limit]

    print("=" * 88)
    print(" 交叉验证：第二裁判")
    print("=" * 88)
    print(f"  裁判 A：{config.chat_model()}（原判定）")
    print(f"  裁判 B：{args.model}")
    print(f"  样本：{len(facts)} 条")

    client = config.get_chat_client()

    agree = 0
    disagree_cases = []
    results = []

    for n, jid in enumerate(facts, 1):
        item = items[jid]
        row = rows[jid]
        r = judge_one(item, row, client, args.model)
        if r is None:
            continue

        a_ok = judge_a[jid]["verdict"] == "CORRECT"
        b_ok = r.verdict == "CORRECT"
        if a_ok == b_ok:
            agree += 1
        else:
            disagree_cases.append((jid, judge_a[jid]["verdict"], r.verdict,
                                   judge_a[jid]["reason"][:80], r.reason[:80]))

        results.append({
            "item_id": jid,
            "judge_a": judge_a[jid]["verdict"],
            "judge_b": r.verdict,
            "agree": a_ok == b_ok,
        })

        if n % 10 == 0 or n == len(facts):
            print(f"  [{n:3d}/{len(facts)}] 一致 {agree}  不一致 {len(disagree_cases)}")

    print()
    print("=" * 88)
    print(" 一致性")
    print("=" * 88)
    print(f"  一致 {agree}/{len(results)}  ({100*agree/max(len(results),1):.1f}%)")

    a_ok_n = sum(1 for jid in facts if judge_a[jid]["verdict"] == "CORRECT")
    b_ok_n = sum(1 for j in results if j["judge_b"] == "CORRECT")
    print(f"  裁判 A 判对  {a_ok_n}/{len(facts)}  ({100*a_ok_n/len(facts):.1f}%)")
    print(f"  裁判 B 判对  {b_ok_n}/{len(results)}  ({100*b_ok_n/max(len(results),1):.1f}%)")

    if disagree_cases:
        print()
        print(f"  不一致案例（{len(disagree_cases)} 条，展示前 6）：")
        for jid, va, vb, ra, rb in disagree_cases[:6]:
            print(f"\n  [{jid}] A={va}  B={vb}")
            print(f"    A 理由：{ra}")
            print(f"    B 理由：{rb}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"  明细已存：{OUT}")
    print()
    print("  ⚠️ 两个裁判都来自通义千问系列，共享训练血统，不完全独立。")
    print("     严格交叉验证需要异源模型（GPT/Claude），本机无对应 key。")
    print("     所以这个一致性只能说明「同族模型下结论稳定」，不能说明「客观正确」。")
    print("=" * 88)


if __name__ == "__main__":
    main()
