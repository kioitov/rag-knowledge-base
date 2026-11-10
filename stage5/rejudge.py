"""用修正后的判定逻辑，重新判定已保存的评测结果。

============================== 为什么这样做 ==============================

生成层跑一次要 11 分钟、130 次 LLM 调用。判定逻辑改了之后重跑一遍纯属浪费。
回答文本已经存在 `data/eval/results/answer.json` 里，
只需要用新判定函数重新算对错即可。

这也是一种验证：如果新判定让正确率从某个值跳到接近 100%，
说明之前的判定确实过严；如果只小幅上升，说明模型本身答得就不够好。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage5"))

from run_eval import answer_contains, is_refusal

RESULT = _ROOT / "data" / "eval" / "results" / "answer.json"
EVAL = _ROOT / "data" / "eval" / "eval_set.json"

rows = json.loads(RESULT.read_text(encoding="utf-8"))

old_ok = sum(1 for r in rows if r["correct"])
print(f"原判定正确数：{old_ok}/{len(rows)}")

# 重新判定
flipped = []
for r in rows:
    text = r["answer"]
    refused = is_refusal(text)

    if r["kind"] == "refuse":
        new = refused
    else:
        kws = r["answer_keywords"]
        new = (not refused) and all(answer_contains(text, k) for k in kws)

    if new != r["correct"]:
        flipped.append((r, r["correct"], new))
    r["correct_old"] = r["correct"]
    r["correct"] = new

new_ok = sum(1 for r in rows if r["correct"])
print(f"新判定正确数：{new_ok}/{len(rows)}")
print(f"判定翻转：{len(flipped)} 条（旧判错→新判对 {sum(1 for _, o, n in flipped if not o and n)}）")

print()
print("=" * 80)
print("翻转案例（旧判错、新判对）—— 验证是否真的是判定过严")
print("=" * 80)
shown = 0
for r, o, n in flipped:
    if not o and n:
        print(f"[{r['item_id']}] {r['question'][:60]}")
        print(f"  期望含：{r['answer_keywords']}")
        print(f"  回答：{r['answer'][:130]}")
        print()
        shown += 1
        if shown >= 6:
            break

# 分类统计
facts = [r for r in rows if r["kind"] == "fact"]
refuses = [r for r in rows if r["kind"] == "refuse"]

print("=" * 80)
print("最终统计")
print("=" * 80)
if facts:
    ok = sum(1 for r in facts if r["correct"])
    ref = sum(1 for r in facts if r["refused"])
    print(f"fact 类 {len(facts)} 条：答案正确 {ok} ({100*ok/len(facts):.1f}%)，误拒答 {ref}")
if refuses:
    ok = sum(1 for r in refuses if r["correct"])
    print(f"refuse 类 {len(refuses)} 条：正确拒答 {ok} ({100*ok/len(refuses):.1f}%)")

# 写回
RESULT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(f"已更新 {RESULT}")
