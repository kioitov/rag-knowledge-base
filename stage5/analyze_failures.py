"""分析剩余的失败案例，归类失败原因。"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = json.load(open("data/eval/results/answer.json", encoding="utf-8"))
facts = [r for r in rows if r["kind"] == "fact"]
fails = [r for r in facts if not r["correct"]]

print(f"fact 失败 {len(fails)}/{len(facts)} 条")
print("=" * 90)

kinds = {"误拒答": [], "答案缺失": [], "答案不完整": []}

for r in fails:
    if r["refused"]:
        kinds["误拒答"].append(r)
    else:
        kws = r["answer_keywords"]
        # 看有几个关键词命中
        from run_eval import answer_contains
        hit_n = sum(1 for k in kws if answer_contains(r["answer"], k))
        if hit_n == 0:
            kinds["答案缺失"].append(r)
        else:
            kinds["答案不完整"].append(r)

for name, items in kinds.items():
    print(f"\n【{name}】{len(items)} 条")
    for r in items[:6]:
        print(f"  [{r['item_id']}] {r['question'][:58]}")
        print(f"      期望含：{r['answer_keywords']}")
        print(f"      回答：{r['answer'][:150]}")
        print()

print("=" * 90)
print("按失败类型汇总：")
for name, items in kinds.items():
    print(f"  {name}: {len(items)}")
