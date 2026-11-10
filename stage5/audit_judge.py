"""核查 LLM 裁判的可信度：逐条检查不一致与一致判错的案例。

============================== 为什么必须做这一步 ==============================

LLM 裁判给出 92.5%，比字符串判定高 15 个百分点。这个跃升必须验证，
否则就是把「换个更宽松的裁判」当成「模型变好了」。

要查两件事：

1. **字符串对但 LLM 判错的 3 条**
   如果这些是裁判错了 → 92.5% 偏低
   如果这些是字符串误判 → 92.5% 可信

2. **两者一致判错的 6 条**
   这些应该是真失败。逐条看是不是真的没答对。

============================== 判定原则 ==============================

我看这些案例时，标准是：**回答是否传达了标准答案的核心事实。**
不要求措辞一致，但要求事实一致。

这是人工复核，结论写进报告。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
RESULT = _ROOT / "data" / "eval" / "results" / "answer.json"
JUDGE = _ROOT / "data" / "eval" / "results" / "judge.json"
EVAL = _ROOT / "data" / "eval" / "eval_set.json"

rows = {r["item_id"]: r for r in json.loads(RESULT.read_text(encoding="utf-8"))}
judges = {j["item_id"]: j for j in json.loads(JUDGE.read_text(encoding="utf-8"))}
items = {i["id"]: i for i in json.loads(EVAL.read_text(encoding="utf-8"))["items"]}

facts = [jid for jid in judges if items.get(jid, {}).get("kind") == "fact"]

# ---- 1. 字符串对但 LLM 判错 ----
print("=" * 92)
print(" 一、字符串判对、但 LLM 判错（可能是裁判错了）")
print("=" * 92)
n = 0
for jid in facts:
    r, j = rows[jid], judges[jid]
    if r["correct"] and j["verdict"] != "CORRECT":
        n += 1
        it = items[jid]
        print(f"\n[{jid}] {it['question'][:66]}")
        print(f"  标准答案：{'；'.join(it['answer_keywords'])[:72]}")
        print(f"  LLM 判定：{j['verdict']}  refused={j['refused']}")
        print(f"  理由：{j['reason'][:140]}")
        print(f"  回答：{r['answer'][:260]}")
print(f"\n  合计 {n} 条")

# ---- 2. 两者一致判错 ----
print()
print("=" * 92)
print(" 二、两者一致判错（应该是真失败）")
print("=" * 92)
n2 = 0
for jid in facts:
    r, j = rows[jid], judges[jid]
    if not r["correct"] and j["verdict"] != "CORRECT":
        n2 += 1
        it = items[jid]
        print(f"\n[{jid}] {it['question'][:66]}")
        print(f"  标准答案：{'；'.join(it['answer_keywords'])[:72]}")
        print(f"  LLM 判定：{j['verdict']}  refused={j['refused']}")
        print(f"  理由：{j['reason'][:140]}")
        print(f"  回答：{r['answer'][:260]}")
print(f"\n  合计 {n2} 条")

# ---- 3. 部分抽看 LLM 推翻的案例，确认是同义改写而非裁判放水 ----
print()
print("=" * 92)
print(" 三、LLM 推翻字符串判定（应确认是同义改写）")
print("=" * 92)
n3 = 0
for jid in facts:
    r, j = rows[jid], judges[jid]
    if not r["correct"] and j["verdict"] == "CORRECT":
        n3 += 1
        if n3 > 6:
            continue
        it = items[jid]
        print(f"\n[{jid}] {it['question'][:62]}")
        print(f"  标准答案：{'；'.join(it['answer_keywords'])[:66]}")
        print(f"  裁判依据：{j['evidence_in_answer'][:110]}")
print(f"\n  合计推翻 {sum(1 for jid in facts if not rows[jid]['correct'] and judges[jid]['verdict']=='CORRECT')} 条")

print()
print("=" * 92)
print(" 汇总")
print("=" * 92)
s_ok = sum(1 for jid in facts if rows[jid]["correct"])
l_ok = sum(1 for jid in facts if judges[jid]["verdict"] == "CORRECT")
print(f"  字符串判定  {s_ok}/{len(facts)}  ({100*s_ok/len(facts):.1f}%)")
print(f"  LLM 裁判    {l_ok}/{len(facts)}  ({100*l_ok/len(facts):.1f}%)")
print()
print("  需人工判断的核心问题：")
print("    - 第一部分那几条，裁判判错了吗？")
print("    - 第三部分那些推翻案例，真的是同义改写吗？")
