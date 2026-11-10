"""剔除有缺陷的评测项，重算最终准确率。

============================== 为什么要剔除 ==============================

人工复核 6 条「一致判错」的案例后，发现三类：

    A. 模型真失败（3 条）
       E039 没答出 CGroup、E079 误判资料缺失、E095 加了错误限定

    B. **评测集缺陷**（3 条）
       E078 问「该 NetworkPolicy 的 Egress 端口」—— 文档里有多条 NetworkPolicy
       示例，答案不唯一，模型答的是另一个合法示例里的端口。
       E102 问「存活探测的另一种方式」—— 文档讲了 HTTP 和 TCP 两种，
       「另一种」指代不明，标准答案只取了一个。
       E103 问「示例 Pod 的镜像」—— 文档有多个示例 Pod，
       模型答 nginx:1.14.2，标准答案是 agnhost，两个都真实存在。

    C. 出题错误（1 条，在「字符串对但LLM判错」组里）
       E094 标准答案写「默认值是 .spec.parallelism」，但原文明确说
       工作队列模式下**不可以设置** .spec.completions。标准答案本身是错的。

============================== 处理原则 ==============================

**只剔除「题目本身有问题」的，不剔除「模型答错」的。**
这条界线必须守住，否则就是给自己刷分。

剔除后重算，并明确标注：
    原始 130 条（含缺陷）  →  准确率 X%
    剔除 N 条缺陷题后      →  准确率 Y%

两个数都报，说明剔除理由和编号。这样别人能复核我的判断。
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
ev = json.loads(EVAL.read_text(encoding="utf-8"))
items = {i["id"]: i for i in ev["items"]}

# 人工复核后确认有缺陷的题目（答案不唯一 / 指代不明 / 标准答案错误）
DEFECTIVE = {
    "E078": "文档含多条 NetworkPolicy 示例，答案不唯一",
    "E094": "标准答案与原文矛盾：原文说明不可设置 completions",
    "E102": "「另一种方式」指代不明，HTTP 与 TCP 都是正确答案",
    "E103": "文档含多个示例 Pod，镜像地址不唯一",
    "E036": "文档给出两种权重配置示例，答案不唯一",
}

facts = [jid for jid in judges if items.get(jid, {}).get("kind") == "fact"]
clean = [jid for jid in facts if jid not in DEFECTIVE]

print("=" * 92)
print(" 剔除缺陷题后重算")
print("=" * 92)
print()
print(f"缺陷题 {len(DEFECTIVE)} 条：")
for jid, why in DEFECTIVE.items():
    print(f"  {jid}  {why}")

print()
print("=" * 92)
print(" fact 类准确率")
print("=" * 92)

s_all = sum(1 for jid in facts if rows[jid]["correct"])
l_all = sum(1 for jid in facts if judges[jid]["verdict"] == "CORRECT")
s_cln = sum(1 for jid in clean if rows[jid]["correct"])
l_cln = sum(1 for jid in clean if judges[jid]["verdict"] == "CORRECT")

print(f"  {'':<32}{'字符串':>12}{'LLM裁判':>12}")
print(f"  {'原始 120 条（含缺陷）':<28}{s_all:>6}/{len(facts):<4}{l_all:>6}/{len(facts):<4}")
print(f"  {'':<32}{100*s_all/len(facts):>11.1f}%{100*l_all/len(facts):>11.1f}%")
print(f"  {'剔除缺陷后 ' + str(len(clean)) + ' 条':<28}{s_cln:>6}/{len(clean):<4}{l_cln:>6}/{len(clean):<4}")
print(f"  {'':<32}{100*s_cln/len(clean):>11.1f}%{100*l_cln/len(clean):>11.1f}%")

# refuse
refuses = [jid for jid in judges if items.get(jid, {}).get("kind") == "refuse"]
r_ok = sum(1 for jid in refuses if judges[jid]["refused"])
print()
print(f"  refuse 类 {r_ok}/{len(refuses)} 正确拒答  ({100*r_ok/len(refuses):.1f}%)")

# 失败明细
print()
print("=" * 92)
print(" 剔除缺陷后仍判错（真失败）")
print("=" * 92)
for jid in clean:
    if judges[jid]["verdict"] != "CORRECT":
        it = items[jid]
        print(f"\n[{jid}] {it['question'][:64]}")
        print(f"  标准答案：{'；'.join(it['answer_keywords'])[:66]}")
        print(f"  LLM：{judges[jid]['verdict']}  理由：{judges[jid]['reason'][:110]}")
        print(f"  回答：{rows[jid]['answer'][:180]}")

print()
print("=" * 92)
print(" 最终结论（写进报告用）")
print("=" * 92)
print(f"  评测集 {len(facts)} 条 fact 题，人工复核发现 {len(DEFECTIVE)} 条题目本身有缺陷，已剔除。")
print(f"  剩余 {len(clean)} 条有效题：")
print(f"    严格字符串匹配  {100*s_cln/len(clean):.1f}%   （保守下界，可完全复现）")
print(f"    LLM 语义裁判    {100*l_cln/len(clean):.1f}%   （识别同义改写）")
print(f"  拒答准确率        {100*r_ok/len(refuses):.1f}%")
print()
print("  ⚠️ 两个数字都给，因为判定方式不同：")
print("     字符串匹配零主观成分但会漏判同义改写；")
print("     LLM 裁判更接近人类判断，但本身依赖模型质量。")
