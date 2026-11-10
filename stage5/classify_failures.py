"""量化「判定方法局限性」的影响：区分真失败与判定假阴性。

============================== 这个分析为什么重要 ==============================

生成层报告 77.5% 正确率。但人工看失败案例，大部分其实是**语义正确但措辞不同**：

    期望「无需调整」      回答「无需再执行存储大小调整」
    期望「使这种解耦成为可能」 回答「从而实现前后端的解耦」
    期望「该Secret 中长期有效的令牌」  回答「该 Secret 中长期有效的令牌」← 只差空格

字符串匹配判不了这类。所以 77.5% 是**下界**，真实能力更高。

但不能因此就说「实际是 95%」——那是自我安慰。
正确做法是把失败案例分类，给出可核查的数字：
    判定假阴性（同义改写）   N 条
    真失败（确实没答出）     M 条

判据：检查回答里是否包含答案的**核心词**（去掉虚词后的关键词）。
这不是完美方案，但比裸字符串匹配好，且可解释。

============================== 局限 ==============================

这个方法仍会误判。所以本脚本的输出只作为「失败原因分类」，
**不作为最终正确率**。最终报告会同时给出：
    - 严格匹配正确率（可复现，偏保守）
    - 分类后的失败原因（说明保守程度）
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage5"))

from run_eval import answer_contains, normalize

rows = json.load(open(_ROOT / "data" / "eval" / "results" / "answer.json", encoding="utf-8"))
facts = [r for r in rows if r["kind"] == "fact"]
fails = [r for r in facts if not r["correct"]]

# 虚词表：判断「核心词」时忽略
STOP = set("的了吧呢啊是把被使将则即其以及或与和对于在于为以所以因此这那该")

def core_tokens(s: str) -> list[str]:
    """抽取核心词：取长度 >=2 的中文片段和英文/数字 token。"""
    s = normalize(s)
    toks = re.findall(r"[a-z0-9][a-z0-9_.\-/]*|[\u4e00-\u9fff]{2,}", s)
    out = []
    for t in toks:
        # 去掉纯虚词构成的长串
        if re.fullmatch(r"[\u4e00-\u9fff]+", t):
            if all(ch in STOP for ch in t):
                continue
        if len(t) >= 2:
            out.append(t)
    return out


def is_semantic_equivalent(answer: str, keyword: str, threshold: float = 0.5) -> bool:
    """粗判：回答是否覆盖了答案的核心词（>=threshold 比例）。

    只看核心词覆盖率，不看语序 —— 这是很粗的近似。
    但比「必须逐字包含」宽松，能识别出「无需调整」vs「无需再执行存储大小调整」
    这类同义改写（核心词「无需」「调整」都出现了）。
    """
    k_toks = core_tokens(keyword)
    if not k_toks:
        return False
    a_norm = normalize(answer)
    hit = sum(1 for t in k_toks if t in a_norm)
    return hit / len(k_toks) >= threshold


buckets = {"误拒答": [], "同义改写(判定假阴性)": [], "真失败": []}

for r in fails:
    if r["refused"]:
        buckets["误拒答"].append(r)
        continue
    kws = r["answer_keywords"]
    if any(is_semantic_equivalent(r["answer"], k) for k in kws):
        buckets["同义改写(判定假阴性)"].append(r)
    else:
        buckets["真失败"].append(r)

print("=" * 90)
print(" 失败原因分类（严格匹配判为失败，共 %d 条）" % len(fails))
print("=" * 90)
for name, items in buckets.items():
    print(f"  {name:<22} {len(items):>3} 条")

print()
print("=" * 90)
print(" 同义改写案例（说明字符串匹配偏保守）")
print("=" * 90)
for r in buckets["同义改写(判定假阴性)"][:10]:
    print(f"[{r['item_id']}] {r['question'][:56]}")
    print(f"   期望：{r['answer_keywords'][0][:70]}")
    print(f"   回答：{r['answer'][:120]}")
    print()

print("=" * 90)
print(" 真失败案例（模型确实没答到）")
print("=" * 90)
for r in buckets["真失败"][:10]:
    print(f"[{r['item_id']}] {r['question'][:56]}")
    print(f"   期望：{r['answer_keywords'][0][:80]}")
    print(f"   回答：{r['answer'][:140]}")
    print()

n_fact = len(facts)
n_ok = sum(1 for r in facts if r["correct"])
n_syn = len(buckets["同义改写(判定假阴性)"])
print("=" * 90)
print(" 结论")
print("=" * 90)
print(f"  严格匹配正确率      {n_ok}/{n_fact}  ({100*n_ok/n_fact:.1f}%)")
print(f"  加上同义改写后估算  {n_ok+n_syn}/{n_fact}  ({100*(n_ok+n_syn)/n_fact:.1f}%)")
print(f"  其中误拒答          {len(buckets['误拒答'])} 条")
print(f"  真失败              {len(buckets['真失败'])} 条")
print()
print("  ⚠️ 「加上同义改写」这一行是启发式估算，不是严格测量。")
print("     报告里两个数字都要写，并说明后者依赖核心词覆盖率判据。")
print("=" * 90)
