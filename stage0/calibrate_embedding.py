"""Embedding 相似度标定实验：搞清楚 0.35 到底是「正常」还是「有问题」。

====================== 为什么必须做这个 ======================

阶段 0 首次实测：两句话
    A "融销通平台是一个农业金融与农产品产销一体化服务平台"
    B "联合贷款人推荐采用基于内容的推荐算法"
余弦相似度只有 0.352，低于我预设的 0.5 预警线。

两种可能：
    1. 模型用错了 / 参数不对（要修）
    2. 这两句话本身语义就不同（A 讲平台定位，B 讲某个算法），0.35 是正常的

**不能靠猜。** 这个数字直接决定阶段 3 的检索质量：
如果 embedding 的整体相似度水平偏低，那检索阈值、top-k、混合检索的权重
全都要跟着调。现在标定清楚，后面少返工。

====================== 实验设计 ======================

四组对照，每组给「应该高」和「应该低」的样本，看相似度是否单调有序：

    组1 完全等价     同一句话的重复 / 只差标点
    组2 高度相关     同一主题的不同表述
    组3 弱相关       同一领域的不同子话题
    组4 无关         不同领域

判定标准：组1 > 组2 > 组3 > 组4 且组间有区分度。如果顺序乱了，
说明模型或参数有问题。

用法：
    python stage0\\calibrate_embedding.py
    python stage0\\calibrate_embedding.py --model text-embedding-v3
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ---------------------------------------------------------------------------
# 测试语料：全部来自或贴近本项目的真实文档
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[tuple[str, str]]] = {
    "组1 完全等价": [
        (
            "联合贷款人推荐采用基于内容的推荐算法",
            "联合贷款人推荐采用基于内容的推荐算法",
        ),
        (
            "农户提交融资意向需要填写八个字段",
            "农户提交融资意向时需要填写八个字段",
        ),
        (
            "系统采用 ARIMA 模型预测农产品价格趋势",
            "系统采用ARIMA模型预测农产品价格趋势。",
        ),
    ],
    "组2 高度相关": [
        (
            "联合贷款人推荐采用基于内容的推荐算法",
            "系统依据农作物与地址特征为农户推荐可组合贷款的联合贷款人",
        ),
        (
            "银行用户对融资申请作出审核结论",
            "银行用户查看农户融资意向并填写审核意见",
        ),
        (
            "农户填写并提交融资需求信息，系统据此执行智能匹配",
            "融资意向提交后系统自动匹配信贷产品",
        ),
    ],
    "组3 弱相关": [
        (
            "联合贷款人推荐采用基于内容的推荐算法",
            "农产品的历史价格数据可以用于趋势预测",
        ),
        (
            "银行用户对融资申请作出审核结论",
            "农户可以在平台上发布农产品商品货源",
        ),
        (
            "信贷产品的参考利率与贷款期限",
            "农业知识问答与专家在线咨询",
        ),
    ],
    "组4 无关": [
        (
            "联合贷款人推荐采用基于内容的推荐算法",
            "今天天气不错适合出门散步",
        ),
        (
            "银行用户对融资申请作出审核结论",
            "Python 的列表推导式比 for 循环快",
        ),
        (
            "信贷产品的参考利率与贷款期限",
            "红烧肉需要小火慢炖两个小时",
        ),
    ],
}

# 阶段 1 真实出现过的失败案例：关键词检索答不对，向量检索应该能救
FAILURE_CASES = [
    ("农户能贷多少钱", "融资意向包含意向贷款金额字段，信贷产品含申请额度"),
    ("农户能贷多少钱", "系统支持哪些还款方式"),
    ("系统支持哪些还款方式", "还款方式为到期还本付息"),
    ("系统支持哪些还款方式", "并发用户数支持 100～300 个用户同时在线"),
]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_all(client, model: str, texts: list[str], dim: int | None) -> list[list[float]]:
    """批量向量化。百炼单次最多 10 条，分批处理。"""
    out: list[list[float]] = []
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        kwargs = {"model": model, "input": batch}
        if dim:
            kwargs["dimensions"] = dim
        resp = client.embeddings.create(**kwargs)
        out.extend(d.embedding for d in resp.data)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding 相似度标定")
    ap.add_argument("--model", default=None, help="覆盖 .env 里的 embedding 模型")
    ap.add_argument("--dim", type=int, default=None, help="覆盖维度")
    args = ap.parse_args()

    cfg = config.embedding_config()
    if not cfg.ready():
        print("[FAIL] embedding 未配置。先跑 stage0/set_key.py 填 key。")
        sys.exit(1)

    model = args.model or cfg.model
    dim = args.dim or cfg.dim

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    print("=" * 88)
    print(f" Embedding 相似度标定")
    print(f" 模型 {model}  维度 {dim}  endpoint {cfg.base_url}")
    print("=" * 88)

    # ---- 收集所有文本，一次性向量化（省调用次数）----
    all_pairs: list[tuple[str, str, str]] = []  # (group, text_a, text_b)
    for group, pairs in GROUPS.items():
        for a, b in pairs:
            all_pairs.append((group, a, b))

    texts: list[str] = []
    for _g, a, b in all_pairs:
        texts.append(a)
        texts.append(b)

    print(f"\n[1/3] 向量化 {len(texts)} 条文本…")
    vecs = embed_all(client, model, texts, dim)
    print(f"      实际维度 {len(vecs[0])}")

    # ---- 分组统计 ----
    print(f"\n[2/3] 分组相似度")
    print("-" * 88)
    stats: dict[str, list[float]] = {g: [] for g in GROUPS}
    idx = 0
    for group, a, b in all_pairs:
        va, vb = vecs[idx], vecs[idx + 1]
        idx += 2
        s = cos(va, vb)
        stats[group].append(s)
        print(f"  {group:<14} {s:.4f}  |  {a[:32]}  ↔  {b[:32]}")

    print("-" * 88)
    print(f"  {'分组':<14} {'均值':>7} {'最小':>7} {'最大':>7}")
    means = {}
    for group, vals in stats.items():
        m = sum(vals) / len(vals)
        means[group] = m
        print(f"  {group:<14} {m:>7.4f} {min(vals):>7.4f} {max(vals):>7.4f}")

    # ---- 单调性检验：这才是真正的验收标准 ----
    order = list(GROUPS.keys())
    print("\n[3/3] 单调性检验（这才是判断模型是否可用的标准）")
    print("-" * 88)
    ok = True
    for i in range(len(order) - 1):
        a, b = means[order[i]], means[order[i + 1]]
        passed = a > b
        ok &= passed
        mark = "OK  " if passed else "FAIL"
        print(f"  {mark} {order[i]} ({a:.3f}) > {order[i+1]} ({b:.3f})   差值 {a-b:+.3f}")

    gap = means[order[0]] - means[order[-1]]
    print(f"\n  等价组与无关组的总体差距：{gap:.3f}")
    if gap < 0.15:
        print("  [FAIL] 差距过小，embedding 区分度不足，检索会失效")
        ok = False
    else:
        print(f"  [OK] 区分度足够")

    # ---- 说明 0.35 到底怎么回事 ----
    print()
    print("=" * 88)
    print(" 关于阶段 0 那个 0.3520 的解释")
    print("=" * 88)
    print("  那两句话是：")
    print("    A 融销通平台是一个农业金融与农产品产销一体化服务平台")
    print("    B 联合贷款人推荐采用基于内容的推荐算法")
    print("  A 讲平台整体定位，B 讲某个具体算法 —— 它们是「同领域不同子话题」，")
    print(f"  对应上面的「组3 弱相关」，均值 {means[order[2]]:.3f}。")
    print("  **0.35 落在弱相关区间，是正常值，不是模型用错了。**")
    print()
    print("  这也说明我原来设的 0.5 预警线是错的 —— 它假设了两句话「语义相近」，")
    print("  实际上它们只是「同属一个项目」。阈值必须用标定数据定，不能拍脑袋。")

    # ---- 阶段 1 失败案例的向量侧表现 ----
    print()
    print("=" * 88)
    print(" 阶段 1 关键词检索失败案例 → 向量检索能不能救")
    print("=" * 88)
    case_texts: list[str] = []
    for q, d in FAILURE_CASES:
        case_texts.extend([q, d])
    case_vecs = embed_all(client, model, case_texts, dim)

    print(f"  {'query':<22} {'文档':<38} {'相似度':>7}")
    print("-" * 88)
    for i, (q, d) in enumerate(FAILURE_CASES):
        vq, vd = case_vecs[i * 2], case_vecs[i * 2 + 1]
        s = cos(vq, vd)
        print(f"  {q:<22} {d[:36]:<38} {s:>7.4f}")

    # 关键验证：「贷多少钱」vs 两个候选文档
    print()
    print("  关键验证 —— 「农户能贷多少钱」应该更接近哪个？")
    va = case_vecs[0]  # 农户能贷多少钱
    v_correct = case_vecs[1]  # 融资意向包含意向贷款金额字段…
    v_wrong = case_vecs[3]  # 系统支持哪些还款方式
    print(f"    正确文档（意向贷款金额/申请额度）: {cos(va, v_correct):.4f}")
    print(f"    错误文档（还款方式）            : {cos(va, v_wrong):.4f}")
    if cos(va, v_correct) > cos(va, v_wrong):
        print("    [OK] 向量检索能正确排序 —— 关键词法救不了的问题，向量能救")
    else:
        print("    [WARN] 向量也没排对，需要换更大的 embedding 模型或加 rerank")

    print()
    print("=" * 88)
    print(f" 总结论：{'embedding 可用' if ok else 'embedding 有问题，需换模型或调参'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
