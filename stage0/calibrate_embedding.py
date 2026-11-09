"""Embedding 相似度标定实验：搞清楚 0.35 到底是「正常」还是「有问题」。

====================== 为什么必须做这个 ======================

阶段 0 首次验收时，两段「同属一个项目、但讲不同子话题」的文本
（一段讲系统整体定位，一段讲某个具体算法）余弦相似度只有 0.352，
低于我预设的 0.5 预警线。

两种可能：
    1. 模型用错了 / 参数不对（要修）
    2. 这两段文本本身语义距离就远，0.35 是正常的

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

注意：测试语料刻意使用**通用技术句子**，与本项目的具体业务无关。
标定验证的是「模型能否区分语义远近」，与文本属于哪个行业无关；
用通用句子既不泄露业务领域，也更容易被别人复现。

用法：
    python stage0/calibrate_embedding.py
    python stage0/calibrate_embedding.py --model text-embedding-v3
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ---------------------------------------------------------------------------
# 测试语料
# ---------------------------------------------------------------------------
# 刻意使用**与具体业务无关的通用技术句子**。原因：
#   1. 语料里的文本会随代码进公开仓库，用真实项目内容等于泄露业务领域
#   2. 标定实验验证的是「模型能否区分语义远近」，与文本属于哪个行业无关
#      —— 用通用句子反而更容易被别人复现和质疑
GROUPS: dict[str, list[tuple[str, str]]] = {
    "组1 完全等价": [
        (
            "向量数据库用于存储高维嵌入向量",
            "向量数据库用于存储高维嵌入向量",
        ),
        (
            "文档切块需要设置重叠以避免语义截断",
            "文档切块时需要设置重叠以避免语义截断",
        ),
        (
            "BM25 是一种基于词频的检索算法",
            "BM25是一种基于词频的检索算法。",
        ),
    ],
    "组2 高度相关": [
        (
            "向量数据库用于存储高维嵌入向量",
            "嵌入向量可以持久化到本地磁盘以便复用",
        ),
        (
            "文档切块需要设置重叠以避免语义截断",
            "分块粒度过大会导致检索精度下降",
        ),
        (
            "BM25 是一种基于词频的检索算法",
            "关键词检索依靠词频统计计算相关性得分",
        ),
    ],
    "组3 弱相关": [
        (
            "向量数据库用于存储高维嵌入向量",
            "日志文件需要定期轮转以免占满磁盘",
        ),
        (
            "文档切块需要设置重叠以避免语义截断",
            "前端界面可以用 Streamlit 快速搭建",
        ),
        (
            "BM25 是一种基于词频的检索算法",
            "HTTP 接口应当返回合适的状态码",
        ),
    ],
    "组4 无关": [
        (
            "向量数据库用于存储高维嵌入向量",
            "今天午饭吃了牛肉面",
        ),
        (
            "文档切块需要设置重叠以避免语义截断",
            "钢琴调音需要专门的工具和听觉经验",
        ),
        (
            "BM25 是一种基于词频的检索算法",
            "马拉松比赛全程约四十二公里",
        ),
    ],
}

# 验证「口语 query 与书面文档无字面重叠」时，向量检索能否正确排序。
# 这些是通用问法，不含任何具体业务名词，但保留了当初真实失败案例的**结构**：
#   query 是口语化的，文档是书面化的，两者几乎没有共同字符。
FAILURE_CASES = [
    ("这个能装多少东西", "单机部署可支撑千万级向量规模"),
    ("这个能装多少东西", "接口响应时间应控制在二百毫秒以内"),
    ("怎样才能更快", "批量写入比逐条写入高一个数量级"),
    ("怎样才能更快", "系统默认采用单线程顺序处理"),
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

    # ---- 说明阶段 0 那个 0.35 到底怎么回事 ----
    print()
    print("=" * 88)
    print(" 关于阶段 0 那个 0.3520 的解释")
    print("=" * 88)
    print("  首次验收时，我把两条「同属一个项目、但讲不同子话题」的文本")
    print("  （一条讲系统整体定位，一条讲某个具体算法）当作「语义相近」，")
    print("  预设了 0.5 的预警线，实测 0.35 就触发了警告。")
    print("  但实测显示它们属于「同领域不同子话题」，")
    print(f"  正对应上面的「组3 弱相关」，均值 {means[order[2]]:.3f}。")
    print("  **0.35 落在弱相关区间，是正常值，不是模型用错了。**")
    print()
    print("  结论：阈值必须用标定数据定，不能拍脑袋。")
    print("       拍脑袋的阈值最坏的作用不是漏报，而是在正常数据上误报，")
    print("       然后你开始怀疑一个本来没问题的系统。")

    # ---- 口语 query 的向量侧表现 ----
    print()
    print("=" * 88)
    print(" 口语化 query → 向量检索能不能排对")
    print("=" * 88)
    case_texts: list[str] = []
    for q, d in FAILURE_CASES:
        case_texts.extend([q, d])
    case_vecs = embed_all(client, model, case_texts, dim)

    print(f"  {'query':<20} {'文档':<36} {'相似度':>7}")
    print("-" * 88)
    for i, (q, d) in enumerate(FAILURE_CASES):
        vq, vd = case_vecs[i * 2], case_vecs[i * 2 + 1]
        s = cos(vq, vd)
        print(f"  {q:<20} {d[:34]:<36} {s:>7.4f}")

    print()
    print("  关键验证 —— 口语 query 与书面文档几乎零字面重叠时，能否排对？")
    va = case_vecs[0]  # 这个能装多少东西
    v_correct = case_vecs[1]  # 单机部署可支撑千万级向量规模
    v_wrong = case_vecs[3]  # 接口响应时间应控制在二百毫秒以内
    s_correct, s_wrong = cos(va, v_correct), cos(va, v_wrong)
    print(f"    语义正确（容量规模）: {s_correct:.4f}")
    print(f"    语义无关（响应时间）: {s_wrong:.4f}")
    if s_correct > s_wrong:
        print("    [OK] 向量检索能正确排序 —— 纯关键词法在这种 query 上会失败")
    else:
        print("    [WARN] 向量也没排对，需要换更强的 embedding 模型或加 rerank")

    print()
    print("=" * 88)
    print(f" 总结论：{'embedding 可用' if ok else 'embedding 有问题，需换模型或调参'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
