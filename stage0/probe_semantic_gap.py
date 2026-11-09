"""验证：口语化 query 与书面文档无字面重叠时，向量检索能否正确排序。

⚠️ **实验性脚本，结论不可直接引用。** 原因见文末「已知局限」。

====================== 为什么做这个 ======================

阶段 1 用纯 BM25 时，有些口语化 query 完全检索不到（如「能装多少」vs
文档写「容量上限」「支撑规模」）。当时判断这是关键词法的物理上限，
但阶段 3 要花时间上向量检索，必须先验证它真的能改善。

首次尝试用「这个能装多少东西」做 query，结果向量反而把
「接口响应时间」排得更前。「装」在中文里有「容纳」和「安装」两义，
query 本身歧义，不能用来做基准 —— 这是测试设计的问题，不是技术的问题。

改用无歧义的口语 query 重测，并加书面问法做对照。

====================== 已知局限（不要过度解读）======================

1. **候选文档只有 5 条。** 真实的 RAG 是在 645 个块里检索。
   5 选 1 的难度远低于 645 选 3，命中率不能外推。
2. **「最大能存多少条」仍失败**，说明口语→书面的鸿沟没有完全跨越。
3. **字面基线实现过于粗糙**：它按「bigram 重叠个数」排序，
   重叠全为 0 时退化成取第一个文档，判定为 OK 纯属运气。
   要公平对比应当用 stage1/retriever.py 里真正的 BM25，
   在真实语料上跑同一批 query。

本脚本的价值在于**暴露问题**（口语 query 确实有鸿沟、歧义句会带偏排序），
而不是证明「向量优于关键词」。正式对比放到阶段 5 的评测集里做。

用法：
    python stage0/probe_semantic_gap.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ---------------------------------------------------------------------------
# 候选文档：通用技术描述，书面语
# ---------------------------------------------------------------------------
DOCS = [
    "单机部署可支撑千万级向量规模",
    "接口响应时间应控制在二百毫秒以内",
    "批量写入比逐条写入高一个数量级",
    "系统默认采用单线程顺序处理",
    "数据定期做全量备份以防丢失",
]

# ---------------------------------------------------------------------------
# 测试 query：语义明确、但措辞与文档零重叠
# 每条 query 标注它「应该最接近」哪条文档（按索引）
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, int]] = [
    # (query, 说明, 期望命中的 DOCS 索引)
    ("最大能存多少条", "口语：容量", 0),
    ("一次请求要等多久", "口语：延迟", 1),
    ("怎么让它快一点", "口语：性能", 2),
    ("能不能同时处理多个", "口语：并发", 3),
    ("数据丢了怎么办", "口语：备份", 4),
    # 书面问法作为对照：应该也命中同一个文档
    ("系统支持的最大数据规模是多少", "书面：容量", 0),
    ("接口的响应时间要求", "书面：延迟", 1),
    ("如何提升写入吞吐量", "书面：性能", 2),
    ("是否支持并发处理", "书面：并发", 3),
    ("备份策略是什么", "书面：备份", 4),
]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    cfg = config.embedding_config()
    if not cfg.ready():
        print("[FAIL] embedding 未配置。先跑 stage0/set_key.py 填 key。")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    print("=" * 90)
    print(f" 语义鸿沟测试：口语 query 能否命中书面文档")
    print(f" 模型 {cfg.model}  维度 {cfg.dim}")
    print("=" * 90)

    # 文档只向量化一次
    resp = client.embeddings.create(model=cfg.model, input=DOCS, dimensions=cfg.dim)
    doc_vecs = [d.embedding for d in resp.data]

    # ---- 逐条 query 测 ----
    print(f"\n{'query':<22} {'类型':<5} {'期望命中':<26} {'实际 top-1':<26} {'判定':<5}")
    print("-" * 90)

    hits = 0
    for query, note, expect_idx in CASES:
        r = client.embeddings.create(model=cfg.model, input=[query], dimensions=cfg.dim)
        qv = r.data[0].embedding
        sims = [cos(qv, dv) for dv in doc_vecs]
        top_idx = max(range(len(sims)), key=lambda i: sims[i])
        ok = top_idx == expect_idx
        hits += ok
        kind = "口语" if note.startswith("口语") else "书面"
        mark = "OK" if ok else "MISS"
        print(
            f"{query:<22} {kind:<5} {DOCS[expect_idx][:24]:<26} "
            f"{DOCS[top_idx][:24]:<26} {mark}"
        )

    total = len(CASES)
    print("-" * 90)
    print(f"  命中 {hits}/{total}  ({100 * hits / total:.0f}%)")

    # ---- 按类型分开统计 ----
    oral = [(q, n, e) for q, n, e in CASES if n.startswith("口语")]
    written = [(q, n, e) for q, n, e in CASES if n.startswith("书面")]
    print(f"  其中口语 {len(oral)} 条 / 书面 {len(written)} 条")

    print()
    print("=" * 90)
    if hits == total:
        print(f" 结果：{hits}/{total} 全部命中")
    elif hits >= total * 0.7:
        print(f" 结果：{hits}/{total} 命中")
        print("       注意：候选仅 5 条，难度远低于真实场景的 645 选 3，不可外推")
    else:
        print(f" 结果：命中率仅 {hits}/{total}，语义匹配在这批 query 上不可靠")
    print(" 结论：本脚本用于暴露问题，不作为「向量优于关键词」的证据。")
    print("       正式对比放到阶段 5 的评测集里，用真实语料和真实 BM25 跑。")
    print("=" * 90)

    # ---- 与关键词法对照 ----
    print()
    print(" 对照：同样的 query，纯字面匹配（bigram）能命中吗？")
    print("-" * 90)
    import re

    def bigrams(s: str) -> set[str]:
        chars = [c for c in re.sub(r"\s+", "", s) if "\u4e00" <= c <= "\u9fff"]
        return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}

    literal_hits = 0
    zero_overlap = 0
    for query, note, expect_idx in CASES:
        qb = bigrams(query)
        overlaps = [len(qb & bigrams(d)) for d in DOCS]
        best_i = max(range(len(overlaps)), key=lambda i: overlaps[i])
        # 记录「所有候选重叠都是 0」的情况 —— 此时排序结果无意义，
        # 命中纯属候选列表顺序的运气，不能算作字面匹配的成功。
        all_zero = max(overlaps) == 0
        if all_zero:
            zero_overlap += 1
        ok = best_i == expect_idx and not all_zero
        literal_hits += ok
        kind = "口语" if note.startswith("口语") else "书面"
        mark = "OK" if ok else ("N/A" if all_zero else "MISS")
        print(f"  {query:<22} {kind:<5} 最大字面重叠 {max(overlaps)} 个  {mark}")

    print("-" * 90)
    print(f"  字面匹配命中 {literal_hits}/{total}  ({100 * literal_hits / total:.0f}%)")
    if zero_overlap:
        print(f"  其中 {zero_overlap} 条与所有候选零字面重叠 —— 字面法无从下手，不计入命中")
    print(f"  向量匹配命中 {hits}/{total}  ({100 * hits / total:.0f}%)")
    print()
    print("  提醒：5 条候选太少，这些百分比不能外推到真实场景。")
    print("       而且字面基线用的是简化的重叠计数，不是 stage1 里的 BM25。")


if __name__ == "__main__":
    main()
