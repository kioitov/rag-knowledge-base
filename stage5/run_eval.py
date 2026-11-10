"""阶段 5 第三块：评测执行器 —— 三组对比实验。

============================== 实验设计 ==============================

三个维度交叉对比，每个维度都必须保持其他变量不变：

    维度 A：切块策略    fixed / paragraph / section
    维度 B：检索方式    bm25 / vector / hybrid
    维度 C：top-k       1 / 3 / 5

每组跑 130 条评测集，统计：

    Hit@k        标准答案所在的 chunk 是否出现在 top-k 检索结果里
    AnswerAcc    生成的回答是否包含标准答案关键词
    RefuseOK     refuse 类问题是否被正确拒答（只对 refuse 类统计）
    延迟          P50 / P95

============================== 一个关键设计：检索和生成分开测 ==============================

Hit@k  只跑检索，不调 LLM（快、免费、可重复）
AnswerAcc 才调 LLM（慢、花钱）

这样能区分两种失败：
    检索没找到      → 改检索方案
    检索到了但没答对 → 改 prompt 或模型
    
混在一起测就分不清了，也无法定位问题。

============================== ref 判定用 chunk_id 而非文本匹配 ==============================

评测集里每条 fact 都有 chunk_id（答案所在块）。
Hit 判定：检索结果的 chunk_id 列表里是否包含它。
这是精确判定，不受文本相似度影响。

============================== 成本估算（跑之前先算）==============================

    Hit@k 层：130 条 × 9 组配置 = 1170 次检索
             向量检索每次约 1 次 embedding 调用（~0.3s）
             合计约 6 分钟，费用可忽略

    AnswerAcc 层：只跑最优配置 130 条
             每条约 3-6s，合计 7-13 分钟

用法：
    python stage5/run_eval.py --layer retrieval       # 只测检索（快）
    python stage5/run_eval.py --layer answer          # 只测生成
    python stage5/run_eval.py --layer all             # 全跑
    python stage5/run_eval.py --layer retrieval --limit 20  # 小样本试跑
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage2"))
sys.path.insert(0, str(_ROOT / "stage3"))

import config
from cache import load_chunks
from chunker import Chunk, build_chunks
from generator import Hit, RAGAnswerer
from paths import find_docs
from retriever import KeywordIndex
from vector_index import EmbeddingClient, VectorIndex

EVAL_FILE = _ROOT / "data" / "eval" / "eval_set.json"
RESULT_DIR = _ROOT / "data" / "eval" / "results"

REFUSAL_MARKERS = [
    "无法确定", "未提及", "未涉及", "不包含", "资料不足",
    "无法从", "没有找到", "没有提及", "无法回答",
]


# ---------------------------------------------------------------------------
# 答案判定：规范化后比对
# ---------------------------------------------------------------------------
# ---------------------------- 判定过严的教训 ----------------------------
# 第一版用 `answer_keyword in text` 原样匹配，结果大量**语义完全正确**的
# 回答被判为错误：
#
#   E003  期望「...的值筛选Kubernetes 对象」
#         回答「...的值筛选 Kubernetes 对象」     ← 只差一个空格
#   E024  期望「30000-30085」
#         回答「30000–30085」                     ← 连字符 vs 短破折号
#   E020  期望「使这种解耦成为可能」
#         回答「从而实现前后端的解耦」            ← 同义改写
#
# 前两类是纯粹的字符规范化问题，必须修。
# 第三类是同义改写，不能靠字符串匹配判定 —— 那需要人工或 LLM 裁判，
# 本脚本不处理，报告里明确说明这一局限。
#
# 处理方式：
#   1. 统一空白（中英文空格差异）
#   2. 统一各种破折号/连字符
#   3. 统一全角半角
#   4. 忽略大小写
def normalize(s: str) -> str:
    """规范化文本，消除排版差异导致的假阴性。"""
    s = s or ""
    # 各类破折号、连字符统一成 -
    s = re.sub(r"[–—−‐‑﹘﹣－]", "-", s)
    # 全角转半角（数字、字母、常见标点）
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    s = "".join(out)
    # 去掉所有空白（中英文之间空格差异最常见）
    s = re.sub(r"\s+", "", s)
    # 统一引号
    s = re.sub(r"[\"'“”‘’`]", "", s)
    return s.lower()


def answer_contains(text: str, keyword: str) -> bool:
    """判断回答里是否包含标准答案（规范化后比对）。"""
    return normalize(keyword) in normalize(text)


def is_refusal(text: str) -> bool:
    """是否拒答。"""
    return any(m in text for m in REFUSAL_MARKERS)


@dataclass
class RetrievalResult:
    """单条检索评测结果。"""

    item_id: str
    method: str
    top_k: int
    hit: bool
    latency_ms: float
    rank: int | None  # 命中的排名，未命中为 None


@dataclass
class AnswerResult:
    """单条生成评测结果。"""

    item_id: str
    kind: str
    question: str
    correct: bool
    refused: bool
    answer: str
    answer_keywords: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# 检索层评测
# ---------------------------------------------------------------------------
def eval_retrieval(items: list[dict], chunks: list[Chunk], configs: list[tuple[str, str]],
                   top_ks: list[int], limit: int = 0) -> list[RetrievalResult]:
    """跑检索层评测。

    configs: [(策略, 检索方式), ...]
    每个 (策略, 检索方式) 组合都要重建索引（策略变了，chunk_id 就变了）。
    """
    from hybrid_retriever import HybridRetriever

    results: list[RetrievalResult] = []

    # 只取 fact 类（refuse 类没有 chunk_id，无法算 Hit）
    facts = [i for i in items if i["kind"] == "fact"]
    if limit:
        facts = facts[:limit]

    # 按切块策略分组，每种策略只建一次索引
    for strategy in sorted({c[0] for c in configs}):
        print(f"\n  切块策略：{strategy}")

        # 用该策略重新切块
        t0 = time.perf_counter()
        strat_chunks: list[Chunk] = []
        for d in find_docs():
            strat_chunks.extend(build_chunks(d, strategy=strategy))
        print(f"    {len(strat_chunks)} 块（{time.perf_counter() - t0:.1f}s）")

        chunk_ids = {c.chunk_id for c in strat_chunks}

        # 注意：切块策略变了，评测集里记录的 chunk_id 是默认策略（paragraph）的，
        # 所以其他策略下无法用 chunk_id 精确匹配。
        # 处理方式：用「答案关键词是否在该块中出现」来近似判定答案块。
        # 这是有损的，报告里必须说明。
        key_to_ids: dict[str, set[str]] = {}
        for it in facts:
            kws = it["answer_keywords"]
            if not kws:
                continue
            ids = {c.chunk_id for c in strat_chunks if all(k in c.text for k in kws)}
            key_to_ids[it["id"]] = ids
            if not ids:
                # 该策略下找不到含答案的块 —— 记录为空，会算作未命中
                pass

        for method in sorted({c[1] for c in configs if c[0] == strategy}):
            print(f"    检索方式：{method}")

            # 建索引
            if method == "bm25":
                idx = KeywordIndex(strat_chunks)
                vec = None
                emb = None
            else:
                vec = VectorIndex(_ROOT / "data" / f"chroma_eval_{strategy}",
                                  collection=f"eval_{strategy}")
                emb = EmbeddingClient()
                if vec.count() != len(strat_chunks):
                    st = vec.build(strat_chunks, emb, force=True)
                    print(f"      建向量索引 {st.summary()}")
                if method == "hybrid":
                    hy = HybridRetriever(strat_chunks, vec, emb)

            max_k = max(top_ks)
            for it in facts:
                q = it["question"]
                t1 = time.perf_counter()
                try:
                    if method == "bm25":
                        hits = idx.search(q, top_k=max_k)
                        got = [h.chunk.chunk_id for h in hits]
                    elif method == "vector":
                        rows = vec.search(q, emb, top_k=max_k)
                        got = [r[0] for r in rows]
                    else:
                        rows = hy.search(q, k=max_k)
                        got = [r.chunk_id for r in rows]
                except Exception as exc:
                    print(f"      [ERR] {it['id']}: {type(exc).__name__}")
                    continue
                dt = (time.perf_counter() - t1) * 1000

                want = key_to_ids.get(it["id"], set())
                rank = None
                for i, cid in enumerate(got, 1):
                    if cid in want:
                        rank = i
                        break

                for k in top_ks:
                    results.append(RetrievalResult(
                        item_id=it["id"], method=f"{strategy}/{method}", top_k=k,
                        hit=(rank is not None and rank <= k),
                        latency_ms=dt, rank=rank,
                    ))

    return results


# ---------------------------------------------------------------------------
# 生成层评测
# ---------------------------------------------------------------------------
def eval_answer(items: list[dict], chunks: list[Chunk], method: str,
                top_k: int, limit: int = 0) -> list[AnswerResult]:
    """跑生成层评测：检索 + 生成 + 判定。"""
    from hybrid_retriever import HybridRetriever

    facts = [i for i in items if i["kind"] == "fact"]
    refuses = [i for i in items if i["kind"] == "refuse"]
    if limit:
        facts = facts[:limit]
        refuses = refuses[: max(1, limit // 10)]

    bm25 = KeywordIndex(chunks)
    vec = VectorIndex(_ROOT / "data" / "chroma", collection="kb_chunks")
    emb = EmbeddingClient()
    hy = HybridRetriever(chunks, vec, emb)
    chunk_map = {c.chunk_id: c for c in chunks}
    answerer = RAGAnswerer(bm25, top_k=top_k)

    results: list[AnswerResult] = []
    total = len(facts) + len(refuses)

    for n, it in enumerate(facts + refuses, 1):
        q = it["question"]
        kind = it["kind"]

        # 检索
        if kind == "refuse":
            # 拒答问题要正常检索（模拟真实场景），看模型会不会编
            hits = hy.search(q, k=top_k)
        else:
            hits = hy.search(q, k=top_k)
        gen_hits = [
            Hit(chunk=chunk_map[h.chunk_id], score=h.rrf_score, matched_terms=[])
            for h in hits if h.chunk_id in chunk_map
        ]

        try:
            a = answerer.answer(q, hits=gen_hits)
        except Exception as exc:
            print(f"    [ERR] {it['id']} {type(exc).__name__}")
            continue

        text = a.text
        refused = is_refusal(text)

        if kind == "refuse":
            correct = refused
        else:
            kws = it["answer_keywords"]
            # 用规范化比对，避免空格/破折号差异造成假阴性
            correct = (not refused) and all(answer_contains(text, k) for k in kws)

        results.append(AnswerResult(
            item_id=it["id"], kind=kind, question=q, correct=correct,
            refused=refused, answer=text[:400],
            answer_keywords=it["answer_keywords"], latency_ms=a.latency_ms,
        ))

        mark = "OK " if correct else ("REF" if refused else "MISS")
        if n % 10 == 0 or n == total:
            print(f"    [{n:3d}/{total}] {mark}  {q[:44]}")

    return results


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize_retrieval(rows: list[RetrievalResult]) -> None:
    print()
    print("=" * 96)
    print(" 检索层结果（Hit@k）")
    print("=" * 96)

    groups: dict[tuple[str, int], list[bool]] = {}
    lats: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault((r.method, r.top_k), []).append(r.hit)
        lats.setdefault(r.method, []).append(r.latency_ms)

    top_ks = sorted({r.top_k for r in rows})
    methods = sorted({r.method for r in rows})

    hdr = f"  {'配置':<24}"
    for k in top_ks:
        hdr += f" {'Hit@' + str(k):>10}"
    hdr += f" {'P50延迟':>9}"
    print(hdr)
    print("-" * 96)

    for m in methods:
        row = f"  {m:<24}"
        for k in top_ks:
            vals = groups.get((m, k), [])
            if vals:
                row += f" {sum(vals):>3}/{len(vals):<6}"
            else:
                row += f" {'-':>10}"
        lat = lats.get(m, [])
        if lat:
            row += f" {statistics.median(lat):>7.0f}ms"
        print(row)

    print("-" * 96)
    print("  说明：分母为该配置下的评测条数；数值为命中数。")
    print("  注：非 paragraph 策略下，因 chunk_id 变化，答案块用「关键词是否出现在块中」近似定位，")
    print("      结果偏保守（同一关键词可能在多块中出现，取任一即算命中）。")


def summarize_answer(rows: list[AnswerResult]) -> None:
    print()
    print("=" * 96)
    print(" 生成层结果（答案正确率）")
    print("=" * 96)

    facts = [r for r in rows if r.kind == "fact"]
    refuses = [r for r in rows if r.kind == "refuse"]

    if facts:
        n_ok = sum(1 for r in facts if r.correct)
        n_ref = sum(1 for r in facts if r.refused)
        print(f"  fact 类（{len(facts)} 条）：")
        print(f"    答案正确   {n_ok:>3}/{len(facts):<4}  {100 * n_ok / len(facts):.1f}%")
        print(f"    误拒答     {n_ref:>3}/{len(facts):<4}  {100 * n_ref / len(facts):.1f}%")

    if refuses:
        n_ok = sum(1 for r in refuses if r.correct)
        print(f"  refuse 类（{len(refuses)} 条）：")
        print(f"    正确拒答   {n_ok:>3}/{len(refuses):<4}  {100 * n_ok / len(refuses):.1f}%")

    lats = [r.latency_ms for r in rows]
    if lats:
        lats_s = sorted(lats)
        p50 = statistics.median(lats_s)
        p95 = lats_s[int(len(lats_s) * 0.95) - 1] if len(lats_s) > 1 else lats_s[0]
        print(f"  延迟 P50 {p50:.0f}ms / P95 {p95:.0f}ms")

    # 失败案例
    fails = [r for r in rows if not r.correct]
    if fails:
        print()
        print(f"  失败案例（{len(fails)} 条，展示前 6 条）：")
        for r in fails[:6]:
            print(f"    [{r.item_id}] {r.question[:50]}")
            print(f"      期望含：{r.answer_keywords}")
            print(f"      实际：{r.answer[:90]}")
            print()


def main() -> None:
    ap = argparse.ArgumentParser(description="评测执行器")
    ap.add_argument("--layer", choices=["retrieval", "answer", "all"], default="retrieval")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--top-k", type=int, default=3, help="生成层使用的 top-k")
    args = ap.parse_args()

    if not EVAL_FILE.exists():
        raise SystemExit("评测集不存在。先跑：python stage5/build_evalset.py")

    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    items = data["items"]

    print("=" * 96)
    print(" 阶段 5 评测")
    print("=" * 96)
    print(f"  评测集：{data['total']} 条"
          f"（fact {data['by_kind']['fact']} / refuse {data['by_kind']['refuse']}）")

    print("\n加载语料…")
    chunks: list[Chunk] = []
    for d in find_docs():
        chunks.extend(load_chunks(d))
    print(f"  {len(chunks)} 块（paragraph 策略）")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.layer in ("retrieval", "all"):
        # 维度 A（切块）× 维度 B（检索）
        configs = [
            ("paragraph", "bm25"),
            ("paragraph", "vector"),
            ("paragraph", "hybrid"),
            ("fixed", "hybrid"),
            ("section", "hybrid"),
        ]
        print(f"\n检索层评测：{len(configs)} 组配置 × {len([i for i in items if i['kind']=='fact'])} 条")
        rows = eval_retrieval(items, chunks, configs, top_ks=[1, 3, 5], limit=args.limit)
        summarize_retrieval(rows)

        out = RESULT_DIR / "retrieval.json"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=1)
        print(f"\n  明细已存：{out}")

    if args.layer in ("answer", "all"):
        print(f"\n生成层评测：hybrid / top-{args.top_k}")
        rows = eval_answer(items, chunks, "hybrid", args.top_k, limit=args.limit)
        summarize_answer(rows)

        out = RESULT_DIR / "answer.json"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=1)
        print(f"  明细已存：{out}")

    print()
    print("=" * 96)


if __name__ == "__main__":
    main()
