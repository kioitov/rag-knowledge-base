"""在打包后的目录里验证部署包能独立运行。

============================== 为什么必须验证 ==============================

打包时容易漏文件——某个 import 的模块没带上，本地能跑（因为根目录什么都有），
上传到云端就 ModuleNotFoundError。

所以这个脚本**把工作目录切到 deploy_pkg/**，屏蔽掉根目录的其他文件，
模拟云端只有这 20 个文件的场景。

只要这里能跑通，上传后大概率也能跑通。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG = Path(r"E:\deepseek work\rag-knowledge-base\deploy_pkg")

if not PKG.exists():
    raise SystemExit(f"部署包不存在：{PKG}\n先跑 python deploy/build_package.py")

# ---- 关键：把 sys.path 限定在包内，模拟云端的干净环境 ----
# 不能把项目根目录放进 path，否则会误用根目录里没打包的文件
_THIS = Path(__file__).resolve().parent
# 移除根目录（如果之前被加进去了）
sys.path = [p for p in sys.path if "rag-knowledge-base" not in p or "deploy_pkg" in p]
sys.path.insert(0, str(PKG))
sys.path.insert(1, str(PKG / "stage1"))
sys.path.insert(2, str(PKG / "stage2"))
sys.path.insert(3, str(PKG / "stage3"))

print("=" * 84)
print(" 部署包独立运行验证")
print("=" * 84)
print(f"  包路径：{PKG}")

# ---- 1. 检查 import 是否都来自包内 ----
print("\n[1] 模块来源检查")
import importlib

mods = ["config", "paths", "chunker", "cache", "md_parser", "pdf_parser",
        "retriever", "generator", "vector_index", "hybrid_retriever"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        src = getattr(mod, "__file__", "")
        in_pkg = "deploy_pkg" in src
        mark = "OK  " if in_pkg else "WARN"
        if not in_pkg:
            bad.append(m)
        print(f"  {mark} {m:<18} {src.replace(str(PKG), '<pkg>')}")
    except Exception as exc:
        bad.append(m)
        print(f"  FAIL {m:<18} {type(exc).__name__}: {str(exc)[:60]}")

if bad:
    print(f"\n  [FAIL] 有 {len(bad)} 个模块没从包内加载：{bad}")
else:
    print("\n  所有模块都从包内加载，无外部依赖")

# ---- 2. 加载数据产物 ----
print("\n[2] 加载部署产物")
import json

from chunker import Chunk

chunks = []
with open(PKG / "data" / "deploy" / "chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        chunks.append(Chunk(
            text=d["text"], source_file=d["source_file"], pages=d["pages"],
            section=d["section"], kind=d["kind"], strategy=d["strategy"],
            chunk_id=d["chunk_id"], meta=d.get("meta", {}),
        ))
print(f"  chunks.jsonl  {len(chunks)} 块")

paged_false = sum(1 for c in chunks if not c.meta.get("paged", True))
print(f"  paged=False   {paged_false} 块（应为 1290，全部是 Markdown）")

from vector_index import VectorIndex
vec = VectorIndex(PKG / "data" / "deploy" / "chroma", collection="kb_chunks")
print(f"  向量集合      {vec.count()} 块")

if vec.count() != len(chunks):
    print(f"  [WARN] 向量数与块数不一致")

# ---- 3. 完整问答链路 ----
print("\n[3] 完整问答链路")
from retriever import KeywordIndex
from hybrid_retriever import HybridRetriever
from vector_index import EmbeddingClient
from generator import RAGAnswerer, Hit

bm25 = KeywordIndex(chunks)
emb = EmbeddingClient()
hy = HybridRetriever(chunks, vec, emb)

q = "Pod 的生命周期有哪几个阶段"
rows = hy.search(q, k=3)
print(f"  检索「{q}」→ {len(rows)} 条")
for r in rows[:3]:
    print(f"    rrf={r.rrf_score:.6f} [{r.sources}]")

cmap = {c.chunk_id: c for c in chunks}
hits = [Hit(chunk=cmap[r.chunk_id], score=r.rrf_score, matched_terms=[])
        for r in rows if r.chunk_id in cmap]

answerer = RAGAnswerer(bm25, top_k=3)
a = answerer.answer(q, hits=hits)

print(f"\n  回答（{a.latency_ms:.0f}ms）：")
for line in a.text.splitlines()[:8]:
    if line.strip():
        print(f"    {line.strip()[:84]}")

print(f"\n  来源：")
for c in a.citations:
    mark = "引用" if c.index in a.used_indices else "未用"
    # 注意：Citation.page_label 是**字符串字段**（构造时已算好），
    # 与 Chunk.page_label()（方法）同名但类型不同。
    # 这里不加括号 —— 加了会 TypeError: 'str' object is not callable（实测踩过）。
    print(f"    [{c.index}] {c.source_file} {c.page_label} · {c.section}  ({mark})")

# ---- 4. 关键断言 ----
print("\n[4] 关键断言")
ok = True

# 页码必须是「节」不是「页」
# Citation.page_label 是字符串字段，不加括号
labels = [c.page_label for c in a.citations]
has_jie = any("节" in l for l in labels)
has_ye = any("页" in l for l in labels)
print(f"  {'OK  ' if has_jie and not has_ye else 'FAIL'} 来源用「节」不用「页」：{labels}")
ok &= (has_jie and not has_ye)

# 引用有效
print(f"  {'OK  ' if a.used_indices else 'FAIL'} 有有效引用：{a.used_indices}")
ok &= bool(a.used_indices)

# 无编造编号
print(f"  {'OK  ' if not a.invalid_indices else 'FAIL'} 无编造编号")
ok &= not a.invalid_indices

# 答案含关键内容
has_pending = "Pending" in a.text
print(f"  {'OK  ' if has_pending else 'FAIL'} 答案提到 Pending")
ok &= has_pending

# ---- 5. AppTest 跑完整 UI ----
print("\n[5] Streamlit UI 测试（AppTest）")
try:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PKG / "deploy" / "app.py"), default_timeout=300)
    at.run()

    if at.exception:
        print(f"  FAIL 启动异常：{str(at.exception[0].value)[:200]}")
        ok = False
    else:
        print(f"  OK   启动无异常，标题 {[t.value for t in at.title]}")

        at.chat_input[0].set_value(q).run(timeout=300)
        if at.exception:
            print(f"  FAIL 交互异常：{str(at.exception[0].value)[:300]}")
            ok = False
        else:
            n_msg = len(at.get("chat_message"))
            print(f"  OK   交互产生 {n_msg} 条 chat_message")
            ok &= n_msg >= 2

            mds = [m.value for m in at.markdown]
            has_jie_ui = any("节" in str(m) for m in mds)
            print(f"  {'OK  ' if has_jie_ui else 'FAIL'} UI 中来源显示「节」")
            ok &= has_jie_ui
except ImportError:
    print("  SKIP 当前 streamlit 无 testing.v1")

print()
print("=" * 84)
print(f" 结论：{'部署包可用' if ok else '有问题，见上面 FAIL 项'}")
print("=" * 84)
sys.exit(0 if ok else 1)
