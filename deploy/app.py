"""Streamlit Cloud 单进程部署入口。

============================== 为什么不再走 HTTP ==============================

阶段 4 的前端是通过 HTTP 调 `stage3/api.py` 的，那是本地开发的合理架构
（前后端解耦、能真实验证 API）。

但 Streamlit Community Cloud **只能跑一个进程**，起不了第二个 FastAPI 服务。
所以这个入口把检索和生成直接 import 进来，在同进程里完成。

**代价**：失去了前后端解耦，改后端要重启前端。
**收益**：单服务部署，冷启动少一次网络往返。

阶段 4 的 `app.py` 保留不动 —— 本地开发仍用它的 HTTP 模式，
两种模式共存，各自服务于不同场景。

============================== 部署环境的三个硬约束 ==============================

1. **文件系统是临时的**
   每次重启容器，写进去的文件全丢。所以：
   - 向量索引必须在启动时重建，或从仓库里读
   - 不能依赖 data/cache 里的解析缓存

   策略：**语料随仓库部署 + 启动时构建索引**。
   1290 块的 embedding 要 ~50 秒 + 129 次 API 调用。太慢且费钱。

   改进：**把向量索引预先构建好，序列化后随仓库部署**。
   Chroma 的持久化目录可以直接打包，启动时直接加载。见 `prebuild_index.py`。

2. **内存约 1–2.7 GB**
   Chroma + 1290 个 1024 维向量 ≈ 1290 × 1024 × 4 字节 ≈ 5.3 MB，可忽略。
   但 pymupdf 解析大 PDF 会吃内存 —— 所以部署版**不解析 PDF**，
   直接用预切好的块（随仓库部署的 JSONL）。

3. **没有 .env 文件**
   Streamlit Cloud 用 Secrets 管理，通过 `st.secrets` 读取。
   需要同时兼容本地 `.env` 和云端 secrets。

============================== 启动流程 ==============================

    首次访问
      ├─ 读 data/deploy/chunks.jsonl（预切好的块，随仓库）
      ├─ 建 BM25 索引（纯内存，0.1 秒）
      ├─ 加载 data/deploy/chroma（预建向量索引，随仓库）
      └─ 缓存到 session_state，后续请求直接复用

用法（本地验证）：
    python -m streamlit run deploy/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 定位项目根目录
# ---------------------------------------------------------------------------
# 这个文件要在两种位置都能跑：
#   本地开发：deploy/app.py        → 根目录是 parent.parent
#   魔搭云端：app.py 在仓库根目录   → 根目录是 parent
#
# 平台强制要求入口文件叫 app.py 且放在根目录，所以部署时必须搬上去。
# 判断依据：如果当前目录下已经有 stage1/ 和 data/，说明自己就在根目录。
_THIS_DIR = Path(__file__).resolve().parent


def _find_root() -> Path:
    """从当前文件位置往上找，第一个同时含 stage1/ 和 data/ 的目录就是根。"""
    for cand in (_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent):
        if (cand / "stage1").is_dir() and (cand / "data").is_dir():
            return cand
    # 兜底：按部署到子目录的老规则
    return _THIS_DIR.parent


_ROOT = _find_root()
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage2"))
sys.path.insert(0, str(_ROOT / "stage3"))

DEPLOY_DIR = _ROOT / "data" / "deploy"


# ---------------------------------------------------------------------------
# 配置：兼容本地 .env 与云端 st.secrets
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """按优先级取配置：st.secrets > 环境变量 > .env。

    Streamlit Cloud 把 Secrets 注入到 st.secrets，
    本地开发则读 .env。两边字段名保持一致。
    """
    cfg = {"api_key": "", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
           "model": "qwen-plus"}

    # 1. st.secrets 优先（云端）
    try:
        sec = st.secrets
        cfg["api_key"] = sec.get("DASHSCOPE_API_KEY", "") or ""
        cfg["base_url"] = sec.get("LLM_BASE_URL", cfg["base_url"])
        cfg["model"] = sec.get("LLM_MODEL", cfg["model"])
    except Exception:
        pass  # 本地没有 secrets.toml 时 st.secrets 会抛异常

    # 2. 环境变量 / .env 兜底
    if not cfg["api_key"]:
        try:
            from dotenv import load_dotenv
            env = _ROOT / ".env"
            if env.exists():
                load_dotenv(env, override=False)
        except ImportError:
            pass
        cfg["api_key"] = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )
        cfg["base_url"] = os.getenv("LLM_BASE_URL", cfg["base_url"])
        cfg["model"] = os.getenv("LLM_MODEL", cfg["model"])

    return cfg


# ---------------------------------------------------------------------------
# 加载语料与索引（缓存，避免每次交互重建）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    """加载切块、BM25 索引、向量索引。

    用 cache_resource：这些是不可序列化的大对象，需要用对象缓存而不是数据缓存。

    返回 (chunks, bm25_index, vector_index, embedder, answerer_factory)
    """
    from chunker import Chunk
    from retriever import KeywordIndex
    from vector_index import EmbeddingClient, VectorIndex

    # --- 1. 读预切块（JSONL，随仓库部署）---
    chunks_file = DEPLOY_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"缺少 {chunks_file}。\n"
            f"请先运行：python deploy/prebuild_index.py"
        )

    chunks: list[Chunk] = []
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(
                text=d["text"],
                source_file=d["source_file"],
                pages=d["pages"],
                section=d["section"],
                kind=d["kind"],
                strategy=d["strategy"],
                chunk_id=d["chunk_id"],
                meta=d.get("meta", {}),
            ))

    # --- 2. BM25（纯内存，很快）---
    bm25 = KeywordIndex(chunks)

    # --- 3. 向量索引（预建，随仓库）---
    vec = VectorIndex(DEPLOY_DIR / "chroma", collection="kb_chunks")

    # --- 4. embedding 客户端 ---
    cfg = load_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "没有配置 API Key。\n"
            "本地：在 .env 里设 DASHSCOPE_API_KEY\n"
            "云端：在 Space Settings → Secrets 里设 DASHSCOPE_API_KEY"
        )
    # 用配置覆盖环境变量，让 EmbeddingClient 能读到
    os.environ["DASHSCOPE_API_KEY"] = cfg["api_key"]
    os.environ["LLM_BASE_URL"] = cfg["base_url"]
    os.environ["LLM_MODEL"] = cfg["model"]
    embedder = EmbeddingClient()

    return chunks, bm25, vec, embedder


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="知识库问答（Kubernetes 中文文档）",
    page_icon="📚",
    layout="wide",
)


def main() -> None:
    st.title("📚 知识库问答")
    st.caption(
        "基于 Kubernetes 官方中文文档（CC BY 4.0）的检索增强问答。"
        "回答带来源定位，资料里没有的内容会说「无法确定」。"
    )

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("设置")
        method = st.selectbox(
            "检索方式",
            ["hybrid", "bm25", "vector"],
            index=0,
            help="hybrid=BM25+向量融合；bm25=纯关键词（最快）；vector=纯语义",
        )
        top_k = st.slider("检索段落数", 1, 6, 3)
        show_debug = st.toggle("显示调试信息", value=False)

        st.divider()
        st.caption("语料：Kubernetes 中文文档 63 篇")
        st.caption("© The Kubernetes Authors · CC BY 4.0")

        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---- 加载引擎 ----
    try:
        with st.spinner("加载索引…"):
            chunks, bm25, vec, embedder = load_engine()
    except Exception as exc:
        st.error(f"初始化失败：{exc}")
        st.stop()

    # 检索器按需构建
    from hybrid_retriever import HybridRetriever

    if "hybrid" not in st.session_state:
        st.session_state.hybrid = HybridRetriever(chunks, vec, embedder)

    # ---- 对话历史 ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_meta(meta: dict) -> None:
        """渲染来源。区分「被引用的」和「检索到但没用的」。"""
        sources = meta.get("sources", [])
        if sources:
            cited = [s for s in sources if s.get("cited")]
            uncited = [s for s in sources if not s.get("cited")]
            if cited:
                with st.expander(f"📎 引用的来源（{len(cited)} 条）", expanded=True):
                    for s in cited:
                        line = f"**[{s['index']}]** 《{s['file']}》 {s['page']}"
                        if s.get("section"):
                            line += f" · {s['section']}"
                        st.markdown(line)
            if uncited:
                with st.expander(f"🔍 检索到但未被引用（{len(uncited)} 条）"):
                    st.caption("这些段落被检索出来了，但模型回答时没有用到。")
                    for s in uncited:
                        st.markdown(f"[{s['index']}] 《{s['file']}》 {s['page']}")

        if show_debug:
            parts = []
            if meta.get("latency_ms"):
                parts.append(f"延迟 {meta['latency_ms']:.0f}ms")
            tok = meta.get("tokens") or {}
            if tok.get("total_tokens"):
                parts.append(f"tokens {tok['total_tokens']}")
            if parts:
                st.caption(" · ".join(parts))
            st.caption(f"引用编号：{meta.get('used_indices') or '无'}")
            if meta.get("invalid_indices"):
                st.error(f"⚠️ 模型标注了不存在的引用编号 {meta['invalid_indices']}")

    # 重放历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                render_meta(msg["meta"])

    # ---- 输入 ----
    if prompt := st.chat_input("问点什么，比如「Pod 的生命周期有哪几个阶段」"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索并生成…"):
                try:
                    # 检索
                    if method == "bm25":
                        raw = bm25.search(prompt, top_k=top_k)
                        hits = raw
                    elif method == "vector":
                        from generator import Hit
                        cmap = {c.chunk_id: c for c in chunks}
                        rows = vec.search(prompt, embedder, top_k=top_k)
                        hits = [
                            Hit(chunk=cmap[cid], score=1 - dist, matched_terms=[])
                            for cid, _doc, _m, dist in rows
                            if cid in cmap
                        ]
                    else:
                        from generator import Hit
                        cmap = {c.chunk_id: c for c in chunks}
                        rows = st.session_state.hybrid.search(prompt, k=top_k)
                        hits = [
                            Hit(chunk=cmap[r.chunk_id], score=r.rrf_score, matched_terms=[])
                            for r in rows
                            if r.chunk_id in cmap
                        ]

                    # 生成
                    from generator import RAGAnswerer
                    answerer = RAGAnswerer(bm25, top_k=top_k)
                    a = answerer.answer(prompt, hits=hits)

                    meta = {
                        "sources": [
                            {
                                "index": c.index,
                                "file": c.source_file,
                                "page": c.page_label,
                                "section": c.section,
                                "cited": c.index in a.used_indices,
                            }
                            for c in a.citations
                        ],
                        "used_indices": a.used_indices,
                        "invalid_indices": a.invalid_indices,
                        "latency_ms": a.latency_ms,
                        "tokens": a.usage,
                    }

                    st.markdown(a.text)
                    render_meta(meta)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": a.text, "meta": meta}
                    )
                except Exception as exc:
                    msg = f"出错了：{type(exc).__name__}: {exc}"
                    st.error(msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg, "meta": None}
                    )


main()
