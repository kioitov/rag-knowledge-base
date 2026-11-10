"""阶段 4：Streamlit 前端 —— 对话 + 来源展示。

============================== 一个必须先定的架构问题 ==============================

前端有两种接法：

    方案 A：直接 import 后端的检索/生成模块（进程内调用）
    方案 B：通过 HTTP 请求 stage3/api.py

我选 **B（HTTP）**，理由：

    1. **前端和后端解耦**。改前端不用重启后端，改后端不用重启前端
       （streamlit 每次交互都会重跑整个脚本，如果 import 后端，
        每次重跑都可能触发重复加载语料 —— 实测 645 块要几秒）。
    2. **能真实验证 API**。前端就是在调用接口，说明接口设计是可用的，
       而不是「有个 API 但没人用」。
    3. **部署形态一致**。上线时前后端本来就是分开部署的，
       现在就按这个形态写，不会到部署时才发现要改。

代价：多一层网络开销。但后端在我自己机器上，局域网延迟可忽略，
只有 embedding 那 ~200ms 是真实成本，跟方案 A 一样。

============================== 前端要解决的问题 ==============================

不追求好看，追求**信息完整**。每条回答必须让用户看到：

    1. 答案正文
    2. 引用了哪几条来源（[1][2]）
    3. 每条来源的文件名 + 页码 + 章节
    4. 哪些来源被正文真正引用了（cited=True），哪些只是检索到但没用上
    5. 延迟和 token 消耗（判断成本和性能）

第 4 点很多 RAG demo 不做 —— 它们把所有检索结果都列成「来源」，
但模型可能一条都没引用。**把「检索到的」和「被引用的」区分开，才是诚实的溯源。**

============================== 几个 Streamlit 的坑 ==============================

1. **脚本会反复重跑**
   每次交互，整个脚本从头执行。所以：
     - 语料/索引这类昂贵资源不能放在模块顶层
     - 用 @st.cache_resource 缓存连接，用 session_state 存对话历史

2. **不能用 st.cache_data 缓存 client 对象**
   cache_data 会序列化返回值，httpx 连接对象不可序列化。
   client 要用 cache_resource。

3. **对话历史要手动管理**
   Streamlit 没有内置的 chat memory，必须自己存进 session_state 并重放。

用法：
    # 先起后端
    python stage3/api.py
    # 再起前端
    python -m streamlit run stage4/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 后端连接（缓存，避免每次交互都新建）
# ---------------------------------------------------------------------------
@st.cache_resource
def get_session():
    """复用一个 requests.Session，避免每次请求重新建 TCP 连接。

    注意用 cache_resource 而不是 cache_data：
        cache_data 会尝试序列化返回值做缓存键，Session 对象不可序列化会报错。
        cache_resource 缓存的是对象本身，适合连接、客户端这类资源。
    """
    import requests

    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def api_get(path: str, params: dict | None = None, timeout: int = 30) -> dict:
    """GET 请求，统一错误处理。"""
    session = get_session()
    try:
        r = session.get(f"{api_base()}{path}", params=params, timeout=timeout)
    except Exception as exc:
        raise ConnectionError(
            f"连不上后端（{api_base()}）。\n"
            f"请先启动：python stage3/api.py\n"
            f"具体错误：{type(exc).__name__}: {exc}"
        ) from exc
    if r.status_code != 200:
        raise RuntimeError(f"后端返回 {r.status_code}：{r.text[:300]}")
    return r.json()


def api_post(path: str, payload: dict, timeout: int = 120) -> dict:
    """POST 请求，统一错误处理。

    timeout 给到 120s：混合检索要调 embedding（~1s）+ 大模型生成（~3-6s），
    但首次调用可能触发连接建立，给足余量。
    """
    session = get_session()
    try:
        r = session.post(f"{api_base()}{path}", json=payload, timeout=timeout)
    except Exception as exc:
        raise ConnectionError(
            f"连不上后端（{api_base()}）。\n"
            f"请先启动：python stage3/api.py\n"
            f"具体错误：{type(exc).__name__}: {exc}"
        ) from exc
    if r.status_code != 200:
        # 把后端的 detail 打出来 —— 502 通常是 LLM 侧问题，503 是索引没就绪
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"后端返回 {r.status_code}：{detail}")
    return r.json()


def api_base() -> str:
    return st.session_state.get("api_base", DEFAULT_API).rstrip("/")


# ---------------------------------------------------------------------------
# 会话状态初始化
# ---------------------------------------------------------------------------
def init_state() -> None:
    """初始化 session_state。Streamlit 每次交互都重跑脚本，状态必须显式初始化。"""
    if "messages" not in st.session_state:
        # 每条：{"role": "user"|"assistant", "content": str, "meta": dict|None}
        st.session_state.messages = []
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API


init_state()


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 知识库问答")
    st.caption("检索增强生成 · 回答带来源页码")

    st.divider()

    st.session_state.api_base = st.text_input(
        "后端地址", value=api_base(), help="stage3/api.py 的监听地址"
    )

    # 连接状态检测
    if st.button("检测连接", use_container_width=True):
        try:
            h = api_get("/health", timeout=10)
            st.success(f"已连接 · {h['chunks']} 块 · 向量 {h['vector_index']}")
            st.session_state.health = h
        except Exception as exc:
            st.error(str(exc))
            st.session_state.health = None

    health = st.session_state.get("health")
    if health:
        col1, col2 = st.columns(2)
        col1.metric("文档块", health["chunks"])
        col2.metric("向量", health["vector_index"])
        st.caption(f"模型 {health['model']}")
        st.caption(f"可用检索：{', '.join(health['method_available'])}")

    st.divider()

    method = st.selectbox(
        "检索方式",
        options=["hybrid", "vector", "bm25"],
        index=0,
        help="hybrid=BM25+向量融合（推荐）；vector=纯语义；bm25=纯关键词",
    )
    top_k = st.slider("检索段落数", 1, 10, 3, help="喂给模型的资料条数")

    st.divider()
    show_debug = st.toggle("显示调试信息", value=False, help="展示延迟/token/引用编号")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("启动顺序：")
    st.code("python stage3/api.py\npython -m streamlit run stage4/app.py",
            language="bash")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 渲染函数
# ---------------------------------------------------------------------------
# ⚠️ 这个函数必须定义在「重放历史」之前。
# 第一版写成了「先 for 循环重放历史、再 def render_meta」，
# 结果只要对话历史里有一条带 meta 的消息，脚本重跑时就会 NameError。
# Streamlit 是自上而下顺序执行的，函数定义位置就是生效位置，不会被提升。
def render_meta(meta: dict, debug: bool) -> None:
    """渲染来源与调试信息。历史重放和实时响应共用同一套渲染。"""
    sources = meta.get("sources", [])
    if sources:
        cited = [s for s in sources if s.get("cited")]
        uncited = [s for s in sources if not s.get("cited")]

        # 正文真正引用的来源 —— 这是「溯源」的核心
        if cited:
            with st.expander(f"📎 引用的来源（{len(cited)} 条）", expanded=True):
                for s in cited:
                    loc = f"**[{s['index']}]** 《{s['file']}》 {s['page']}"
                    if s.get("section"):
                        loc += f" · {s['section']}"
                    st.markdown(loc)

        # 检索到但没被引用的 —— 区分开才诚实
        if uncited:
            with st.expander(f"🔍 检索到但未被引用（{len(uncited)} 条）", expanded=False):
                st.caption("这些段落被检索出来了，但模型回答时没有用到。")
                for s in uncited:
                    loc = f"[{s['index']}] 《{s['file']}》 {s['page']}"
                    if s.get("section"):
                        loc += f" · {s['section']}"
                    st.markdown(loc)

    if debug:
        parts = []
        if meta.get("latency_ms"):
            parts.append(f"延迟 {meta['latency_ms']:.0f}ms")
        tok = meta.get("tokens") or {}
        if tok.get("total_tokens"):
            parts.append(
                f"tokens {tok['total_tokens']}"
                f"（入 {tok.get('prompt_tokens', '?')} / 出 {tok.get('completion_tokens', '?')}）"
            )
        if parts:
            st.caption(" · ".join(parts))

        used = meta.get("used_indices") or []
        invalid = meta.get("invalid_indices") or []
        st.caption(f"引用编号：{used or '无'}")

        # 编造编号是严重问题，必须显眼提示
        if invalid:
            st.error(
                f"⚠️ 模型标注了不存在的引用编号 {invalid} —— "
                f"说明它引用了不存在的资料，回答可信度存疑。"
            )
        elif not used and sources:
            st.warning("⚠️ 检索到了资料，但回答里没有任何有效引用。")


# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
st.title("知识库问答")
st.caption("提问后会检索文档并给出带来源的回答。资料里没有的内容，系统会说「无法确定」。")

# 重放历史（此时 render_meta 已定义）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            render_meta(meta, show_debug)


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
if prompt := st.chat_input("问点什么，比如「这个系统能装多少人同时用」"):
    # 1. 立刻显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调后端
    with st.chat_message("assistant"):
        with st.spinner(f"检索（{method}）并生成…"):
            try:
                resp = api_post("/chat", {
                    "question": prompt,
                    "top_k": top_k,
                    "method": method,
                })
                answer = resp["answer"]
                meta = {
                    "sources": resp.get("sources", []),
                    "used_indices": resp.get("used_indices", []),
                    "invalid_indices": resp.get("invalid_indices", []),
                    "latency_ms": resp.get("latency_ms", 0),
                    "tokens": resp.get("tokens", {}),
                }
                st.markdown(answer)
                render_meta(meta, show_debug)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": meta,
                })
            except Exception as exc:
                # 错误要显示成用户能看懂的话，同时保留原始信息便于排查
                msg = str(exc)
                st.error(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"请求失败：{msg}",
                    "meta": None,
                })
