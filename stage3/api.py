"""阶段 3 第三块：FastAPI 接口，用 /docs 直接测试问答。

============================== 交接文档的要求 ==============================

阶段 3 原文：「用 FastAPI 包一个 /chat 接口，打开 /docs 能测试，
验收：curl 或浏览器能问问题并拿到答案。」

============================== 接口设计 ==============================

    GET  /health          健康检查 + 索引状态
    POST /chat            问答（主接口）
    GET  /search          只检索不生成（调试用，省 token）

为什么分 /chat 和 /search：
    调试检索质量时不想每次都花钱调模型。/search 只返回检索结果，
    排查「答得不对是检索的问题还是模型的问题」时非常有用。

============================== 几个工程细节 ==============================

1. **模型客户端与索引在启动时初始化一次**
   放在 lifespan 里，不要每个请求新建。否则每次请求都要重新
   load 语料（实测 645 块要几秒），接口会慢到不可用。

2. **请求体用 Pydantic 校验**
   question 长度限制、top_k 范围限制。不校验的话，
   一个空字符串或 top_k=10000 就能打挂服务。

3. **同步执行，不装异步**
   embedder 和 LLM 调用都是同步阻塞的。用 async def 但内部调同步函数，
   会阻塞事件循环，反而更慢。这里用 def（FastAPI 会自动丢线程池）。
   要真正支持高并发，得用 AsyncOpenAI —— 但那是优化，不是现在该做的。

4. **错误要可读**
   检索失败、LLM 超时、key 没配，分别返回不同的错误信息，
   而不是统一的 500。

用法：
    python stage3/api.py
    然后打开 http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage2"))
sys.path.insert(0, str(_ROOT / "stage3"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from cache import load_chunks
from chunker import Chunk
from generator import Hit, RAGAnswerer, verify_citations
from paths import find_pdfs
from retriever import KeywordIndex

# 全局状态，在 lifespan 里初始化
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载语料和索引，关闭时释放。

    放这里而不是模块顶层，是因为模块被 import 时（如 uvicorn --reload
    的检测进程）不该触发昂贵的加载。
    """
    print("启动中：加载语料…")
    pdfs = find_pdfs()
    if not pdfs:
        print("[WARN] data/docs 下没有 PDF，服务将以空索引启动")
        chunks: list[Chunk] = []
    else:
        chunks = []
        for p in pdfs:
            chunks.extend(load_chunks(p))
        print(f"  {len(pdfs)} 份文档 / {len(chunks)} 块")

    STATE["chunks"] = chunks
    STATE["chunk_map"] = {c.chunk_id: c for c in chunks}
    STATE["bm25"] = KeywordIndex(chunks)

    # 向量索引与混合检索（可选：没配 embedding 时降级为纯 BM25）
    try:
        from hybrid_retriever import HybridRetriever
        from vector_index import EmbeddingClient, VectorIndex

        vec = VectorIndex(_ROOT / "data" / "chroma")
        embedder = EmbeddingClient()
        STATE["vec"] = vec
        STATE["embedder"] = embedder
        STATE["hybrid"] = HybridRetriever(chunks, vec, embedder)
        print(f"  向量集合 {vec.count()} 块，混合检索就绪")
    except Exception as exc:
        print(f"  [WARN] 向量检索不可用（{type(exc).__name__}），降级为纯 BM25")
        STATE["hybrid"] = None

    STATE["answerer"] = RAGAnswerer(STATE["bm25"], top_k=3)
    print(f"就绪。模型：{config.chat_model()}")
    yield
    STATE.clear()


app = FastAPI(
    title="企业知识库 RAG 问答服务",
    description="基于检索增强生成的文档问答，回答带来源页码。",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="检索段落数，1-10")
    method: str = Field("hybrid", pattern="^(hybrid|bm25|vector)$",
                        description="检索方式")


class SourceItem(BaseModel):
    index: int = Field(..., description="对应回答里的 [n]")
    file: str
    page: str
    section: str
    cited: bool = Field(..., description="正文是否真的引用了它")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    used_indices: list[int]
    invalid_indices: list[int] = Field(
        default_factory=list, description="模型编造的、不存在的引用编号"
    )
    latency_ms: float
    tokens: dict = Field(default_factory=dict)


class SearchHitItem(BaseModel):
    score: float
    file: str
    page: str
    section: str
    kind: str
    text: str


class SearchResponse(BaseModel):
    question: str
    method: str
    hits: list[SearchHitItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    chunks: int
    vector_index: int
    method_available: list[str]
    model: str


# ---------------------------------------------------------------------------
# 检索适配
# ---------------------------------------------------------------------------
def _retrieve(question: str, top_k: int, method: str) -> list[Hit]:
    """按 method 取检索结果，统一成 generator 需要的 Hit。"""
    chunk_map = STATE["chunk_map"]

    if method == "bm25" or not STATE.get("hybrid"):
        return STATE["bm25"].search(question, top_k=top_k)

    if method == "vector":
        raw = STATE["vec"].search(question, STATE["embedder"], top_k=top_k)
        return [
            Hit(chunk=chunk_map[cid], score=1 - dist, matched_terms=[])
            for cid, _doc, _m, dist in raw
            if cid in chunk_map
        ]

    # hybrid
    hr = STATE["hybrid"].search(question, k=top_k)
    return [
        Hit(chunk=chunk_map[h.chunk_id], score=h.rrf_score, matched_terms=[])
        for h in hr
        if h.chunk_id in chunk_map
    ]


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    """返回服务与索引状态。"""
    if not STATE.get("chunks"):
        raise HTTPException(503, "索引尚未就绪")
    methods = ["bm25"]
    if STATE.get("hybrid"):
        methods += ["vector", "hybrid"]
    return HealthResponse(
        status="ok",
        chunks=len(STATE["chunks"]),
        vector_index=STATE["vec"].count() if STATE.get("vec") else 0,
        method_available=methods,
        model=config.chat_model(),
    )


@app.post("/chat", response_model=ChatResponse, summary="问答（带来源）")
def chat(req: ChatRequest) -> ChatResponse:
    """检索 + 生成，返回带来源标注的回答。"""
    if not STATE.get("chunks"):
        raise HTTPException(503, "索引尚未就绪，请检查 data/docs 下是否有 PDF")

    try:
        hits = _retrieve(req.question, req.top_k, req.method)
    except Exception as exc:
        raise HTTPException(500, f"检索失败：{type(exc).__name__}: {exc}") from exc

    try:
        a = STATE["answerer"].answer(req.question, hits=hits)
    except Exception as exc:
        # 常见原因：key 无效 / 限流 / 网络。把类型带出去便于排查
        raise HTTPException(502, f"生成失败：{type(exc).__name__}: {exc}") from exc

    sources = [
        SourceItem(
            index=c.index,
            file=c.source_file,
            page=c.page_label,
            section=c.section,
            cited=c.index in a.used_indices,
        )
        for c in a.citations
    ]

    return ChatResponse(
        answer=a.text,
        sources=sources,
        used_indices=a.used_indices,
        invalid_indices=a.invalid_indices,
        latency_ms=round(a.latency_ms, 1),
        tokens=a.usage,
    )


@app.get("/search", response_model=SearchResponse, summary="只检索不生成（调试用）")
def search(q: str, top_k: int = 3, method: str = "hybrid") -> SearchResponse:
    """只返回检索结果，不调模型。调试检索质量时用，省 token。"""
    if not STATE.get("chunks"):
        raise HTTPException(503, "索引尚未就绪")
    if not (1 <= top_k <= 20):
        raise HTTPException(400, "top_k 必须在 1-20 之间")
    if method not in ("bm25", "vector", "hybrid"):
        raise HTTPException(400, "method 必须是 bm25 / vector / hybrid")

    import time

    t0 = time.perf_counter()
    hits = _retrieve(q, top_k, method)
    dt = (time.perf_counter() - t0) * 1000

    items = [
        SearchHitItem(
            score=round(h.score, 4),
            file=h.chunk.source_file,
            page=h.chunk.page_label(),
            section=h.chunk.section,
            kind=h.chunk.kind,
            text=h.chunk.text[:300],
        )
        for h in hits
    ]
    return SearchResponse(
        question=q, method=method, hits=items, latency_ms=round(dt, 1)
    )


@app.get("/", summary="根路径")
def root() -> dict:
    return {
        "service": "企业知识库 RAG 问答服务",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
        "search": "GET /search?q=...",
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print(" 启动 API 服务")
    print("=" * 70)
    print("  打开 http://127.0.0.1:8000/docs 直接测试")
    print("  或 curl -X POST http://127.0.0.1:8000/chat \\")
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"question":"系统采用什么架构"}\'')
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
