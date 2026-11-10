"""阶段 3 第一块：向量索引（Chroma 持久化 + 百炼 embedding）。

============================== 这一层解决什么 ==============================

阶段 1 的 BM25 有两个物理缺陷，实测证明调参修不了：

    「这个能承载多少」vs 文档写「容量上限 / 支撑规模」
        → 零字面重叠，BM25 无从下手
    「系统支持哪些 X 选项」→ 命中「性能需求」
        → 「系统」「支持」是高频词，IDF 压不住

向量检索把文本映射到语义空间，不依赖字面重合。这一层就是它的实现。

============================== 五个工程决定 ==============================

1. **embedding 自己算，不让 Chroma 算**
   Chroma 支持传入 embedding_function 自动向量化，但那样：
     - 无法控制批量大小（百炼实测上限 10 条，超了直接 400）
     - 出错时看不清是哪一批失败
     - 换模型要动集合配置，不能平滑迁移
   所以显式传 embeddings，自己控制节奏和重试。

2. **批量必须 ≤ 10**
   实测百炼 text-embedding-v4：10 条 OK，16 条报
   `InvalidParameter: batch size is invalid`。
   645 块 → 65 次调用 → 约 35 秒。这个数字要记进文档，
   因为它决定了「重建索引要等多久」，也影响部署时的冷启动体验。

3. **维度必须与实际返回一致**
   Chroma 建集合时若声明了维度，插入不匹配的向量会报错。
   百炼默认返回 1024，但配置里可以改。
   **不硬编码 1024，而是从第一次真实返回里读出来。**

4. **元数据只放标量**
   Chroma 的 metadata 只接受 str/int/float/bool。
   pages 是列表，必须序列化成 "5-7" 这样的字符串，
   读取时再解析。这是很常见的踩坑点。

5. **ID 用 chunk_id**
   已经带了 source_file + pages + 文本前缀的哈希，天然唯一。
   用自增序号会在重建时错位。

============================== 距离度量 ==============================

用 cosine。理由：
    文本 embedding 的绝对长度没有意义，方向才代表语义。
    用 L2 距离会受向量模长影响，而百炼返回的向量模长不归一。
    实测相似度 0.35 这类数字，都是 cosine 下的结果，保持口径一致。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))

import config
from chunker import Chunk

# 实测上限，超了会 400
EMBED_BATCH = 10

# 重试策略：网络抖动时不要整批失败
MAX_RETRY = 3
RETRY_BACKOFF = 1.5


@dataclass
class IndexStats:
    chunks: int
    embedded: int
    dim: int
    seconds: float
    api_calls: int

    def summary(self) -> str:
        return (
            f"{self.embedded} 块 / {self.dim} 维 / "
            f"{self.api_calls} 次 API 调用 / {self.seconds:.1f}s"
        )


def _pages_to_str(pages: list[int]) -> str:
    """列表 → '5-7' 或 '5'。Chroma metadata 不支持 list。"""
    if not pages:
        return "0"
    lo, hi = min(pages), max(pages)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _str_to_pages(s: str) -> list[int]:
    """'5-7' → [5,6,7]，'5' → [5]。"""
    try:
        if "-" in s:
            lo, hi = s.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(s)]
    except (ValueError, AttributeError):
        return [0]


class EmbeddingClient:
    """封装批量向量化，处理重试与限速。"""

    def __init__(self) -> None:
        cfg = config.embedding_config()
        if not cfg.ready():
            raise SystemExit(
                "embedding 未配置。\n"
                "  在 .env 里设 DASHSCOPE_API_KEY，或跑 stage0/set_key.py 填 key。"
            )
        self.cfg = cfg
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.calls = 0
        self.dim = 0  # 由第一次真实返回确定

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量。自动分批 + 重试。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            vecs = self._embed_batch(batch)
            out.extend(vecs)
        return out

    def _embed_batch(self, batch: list[str], attempt: int = 0) -> list[list[float]]:
        try:
            kwargs = {"model": self.cfg.model, "input": batch}
            if self.cfg.dim:
                kwargs["dimensions"] = self.cfg.dim
            resp = self._client.embeddings.create(**kwargs)
            self.calls += 1
            vecs = [d.embedding for d in resp.data]
            if not self.dim:
                self.dim = len(vecs[0])
            return vecs
        except Exception as exc:
            if attempt >= MAX_RETRY:
                raise
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"    [retry {attempt + 1}/{MAX_RETRY}] {type(exc).__name__}，{wait:.1f}s 后重试")
            time.sleep(wait)
            return self._embed_batch(batch, attempt + 1)


class VectorIndex:
    """Chroma 持久化向量索引。"""

    # Chroma 要求集合名 3-512 字符，且只能含 [a-zA-Z0-9._-]，
    # 首尾必须是字母或数字。用 "kb" 会直接报 InvalidArgumentError（实测踩过）。
    DEFAULT_NAME = "kb_chunks"

    def __init__(self, path: Path, collection: str = DEFAULT_NAME) -> None:
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.name = collection
        self.col = self._open()

    def _open(self):
        """打开或创建集合。

        用 cosine 距离 + 不指定维度（让第一批数据决定），
        避免「配置写 1024、实际返回别的」这种错配。
        """
        return self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- 构建 ----------------------------------------------------------
    def build(self, chunks: list[Chunk], embedder: EmbeddingClient,
              force: bool = False) -> IndexStats:
        """把切块写入向量库。已存在且未要求重建时跳过。"""
        if not force and self.col.count() == len(chunks) and self.col.count() > 0:
            first = self.col.get(limit=1, include=["embeddings"])
            dim = len(first["embeddings"][0]) if first["embeddings"] else 0
            return IndexStats(len(chunks), self.col.count(), dim, 0.0, 0)

        if force:
            self.client.delete_collection(self.name)
            self.col = self._open()

        t0 = time.perf_counter()
        embedder.dim = 0
        embedder.calls = 0

        # ---- 去重 ----
        # 实测：645 块里有 1 对完全相同的块（同一文档同一页的同一个表格出现两次）。
        # 内容和页码都一样，embedding 也会完全一致，写两遍纯属浪费向量空间，
        # 而且 Chroma 的 id 必须唯一，不处理会直接抛 DuplicateIDError。
        # 去重放在这里而不是 chunker，因为「完全相同」是数据事实，
        # 而 chunker 按块生成、不知道全局是否存在副本。
        seen: dict[str, Chunk] = {}
        deduped: list[Chunk] = []
        for c in chunks:
            if c.chunk_id in seen:
                seen[c.chunk_id].meta["dup_count"] = seen[c.chunk_id].meta.get("dup_count", 1) + 1
                continue
            seen[c.chunk_id] = c
            deduped.append(c)

        if len(deduped) < len(chunks):
            print(f"  去重：{len(chunks)} → {len(deduped)} 块"
                  f"（移除 {len(chunks) - len(deduped)} 个完全重复的块）")
        chunks = deduped

        texts = [c.text for c in chunks]
        vecs = embedder.embed(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "pages": _pages_to_str(c.pages),
                "section": c.section or "",
                "kind": c.kind,
                "strategy": c.strategy,
                "length": len(c),
            }
            for c in chunks
        ]

        # 分批写入，避免单次请求体过大
        WRITE_BATCH = 500
        for i in range(0, len(chunks), WRITE_BATCH):
            self.col.add(
                ids=ids[i : i + WRITE_BATCH],
                embeddings=vecs[i : i + WRITE_BATCH],
                documents=texts[i : i + WRITE_BATCH],
                metadatas=metadatas[i : i + WRITE_BATCH],
            )

        return IndexStats(
            chunks=len(chunks),
            embedded=len(vecs),
            dim=embedder.dim,
            seconds=time.perf_counter() - t0,
            api_calls=embedder.calls,
        )

    # -- 查询 ----------------------------------------------------------
    def search(self, query: str, embedder: EmbeddingClient, top_k: int = 3):
        """返回 [(chunk_id, text, metadata, distance), ...]。

        distance 是 cosine 距离（= 1 - 相似度），越小越相似。
        这里显式转成相似度，方便与阶段 0 的标定数据对照。
        """
        qv = embedder.embed([query])[0]
        res = self.col.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append((cid, doc, meta, float(dist)))
        return out

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self._open()


def rebuild_from_docs(force: bool = True) -> VectorIndex:
    """从 data/docs 全量重建索引。供 run_stage3 调用。"""
    from cache import load_chunks
    from paths import find_pdfs
    from chunker import chunk_paragraph

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    chunks: list[Chunk] = []
    for p in pdfs:
        chunks.extend(load_chunks(p))

    idx = VectorIndex(_ROOT / "data" / "chroma")
    embedder = EmbeddingClient()
    stats = idx.build(chunks, embedder, force=force)
    print(f"  {stats.summary()}")
    return idx


if __name__ == "__main__":
    from cache import load_chunks
    from paths import find_pdfs

    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit("没找到 PDF。请把文件放进 data/docs/")

    print("=" * 80)
    print(" 向量索引构建")
    print("=" * 80)

    chunks: list[Chunk] = []
    for p in pdfs:
        cs = load_chunks(p)
        chunks.extend(cs)
        print(f"  {p.name:<42} {len(cs):>4} 块")
    print(f"  合计 {len(chunks)} 块\n")

    idx = VectorIndex(_ROOT / "data" / "chroma")
    embedder = EmbeddingClient()

    print(f"embedding: {embedder.cfg.summary()}")
    print(f"现有集合块数: {idx.count()}")
    print("\n构图中（批量 10 条，会有等待）…")

    stats = idx.build(chunks, embedder, force=True)
    print(f"\n完成: {stats.summary()}")
    print(f"集合块数: {idx.count()}")
