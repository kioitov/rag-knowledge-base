"""预构建部署产物：把切块和向量索引打包，随仓库部署。

============================== 为什么必须预构建 ==============================

Streamlit Cloud 的容器文件系统是临时的，且免费层冷启动时间敏感。
如果启动时才解析语料 + 调 embedding：

    解析 63 篇 Markdown    0.1 秒（快，不是问题）
    构建向量索引            1290 块 → 129 次 API 调用 → 约 50 秒
    加上 API 限流和网络抖动  实际可能 1-2 分钟

每次容器重启都要重来一遍，用户体验很差，而且白白烧 token。

所以：**在本地构建一次，产物提交到仓库，线上直接加载。**

============================== 产物组成 ==============================

    data/deploy/
    ├── chunks.jsonl     预切好的块（含 chunk_id、页码、章节）
    └── chroma/          Chroma 持久化目录（向量 + 元数据）

体积估算：
    1290 块 × 平均 444 字 ≈ 570 KB（文本）
    向量 1290 × 1024 × 4 字节 ≈ 5.3 MB
    加上 Chroma 的 sqlite 索引，总共约 10-15 MB

这个体积提交到 git 可以接受（没到需要 LFS 的程度）。

============================== 一个需要注意的点 ==============================

chunks.jsonl 里必须保留 `meta` 字段（尤其是 `paged: False`），
否则线上加载后 `page_label()` 会退化成「第 N 页」而不是「第 N 节」——
又把 Markdown 的章节序号伪装成页码，破坏溯源准确性。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))
sys.path.insert(0, str(_ROOT / "stage3"))

from cache import load_chunks
from chunker import Chunk
from paths import find_docs
from vector_index import EmbeddingClient, VectorIndex

DEPLOY_DIR = _ROOT / "data" / "deploy"


def main() -> None:
    print("=" * 84)
    print(" 预构建部署产物")
    print("=" * 84)

    docs = find_docs()
    if not docs:
        raise SystemExit(
            "没找到语料。先跑：python stage5/fetch_corpus.py"
        )

    # ---- 1. 切块 ----
    print(f"\n[1/3] 切块（{len(docs)} 篇文档）")
    chunks: list[Chunk] = []
    # 部署用 paragraph 策略：评测显示它在多数指标上最稳
    for d in docs:
        chunks.extend(load_chunks(d, strategy="paragraph"))
    print(f"  {len(chunks)} 块 / {sum(len(c.text) for c in chunks):,} 字")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 2. 写 chunks.jsonl ----
    print(f"\n[2/3] 写切块清单")
    chunks_file = DEPLOY_DIR / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps({
                "text": c.text,
                "source_file": c.source_file,
                "pages": c.pages,
                "section": c.section,
                "kind": c.kind,
                "strategy": c.strategy,
                "chunk_id": c.chunk_id,
                # meta 必须带上，否则 paged 标记丢失，页码会退化
                "meta": c.meta,
            }, ensure_ascii=False) + "\n")
    size_kb = chunks_file.stat().st_size / 1024
    print(f"  {chunks_file.name}  {size_kb:.0f} KB")

    # 验证 paged 标记
    n_paged_false = sum(1 for c in chunks if not c.meta.get("paged", True))
    print(f"  其中 paged=False（Markdown 无页码）：{n_paged_false} 块")

    # ---- 3. 建向量索引 ----
    print(f"\n[3/3] 构建向量索引")
    vec_dir = DEPLOY_DIR / "chroma"
    if vec_dir.exists():
        shutil.rmtree(vec_dir)
        print(f"  清空旧索引 {vec_dir.name}")

    vec = VectorIndex(vec_dir, collection="kb_chunks")
    embedder = EmbeddingClient()
    t0 = time.perf_counter()
    stats = vec.build(chunks, embedder, force=True)
    print(f"  {stats.summary()}")
    print(f"  集合块数 {vec.count()}")

    # ---- 汇总 ----
    total = sum(f.stat().st_size for f in DEPLOY_DIR.rglob("*") if f.is_file())
    print()
    print("=" * 84)
    print(f" 产物目录 {DEPLOY_DIR}")
    print(f" 总大小 {total / 1024 / 1024:.1f} MB")
    print(f" 文件数 {len([f for f in DEPLOY_DIR.rglob('*') if f.is_file()])}")
    print()
    print(" 下一步：")
    print("   1. 确认 .gitignore 没有排除 data/deploy/")
    print("   2. git add data/deploy/ && git commit")
    print("   3. 推送后在 Streamlit Cloud 新建 app，入口填 deploy/app.py")
    print("=" * 84)


if __name__ == "__main__":
    main()
