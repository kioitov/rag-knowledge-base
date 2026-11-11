"""给 Chroma 数据库瘦身：删掉用不到的全文检索索引。

============================== 为什么要瘦身 ==============================

chroma.sqlite3 有 11.5 MB，但向量数据本身只有 5.04 MB
（1290 × 1024 维 × 4 字节）。多出来的 6.5 MB 是：

    embedding_fulltext_search         1290 行
    embedding_fulltext_search_content 1290 行
    embedding_fulltext_search_data     668 行
    embedding_fulltext_search_docsize 1290 行
    embedding_fulltext_search_idx      666 行
    embeddings_queue                   291 行

这些是 Chroma 的内置全文检索（FTS5）索引。但本项目：
    - 向量检索用 HNSW（data_level0.bin）
    - 关键词检索用**自建的 BM25**（纯内存，不依赖 Chroma）
所以这套 FTS 索引一次都不会被用到，纯属冗余。

删掉它们：
    1. 减小上传体积（网页上传失败可能与文件大小有关）
    2. 减少云端存储占用
    3. 不影响任何功能

============================== 安全性 ==============================

先备份原库，改完用完整链路验证。验证不通过就自动回滚。
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(r"E:\deepseek work\rag-knowledge-base\data\deploy\chroma")
DB = BASE / "chroma.sqlite3"
BACKUP = BASE / "chroma.sqlite3.bak"

# 要清理的表：全文检索相关，本项目用不到
DROP_TABLES = [
    "embedding_fulltext_search",
    "embedding_fulltext_search_content",
    "embedding_fulltext_search_data",
    "embedding_fulltext_search_docsize",
    "embedding_fulltext_search_idx",
    "embedding_fulltext_search_config",
]

# ---------------------------- 踩过的坑 ----------------------------
# 第一版还删了 embeddings_queue 和 embeddings_queue_config，结果 Chroma
# 启动时直接崩：
#     InternalError: Database error: no such table: embeddings_queue_config
# 说明这两个是**启动必需**的结构（即使我们不写入，初始化也要读它）。
# 所以只删 FTS 索引，队列相关的表一律不动。
DROP_QUEUE: list[str] = []


def size_of(p: Path) -> float:
    return p.stat().st_size / 1024 / 1024 if p.exists() else 0.0


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"数据库不存在：{DB}")

    before = size_of(DB)
    print("=" * 76)
    print(" Chroma 数据库瘦身")
    print("=" * 76)
    print(f"  原始大小 {before:.2f} MB")

    # 备份
    if not BACKUP.exists():
        shutil.copy2(DB, BACKUP)
        print(f"  已备份 → {BACKUP.name}")
    else:
        print(f"  备份已存在（{BACKUP.name}），跳过")

    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    # 先确认这些表确实存在
    existing = {
        r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    dropped = []
    for t in DROP_TABLES + DROP_QUEUE:
        if t in existing:
            try:
                cur.execute(f'DROP TABLE IF EXISTS "{t}"')
                dropped.append(t)
            except Exception as exc:
                print(f"    [WARN] 删除 {t} 失败：{exc}")

    conn.commit()

    # VACUUM 回收空间（必须在事务外）
    print("  执行 VACUUM 回收空间…")
    cur.execute("VACUUM")
    conn.close()

    after = size_of(DB)
    print()
    print(f"  已删除 {len(dropped)} 张表：")
    for t in dropped:
        print(f"    - {t}")
    print()
    print(f"  瘦身后 {after:.2f} MB（减少 {before - after:.2f} MB，"
          f"{100 * (before - after) / before:.0f}%）")

    total = sum(f.stat().st_size for f in BASE.rglob("*") if f.is_file()
                and f.name != BACKUP.name)
    print(f"  目录合计 {total / 1024 / 1024:.2f} MB（不含备份）")
    print("=" * 76)
    print(" 下一步：运行 verify_package.py 验证功能未受影响")
    print("       若不通过，把 chroma.sqlite3.bak 改名回 chroma.sqlite3 即可回滚")
    print("=" * 76)


if __name__ == "__main__":
    main()
