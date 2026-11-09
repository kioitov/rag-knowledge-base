"""解析结果缓存：把 pymupdf 解析 + 切块的产物落到磁盘，避免每次重跑都等 25 秒。

为什么要有这个：
    实测 4 份 PDF 全量解析 25.4s，其中 find_tables() 占 6.8s/份、
    get_text("dict") 占 3.0s/份。调检索参数（size/overlap/策略）时
    每次都重新解析纯属浪费 —— 调 10 次参数就是 4 分钟白白耗掉。

缓存键：
    文件绝对路径 + mtime + size + 策略 + size + overlap
    改了切块参数会自动失效，不用手动清。

存储：
    data/cache/<sha1>.json
    人可读的 JSON，方便排查「缓存里到底是什么」。
    刻意不用 pickle —— 二进制缓存出了问题无法人眼检查。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import Chunk, build_chunks
from pdf_parser import Segment, parse_pdf

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"


def _file_sig(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(path: Path, strategy: str, size: int, overlap: int) -> Path:
    key = f"{_file_sig(path)}_{strategy}_{size}_{overlap}"
    return CACHE_DIR / f"{key}.json"


def load_chunks(
    pdf_path: Path,
    strategy: str = "paragraph",
    size: int = 500,
    overlap: int = 60,
    use_cache: bool = True,
) -> list[Chunk]:
    """读缓存或现算，返回切块结果。"""
    cache_file = _cache_path(pdf_path, strategy, size, overlap)

    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            chunks = [
                Chunk(
                    text=d["text"],
                    source_file=d["source_file"],
                    pages=d["pages"],
                    section=d["section"],
                    kind=d["kind"],
                    strategy=d["strategy"],
                    chunk_id=d["chunk_id"],
                )
                for d in data["chunks"]
            ]
            print(f"  [cache] {pdf_path.name} → {len(chunks)} 块")
            return chunks
        except Exception as exc:
            print(f"  [cache] 读取失败({type(exc).__name__})，重新解析")

    chunks = build_chunks(pdf_path, strategy=strategy, size=size, overlap=overlap)

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "source": str(pdf_path),
                "strategy": strategy,
                "size": size,
                "overlap": overlap,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "text": c.text,
                        "source_file": c.source_file,
                        "pages": c.pages,
                        "section": c.section,
                        "kind": c.kind,
                        "strategy": c.strategy,
                        "chunk_id": c.chunk_id,
                    }
                    for c in chunks
                ],
            }
            cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception as exc:
            print(f"  [cache] 写入失败({type(exc).__name__})，不影响本次结果")

    return chunks


def clear_cache() -> int:
    """删掉所有缓存文件，返回删除数量。"""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)


if __name__ == "__main__":
    import time

    pdf = Path(
        r"E:\deepseek work\融销通项目\成果文档\系统详细设计文档\融销通平台系统详细设计文档.pdf"
    )
    print("首次（现算）：")
    t0 = time.perf_counter()
    cs1 = load_chunks(pdf)
    t1 = time.perf_counter() - t0
    print(f"  {len(cs1)} 块，{t1:.2f}s")

    print("\n第二次（缓存）：")
    t0 = time.perf_counter()
    cs2 = load_chunks(pdf)
    t2 = time.perf_counter() - t0
    print(f"  {len(cs2)} 块，{t2:.3f}s  加速 {t1 / max(t2, 1e-9):.0f}x")

    assert len(cs1) == len(cs2)
    assert cs1[0].text == cs2[0].text
    print("\n[OK] 缓存内容与原结果一致")
