"""文档质量探测：在看切块效果之前，先搞清楚手上这批文档是什么货色。

为什么先做这一步：
    切块的前提是「有干净的文字流」。扫描版 PDF 没有文字层，pymupdf 读出来是空字符串，
    如果不先探测就写切块代码，你会得到一堆空块，然后花两小时怀疑是切块逻辑写错了。

探测项（每份文档）：
    页数 / 总字符数 / 每页平均字符数 / 空页占比 / 内嵌图片数
判定：
    平均字符 < 50 且 图片数多  → 疑似扫描件，需要 OCR，阶段 1 先排除
    平均字符正常              → 可用

用法：
    python stage1\\inspect_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf  # 不用 fitz，见 NOTES.md 问题 4
from docx import Document

ROOT = Path(__file__).resolve().parent.parent

# 素材放 data/docs/<任意子目录>/<任意名>.pdf
DOC_DIR = ROOT / "data" / "docs"

# 若按下面这套子目录布局放置，报告里的对照数字可直接复现
CANDIDATES = [
    "可行性研究报告/可行性研究报告.pdf",
    "软件需求规格说明书/软件需求规格说明书.pdf",
    "系统概要设计说明书/系统概要设计说明书.pdf",
    "系统详细设计文档/系统详细设计文档.pdf",
]


def inspect_pdf(path: Path) -> dict:
    """返回单份 PDF 的质量指标。"""
    doc = pymupdf.open(path)
    page_chars: list[int] = []
    total_images = 0

    for page in doc:
        text = page.get_text("text")
        page_chars.append(len(text.strip()))
        total_images += len(page.get_images(full=True))

    pages = len(doc)
    doc.close()

    non_empty = [c for c in page_chars if c > 0]
    empty_pages = pages - len(non_empty)

    return {
        "path": path,
        "pages": pages,
        "total_chars": sum(page_chars),
        "avg_chars": sum(page_chars) / pages if pages else 0,
        "empty_pages": empty_pages,
        "images": total_images,
        "min_page_chars": min(page_chars) if page_chars else 0,
        "max_page_chars": max(page_chars) if page_chars else 0,
    }


def judge(r: dict) -> str:
    """给一份文档下结论：能不能进阶段 1。"""
    if r["avg_chars"] < 50:
        return "疑似扫描件 / 无文字层 → 阶段 1 排除，需要 OCR"
    if r["avg_chars"] < 200:
        return "文字偏少 → 可用，但切块要调小粒度"
    if r["empty_pages"] / max(r["pages"], 1) > 0.3:
        return "空页过多 → 可用，但需过滤空页"
    return "文字层健康 → 直接用"


def main() -> None:
    print("=" * 96)
    print(" 阶段 1 前置：文档质量探测")
    print(f" 目录：{DOC_DIR}")
    print("=" * 96)

    if not DOC_DIR.exists():
        print(f"[FAIL] 目录不存在：{DOC_DIR}")
        sys.exit(1)

    rows = []
    for rel in CANDIDATES:
        p = DOC_DIR / rel
        if not p.exists():
            print(f"[MISS] {rel}")
            continue
        r = inspect_pdf(p)
        r["verdict"] = judge(r)
        rows.append(r)

    print(f"\n{'文档':<28} {'页数':>4} {'总字符':>7} {'均字符':>6} {'空页':>4} {'图片':>4}  结论")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['path'].stem[:26]:<28} {r['pages']:>4} {r['total_chars']:>7} "
            f"{r['avg_chars']:>6.0f} {r['empty_pages']:>4} {r['images']:>4}  {r['verdict']}"
        )

    usable = [r for r in rows if "排除" not in r["verdict"]]
    print("-" * 96)
    print(f"可用文档：{len(usable)} / {len(rows)} 份")
    print(f"可用总字符：{sum(r['total_chars'] for r in usable):,} 字")

    # docx 也顺手探一下（同批项目的 Word 版）
    docx_path = DOC_DIR / "软件需求规格说明书" / "融销通平台软件需求规格说明书.docx"
    if docx_path.exists():
        d = Document(docx_path)
        paras = [p.text.strip() for p in d.paragraphs if p.text.strip()]
        chars = sum(len(p) for p in paras)
        print(f"\ndocx 对照：{docx_path.name}")
        print(f"  段落 {len(paras)} 段 / {chars:,} 字 / 表格 {len(d.tables)} 个")

    print("=" * 96)


if __name__ == "__main__":
    main()
