"""看真实抽出来的文本长什么样 —— 切块前的第一次「人眼验收」。

为什么必须做：
    交接文档里说「把每块打印出来，人眼看切得对不对」。但在切块之前，
    得先看清楚 PDF 抽出来的原始文本是什么形态：段落有没有连成一整块、
    表格是不是变成了散乱的数字、页眉页脚有没有混进去。
    这些问题不解决，切块调参就是瞎调。

用法：
    python stage1/peek_text.py            # 默认看第 5 页
    python stage1/peek_text.py 5 12       # 看第 5~12 页
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "docs"


def _first_pdf() -> Path:
    pdfs = sorted(DOC_DIR.rglob("*.pdf")) if DOC_DIR.exists() else []
    if not pdfs:
        raise SystemExit(f"没找到 PDF。把文件放进 {DOC_DIR}/ 即可（支持子目录）。")
    return pdfs[0]


PDF = None  # 由 main 惰性获取


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1

    pdf = _first_pdf()
    doc = pymupdf.open(pdf)
    print(f"文件：{pdf.name}  共 {len(doc)} 页")
    print(f"查看第 {start} ~ {end - 1} 页（0 基索引）")
    print("=" * 90)

    for i in range(start, min(end, len(doc))):
        page = doc[i]
        text = page.get_text("text")
        print(f"\n{'─' * 90}")
        print(f"[第 {i} 页] 字符数 {len(text.strip())}  行数 {len(text.strip().splitlines())}")
        print(f"{'─' * 90}")
        print(text)
        # 顺便看看这一页到底有没有被识别成表格
        tables = page.find_tables()
        if tables.tables:
            print(f">>> 本页检测到 {len(tables.tables)} 个表格结构")
            for ti, tb in enumerate(tables.tables):
                data = tb.extract()
                print(f"    表 {ti}: {len(data)} 行 x {len(data[0]) if data else 0} 列")
                for row in data[:3]:
                    print(f"      {row}")

    doc.close()
    print("=" * 90)


if __name__ == "__main__":
    main()
