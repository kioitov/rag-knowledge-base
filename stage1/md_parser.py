"""Markdown 文档解析：把 .md 文件转成与 PDF 解析器一致的 Segment 流。

============================== 为什么需要这个 ==============================

原来整条链路只认 PDF（`find_pdfs` + `pdf_parser`）。换语料时发现
K8s 文档是 Markdown 源文件，比 PDF 干净得多 —— 但流水线跑不了。

这暴露了一个设计问题：**解析器和文档格式耦合了。**
正确做法是让「文档 → Segment」这一步可替换，下游的切块、检索、
索引完全不关心来源是 PDF 还是 Markdown。

============================== Markdown 解析比 PDF 简单在哪 ==============================

PDF 需要自己做的脏活（见 stage1/pdf_parser.py）：
    - 行 → 段聚类（按字号和行距）
    - 识别标题（字号 ≥ 11pt）
    - 表格还原（find_tables）
    - 页眉页脚剔除（按 y 坐标）
    - 阅读顺序重排

Markdown 全都有显式标记：
    - `#` 就是标题层级
    - 空行就是段落边界
    - `|` 表格就是表格
    - 没有页眉页脚

所以这个解析器只有 PDF 解析器 1/4 的长度，且没有启发式猜测。

============================== 但仍有两个真实问题 ==============================

1. **中文换行**
   中文 Markdown 常把一句话折成多行（源码里为了阅读方便），
   合并时不能插入空格，否则「Pod 的\n生命周期」变成「Pod 的 生命周期」。
   处理方式沿用 PDF 解析器里验证过的 `merge_broken_lines` 逻辑。

2. **页码不存在**
   PDF 有物理页码，Markdown 没有。但下游（generator、前端）都依赖
   `page_label()` 显示「第 N 页」。
   这里的处理：用**章节序号**代替页码，显示为「第 3 节」，
   并在 meta 里标 `paged=False`，让展示层知道这不是物理页码。
   **不能假装有页码** —— 那是在编造溯源信息。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "stage1"))

from pdf_parser import Segment, merge_broken_lines

# 标题：ATX 风格 # ## ###，K8s 文档还会带 {#anchor}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*(?:\{#[^}]*\})?\s*$")
# 代码围栏
FENCE_RE = re.compile(r"^```")
# 表格行
TABLE_ROW_RE = re.compile(r"^\s*\|")
# 列表项
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


@dataclass
class _Block:
    """解析过程中的中间块。"""

    kind: str  # "heading" | "para" | "code" | "table" | "list"
    lines: list[str] = field(default_factory=list)
    level: int = 0  # heading 层级
    index: int = 0  # 章节序号（用于替代页码）


def parse_markdown(path: Path) -> list[Segment]:
    """把 Markdown 文件解析成 Segment 列表。

    产出的 Segment.kind 与 PDF 解析器保持一致（"text" / "table"），
    这样下游 chunker 不用改。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    blocks: list[_Block] = []
    cur: _Block | None = None
    in_fence = False
    heading_index = 0

    def flush() -> None:
        nonlocal cur
        if cur and cur.lines:
            blocks.append(cur)
        cur = None

    for line in lines:
        # --- 代码围栏：整体保留，不参与断段 ---
        if FENCE_RE.match(line):
            if in_fence:
                # 结束围栏
                if cur and cur.kind == "code":
                    flush()
                in_fence = False
            else:
                flush()
                cur = _Block(kind="code")
                in_fence = True
            continue
        if in_fence:
            if cur is None:
                cur = _Block(kind="code")
            cur.lines.append(line)
            continue

        # --- 标题 ---
        m = HEADING_RE.match(line)
        if m:
            flush()
            heading_index += 1
            b = _Block(kind="heading", lines=[m.group(2).strip()],
                       level=len(m.group(1)), index=heading_index)
            blocks.append(b)
            continue

        # --- 表格 ---
        if TABLE_ROW_RE.match(line):
            if cur and cur.kind != "table":
                flush()
            if cur is None:
                cur = _Block(kind="table")
            cur.lines.append(line.rstrip())
            continue

        # --- 空行：断段 ---
        if not line.strip():
            flush()
            continue

        # --- 列表项 ---
        if LIST_RE.match(line):
            if cur and cur.kind != "list":
                flush()
            if cur is None:
                cur = _Block(kind="list")
            cur.lines.append(line.rstrip())
            continue

        # --- 普通正文：续到当前段，或开新段 ---
        if cur and cur.kind in ("para", "list"):
            cur.lines.append(line.rstrip())
        else:
            flush()
            cur = _Block(kind="para", lines=[line.rstrip()])

    flush()

    # ---- 转成 Segment ----
    segments: list[Segment] = []
    current_section = ""
    section_index = 0

    for b in blocks:
        if b.kind == "heading":
            title = b.lines[0]
            # 一级标题通常是文档标题，作为整体章节名
            if b.level == 1:
                current_section = title
            elif b.level <= 3:
                current_section = title
            section_index = b.index
            # 标题本身也作为一个段（供检索命中章节名），但标记出来
            segments.append(Segment(
                text=title,
                page=section_index,
                kind="text",
                section=current_section,
                meta={"heading": True, "level": b.level, "paged": False},
            ))
            continue

        if b.kind == "table":
            # 表格转 Markdown 行，保持结构
            md = "\n".join(b.lines)
            if len(b.lines) >= 2:
                segments.append(Segment(
                    text=md, page=section_index, kind="table",
                    section=current_section,
                    meta={"paged": False},
                ))
            continue

        if b.kind == "code":
            # 代码块：保留，但单独标记 kind。
            # 代码对「怎么配置」类问题价值高，不能丢。
            code = "\n".join(b.lines).strip()
            if len(code) >= 20:
                segments.append(Segment(
                    text=code, page=section_index, kind="text",
                    section=current_section,
                    meta={"is_code": True, "paged": False},
                ))
            continue

        # 正文/列表：合并断行
        merged = merge_broken_lines(b.lines).strip()
        if merged and len(merged) >= 8:
            segments.append(Segment(
                text=merged, page=section_index, kind="text",
                section=current_section,
                meta={"paged": False},
            ))

    return segments


def parse_doc(path: Path) -> list[Segment]:
    """统一入口：按扩展名分派到对应解析器。

    这是「文档格式可替换」的关键 —— 下游只调这个函数。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pdf_parser import parse_pdf
        return parse_pdf(path)
    if suffix in (".md", ".markdown"):
        return parse_markdown(path)
    raise ValueError(f"不支持的格式：{path.suffix}（{path.name}）")


if __name__ == "__main__":
    # 自测：解析一篇，人工看切分质量
    base = _ROOT / "data" / "docs"
    md_files = sorted(base.rglob("*.md")) if base.exists() else []
    md_files = [f for f in md_files if not f.name.startswith("_")]

    if not md_files:
        raise SystemExit("没找到 .md 语料。先跑 stage5/fetch_corpus.py")

    target = md_files[0]
    print("=" * 88)
    print(f" 解析自测：{target.name}")
    print("=" * 88)

    segs = parse_doc(target)
    texts = [s for s in segs if s.kind == "text"]
    tables = [s for s in segs if s.kind == "table"]
    heads = [s for s in texts if s.meta.get("heading")]
    codes = [s for s in texts if s.meta.get("is_code")]

    print(f"  Segment 总数 {len(segs)}")
    print(f"    标题 {len(heads)} / 正文 {len(texts) - len(heads) - len(codes)}"
          f" / 代码 {len(codes)} / 表格 {len(tables)}")
    print(f"  总字符 {sum(len(s.text) for s in segs):,}")
    print(f"  章节数 {len({s.section for s in segs if s.section})}")
    print()
    print("  前 6 个正文段：")
    body = [s for s in texts if not s.meta.get("heading")]
    for s in body[:6]:
        print(f"    [{len(s.text):4d}字 | {s.section[:20]}] {s.text[:70]!r}")
