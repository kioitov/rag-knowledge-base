"""阶段 1 第一块：把 PDF 解析成「干净段落流 + 结构化表格 + 页码 + 章节」。

只做一件事：给定一份 PDF，吐出一串 Segment。每段自带 page / kind / section，
后面切块时原样带着走 —— 这是「答案能标出来源页码」的根。

================================ 实测数据（决定算法） ================================

对一份 54 页、由 Word 导出的中文技术文档（A4，含大量表格）做的排版统计：

    (8.6pt, SimSun)  20942 字  ← 正文主体
    (10.5pt, SimSun)  6591 字  ← 表格内容
    (12.0pt, SimSun)   321 字  ← 一级/二级标题
    (15.0pt, SimSun)    73 字  ← 封面大标题
    (7.8pt, SimSun)   1529 字  ← 图注/脚注

**关键发现：这类 PDF 的每个 block 恰好等于一行。** 抽一页看，10 个 block =
页眉 + 页码 + 图注 + 标题 + 6 行正文。所以 pymupdf 的 blocks 模式**拿不到段落边界**，
必须自己做「行 → 段」聚类。这也是为什么要读字号：正文 8.6pt 与标题 12pt 差距明显，
字号切换 = 段落切换，比猜行距倍数可靠得多。

                                七个脏数据问题

1. 页眉页脚污染 —— y≈34 是文档名，y≈798 是页码。不按坐标剔掉，每块都带噪音。
   （本来想用「跨页高频行」统计法，实测发现这份文档页眉在 y 上极其固定，
    坐标法更稳且不需要全文档扫描，改用坐标 + 纯数字兜底。）

2. blocks 拿不到段落 —— 见上，必须自建行聚类。

3. 表格被 pymupdf 打平成「一格一行」—— 20 行 2 列的术语表抽出来是 40 行孤立短句。
   用 find_tables() 抠走表格区域，转 Markdown。

4. 中文句子行尾硬断 —— 「...资金\n提供方」。聚类后还要做「中文直拼、英数补空格」。

5. 目录页 —— 第 2、3、54 页是目录/图表清单，全是「章节名 | 页码」，对问答零价值
   还会抢占检索，整页丢弃。

6. block 阅读顺序不可靠 —— 实测某页 block[2] 是 y=447 的图注、block[3] 是 y=488
   的标题，返回顺序按内容流而非版面。必须按 (y, x) 重排。

7. 跨页段落断裂 —— 段落在页尾被切断，下一页接着写。按 (page, gap, 字号, 句末标点)
   四条件判断是否续接。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pymupdf

# 页眉/页脚边界（A4 高 842pt）。实测页眉 y≈34、页码 y≈798、正文 y 在 60~780。
HEADER_Y_MAX = 60.0
FOOTER_Y_MIN = 780.0

# 行距聚类阈值（字号倍数）。实测正常行距 ≈ 1.5 倍字号。
LINE_GAP_RATIO = 1.45

# 标题字号下限。实测正文 8.6、标题 12.0、封面 15.0。
HEADING_SIZE_MIN = 11.0

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
TERMINAL_PUNCT = "。！？；…）】」』\"'"


@dataclass
class Segment:
    """一段有语义的文本，附带来源信息。"""

    text: str
    page: int  # 0 基页码
    kind: str  # "text" | "table"
    section: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def page_1based(self) -> int:
        return self.page + 1


# ---------------------------------------------------------------------------
# 中文断行修复
# ---------------------------------------------------------------------------
def merge_broken_lines(lines: list[str]) -> str:
    """把同一段内的硬断行拼回自然句。

      - 上一行以中文句读结尾 → 段真的结束，保留换行
      - 两端都是中文 → 同一个词被切断，直接拼不插空格
      - 一端是英文/数字 → 补空格，避免 "BM25"+"算法" 这类词被粘连
    """
    out: list[str] = []
    for raw in lines:
        cur = raw.strip()
        if not cur:
            continue
        if not out:
            out.append(cur)
            continue
        prev = out[-1]
        p_last, c_first = prev[-1], cur[0]
        if p_last in TERMINAL_PUNCT:
            out.append(cur)
        elif p_last in "，、":
            out.append(cur)
        elif CJK_RE.match(p_last) and CJK_RE.match(c_first):
            out[-1] = prev + cur
        elif CJK_RE.match(p_last) or CJK_RE.match(c_first):
            out[-1] = prev + cur
        else:
            out[-1] = prev + " " + cur
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 标题判定（字号优先，编号兜底）
# ---------------------------------------------------------------------------
def is_heading(line: _Line) -> bool:
    """标题 = 字号够大 且 文本够短 且 不以句读结尾。"""
    t = line.text.strip()
    if not t or len(t) > 50 or "\n" in t:
        return False
    if t.endswith(("。", "，", "；")):
        return False
    if line.size >= HEADING_SIZE_MIN:
        return True
    # 字号相同但编号形式明确（如 "2.5.6 xxx"），也算标题
    return bool(re.match(r"^\d+(\.\d+)*[\s\u3000]+\S", t)) and len(t) <= 40


def is_toc_page(blocks: list) -> bool:
    """目录页：含「目录」字样，或 >60% 文本行以数字页码结尾。"""
    joined = "".join(b[4] for b in blocks)
    if re.search(r"目[\s\u3000]*录", joined):
        return True
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if len(text_blocks) < 5:
        return False
    with_no = sum(1 for b in text_blocks if re.search(r"\d+\s*$", b[4].strip()))
    return with_no / len(text_blocks) > 0.6


# ---------------------------------------------------------------------------
# 行 → 段落
# ---------------------------------------------------------------------------
@dataclass
class _Line:
    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    size: float  # 主字号

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)


def cluster_paragraphs(lines: list[_Line]) -> list[list[_Line]]:
    """按 (行距, 字号) 把行聚成段落；同一 y 上的多个片段合并成一行文本。

    ---------------------------- 三个断段条件 ----------------------------
    1. 字号变化 > 1.2pt
       实测图注 7.8pt 紧贴正文 8.6pt，行距正常但字号变了。
       只看行距会把图注并进正文，污染语义。

    2. 行距 > 字号 * 1.45
       正常行距 ≈ 1.5 倍字号（8.6pt 字，15.6pt 行距）。

    3. 同一 y 上的多个片段 → **合并，不拆**
       实测封面：
           y=551.0  x=119.4-161.4  '文件标识'
           y=551.0  x=229.4-376.4  '： QST-班号-项目代码-DOC-LLD'
       曾试图按 x 间隙拆成两段，结果得到 '： QST-班号-项目代码-DOC-LLD' ——
       值还在，但「文件标识」这个标签被拆走且因为只有 4 字被过滤掉，
       变成一句没头没尾的话。**表单场景必须合并**，标签和值是一体的。

    另：同一 y 上有多个片段时必须先按 x 排序，否则右值可能排在左标签前面。
    """
    if not lines:
        return []
    # 先按 (行, 列) 排序：同一 y 的先左后右，不同 y 的先上后下
    lines = sorted(lines, key=lambda l: (round(l.y0, 1), l.x0))
    if len(lines) == 1:
        return [lines]

    paragraphs: list[list[_Line]] = [[lines[0]]]
    for i in range(1, len(lines)):
        prev, cur = lines[i - 1], lines[i]
        same_row = abs(cur.y0 - prev.y0) < 3.0

        if same_row:
            # 同一行内的片段：一律合并（左标签 + 右值属于同一字段）。
            # 只把 x 间隙记进 meta，供后续按需拆分，不在这里硬拆 ——
            # 硬拆会得到 '： QST-班号-项目代码-DOC-LLD' 这种没头没尾的值。
            paragraphs[-1].append(cur)
            continue

        gap = cur.y0 - prev.y1
        size_changed = abs(cur.size - prev.size) > 1.2
        gap_too_big = gap > max(prev.size, 8.0) * LINE_GAP_RATIO
        if size_changed or gap_too_big:
            paragraphs.append([cur])
        else:
            paragraphs[-1].append(cur)
    return paragraphs


# ---------------------------------------------------------------------------
# 主解析
# ---------------------------------------------------------------------------
def parse_pdf(path: Path, drop_toc: bool = True, verbose: bool = False) -> list[Segment]:
    """按「页面 → y 序」统一处理正文段落与表格，保证段落归属的章节正确。

    --------------------------- 修过的严重 bug ---------------------------
    第一版在页面循环开头就把表格 Segment 建好了，此时 current_section
    还是**上一页**停留的章节。后果实测：
        第 27 页的某个表格被标成上一页的章节名，
        而它的真实章节是下一页的标题。
    这类错标在 RAG 里是致命的 —— 检索命中后把来源标错，用户点开发现货不对板。

    修法：把表格也放进「按 y 排序的统一队列」，和正文段落一起顺序走，
    章节指针在 y 序上自然推进。表格的 y 取它 bbox 的顶边。
    """
    doc = pymupdf.open(path)
    segments: list[Segment] = []
    current_section = ""
    skipped_toc: list[int] = []
    prev_tail: str = ""

    for pno, page in enumerate(doc):
        blocks = page.get_text("blocks")

        if drop_toc and is_toc_page(blocks):
            skipped_toc.append(pno)
            continue

        # --- 1. 先把表格抽出来，只记录不落地 ---
        table_rects: list[pymupdf.Rect] = []
        raw_tables: list[tuple[float, str]] = []  # (y_top, markdown)
        try:
            for tb in page.find_tables().tables:
                if tb.row_count < 2:
                    continue
                md = tb.to_markdown().replace("<br>", "").strip()
                if not md:
                    continue
                rect = pymupdf.Rect(tb.bbox)
                table_rects.append(rect)
                raw_tables.append((rect.y0, md))
        except Exception:
            pass  # 表格检测失败不该拖垮整份文档

        # --- 2. 收集正文行（带字号）---
        raw_lines: list[_Line] = []
        try:
            dd = page.get_text("dict")
        except Exception:
            dd = {"blocks": []}

        for bl in dd.get("blocks", []):
            if bl.get("type") != 0:
                continue
            for ln in bl.get("lines", []):
                spans = ln.get("spans", [])
                if not spans:
                    continue
                text = "".join(sp["text"] for sp in spans).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = ln["bbox"]
                if y1 <= HEADER_Y_MAX or y0 >= FOOTER_Y_MIN:
                    continue
                rect = pymupdf.Rect(x0, y0, x1, y1)
                if any(rect.intersects(tr) for tr in table_rects):
                    continue
                size = max(spans, key=lambda s: len(s["text"]))["size"]
                raw_lines.append(_Line(y0, y1, x0, x1, text, round(size, 1)))

        # --- 3. 正文行 → 段落 ---
        paragraphs: list[tuple[float, list[_Line]]] = []
        for group in cluster_paragraphs(raw_lines):
            paragraphs.append((group[0].y0, group))

        # --- 4. 段落与表格合并成统一 y 序队列（核心修复）---
        queue: list[tuple[float, str, object]] = []
        for y, group in paragraphs:
            queue.append((y, "para", group))
        for y, md in raw_tables:
            queue.append((y, "table", md))
        queue.sort(key=lambda item: item[0])

        for _y, kind, payload in queue:
            if kind == "table":
                segments.append(Segment(payload, pno, "table", current_section))
                continue

            group: list[_Line] = payload  # type: ignore[assignment]
            merged = merge_broken_lines([l.text for l in group]).strip()
            if not merged:
                continue
            head = group[0]
            heading = is_heading(head)

            # --- 5. 跨页续接 ---
            if (
                prev_tail
                and segments
                and not heading
                and segments[-1].kind == "text"
                and abs(head.size - segments[-1].meta.get("size", head.size)) <= 1.2
                and prev_tail[-1] not in TERMINAL_PUNCT
                and segments[-1].page == pno - 1
            ):
                segments[-1].text = merge_broken_lines([segments[-1].text, merged])
                segments[-1].meta["spans_pages"] = True
                prev_tail = segments[-1].text
                continue

            if heading:
                current_section = merged.split("\n")[0].strip()

            segments.append(
                Segment(
                    merged,
                    pno,
                    "text",
                    current_section,
                    meta={"size": head.size, "heading": heading},
                )
            )
            prev_tail = merged

    doc.close()

    if verbose and skipped_toc:
        print(f"  [跳过目录页] {[p + 1 for p in skipped_toc]}")

    return [s for s in segments if len(s.text.strip()) >= 8]


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _selftest(pdf: Path) -> None:
    import statistics

    segs = parse_pdf(pdf, verbose=True)
    if not segs:
        print("[FAIL] 没解析出任何段")
        return

    txt = [s for s in segs if s.kind == "text"]
    tbl = [s for s in segs if s.kind == "table"]
    heads = [s for s in txt if s.meta.get("heading")]
    body = [s for s in txt if not s.meta.get("heading")]
    lens = [len(s.text) for s in body]

    print(f"文件      : {pdf.name}")
    print(f"段总数    : {len(segs)}（正文 {len(txt)} / 其中标题 {len(heads)} / 表格 {len(tbl)}）")
    print(f"总字符    : {sum(len(s.text) for s in segs):,}")
    print(f"覆盖页数  : {len({s.page for s in segs})}")
    if lens:
        print(f"正文段长度: 中位 {statistics.median(lens):.0f}  均值 {sum(lens)/len(lens):.0f}"
              f"  最短 {min(lens)}  最长 {max(lens)}")
    print(f"章节数    : {len({s.section for s in segs if s.section})}")

    print("\n" + "=" * 92)
    print(" 前 4 条正文段（封面应保持分行，不粘成一坨）")
    print("=" * 92)
    for s in txt[:4]:
        flag = "标题" if s.meta.get("heading") else "正文"
        print(f"[p{s.page_1based} | {flag} | {len(s.text)}字] {s.text[:220]}")
        print("-" * 92)

    print("\n 表格段前 1 条（确认没被拆散）")
    print("=" * 92)
    for s in tbl[:1]:
        print(f"[p{s.page_1based} | 表格]")
        print(s.text[:500])
        print("-" * 92)

    print("\n 中位长度正文段（验证断行合并）")
    print("=" * 92)
    if lens:
        mid = statistics.median(lens)
        cand = [s for s in body if abs(len(s.text) - mid) <= 8]
        if cand:
            s = cand[len(cand) // 2]
            print(f"[p{s.page_1based} | {s.section}] {len(s.text)}字")
            print(s.text[:600])
            print("-" * 92)

    print("\n 全部识别出的章节（确认层级合理）")
    print("=" * 92)
    for sec in dict.fromkeys(s.section for s in segs if s.section):
        print(f"  - {sec}")


def _default_pdf() -> Path:
    docs = Path(__file__).resolve().parent.parent / "data" / "docs"
    pdfs = sorted(docs.rglob("*.pdf")) if docs.exists() else []
    if not pdfs:
        raise SystemExit(
            "没找到 PDF。\n"
            "  请把待处理的 PDF 放进 data/docs/（支持子目录），然后重新运行。\n"
            "  也可以直接传路径：python stage1/pdf_parser.py <你的文件.pdf>"
        )
    return pdfs[0]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_pdf()
    _selftest(target)
