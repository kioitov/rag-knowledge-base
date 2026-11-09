"""阶段 1 第二块：切块器（chunker）。

三套切法放在一起，是为了后面能拿真实数字对比 —— 这是交接文档阶段 5
「固定分块 vs 语义分块」的实验基础，现在就把接口和测量方式留好。

    FixedChunker     按字符数硬切 + 重叠。最朴素，是基线。
    ParagraphChunker 以段落为最小单位装箱，装满才出块，不切断句子。
    SectionChunker   按章节组织，块首注入章节名，章节内再按段落装箱。

------------------------------- 两个设计决定 -------------------------------

1. **表格整体不切**
   pymupdf 把表格还原成 Markdown 行。硬切会把 `|` 行劈开，
   检索时只拿到半个表格，模型看不懂。表格段一律整块保留，即使超长。

2. **记录的是 pages 列表，不是单个 page**
   500 字的块必然跨页（这份文档正文每页约 700 字）。
   实测：段落装箱后一个块经常横跨 2~3 页。只记一个页码等于说谎 ——
   用户点开来源发现答案不在那一页，"可溯源"这个卖点当场破产。
   pages 是列表，展示时渲染成「第 3-5 页」。

------------------------------- 修过的 bug --------------------------------
第一版三种策略输出完全一致（288 块 / 36136 字），原因是把累加缓冲区 `buf`
写在了 `for seg in segs` 循环体内 —— 每换一个 Segment 就清零，
等于每段独立切，装箱逻辑完全没生效。正确做法是把 buf 提到循环外，
按「流」处理，并在换页/换章节时决定是否强制断块。
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdf_parser import Segment, parse_pdf

SENT_END = "。！？；…"
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；…!?;])\s*")


@dataclass
class Chunk:
    """切块结果，带完整溯源信息。"""

    text: str
    source_file: str
    pages: list[int]  # 0 基，可能跨页
    section: str
    kind: str  # "text" | "table"
    strategy: str
    chunk_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.source_file}|{self.pages}|{self.text[:80]}".encode()
            ).hexdigest()[:10]

    @property
    def page(self) -> int:
        return self.pages[0] if self.pages else 0

    @property
    def page_1based(self) -> int:
        return self.page + 1

    def page_label(self) -> str:
        """单页 "第 3 页"，跨页 "第 3-5 页"。"""
        if not self.pages:
            return "页码未知"
        lo, hi = min(self.pages) + 1, max(self.pages) + 1
        return f"第 {lo} 页" if lo == hi else f"第 {lo}-{hi} 页"

    def cite(self) -> str:
        return f"{self.source_file} {self.page_label()}"

    def __len__(self) -> int:
        return len(self.text)


def split_sentences(text: str) -> list[str]:
    """按中英文句读断句，句读符跟随前句。"""
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]


def _attach_headings(segs: list[Segment]) -> list[Segment]:
    """把「纯标题段」并入它统领的下一个正文段，避免标题单独成块。

    ---------------------------- 为什么必须做 ----------------------------
    实测（修之前）：199 个块里有 75 个小于 100 字，其中 58 个是正文块。
    抽样看，绝大多数是这种：
        [17] 第 15 页 〔2.3.2 用户登录（UR-02）〕 '2.3.2 用户登录（UR-02）'
        [19] 第 16 页 〔2.3.3 退出登录（UR-03）〕 '2.3.3 退出登录（UR-03）'
        [22] 第 16 页 〔2.3.4 个人信息查看与维护（UR-04）〕 ...

    这是纯粹的浪费：一个 17 字的标题块，embedding 之后主题信息极弱，
    却占掉 top-k 的一个名额。用户问「用户登录怎么实现的」，
    检索很可能命中这个空壳标题，然后模型拿着标题答不出任何东西。

    修法：如果一个 Segment 是标题（meta["heading"]）且长度 < 40 字，
    把它的文本用换行拼到下一个正文段开头，并沿用下一段的 kind。
    章节名本身已经通过 section 字段保留，不会丢信息。

    表格不动 —— 表格前面挂个标题也没坏处，但不合并更安全（表格是结构化的）。
    """
    out: list[Segment] = []
    pending_heading = ""

    for seg in segs:
        is_bare_heading = (
            seg.kind == "text"
            and seg.meta.get("heading")
            and len(seg.text.strip()) < 40
            and "\n" not in seg.text.strip()
        )
        if is_bare_heading:
            # 已经在 pending 里有标题，说明上一个标题没有正文跟着（连续标题），
            # 直接串起来，别丢
            pending_heading = (
                f"{pending_heading}\n{seg.text.strip()}" if pending_heading else seg.text.strip()
            )
            continue

        if pending_heading:
            merged_text = f"{pending_heading}\n{seg.text.strip()}"
            out.append(
                Segment(
                    merged_text,
                    seg.page,
                    seg.kind,
                    seg.section,
                    meta=dict(seg.meta),
                )
            )
            pending_heading = ""
        else:
            out.append(seg)

    # 文档末尾孤立的标题（后面没有正文了）：保留，不能吞掉
    if pending_heading:
        last_page = segs[-1].page if segs else 0
        out.append(Segment(pending_heading, last_page, "text", "", meta={"heading": True}))

    return out


# ---------------------------------------------------------------------------
# 公共：把段落流装进 size 大小的箱子
# ---------------------------------------------------------------------------
def _pack(
    segs: list[Segment],
    size: int,
    overlap: int,
    strategy: str,
    force_break_on_section: bool = False,
    inject_section: bool = False,
) -> list[Chunk]:
    """核心装箱器：把一串 Segment 累加成不超过 size 的块。

    缓冲区在循环外，跨 Segment 累积 —— 这是修 bug 的关键。
    换页不强制断块（语义连续），换章节才断（章节是天然的语义边界）。

    状态机（三个状态，不能混）：
        buf          当前正在装的块
        carry        上一块的尾部重叠，等下一次 flush 后注入新块开头
        pending_*    出块瞬间要立即写入的表格
    早期版本把「出块」和「设置重叠」混在同一个闭包里，导致
    (a) 每次 flush 都残留尾巴 → 内容重复
    (b) 最后一块的尾巴被当独立块 → 内容截断
    """
    out: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    buf_pages: list[int] = []
    buf_section = ""
    buf_kind = "text"
    carry = ""  # 待注入的尾部重叠

    def emit(text: str, pages: list[int], section: str, kind: str) -> None:
        if not text.strip():
            return
        prefix = f"【{section}】\n" if inject_section and section and kind == "text" else ""
        out.append(
            Chunk(
                text=prefix + text,
                source_file="",
                pages=sorted(set(pages)) or [0],
                section=section,
                kind=kind,
                strategy=strategy,
            )
        )

    def flush() -> None:
        """把当前 buf 出块，并设置尾部重叠。"""
        nonlocal buf, buf_len, buf_pages, carry
        text = "".join(buf).strip()
        if text:
            emit(text, buf_pages, buf_section, buf_kind)
            carry = text[-overlap:] if overlap > 0 and len(text) > overlap else ""
        else:
            carry = ""
        buf, buf_len, buf_pages = [], 0, []

    for seg in segs:
        # --- 表格：整块独立，永不切碎 ---
        if seg.kind == "table":
            flush()
            emit(seg.text.strip(), [seg.page], seg.section, "table")
            carry = ""  # 表格后面不接重叠，避免把表格末行塞进正文块
            continue

        text = seg.text.strip()
        if not text:
            continue

        # --- 章节切换：章节是语义硬边界，强制断块 ---
        if force_break_on_section and seg.section != buf_section and buf:
            flush()
            carry = ""
        buf_section = seg.section

        # --- 超长段落先按句子拆 ---
        units = [text] if len(text) <= size else []
        if not units:
            for sent in split_sentences(text):
                if len(sent) <= size:
                    units.append(sent)
                else:
                    units.extend(sent[i : i + size] for i in range(0, len(sent), size))

        for u in units:
            add = (carry + u) if carry else u
            if buf_len + len(add) > size and buf:
                flush()
                add = (carry + u) if carry else u
            buf.append(add)
            buf_len += len(add)
            carry = ""  # 用过就消费掉
            if seg.page not in buf_pages:
                buf_pages.append(seg.page)

    flush()
    return out


# ---------------------------------------------------------------------------
# 三种策略
# ---------------------------------------------------------------------------
def chunk_fixed(segs: list[Segment], size: int = 500, overlap: int = 60) -> list[Chunk]:
    """基线：**真正粗糙**的固定长度切法 —— 把全文拼成一条流，按字符数硬切。

    刻意不尊重段落和句子边界，就是要让它切出半截话。
    这是「改进前」的对照组：阶段 5 要用数字证明
    「尊重边界的切法」到底好多少。如果基线也做得聪明，对比就失去意义。

    表格仍然整块保留（否则评测会假阴性 —— 问题不在策略上，
    而在于表格被劈开这件事本身没有意义）。
    """
    out: list[Chunk] = []

    # 表格直接出块，正文拼成长流
    stream_parts: list[tuple[str, int, str]] = []  # (text, page, section)
    for seg in segs:
        if seg.kind == "table":
            out.append(
                Chunk(seg.text.strip(), "", [seg.page], seg.section, "table", "fixed")
            )
        else:
            stream_parts.append((seg.text.strip(), seg.page, seg.section))

    # 拼流并记录每个字符属于哪一页哪一章
    full = ""
    page_map: list[tuple[int, int, int, str]] = []  # (start, end, page, section)
    for text, page, section in stream_parts:
        if not text:
            continue
        s = len(full)
        full += text
        page_map.append((s, len(full), page, section))

    def locate(lo: int, hi: int) -> tuple[list[int], str]:
        pages = sorted({p for (s, e, p, _sec) in page_map if s < hi and e > lo})
        sec = ""
        for (s, e, _p, _sec) in page_map:
            if s <= lo < e:
                sec = _sec
                break
        return pages or [0], sec

    start = 0
    while start < len(full):
        piece = full[start : start + size]
        if piece.strip():
            pages, sec = locate(start, start + len(piece))
            out.append(Chunk(piece, "", pages, sec, "text", "fixed"))
        if start + size >= len(full):
            break
        start += size - overlap

    return out


def chunk_paragraph(segs: list[Segment], size: int = 500, overlap: int = 60) -> list[Chunk]:
    """推荐默认：以段落为单位装箱，不切断句子，跨段累积。"""
    return _pack(segs, size, overlap, "paragraph")


def chunk_section(segs: list[Segment], size: int = 500, overlap: int = 60) -> list[Chunk]:
    """章节感知：块首注入【章节名】，章节之间强制断块。

    为什么注入章节名很重要：
        一个块如果只写「农户填写融资意向，系统按 w1/w2/w3 权重计算相似度」，
        读者（和模型）不知道这是哪个功能。加上【2.6.2 联合贷款人推荐】，
        答案归属立刻清晰。代价是每块多 ~20 字。

    **短章节合并（修过的坑）**：
        实测第一版产生 96 个 <100 字的碎块。原因这份文档有大量
        「2.6.1 贷款产品智能匹配（IM-01）」这种标题下面只跟一两行说明的章节。
        纯粹按章节断，就把 500 字的容量浪费掉了。
        → 引入 min_size：章节内容不足 min_size 且与上一块同章节层级时，
          合并进上一个块，而不是独立成块。
    """
    # 先按章节分组，保留原始顺序
    groups: list[tuple[str, list[Segment]]] = []
    for s in segs:
        key = s.section or "(无章节)"
        if groups and groups[-1][0] == key:
            groups[-1][1].append(s)
        else:
            groups.append((key, [s]))

    # 合并过短的相邻组（同页优先，避免跨太远）
    merged: list[tuple[str, list[Segment]]] = []
    for sec, group_segs in groups:
        body_len = sum(len(s.text) for s in group_segs if s.kind == "text")
        has_table = any(s.kind == "table" for s in group_segs)
        if (
            merged
            and body_len < size // 3
            and not has_table
            and merged[-1][1]
            and not any(s.kind == "table" for s in merged[-1][1][-1:])
        ):
            prev_sec, prev_segs = merged[-1]
            prev_len = sum(len(s.text) for s in prev_segs if s.kind == "text")
            if prev_len + body_len <= size:
                merged[-1] = (prev_sec, prev_segs + group_segs)
                continue
        merged.append((sec, group_segs))

    out: list[Chunk] = []
    for sec, group_segs in merged:
        inner = _pack(
            group_segs, size - 40, overlap, "section",
            force_break_on_section=False, inject_section=False,
        )
        for c in inner:
            if c.kind == "table":
                out.append(c)
            else:
                prefix = f"【{sec}】\n" if sec != "(无章节)" else ""
                out.append(
                    Chunk(prefix + c.text, "", c.pages, sec, "text", "section")
                )
    return out


STRATEGIES = {
    "fixed": chunk_fixed,
    "paragraph": chunk_paragraph,
    "section": chunk_section,
}


def build_chunks(
    pdf_path: Path,
    strategy: str = "paragraph",
    size: int = 500,
    overlap: int = 60,
) -> list[Chunk]:
    segs = parse_pdf(pdf_path)
    # 三种策略都先做「标题并入正文」，否则标题会单独成块。
    # fixed 策略是刻意粗糙的基线，同样要过这一步 ——
    # 否则对比实验会把「标题碎块」的锅算到「固定切分」头上，不公平。
    segs = _attach_headings(segs)
    chunks = STRATEGIES[strategy](segs, size=size, overlap=overlap)
    for c in chunks:
        c.source_file = pdf_path.name
    return chunks


# ---------------------------------------------------------------------------
# 自测：三种策略横向对比
# ---------------------------------------------------------------------------
def _selftest(pdf: Path) -> None:
    import statistics

    print(f"文档：{pdf.name}")
    print(f"参数：size=500 overlap=60\n")
    header = f"{'策略':<12} {'块数':>5} {'总字符':>7} {'中位':>5} {'均值':>5} {'最小':>4} {'最大':>5} {'<100字':>7} {'跨页块':>6}"
    print(header)
    print("-" * len(header))

    results = {}
    for strat in STRATEGIES:
        cs = build_chunks(pdf, strategy=strat)
        results[strat] = cs
        lens = [len(c) for c in cs]
        small = sum(1 for n in lens if n < 100)
        multi = sum(1 for c in cs if len(set(c.pages)) > 1)
        print(
            f"{strat:<12} {len(cs):>5} {sum(lens):>7} {statistics.median(lens):>5.0f} "
            f"{sum(lens)/len(lens):>5.0f} {min(lens):>4} {max(lens):>5} {small:>7} {multi:>6}"
        )

    # 抽样展示：同一个位置三种切法的差异
    print("\n" + "=" * 90)
    print(" 抽样对比：三种切法在同一主题上的表现")
    print("=" * 90)
    for strat in STRATEGIES:
        cs = [c for c in results[strat] if c.kind == "text" and "联合贷款人" in c.text]
        if not cs:
            print(f"\n[{strat}] 未命中关键词")
            continue
        c = cs[0]
        print(f"\n[{strat}] {c.page_label()} | {c.section} | {len(c)}字")
        print(c.text[:420].replace("\n", " ⏎ "))
        print("-" * 90)


def _default_pdf() -> Path:
    docs = Path(__file__).resolve().parent.parent / "data" / "docs"
    pdfs = sorted(docs.rglob("*.pdf")) if docs.exists() else []
    if not pdfs:
        raise SystemExit(f"没找到 PDF。把文件放进 {docs}/ 即可（支持子目录）。")
    return pdfs[0]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_pdf()
    _selftest(target)
