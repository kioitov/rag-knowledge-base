"""阶段 5 第一块：评测集构建——从语料中抽取可自动判分的事实型问题。

============================== 核心设计：不依赖人工领域判断 ==============================

使用者（YG）明确说不了解 K8s 内容，无法判断「这条答案算不算对」。
所以评测集不能建立在人工主观判断上。

改用**可自动验证的事实型问题**：

    类型                    例子                                    判定方式
    ─────────────────────────────────────────────────────────────────────────
    default_value      「Pod 的 terminationGracePeriodSeconds 默认值是多少？」  答案含 "30"
    param_name         「控制 Pod 重启策略的字段叫什么？」                  答案含 "restartPolicy"
    enum_values        「Pod 的 phase 有哪几种？」                        答案含 Pending/Running/...
    numeric_limit      「单节点默认最多跑多少个 Pod？」                    答案含 "110"
    component_role     「kube-scheduler 负责什么？」                      答案含关键词
    refuse             「Kubernetes 的许可证编号是多少？」                 必须拒答

每条评测项的「正确答案关键词」都从语料原文摘录，可逐条核验：
    - 打开对应 chunk 就能看到那个词
    - 判定是字符串包含，不含任何主观成分

============================== 为什么不能靠 LLM 自动生成全集 ==============================

LLM 生成问题很方便，但有两个坑：
    1. 它可能编出语料里没有的「事实」，生成的问题答案压根不在语料里
    2. 它生成的答案可能与原文不符
所以本脚本采用**先抽取原文事实、再据此构造问题**的路径：
    - 用正则从语料里找「X 的默认值是 Y」这类句式
    - 把 Y 作为标准答案关键词，把 X 作为问题主体
    - 每一条都保留出处 chunk_id，可人工抽查

LLM 只用来做「把陈述改写成自然问句」这一步，答案关键词始终来自原文。

============================== 判定口径（先定死）==============================

    Hit@k      正确答案所在 chunk 出现在 top-k 检索结果里
    AnswerAcc  生成的回答里包含标准答案关键词
    RefuseOK   该拒答的问题确实拒答了

三个指标分开统计，不混在一起 —— 混在一起会掩盖问题：
「检索到了但回答没引用到」和「压根没检索到」是两种完全不同的失败。
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1"))

from cache import load_chunks
from chunker import Chunk
from paths import find_docs

OUT_DIR = _ROOT / "data" / "eval"
OUT_FILE = OUT_DIR / "eval_set.json"


# ---------------------------------------------------------------------------
# 事实抽取器
# ---------------------------------------------------------------------------
@dataclass
class Fact:
    """从原文抽出的一条事实。"""

    kind: str  # default_value / numeric_limit / enum_values / param_name / component_role
    subject: str  # 问题主体，如 "terminationGracePeriodSeconds"
    answer: str  # 标准答案关键词
    evidence: str  # 原文片段（用于人工核验）
    chunk_id: str
    source_file: str
    section: str
    page_label: str


# 抽取模式：(kind, 正则)
# 每条正则必须同时捕获 subject 和 answer
PATTERNS: list[tuple[str, str, int]] = [
    # 「X 的默认值是 Y」/「X 默认为 Y」
    ("default_value",
     r"`?([A-Za-z][A-Za-z0-9_.\-]{2,})`?[^。\n]{0,12}?(?:的)?默认(?:值)?(?:为|是)\s*`?([^`。\n，,；;]{1,40})",
     2),
    # 「默认 X 为 Y」
    ("default_value",
     r"默认(?:值)?(?:为|是)\s*`?([^`。\n，,；;]{1,30})`?[^。\n]{0,10}?`?([A-Za-z][A-Za-z0-9_.\-]{2,})`?",
     2),
    # 「最多 N 个」/「最大 N」
    ("numeric_limit",
     r"`?([A-Za-z][A-Za-z0-9_.\- ]{2,30})`?[^。\n]{0,20}?最多\s*([0-9]{1,6})\s*(?:个|条|秒|分钟|MB|GB|核)",
     2),
]


def extract_facts(chunks: list[Chunk]) -> list[Fact]:
    """从所有 chunk 里抽事实。"""
    facts: list[Fact] = []
    seen: set[str] = set()

    for c in chunks:
        if c.kind != "text":
            continue
        text = c.text

        for kind, pat, _n in PATTERNS:
            for m in re.finditer(pat, text):
                subject = (m.group(1) or "").strip()
                answer = (m.group(2) or "").strip()

                # 过滤噪音：太短、纯符号、明显不是答案
                if len(answer) < 1 or len(subject) < 2:
                    continue
                if answer in ("的", "是", "为", "值", "情况下"):
                    continue
                if not re.search(r"[A-Za-z0-9]", answer) and len(answer) < 2:
                    continue

                key = f"{kind}|{subject}|{answer}"
                if key in seen:
                    continue
                seen.add(key)

                # 抓取上下文作为证据（前后各 80 字）
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                evidence = text[start:end].replace("\n", " ").strip()

                facts.append(Fact(
                    kind=kind,
                    subject=subject,
                    answer=answer,
                    evidence=evidence,
                    chunk_id=c.chunk_id,
                    source_file=c.source_file,
                    section=c.section,
                    page_label=c.page_label(),
                ))

    return facts


def main() -> None:
    print("=" * 88)
    print(" 阶段 5：从语料抽取事实型评测素材")
    print("=" * 88)

    docs = find_docs()
    if not docs:
        raise SystemExit("没找到语料。先跑 stage5/fetch_corpus.py")

    print(f"\n加载 {len(docs)} 篇文档…")
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(load_chunks(d))
    print(f"  {len(chunks)} 块")

    print("\n抽取事实…")
    facts = extract_facts(chunks)
    print(f"  抽出 {len(facts)} 条候选")

    # 按类型统计
    from collections import Counter
    by_kind = Counter(f.kind for f in facts)
    for k, n in by_kind.most_common():
        print(f"    {k:<18} {n} 条")

    # 抽样展示，人工看质量
    print("\n" + "=" * 88)
    print(" 抽样检查（人工看这些能不能当评测项）")
    print("=" * 88)
    for kind in by_kind:
        samples = [f for f in facts if f.kind == kind][:3]
        print(f"\n[{kind}]")
        for f in samples:
            print(f"  主体：{f.subject[:44]}")
            print(f"  答案：{f.answer[:44]}")
            print(f"  证据：{f.evidence[:100]}")
            print(f"  出处：{f.source_file[:36]} {f.page_label} · {f.section[:18]}")
            print()

    # 保存原始素材（还没转成问题，先让人看质量）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = OUT_DIR / "facts_raw.json"
    with open(raw_file, "w", encoding="utf-8", newline="\n") as fp:
        json.dump([asdict(f) for f in facts], fp, ensure_ascii=False, indent=1)
    print("=" * 88)
    print(f" 原始素材已存：{raw_file}（{len(facts)} 条）")
    print(" 下一步：人工筛掉噪音，再用 LLM 改写成自然问句")
    print("=" * 88)


if __name__ == "__main__":
    main()
