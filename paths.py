"""路径与错误提示统一工具。

====================== 为什么需要这个 ======================

错误的提示信息里打印绝对路径有两个问题：
    1. 暴露本机的目录结构（账号名、盘符、项目在磁盘上的位置）
    2. 对别人没用 —— 他 clone 到别的位置后，看到的是与你机器相关的路径

正确做法是显示**相对于项目根目录**的路径，或者干脆只显示约定位置
（如 `data/docs/`）。这样信息量不变，但不泄露也不误导。

用法：
    from paths import rel, DOCS_HINT
    raise SystemExit(f"没找到 PDF。把文件放进 {DOCS_HINT} 即可。")
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 约定位置：用户放文档的地方。提示里只说这个，不拼绝对路径。
DOCS_HINT = "data/docs/"


def rel(path: Path, base: Path | None = None) -> str:
    """返回相对项目根目录的路径；在根目录之外时退回文件名。

    为什么要有 fallback：用户的文档可能在项目目录之外（比如桌面），
    这时相对路径会变成一长串 ../..，既难看也还是泄露了上层结构。
    这种情况只给文件名——用户自己知道文件在哪。
    """
    base = base or ROOT
    try:
        return str(Path(path).resolve().relative_to(base)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def docs_dir() -> Path:
    """文档目录（不保证存在）。"""
    return ROOT / "data" / "docs"


# 支持的语料格式。
# 原来整条链路只认 PDF，换 Markdown 语料时整条流水线跑不了 ——
# 这是耦合问题，不是格式问题。列在这里，下游统一按扩展名分派解析器。
SUPPORTED_SUFFIXES = (".pdf", ".md", ".markdown")

# 以 _ 开头的文件是元数据（_MANIFEST.json、_ATTRIBUTION.md），不是语料
SKIP_PREFIXES = ("_",)


def find_docs(subdir: str | None = None) -> list[Path]:
    """在 data/docs（或其子目录）下找所有支持的文档，按路径排序。

    取代原来的 find_pdfs。保留 find_pdfs 作为别名，避免各处调用全改。
    跳过以 `_` 开头的文件（清单、署名等元数据）。
    """
    base = docs_dir()
    if subdir:
        base = base / subdir
    if not base.exists():
        return []

    out: list[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        for p in base.rglob(f"*{suffix}"):
            if p.name.startswith(SKIP_PREFIXES):
                continue
            out.append(p)
    return sorted(set(out))


def find_pdfs(subdir: str | None = None) -> list[Path]:
    """兼容旧名。实际返回所有支持的格式，不只是 PDF。

    改名初期很多脚本调用 find_pdfs；保留它避免一次性改动过大。
    但注意语义已经变了 —— 现在会返回 .md 文件。
    """
    return find_docs(subdir)


def first_doc_or_exit(subdir: str | None = None) -> Path:
    """取第一个文档；没有就给一句不含绝对路径的提示后退出。"""
    docs = find_docs(subdir)
    if not docs:
        raise SystemExit(
            f"没找到文档。\n"
            f"  请把待处理的 PDF / Markdown 放进 {DOCS_HINT}（支持子目录），然后重新运行。\n"
            f"  也可以直接传路径：python <脚本> <你的文件>"
        )
    return docs[0]


# 兼容旧名
first_pdf_or_exit = first_doc_or_exit
