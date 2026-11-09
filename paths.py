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


def find_pdfs(subdir: str | None = None) -> list[Path]:
    """在 data/docs（或其子目录）下找 PDF，按路径排序。"""
    base = docs_dir()
    if subdir:
        base = base / subdir
    if not base.exists():
        return []
    return sorted(base.rglob("*.pdf"))


def first_pdf_or_exit(subdir: str | None = None) -> Path:
    """取第一个 PDF；没有就给一句不含绝对路径的提示后退出。"""
    pdfs = find_pdfs(subdir)
    if not pdfs:
        raise SystemExit(
            f"没找到 PDF。\n"
            f"  请把待处理的 PDF 放进 {DOCS_HINT}（支持子目录），然后重新运行。\n"
            f"  也可以直接传路径：python <脚本> <你的文件.pdf>"
        )
    return pdfs[0]
