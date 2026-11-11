"""打包魔搭创空间部署包：只挑运行必需的文件。

============================== 为什么要单独打包 ==============================

项目根目录有 68 MB，其中大部分线上用不到：

    data/docs_old/     旧语料备份       ~27 MB
    data/chroma/       本地向量索引     ~16 MB  ← 部署用 data/deploy/chroma
    data/cache/        解析缓存         ~2 MB
    data/eval/         评测集与结果     ~0.4 MB
    __pycache__        Python 字节码    ~1 MB
    .git/              版本库           ~30 MB

直接全传：
    1. 慢（上传几十分钟）
    2. 魔搭对单文件/总大小有限制
    3. 把评测集、旧语料这些内部文件传上去，等于泄露

所以只挑**运行这条链路真正需要**的文件：

    config.py paths.py requirements.txt
    stage1/  chunker, cache, pdf_parser, md_parser, retriever
    stage2/  generator
    stage3/  vector_index, hybrid_retriever
    deploy/  app.py + prebuild 产出的 chunks.jsonl 和 chroma/

============================== 关键：路径结构必须保持不变 ==============================

Magic Studio 的 base_image 里工作目录是 /app，
如果为了「扁平化」把 stage1/chunker.py 挪到根目录，
`from chunker import Chunk` 这类 import 就全断了。

所以打包时**保留原目录层级**，只删掉不需要的文件。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy_pkg"


# 必须原样带上的文件（相对路径）
FILES = [
    "config.py",
    "paths.py",
    "requirements.txt",
    "ms_deploy.json",
    "stage1/chunker.py",
    "stage1/cache.py",
    "stage1/pdf_parser.py",
    "stage1/md_parser.py",
    "stage1/retriever.py",
    "stage2/generator.py",
    "stage3/vector_index.py",
    "stage3/hybrid_retriever.py",
    "deploy/app.py",
]

# 必须带上的目录（整个复制）
DIRS = [
    "data/deploy",  # chunks.jsonl + chroma/
]


def main() -> None:
    print("=" * 80)
    print(" 打包魔搭创空间部署包")
    print("=" * 80)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    total = 0
    missing = []

    print("\n[1/2] 复制文件")
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        total += size
        print(f"  {size:>9,} B  {rel}")

    print("\n[2/2] 复制目录")
    for rel in DIRS:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT / rel
        shutil.copytree(src, dst)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        n = len([f for f in dst.rglob("*") if f.is_file()])
        total += size
        print(f"  {size:>9,} B  {rel}/  ({n} 个文件)")

    if missing:
        print(f"\n[WARN] 缺失 {len(missing)} 项：")
        for m in missing:
            print(f"  {m}")

    print()
    print("=" * 80)
    print(f" 输出 {OUT}")
    print(f" 总大小 {total / 1024 / 1024:.1f} MB")
    print(f" 文件数 {len([f for f in OUT.rglob('*') if f.is_file()])}")

    # 结构校验：确认 import 路径能对上
    print()
    print(" 结构校验：")
    checks = [
        ("stage1/chunker.py", "chunker 在 stage1/ 下"),
        ("stage2/generator.py", "generator 在 stage2/ 下"),
        ("stage3/vector_index.py", "vector_index 在 stage3/ 下"),
        ("data/deploy/chunks.jsonl", "切块清单"),
        ("data/deploy/chroma", "向量索引目录"),
        ("ms_deploy.json", "魔搭部署配置"),
    ]
    ok = True
    for rel, desc in checks:
        p = OUT / rel
        exists = p.exists()
        ok &= exists
        print(f"   {'OK  ' if exists else 'MISS'} {rel:<34} {desc}")

    print()
    if ok:
        print(" 打包完成，可以上传到魔搭创空间")
        print(" 详细步骤见 deploy_pkg/DEPLOY-MODELSCOPE.md")
    else:
        print(" 有缺失，检查上面的 MISS 项")
    print("=" * 80)


if __name__ == "__main__":
    main()
