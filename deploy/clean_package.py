"""清理部署包：移除会引起配置冲突的文件。

============================== 问题 ==============================

部署时报错：
    json: cannot unmarshal object into Go value of type int

这是平台后端的 JSON 解析错误。查了官方 schema
(https://modelscope.cn/api/v1/studios/deploy_schema.json)，
`resource_configuration` 只允许三个枚举值，其中免费档是
`platform/2v-cpu-16g-mem`。

但页面上实际显示的是 "2v CPU / 8G内存"，说明平台做了自己的映射，
而 `ms_deploy.json` 里我写的是 16g，两边对不上。

更关键的是：**可视化表单里已经把所有配置都填对了**
（框架 Streamlit、免费 CPU、镜像、三条环境变量全部就绪），
此时 ms_deploy.json 只会和表单抢着生效，制造冲突。

============================== 处理 ==============================

删掉这几个文件：
    ms_deploy.json        与可视化表单冲突
    DEPLOY-MODELSCOPE.md  纯文档，部署不需要
    description.txt       纯文档，部署不需要

保留的是运行必需文件。
"""
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG = Path(r"E:\deepseek work\rag-knowledge-base\deploy_pkg")

REMOVE = [
    "ms_deploy.json",
    "DEPLOY-MODELSCOPE.md",
    "description.txt",
]

print("=" * 76)
print(" 清理部署包（移除会与表单冲突的配置）")
print("=" * 76)

removed = []
for name in REMOVE:
    p = PKG / name
    if p.exists():
        p.unlink()
        removed.append(name)
        print(f"  已删除 {name}")
    else:
        print(f"  跳过   {name}（不存在）")

# 同步删除根目录的 ms_deploy.json（源文件，避免下次打包又带回去）
root_ms = PKG.parent / "ms_deploy.json"
if root_ms.exists():
    root_ms.unlink()
    print(f"  已删除项目根的 ms_deploy.json（源文件）")

# 确保 app.py 在根目录
app_root = PKG / "app.py"
app_sub = PKG / "deploy" / "app.py"
if app_root.exists() and app_sub.exists():
    print(f"  app.py 已在根目录（部署入口）")

print()
print("=" * 76)
print(" 当前包内容")
print("=" * 76)

files = sorted(f for f in PKG.rglob("*") if f.is_file())
for f in files:
    rel = f.relative_to(PKG)
    print(f"  {f.stat().st_size:>9,} B  {rel}")

total = sum(f.stat().st_size for f in files)
print()
print(f"  {len(files)} 个文件 / {total / 1024 / 1024:.1f} MB")

# 校验必需项
print()
print("  必需项检查：")
need = {
    "app.py": "入口（必须在根目录）",
    "requirements.txt": "依赖清单",
    "config.py": "配置读取",
    "paths.py": "路径工具",
    "stage1/chunker.py": "切块",
    "stage2/generator.py": "生成",
    "stage3/vector_index.py": "向量",
    "data/deploy/chunks.jsonl": "切块产物",
    "data/deploy/chroma/chroma.sqlite3": "向量库",
}
ok = True
for rel, desc in need.items():
    exists = (PKG / rel).exists()
    ok &= exists
    print(f"   {'OK  ' if exists else 'MISS'} {rel:<36} {desc}")

print()
print("=" * 76)
print(f" {'可以重新上传' if ok else '有缺失，先补齐'}")
print("=" * 76)
