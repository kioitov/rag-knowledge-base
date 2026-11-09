"""环境自检：一次把「库装没装 / Key 配没配 / 网络通不通」全查完。

排查思路：RAG 项目报错九成来自这三层之一，所以先分层定位，
不要一上来就改代码。

用法：
    python stage0\\check_env.py
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文会乱码。强制 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ("fastapi", "Web 服务框架"),
    ("uvicorn", "ASGI 服务器"),
    ("streamlit", "前端"),
    ("openai", "调大模型的 SDK"),
    ("chromadb", "向量库"),
    ("dotenv", "读 .env"),
    ("pymupdf", "PDF 解析"),
    ("docx", "Word 解析"),
    ("numpy", "向量运算"),
]

OPTIONAL = [
    ("langchain", "框架，第一版刻意不用"),
]


def check_libs() -> int:
    print("[1/3] Python 库")
    print(f"      解释器 {sys.version.split()[0]} @ {sys.executable}")
    missing = 0
    for name, desc in REQUIRED:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"      OK   {name:<12} {ver:<12} {desc}")
        except Exception as exc:
            missing += 1
            print(f"      MISS {name:<12} {'':<12} {desc}  ({type(exc).__name__})")
    for name, desc in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"      opt  {name:<12} {getattr(mod, '__version__', '?'):<12} {desc}")
        except Exception:
            print(f"      --   {name:<12} {'':<12} {desc}（没装，正常）")
    return missing


def check_key() -> int:
    print("\n[2/3] API Key")
    from dotenv import load_dotenv
    import os

    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"      OK   .env 存在：{env_file}")
    else:
        print(f"      MISS 没有 .env，请把 .env.example 复制成 .env 再填 Key")

    key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        print("      MISS 环境变量里没有 DEEPSEEK_API_KEY")
        return 1
    if key.startswith("sk-") and len(key) > 20:
        print(f"      OK   Key 格式正常：{key[:6]}...{key[-4:]}（长度 {len(key)}）")
        return 0
    print(f"      WARN Key 长度/前缀可疑：{key[:8]}...（DeepSeek 一般是 sk- 开头，32 位以上）")
    return 0


def check_net() -> int:
    print("\n[3/3] 网络")
    hosts = [("api.deepseek.com", 443), ("pypi.tuna.tsinghua.edu.cn", 443)]
    failed = 0
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=8):
                print(f"      OK   {host}:{port} 可连通")
        except Exception as exc:
            failed += 1
            print(f"      FAIL {host}:{port} -> {type(exc).__name__}: {exc}")
    return failed


def main() -> None:
    print("=" * 68)
    print(" RAG 知识库 · 阶段 0 环境自检")
    print("=" * 68)
    missing = check_libs()
    key_bad = check_key()
    net_bad = check_net()

    print("\n" + "=" * 68)
    if missing or key_bad or net_bad:
        print(f" 结论：还有 {missing} 个库缺失 / Key 状态 {key_bad} / {net_bad} 个网络目标不通")
        if missing:
            print(" 修复： python -m pip install fastapi uvicorn streamlit openai chromadb python-dotenv")
        if key_bad:
            print(" 修复： 编辑 .env，填 DEEPSEEK_API_KEY=sk-xxxx")
        print("=" * 68)
        sys.exit(1)
    print(" 结论：环境就绪，可以跑 hello_llm.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
