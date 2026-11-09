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
    sys.path.insert(0, str(ROOT))
    import config  # 用统一配置层，避免这里和业务代码判断逻辑不一致

    print(f"      服务商 : {config.provider()}")
    print(f"      chat   : {config.chat_model()}")

    key = config.api_key()
    var = config.api_key_var_name()
    if not key:
        print(f"      MISS 没读到 {var}")
        print(f"           请在 {config.ENV_FILE} 里填 {var}=your-key")
        print(f"           百炼申请：https://bailian.console.aliyun.com/  → API-KEY 管理")
        return 1
    if len(key) < 20:
        print(f"      WARN {var} 只有 {len(key)} 位，看着不像完整 key")
        return 1
    print(f"      OK   {var} = {key[:6]}...{key[-4:]}（{len(key)} 位）")

    emb = config.embedding_config()
    if emb.ready():
        print(f"      OK   embedding: {emb.summary()}")
    elif emb.configured():
        print(f"      WARN embedding 缺 key：{emb.model}")
    else:
        print(f"      --   embedding 未配置（阶段 3 才需要）")
    return 0


def check_net() -> int:
    print("\n[3/3] 网络")
    sys.path.insert(0, str(ROOT))
    import config

    # 从当前配置的 base_url 推导要探测的主机，而不是写死
    from urllib.parse import urlparse

    target = urlparse(config.base_url()).hostname or "dashscope.aliyuncs.com"
    hosts = [(target, 443), ("pypi.tuna.tsinghua.edu.cn", 443)]
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
            sys.path.insert(0, str(ROOT))
            import config
            print(f" 修复： 编辑 {config.ENV_FILE}，填 {config.api_key_var_name()}=your-key")
        print("=" * 68)
        sys.exit(1)
    print(" 结论：环境就绪，可以跑 hello_llm.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
