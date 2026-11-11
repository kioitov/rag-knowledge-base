"""统一配置层：一个文件决定用哪家模型服务，其余代码不关心。

============================ 为什么需要这层 ============================

原来的做法是 `hello_llm.py` 里直接读 `DEEPSEEK_API_KEY`，接了百炼就要改代码。
实际做项目时换服务商是常态（额度用完、限流、比价、客户指定），
把「读哪个 key / 用哪个 base_url / 用哪个模型名」硬编码进业务代码，
每换一次就要翻遍全仓库改字符串 —— 这是典型的耦合。

所以抽出这个模块，暴露三个入口：
    get_chat_client()     → OpenAI 兼容客户端（chat 用）
    get_chat_model()      → 模型名
    get_embedding_config()→ embedding 的 base_url / model / dim

业务代码只调这三个，换服务商只改 .env。

============================ 为什么默认百炼 ============================

对比过两家，百炼在「一个人做完整个 RAG 项目」这个场景下更合适：

                    阿里云百炼             DeepSeek
    chat            qwen-plus 等           deepseek-chat
    embedding       text-embedding-v4      **没有官方 embedding 接口**
    rerank          有（qwen3-reranker）    没有
    免费额度        有（新用户 token 赠送）  充钱

关键差异是 embedding。阶段 3 要把 645 个块转成向量，阶段 5 要做
「纯向量 vs 混合检索」的对比，都必须有 embedding。用 DeepSeek 就得再找一家
（BGE 本地部署 / 硅基流动 / 通义），多一套 key、多一个失败点。
百炼一家全包，少一个变量。

**但配置层仍然支持 DeepSeek 和其他 OpenAI 兼容服务** —— 这不是过度设计，
是因为阶段 5 要做对比实验，多服务商能力本身就有价值。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

# Streamlit Cloud / 魔搭创空间等平台把密钥注入 st.secrets，
# 而不是提供 .env 文件。这里在没有 .env 时尝试从 st.secrets 同步到环境变量，
# 让下游所有 `os.getenv(...)` 的代码不用改。
# 存在的意义：部署包里不带 .env（那会泄露密钥），只能靠平台注入。
def _sync_from_streamlit_secrets() -> None:
    if ENV_FILE.exists():
        return  # 本地有 .env，优先用它
    try:
        import streamlit as st  # type: ignore
        secrets = getattr(st, "secrets", None)
        if not secrets:
            return
        # 只同步已知的几个键，避免把无关配置塞进环境
        for key in (
            "DASHSCOPE_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
            "LLM_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM",
            "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY",
        ):
            try:
                val = secrets[key]
            except Exception:
                continue
            if val and not os.getenv(key):
                os.environ[key] = str(val)
    except Exception:
        # streamlit 未安装 / 不在 runtime / secrets 未配置，都静默跳过
        pass


_sync_from_streamlit_secrets()

# ---------------------------------------------------------------------------
# 服务商预设
# ---------------------------------------------------------------------------
# 这些值来自各厂商当前公开文档，实测可用性由 check_setup.py 验证。
# 注意：百炼北京地域已推出业务空间专属域名
#   https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
# 官方说明「现有域名仍可正常使用」，所以默认用老的通用域名（不需要 WorkspaceId，
# 少一个必填项就少一个配置错误来源）。要用专属域名就在 .env 里覆盖。
PRESETS: dict[str, dict[str, str]] = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "embedding_model": "text-embedding-v4",
        "embedding_dim": "1024",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "embedding_model": "",  # DeepSeek 没有官方 embedding
        "embedding_dim": "",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": "1536",
    },
}

DEFAULT_PROVIDER = "dashscope"


def _load_env() -> None:
    """加载 .env，但**不覆盖已存在的真实环境变量**。

    ---------------------------- 这里踩过坑 ----------------------------
    第一版用了 `load_dotenv(ENV_FILE, override=True)`，理由是「改了 .env 立刻生效」。
    结果实测发现：在 shell 里 `$env:LLM_PROVIDER='deepseek'` 后跑脚本，
    读到的仍然是 .env 里的 dashscope —— **override=True 把真实环境变量覆盖了**。

    这违反 12-factor 原则：环境变量优先级应高于配置文件。
    后果是没法临时切换服务商做对比测试，也没法在 CI 里注入配置。

    而且 override=True 想解决的问题本来就不存在：
    每次跑脚本都是新进程，自然读到 .env 的最新值，不需要 override。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)


_load_env()


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def _preset() -> dict[str, str]:
    p = provider()
    if p not in PRESETS:
        raise ValueError(
            f"未知的服务商 '{p}'，可选：{', '.join(PRESETS)}。"
            f"改 .env 里的 LLM_PROVIDER。"
        )
    return PRESETS[p]


def api_key() -> str:
    """按优先级取 key：当前服务商的专用变量 → 通用变量。

    允许通用变量 LLM_API_KEY 兜底，方便临时切换测试不同服务商而不用改多个变量名。
    """
    preset = _preset()
    for var in (preset["api_key_env"], "LLM_API_KEY"):
        v = (os.getenv(var) or "").strip()
        if v:
            return v
    return ""


def api_key_var_name() -> str:
    """返回「应该填哪个变量名」，用于报错提示。"""
    return _preset()["api_key_env"]


def base_url() -> str:
    return (os.getenv("LLM_BASE_URL") or _preset()["base_url"]).rstrip("/")


def chat_model() -> str:
    return (os.getenv("LLM_MODEL") or _preset()["chat_model"]).strip()


def embedding_config() -> "EmbeddingConfig":
    """embedding 配置。允许和 chat 用不同服务商（实际项目中很常见：
    chat 用便宜快的，embedding 用效果好的）。"""
    preset = _preset()

    # 单独的 embedding 服务商（可选）
    emb_provider = (os.getenv("EMBEDDING_PROVIDER") or provider()).strip().lower()
    emb_preset = PRESETS.get(emb_provider, preset)

    emb_key = (
        os.getenv(f"{emb_provider.upper()}_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or api_key()
    ).strip()

    return EmbeddingConfig(
        api_key=emb_key,
        base_url=(os.getenv("EMBEDDING_BASE_URL") or emb_preset["base_url"]).rstrip("/"),
        model=(os.getenv("EMBEDDING_MODEL") or emb_preset["embedding_model"]).strip(),
        dim=int(os.getenv("EMBEDDING_DIM") or emb_preset["embedding_dim"] or 1024),
    )


@dataclass
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    dim: int

    def configured(self) -> bool:
        """配置里是否指定了 embedding 模型（不看 key）。

        dry-run 模式要看这个 —— 它只验证配置形状对不对，不验证凭证。
        """
        return bool(self.model)

    def ready(self) -> bool:
        """能否真正调用（模型名 + key 都有）。"""
        return bool(self.api_key and self.model)

    def summary(self) -> str:
        return f"{self.model} ({self.dim} 维) @ {self.base_url}"


def missing_embedding_hint() -> str:
    """当前服务商不能用 embedding 时，给出具体该怎么办。"""
    p = provider()
    if p == "deepseek":
        return (
            "DeepSeek 没有官方 embedding 接口。两条路：\n"
            "  A. 把 LLM_PROVIDER 改成 dashscope（百炼），一家全包\n"
            "  B. 保留 deepseek 做 chat，在 .env 里另设\n"
            "     EMBEDDING_PROVIDER=dashscope  +  DASHSCOPE_API_KEY=xxx\n"
            "     （chat 和 embedding 用不同服务商是正常做法）"
        )
    return (
        f"服务商 {p} 未配置 embedding 模型。\n"
        f"  在 .env 里设 EMBEDDING_MODEL=<模型名>，\n"
        f"  或改用 EMBEDDING_PROVIDER=dashscope + DASHSCOPE_API_KEY=xxx"
    )


def get_chat_client():
    """返回 OpenAI 兼容客户端。key 缺失时给出可操作的报错，不抛栈。"""
    key = api_key()
    if not key:
        raise RuntimeError(
            f"没读到 API Key。请在 {ENV_FILE} 里填 {api_key_var_name()}=sk-xxxx"
        )
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base_url())


def summary() -> str:
    """一行描述当前配置，用于每个脚本开头打印，出问题一眼看出用错了哪家。"""
    preset = _preset()
    k = api_key()
    masked = f"{k[:6]}...{k[-4:]}({len(k)}位)" if k else "(未配置)"
    lines = [
        f"服务商  : {provider()}",
        f"chat    : {chat_model()}  @ {base_url()}",
        f"key     : {masked}  ({api_key_var_name()})",
    ]
    emb = embedding_config()
    if emb.ready():
        lines.append(f"embedding: {emb.summary()}")
    elif emb.configured():
        lines.append(f"embedding: {emb.summary()}  (缺 key)")
    else:
        lines.append("embedding: (当前服务商不支持，阶段 3 才需要处理)")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(" 当前配置")
    print("=" * 70)
    print(summary())
    print("=" * 70)
    if not api_key():
        print(f"\n下一步：编辑 {ENV_FILE}")
        print(f"把 {api_key_var_name()}= 后面填上百炼的 key")
        print("申请地址：https://bailian.console.aliyun.com/  →  API-KEY 管理")
