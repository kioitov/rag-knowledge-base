"""阶段 0 验收脚本：跑通一次大模型 API 调用，终端打印模型回复。

面试价值：这是整条 RAG 链路的最后一环（生成）。先把它单独跑通，
后面接检索时出错，就知道问题一定在检索侧，不在模型侧。

用法：
    python stage0\\hello_llm.py
    python stage0\\hello_llm.py "用一句话解释什么是 RAG"
    python stage0\\hello_llm.py --dry-run   # 不花钱，只看请求体
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，中文会乱码。强制 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. 配置：.env 与本文件同级的上一级目录
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

DEFAULT_QUESTION = "用三句话说明：企业知识库为什么要做检索增强（RAG），而不是把整份文档塞给模型？"


def load_env() -> dict[str, str]:
    """读取 .env，返回配置字典。缺 key 直接给可操作的报错，不抛栈。"""
    from dotenv import load_dotenv

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        print(f"[warn] 没找到 {ENV_FILE}，尝试直接读系统环境变量")

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    return {
        "api_key": api_key.strip(),
        "base_url": (os.getenv("LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        "model": os.getenv("LLM_MODEL") or "deepseek-chat",
    }


# ---------------------------------------------------------------------------
# 2. 调用
# ---------------------------------------------------------------------------
def call_llm(cfg: dict[str, str], question: str, dry_run: bool = False) -> None:
    if dry_run:
        print("=== --dry-run：以下是将要发出的请求，不会产生费用 ===")
        print(f"POST {cfg['base_url']}/chat/completions")
        print(f"Authorization: Bearer {'***' + cfg['api_key'][-4:] if cfg['api_key'] else '(未配置)'}")
        print(f"model      : {cfg['model']}")
        print(f"messages   : [{{'role': 'user', 'content': {question!r}}}]")
        return

    if not cfg["api_key"]:
        print("[FAIL] 没读到 API Key。")
        print(f"       请把 {ENV_FILE} 里的 DEEPSEEK_API_KEY= 后面填上你的 key。")
        print("       没申请过：https://platform.deepseek.com/api_keys （充 10 元够跑完整个项目）")
        sys.exit(2)

    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    print(f"model : {cfg['model']}")
    print(f"base  : {cfg['base_url']}")
    print(f"问    : {question}")
    print("-" * 60)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": question}],
            temperature=0.3,
        )
    except Exception as exc:  # 网络/鉴权/额度的报错要能一眼看懂
        elapsed = time.perf_counter() - t0
        print(f"[FAIL] 调用失败（耗时 {elapsed:.2f}s）")
        print(f"       类型 : {type(exc).__name__}")
        print(f"       信息 : {exc}")
        hint = {
            "AuthenticationError": "Key 错了或没生效，去 platform.deepseek.com 重新生成",
            "RateLimitError": "余额不足或触发限流，检查充值",
            "APIConnectionError": "网络不通，试试关掉代理或换网络",
            "NotFoundError": "model 名字不对，DeepSeek 用 deepseek-chat",
        }.get(type(exc).__name__)
        if hint:
            print(f"       建议 : {hint}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    answer = resp.choices[0].message.content

    print(answer)
    print("-" * 60)
    usage = resp.usage
    print(f"耗时   : {elapsed:.2f}s")
    if usage:
        print(f"tokens : 输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens} / 合计 {usage.total_tokens}")
    print("[OK] 阶段 0 验收通过：模型能通、Key 有效、SDK 装对了。")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 0：最小 LLM 调用验收")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="要问的问题")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求体，不实际调用")
    args = parser.parse_args()

    cfg = load_env()
    call_llm(cfg, args.question, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
