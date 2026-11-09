"""阶段 0 验收脚本：验证「配置 → chat API → embedding API」整条链路。

比原 `hello_llm.py` 多做两件事：
    1. 接入了 config.py，换服务商不用改代码
    2. 同时验证 embedding 接口 —— 阶段 3 要用，现在不通就提前暴露

用法：
    python stage0\\hello_llm.py                    # 默认问题
    python stage0\\hello_llm.py "你好"              # 自定义问题
    python stage0\\hello_llm.py --dry-run           # 只看请求体，不花钱
    python stage0\\hello_llm.py --skip-embedding    # 只测 chat
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

DEFAULT_QUESTION = "用三句话说明：企业知识库为什么要做检索增强（RAG），而不是把整份文档塞给模型？"

# 常见异常 → 可操作的排查建议。这些是最容易踩的坑，写死在这里省得每次查文档。
ERROR_HINTS = {
    "AuthenticationError": "Key 无效或没生效。检查：是否复制完整（含前后空格）、是否属于当前地域。",
    "PermissionDeniedError": "Key 有效但没权限。检查百炼控制台里该模型是否已开通。",
    "NotFoundError": "模型名不存在。检查 .env 里 LLM_MODEL 拼写，或该模型是否在当前地域可用。",
    "RateLimitError": "限流或额度用尽。检查百炼控制台余额与免费额度。",
    "APIConnectionError": "网络不通。检查代理设置，或换网络。",
    "BadRequestError": "请求参数有问题。常见：模型不支持该参数、输入过长。",
    "APIStatusError": "服务端返回错误。看上面的 message 定位。",
}


def explain(exc: Exception) -> None:
    name = type(exc).__name__
    print(f"  类型   : {name}")
    print(f"  信息   : {exc}")
    hint = ERROR_HINTS.get(name)
    if hint:
        print(f"  排查   : {hint}")


def test_chat(question: str, dry_run: bool) -> bool:
    print("─" * 72)
    print("[1/2] Chat 接口")
    print("─" * 72)
    print(f"  服务商 : {config.provider()}")
    print(f"  模型   : {config.chat_model()}")
    print(f"  endpoint: {config.base_url()}/chat/completions")

    if dry_run:
        print(f"  [dry-run] 不实际调用。将要发送：")
        print(f"    messages = [{{'role':'user','content':{question!r}}}]]")
        return True

    if not config.api_key():
        print(f"  [FAIL] 没读到 Key。请在 .env 里填 {config.api_key_var_name()}")
        return False

    try:
        client = config.get_chat_client()
    except Exception as exc:
        explain(exc)
        return False

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=config.chat_model(),
            messages=[{"role": "user", "content": question}],
            temperature=0.3,
        )
    except Exception as exc:
        print(f"  [FAIL] 调用失败（{time.perf_counter() - t0:.2f}s）")
        explain(exc)
        return False

    elapsed = time.perf_counter() - t0
    answer = resp.choices[0].message.content

    print(f"  [OK] 耗时 {elapsed:.2f}s")
    print()
    print("  问题：", question)
    print("  回答：")
    for line in (answer or "").strip().splitlines():
        print("   ", line)

    u = resp.usage
    if u:
        print()
        print(f"  tokens: 输入 {u.prompt_tokens} / 输出 {u.completion_tokens} / 合计 {u.total_tokens}")
    return True


def test_embedding(dry_run: bool) -> bool:
    print()
    print("─" * 72)
    print("[2/2] Embedding 接口（阶段 3 要用，现在先确认通）")
    print("─" * 72)

    cfg = config.embedding_config()
    if dry_run:
        # dry-run 只看配置形状，不看 key
        if not cfg.configured():
            print(f"  [SKIP] 当前服务商没有 embedding 模型")
            print("  " + config.missing_embedding_hint().replace("\n", "\n  "))
            return True
        print(f"  模型   : {cfg.model}")
        print(f"  维度   : {cfg.dim}")
        print(f"  endpoint: {cfg.base_url}/embeddings")
        print("  [dry-run] 不实际调用")
        return True

    if not cfg.ready():
        print(f"  [SKIP] 当前服务商（{config.provider()}）没有可用的 embedding 配置。")
        print("  " + config.missing_embedding_hint().replace("\n", "\n  "))
        return True  # 阶段 0 不强制要求 embedding

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    # 探测文本用通用技术句子，与具体业务无关。
    # 验收目标是「接口能通、维度正确、向量有区分度」，与文本主题无关；
    # 用通用句子既不泄露业务领域，别人 clone 下来也更容易理解。
    probe = [
        "向量数据库用于存储高维嵌入向量",
        "文档切块需要设置重叠以避免语义截断",
    ]
    t0 = time.perf_counter()
    try:
        resp = client.embeddings.create(model=cfg.model, input=probe)
    except Exception as exc:
        print(f"  [FAIL] 调用失败（{time.perf_counter() - t0:.2f}s）")
        explain(exc)
        return False

    elapsed = time.perf_counter() - t0
    vecs = [d.embedding for d in resp.data]
    actual_dim = len(vecs[0])

    # 余弦相似度：两句同属「RAG 技术」这一子领域，应落在弱相关到高度相关之间
    import math

    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    sim = cos(vecs[0], vecs[1])

    print(f"  [OK] 耗时 {elapsed:.2f}s")
    print(f"  返回维度: {actual_dim}（配置写的是 {cfg.dim}）")
    print(f"  两句话余弦相似度: {sim:.4f}")
    if actual_dim != cfg.dim:
        print(f"  [WARN] 维度与配置不一致。用百炼 text-embedding-v4 时，")
        print(f"         不显式传 dimensions 参数可能返回默认 1024 以外的值。")
        print(f"         阶段 3 写 Chroma 集合时必须按实际返回维度建索引，否则插入会报错。")

    # ---- 相似度判定 ----
    # 阈值来自实测标定（stage0/calibrate_embedding.py，text-embedding-v4 / 1024 维，
    # 四组对照共 24 条通用技术文本）：
    #     完全等价  0.993
    #     高度相关  0.751
    #     弱相关    0.304
    #     无关      0.223
    # 最初我按「语义相近」的直觉预设了 0.5 的预警线，结果在两条
    # 「同属一个项目、但讲不同子话题」的文本上（0.352）误报。
    # 标定之后才确认：那个值本来就属于弱相关区间，模型没有问题。
    # 真正的异常信号是「低于无关组最低值」或「等于 1」。
    if sim < 0.15:
        print(f"  [FAIL] 相似度 {sim:.3f} 低到异常区间（无关组实测 0.223）。")
        print(f"         检查是否用错模型，或返回的是零向量。")
        return False
    if sim > 0.98:
        print(f"  [WARN] 相似度 {sim:.3f} 几乎等于 1，两句不同的话不该这样。")
        print(f"         检查是否把同一条文本传了两次。")
    print(f"  [OK] 相似度在健康区间（弱相关实测 0.30 / 无关 0.22 / 相关 0.75）")
    print(f"       若要完整验证区分度，跑 stage0/calibrate_embedding.py")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 0 验收：配置 + chat + embedding")
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--dry-run", action="store_true", help="只打印配置和请求体，不调用")
    ap.add_argument("--skip-embedding", action="store_true", help="跳过 embedding 检测")
    args = ap.parse_args()

    print("=" * 72)
    print(" 阶段 0 验收 · 配置与 API 连通性")
    print("=" * 72)
    print(config.summary())
    print()

    ok_chat = test_chat(args.question, args.dry_run)
    ok_emb = True if args.skip_embedding else test_embedding(args.dry_run)

    print()
    print("=" * 72)
    if args.dry_run:
        # dry-run 只验配置形状，不能宣称「验收通过」——
        # 没 Key 的情况下说通过是误导，下一阶段会立刻卡住。
        print(" 结论：dry-run 完成，配置形状正确")
        if not config.api_key():
            print(f" 但还没填 Key：在 .env 里设 {config.api_key_var_name()}=your-key")
            print(" 填完去掉 --dry-run 再跑一次，才叫阶段 0 验收通过。")
        else:
            print(" 去掉 --dry-run 再跑一次，验证真实连通。")
    elif ok_chat and ok_emb:
        print(" 结论：阶段 0 验收通过")
        print(" 下一步：阶段 2 —— 把检索到的块拼进 prompt，生成带来源的回答")
    else:
        print(" 结论：未通过，按上面的「排查」提示处理后重跑")
    print("=" * 72)


if __name__ == "__main__":
    main()
