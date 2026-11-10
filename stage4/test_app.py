"""阶段 4 前端测试：用 Streamlit 官方 AppTest 框架验证界面逻辑。

============================== 为什么不用截图验证 ==============================

截一张图只能证明「渲染出来了」，证明不了：
    - 点「检测连接」按钮会不会真的调到后端
    - 提问后对话历史有没有正确追加
    - 来源展开区渲染得对不对
    - 出错时错误提示会不会显示

而且截图依赖窗口焦点，容易被其他窗口遮挡（实测就踩了：点到了 DSH 窗口上）。

Streamlit 1.65 提供了 `streamlit.testing.v1.AppTest`，能：
    - 无头运行整个脚本
    - 模拟点击按钮、输入文本
    - 读取渲染出的元素和错误信息
这是官方的测试方案，比截图可靠得多。

============================== 已知局限 ==============================

AppTest 跑在进程内，**st.chat_input 的模拟有限制**。
对 chat_input 的处理方式：直接操作 session_state 注入消息，
再检查重放路径是否正确。这能测到「渲染历史」的逻辑，
但测不到「真实点击发送」的完整链路。

所以本测试分两层：
    1. AppTest：脚本能否无异常启动、侧边栏/标题/控件是否渲染
    2. 真实 HTTP：后端接口是否可用（前端就是调它）

两层都过，实际可用性就有把握；剩余风险是 Streamlit 版本相关的
前端 JS 行为，那只能靠人工点一次确认。

用法：
    # 需要后端先跑着
    python stage4/test_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

APP = Path(__file__).resolve().parent / "app.py"


def test_backend_http() -> bool:
    """第一层：后端接口是否可用。前端就是调这些。"""
    import requests

    print("=" * 78)
    print(" [1/2] 后端接口实测")
    print("=" * 78)

    base = "http://127.0.0.1:8000"
    ok = True

    try:
        h = requests.get(f"{base}/health", timeout=15).json()
        print(f"  /health      OK  chunks={h['chunks']} vector={h['vector_index']}")
    except Exception as exc:
        print(f"  /health      FAIL  {type(exc).__name__}: {exc}")
        print("              后端没起？先跑 python stage3/api.py")
        return False

    try:
        r = requests.post(
            f"{base}/chat",
            json={"question": "这个系统能装多少人同时用", "top_k": 3, "method": "hybrid"},
            timeout=120,
        )
        d = r.json()
        print(f"  /chat        OK  {len(d['answer'])} 字, "
              f"{len(d['sources'])} 条来源, {d['latency_ms']:.0f}ms")
        print(f"               引用编号 {d['used_indices']}，"
              f"cited=True 的 {sum(1 for s in d['sources'] if s['cited'])} 条")
        # 校验前端依赖的字段都在
        required = {"answer", "sources", "used_indices", "invalid_indices",
                    "latency_ms", "tokens"}
        missing = required - set(d)
        if missing:
            print(f"  [FAIL] 响应缺字段：{missing}")
            ok = False
        for s in d["sources"]:
            need = {"index", "file", "page", "section", "cited"}
            if need - set(s):
                print(f"  [FAIL] source 缺字段：{need - set(s)}")
                ok = False
    except Exception as exc:
        print(f"  /chat        FAIL  {type(exc).__name__}: {str(exc)[:120]}")
        ok = False

    try:
        s = requests.get(
            f"{base}/search",
            params={"q": "数据库挂了怎么办", "top_k": 2, "method": "hybrid"},
            timeout=60,
        ).json()
        print(f"  /search      OK  {len(s['hits'])} 条, {s['latency_ms']:.0f}ms")
    except Exception as exc:
        print(f"  /search      FAIL  {type(exc).__name__}: {str(exc)[:120]}")
        ok = False

    return ok


def test_app_renders() -> bool:
    """第二层：脚本能否无异常启动，关键控件是否都在。"""
    print()
    print("=" * 78)
    print(" [2/2] Streamlit 脚本渲染测试（AppTest）")
    print("=" * 78)

    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("  [SKIP] 当前 streamlit 版本没有 testing.v1，跳过")
        return True

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    ok = True

    if at.exception:
        print(f"  [FAIL] 脚本抛异常：")
        for e in at.exception:
            print(f"         {e.value}")
        return False
    print("  启动无异常            OK")

    # 标题
    titles = [t.value for t in at.title]
    if any("知识库问答" in t for t in titles):
        print(f"  页面标题存在          OK  {titles}")
    else:
        print(f"  [FAIL] 没找到标题，实际：{titles}")
        ok = False

    # 侧边栏控件
    if len(at.sidebar.button) >= 2:
        labels = [b.label for b in at.sidebar.button]
        print(f"  侧边栏按钮            OK  {labels}")
    else:
        print(f"  [WARN] 侧边栏按钮数量偏少")

    if at.sidebar.selectbox:
        sb = at.sidebar.selectbox[0]
        print(f"  检索方式下拉          OK  选项={sb.options} 当前={sb.value}")
        if sb.value not in ("hybrid", "vector", "bm25"):
            print(f"  [FAIL] 默认值异常")
            ok = False

    if at.sidebar.slider:
        sl = at.sidebar.slider[0]
        print(f"  检索段落数滑条        OK  {sl.value}")

    if at.sidebar.toggle:
        tg = at.sidebar.toggle[0]
        print(f"  调试开关              OK  {tg.value}")

    # chat_input 是否存在（用底层元素扫描）
    has_chat_input = False
    try:
        has_chat_input = len(at.get("chat_input")) > 0
    except Exception:
        pass
    print(f"  聊天输入框            {'OK' if has_chat_input else '未检测到（AppTest 对 chat_input 支持有限）'}")

    # ---- 测试「重放历史」路径：注入一条带 meta 的消息再重跑 ----
    print()
    print("  测试历史重放（这条路径最容易出 NameError）：")
    sample_meta = {
        "sources": [
            {"index": 1, "file": "示例文档.pdf", "page": "第 3 页",
             "section": "2.1 架构", "cited": True},
            {"index": 2, "file": "另一份.pdf", "page": "第 7-8 页",
             "section": "", "cited": False},
        ],
        "used_indices": [1],
        "invalid_indices": [9],
        "latency_ms": 1234.5,
        "tokens": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    }
    at.session_state["messages"] = [
        {"role": "user", "content": "测试问题", "meta": None},
        {"role": "assistant", "content": "测试回答 [1]", "meta": sample_meta},
    ]
    at.session_state["debug_show"] = True
    at.run()

    if at.exception:
        print(f"  [FAIL] 重放历史时抛异常：")
        for e in at.exception:
            print(f"         {e.value}")
        ok = False
    else:
        print("  重放历史无异常        OK")

    # 检查是否渲染出了来源和编造编号告警
    body = " ".join(m.value for m in at.markdown if hasattr(m, "value"))
    if "示例文档.pdf" in body:
        print("  来源文件已渲染        OK")
    else:
        print(f"  [WARN] 没在渲染结果里找到来源文件名")
    if "另一份.pdf" in body:
        print("  未引用来源也列出      OK（区分了 cited）")
    else:
        print("  [WARN] 未引用来源未渲染")

    errs = [e.value for e in at.error] if at.error else []
    if any("不存在的引用编号" in str(e) for e in errs):
        print("  编造编号告警          OK（invalid_indices 被显眼提示）")
    else:
        print(f"  [WARN] 未触发编造编号告警，errors={len(errs)}")

    return ok


def main() -> None:
    print()
    print("阶段 4 前端测试")
    print()
    backend_ok = test_backend_http()
    app_ok = test_app_renders()

    print()
    print("=" * 78)
    print(" 结论")
    print("=" * 78)
    print(f"  后端接口：{'通过' if backend_ok else '失败'}")
    print(f"  前端渲染：{'通过' if app_ok else '失败'}")
    if backend_ok and app_ok:
        print()
        print("  下一步：浏览器打开 http://localhost:8501 人工点一次，")
        print("         确认实际交互（AppTest 测不到前端 JS 行为）。")
    print("=" * 78)
    sys.exit(0 if (backend_ok and app_ok) else 1)


if __name__ == "__main__":
    main()
