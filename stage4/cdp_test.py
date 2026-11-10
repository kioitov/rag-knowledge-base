"""用 CDP 驱动 Edge 实测 Streamlit 前端（绕开 bsk 插件）。

============================== 为什么写这个 ==============================

前置尝试都不成：
    - computer_screenshot：截图会被 DSH 窗口遮挡
    - computer_window_capture：Chromium GPU 合成，PrintWindow 返回纯黑
    - bsk_* 插件：提示「需要升级 bsk 浏览器服务」，连不上

CDP（Chrome DevTools Protocol）是最底层的方案：
直接连 Edge 的调试端口，用 WebSocket 发命令控制页面。
不依赖任何插件，也不需要窗口焦点。

============================== 依赖 ==============================

不需要 pip 安装额外包：用 websocket-client（streamlit 的依赖里已有）
手写最小的 CDP 客户端。这样少一个依赖，也便于别人复现。

用法：
    # 先起带调试端口的 Edge
    msedge.exe --remote-debugging-port=9222 --user-data-dir=<某目录> http://127.0.0.1:8501
    # 再跑本脚本
    python stage4/cdp_test.py
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CDP_PORT = 9222
TARGET_URL = "127.0.0.1:8501"
OUT_DIR = Path(__file__).resolve().parent / "shots"


class CDP:
    """极简 CDP 客户端：求值、截图、事件派发。"""

    def __init__(self, ws_url: str, timeout: float = 90) -> None:
        from websocket import create_connection

        # 超时给到 90s：实测 Page.captureScreenshot 在大页面上可能耗时较久，
        # 30s 会超时。截图失败不该让整个测试崩掉（见 screenshot 的容错）。
        self.ws = create_connection(ws_url, timeout=timeout)
        self._id = 0

    def send(self, method: str, params: dict | None = None,
             timeout: float | None = None) -> dict:
        self._id += 1
        msg_id = self._id
        self.ws.send(json.dumps({"id": msg_id, "method": method,
                                 "params": params or {}}))
        deadline = time.time() + (timeout or 90)
        while time.time() < deadline:
            try:
                raw = self.ws.recv()
            except Exception as exc:
                raise TimeoutError(
                    f"{method} 等待响应失败（{type(exc).__name__}）。"
                    f"若发生在截图上，通常是页面太重或 Edge 卡住。"
                ) from exc
            data = json.loads(raw)
            if data.get("id") == msg_id:
                return data
        raise TimeoutError(f"{method} 超时")

    def eval(self, expr: str, await_promise: bool = True):
        r = self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": await_promise,
        })
        result = r.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def screenshot(self, path: Path) -> bool:
        """截图。失败不抛异常，返回 False —— 截图是辅助证据，不是测试必需项。

        ⚠️ 扩展名要和实际编码一致：本函数用 jpeg 编码，
        所以必须存成 .jpg。第一版传的是 .png 文件名，
        结果文件内容是 JPEG，read_image 直接报「扩展名与内容不符」。
        """
        try:
            r = self.send("Page.captureScreenshot",
                          {"format": "jpeg", "quality": 80,
                           "captureBeyondViewport": False},
                          timeout=120)
            data = r.get("result", {}).get("data")
            if not data:
                return False
            # 强制用 .jpg 后缀，避免扩展名与内容不一致
            path = path.with_suffix(".jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(data))
            return True
        except Exception as exc:
            print(f"       [截图失败] {type(exc).__name__}: {str(exc)[:60]}")
            return False

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def find_target() -> str:
    """找到 Streamlit 页面的 WebSocket 调试地址。"""
    with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=10) as r:
        targets = json.load(r)
    for t in targets:
        if t.get("type") == "page" and TARGET_URL in (t.get("url") or ""):
            return t["webSocketDebuggerUrl"]
    # 没找到就列出所有页面，便于排查
    print("  当前页面列表：")
    for t in targets:
        print(f"    [{t.get('type')}] {t.get('url')}")
    raise RuntimeError(f"没找到 {TARGET_URL} 对应的页面，可能没打开")


def wait_for(js_cond: str, timeout: float = 30, interval: float = 0.5) -> bool:
    """轮询等待 JS 条件成立。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if CDP_INSTANCE.eval(js_cond):
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


CDP_INSTANCE: CDP | None = None


def main() -> None:
    global CDP_INSTANCE

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" CDP 驱动前端实测")
    print("=" * 78)

    ws = find_target()
    print(f"  连接 {ws}")

    cdp = CDP(ws)
    CDP_INSTANCE = cdp

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))

    try:
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")

        # ---- 1. 页面加载完成 ----
        loaded = wait_for("document.readyState === 'complete'", 30)
        record("页面加载完成", loaded)

        # ---- 2. 关键控件存在 ----
        has_title = bool(cdp.eval("!!document.body.innerText.includes('知识库问答')"))
        record("标题渲染", has_title)

        has_sidebar = bool(cdp.eval(
            "!!document.querySelector('[data-testid=\"stSidebar\"]')"
        ))
        record("侧边栏存在", has_sidebar, "" if has_sidebar else "选择器可能变化")

        has_chat_input = bool(cdp.eval(
            "!!document.querySelector('[data-testid=\"stChatInput\"] textarea') "
            "|| !!document.querySelector('[data-testid=\"stChatInput\"]')"
        ))
        record("聊天输入框存在", has_chat_input)

        cdp.screenshot(OUT_DIR / "01_loaded.png")

        # ---- 3. 点「检测连接」按钮 ----
        # Streamlit 的按钮文案在 button 元素里
        clicked = cdp.eval("""
        (() => {
          const btns = Array.from(document.querySelectorAll('button'));
          const target = btns.find(b => b.innerText.trim().includes('检测连接'));
          if (!target) return false;
          target.click();
          return true;
        })()
        """)
        record("点击「检测连接」", bool(clicked))

        # 等后端响应（会出现 metric 或 success/error）
        got_health = wait_for("""
        (() => {
          const t = document.body.innerText;
          return t.includes('已连接') || t.includes('文档块') || t.includes('连不上');
        })()
        """, 40)
        record("健康检查有响应", got_health)

        health_text = cdp.eval("""
        (() => {
          const t = document.body.innerText;
          const m = t.match(/已连接[^\\n]*/);
          return m ? m[0] : '';
        })()
        """)
        if health_text:
            print(f"       返回：{health_text}")

        cdp.screenshot(OUT_DIR / "02_health.png")

        # ---- 4. 提问 ----
        # ⚠️ 第一版用 JS 直接改 textarea.value + 派发 input 事件，失败了：
        #    截图显示输入框被清空、但主区域没有任何对话。
        #    原因：Streamlit 的 chat_input 是 React 受控组件，
        #    必须在真实焦点下用**输入事件序列**触发，单纯 set value 会被框架状态覆盖。
        #    正确做法是用 CDP 的 Input.dispatchKeyEvent 模拟真实键盘。
        question = "这个系统能装多少人同时用"

        # 4a. 先点进输入框拿到焦点
        focused = cdp.eval("""
        (() => {
          const ta = document.querySelector('[data-testid="stChatInput"] textarea')
                  || document.querySelector('textarea');
          if (!ta) return false;
          ta.focus();
          ta.click();
          return document.activeElement === ta;
        })()
        """)
        record("输入框获得焦点", bool(focused))

        # 4b. 逐字符输入（真实键盘事件）
        for ch in question:
            cdp.send("Input.dispatchKeyEvent", {"type": "char", "text": ch})
            time.sleep(0.012)

        typed_ok = cdp.eval("""
        (() => {
          const ta = document.querySelector('[data-testid="stChatInput"] textarea')
                  || document.querySelector('textarea');
          return ta ? ta.value : '';
        })()
        """)
        record("填入问题", typed_ok.strip() != "", f"输入框内容长度 {len(typed_ok or '')}")

        # 4c. 用真实回车提交（rawKeyDown + keyUp，Streamlit 认这个）
        cdp.send("Input.dispatchKeyEvent", {
            "type": "rawKeyDown", "key": "Enter", "code": "Enter",
            "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
        })
        cdp.send("Input.dispatchKeyEvent", {
            "type": "keyUp", "key": "Enter", "code": "Enter",
            "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
        })
        record("提交问题", True, "已发送回车")

        # 4d. 等待**用户消息出现在对话区**（比等文字更可靠的信号）
        user_msg_shown = wait_for("""
        (() => {
          // 用户气泡有自己的 testid
          const bubbles = document.querySelectorAll(
            '[data-testid="stChatMessage"]');
          return bubbles.length >= 1;
        })()
        """, 60)
        record("用户消息已出现", user_msg_shown)

        # 4e. 等助手回答（spinner 消失 + 出现来源区或拒答文本）
        got_answer = wait_for("""
        (() => {
          const t = document.body.innerText;
          const hasSources = t.includes('引用的来源') || t.includes('未被引用');
          const hasRefusal = t.includes('无法确定');
          const hasData = t.includes('个用户同时在线');
          return hasSources || hasRefusal || hasData;
        })()
        """, 180)
        record("收到回答", got_answer)

        cdp.screenshot(OUT_DIR / "03_answer.png")

        # ---- 5. 检查来源是否渲染 ----
        src_text = cdp.eval("""
        (() => {
          const t = document.body.innerText;
          const m = t.match(/引用的来源（\\d+ 条）/);
          return m ? m[0] : '';
        })()
        """)
        record("来源区块渲染", bool(src_text), src_text or "未找到")

        answer_text = cdp.eval(
            "(() => { const t = document.body.innerText; "
            "const i = t.indexOf('" + question + "'); "
            "return i >= 0 ? t.slice(i, i + 500) : ''; })()"
        )
        if answer_text:
            print()
            print("  ── 页面上的问答内容 ──")
            for line in answer_text.splitlines()[:14]:
                if line.strip():
                    print(f"    {line.strip()[:88]}")

        # ---- 6. 展开「检索到但未被引用」 ----
        expanded = cdp.eval("""
        (() => {
          const dets = Array.from(document.querySelectorAll('details'));
          const t = dets.find(d => d.innerText.includes('未被引用'));
          if (!t) return 'not_found';
          t.open = true;
          return 'opened';
        })()
        """)
        record("未引用来源可展开", expanded != "not_found", str(expanded))
        cdp.screenshot(OUT_DIR / "04_expanded.png")

    finally:
        cdp.close()

    # ---- 汇总 ----
    print()
    print("=" * 78)
    print(" 汇总")
    print("=" * 78)
    n_ok = sum(1 for _n, ok, _d in results if ok)
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"    {detail}" if detail else ""))
    print("-" * 78)
    print(f"  通过 {n_ok}/{len(results)}")
    print(f"  截图已保存到 {OUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
