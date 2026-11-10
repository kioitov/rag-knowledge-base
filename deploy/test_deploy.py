"""CDP 实测部署版 Streamlit（端口 8502）。

用与 stage4/cdp_test.py 相同的技术，但目标是 deploy/app.py。
目的：证明预构建产物 + 单进程部署版能真正跑起来。
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CDP_PORT = 9223
TARGET = "127.0.0.1:8502"
OUT = Path(__file__).resolve().parent / "shots"


class CDP:
    def __init__(self, ws_url: str, timeout: float = 90) -> None:
        from websocket import create_connection
        self.ws = create_connection(ws_url, timeout=timeout)
        self._id = 0

    def send(self, method: str, params: dict | None = None, timeout: float = 90) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ws.recv()
            d = json.loads(raw)
            if d.get("id") == mid:
                return d
        raise TimeoutError(method)

    def eval(self, expr: str):
        r = self.send("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True,
        })
        return r.get("result", {}).get("result", {}).get("value")

    def shot(self, name: str) -> bool:
        try:
            r = self.send("Page.captureScreenshot",
                          {"format": "jpeg", "quality": 80}, timeout=120)
            data = r.get("result", {}).get("data")
            if not data:
                return False
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{name}.jpg").write_bytes(base64.b64decode(data))
            return True
        except Exception as exc:
            print(f"    [截图失败] {type(exc).__name__}")
            return False

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def find_target() -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=10) as r:
        targets = json.load(r)
    for t in targets:
        if t.get("type") == "page" and TARGET in (t.get("url") or ""):
            return t["webSocketDebuggerUrl"]
    for t in targets:
        print(f"  [{t.get('type')}] {t.get('url')}")
    raise RuntimeError("没找到目标页面")


CDP_I: CDP | None = None


def wait_js(cond: str, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if CDP_I and CDP_I.eval(cond):
                return True
        except Exception:
            pass
        time.sleep(0.6)
    return False


def main() -> None:
    global CDP_I
    ws = find_target()
    cdp = CDP(ws)
    CDP_I = cdp

    results = []

    def rec(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))

    print("=" * 80)
    print(" 部署版实测（预构建产物 + 单进程）")
    print("=" * 80)

    try:
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")

        rec("页面加载", wait_js("document.readyState === 'complete'", 40))

        # 等索引加载完成（出现 sidebar 或标题）
        loaded = wait_js(
            "document.body.innerText.includes('知识库问答')", 120)
        rec("索引加载并渲染", loaded)

        # 检查是否有错误提示
        err = cdp.eval("""
        (() => {
          const t = document.body.innerText;
          const m = t.match(/初始化失败[^\\n]*/);
          return m ? m[0] : '';
        })()
        """)
        rec("无初始化错误", not err, err[:80] if err else "")

        cdp.shot("01_loaded")

        # 填入问题并提交
        q = "Pod 的生命周期有哪几个阶段"
        cdp.eval("""
        (() => {
          const ta = document.querySelector('[data-testid="stChatInput"] textarea')
                  || document.querySelector('textarea');
          if (!ta) return false;
          ta.focus(); ta.click(); return true;
        })()
        """)
        for ch in q:
            cdp.send("Input.dispatchKeyEvent", {"type": "char", "text": ch})
            time.sleep(0.012)

        typed = cdp.eval("""
        (() => {
          const ta = document.querySelector('[data-testid="stChatInput"] textarea')
                  || document.querySelector('textarea');
          return ta ? ta.value : '';
        })()
        """)
        rec("输入问题", bool(typed.strip()), f"长度 {len(typed or '')}")

        # ---- 提交 ----
        # ⚠️ 只靠 Input.dispatchKeyEvent 发 Enter 不触发 Streamlit 的提交。
        # 第一版就是这样：输入框被清空、但没有任何消息出现，
        # 而我的等待条件太宽（匹配到了侧边栏文字）就误判为成功。
        #
        # 更可靠的做法：**点发送按钮**。Streamlit 的 chat_input 有个
        # submit 按钮（data-testid="stChatInputSubmitButton"）。
        clicked = cdp.eval("""
        (() => {
          const btn = document.querySelector('[data-testid="stChatInputSubmitButton"]')
                   || document.querySelector('[data-testid="stChatInput"] button');
          if (!btn) return 'no_button';
          btn.click();
          return 'clicked';
        })()
        """)
        if clicked != "clicked":
            # 退路：回车
            cdp.send("Input.dispatchKeyEvent", {
                "type": "rawKeyDown", "key": "Enter", "code": "Enter",
                "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
            cdp.send("Input.dispatchKeyEvent", {
                "type": "keyUp", "key": "Enter", "code": "Enter",
                "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
        rec("提交问题", True, str(clicked))

        # 等待条件必须严格：**assistant 消息数 >= 1**
        # 之前用「页面含某段文字」会被侧边栏误匹配，是典型的假成功。
        got = wait_js("""
        (() => {
          const msgs = document.querySelectorAll('[data-testid="stChatMessage"]');
          return msgs.length >= 2;
        })()
        """, 240)
        rec("收到回答", got)

        src = cdp.eval("""
        (() => {
          const m = document.body.innerText.match(/引用的来源（\\d+ 条）/);
          return m ? m[0] : '';
        })()
        """)
        rec("来源区块渲染", bool(src), src or "未找到")

        # 检查页码是否显示为「节」而不是「页」
        page_info = cdp.eval("""
        (() => {
          const t = document.body.innerText;
          const jie = (t.match(/第 \\d+ 节/g) || []).length;
          const ye = (t.match(/第 \\d+ 页/g) || []).length;
          return {jie: jie, ye: ye};
        })()
        """)
        if page_info:
            rec("页码展示正确（用「节」不用「页」）",
                page_info.get("jie", 0) > 0 and page_info.get("ye", 0) == 0,
                f"节×{page_info.get('jie')} 页×{page_info.get('ye')}")

        cdp.shot("02_answer")

        # 打印看到的问答
        txt = cdp.eval("""
        (() => {
          const t = document.body.innerText;
          const i = t.indexOf('Pod 的生命周期有哪几个阶段');
          return i >= 0 ? t.slice(i, i + 420) : '';
        })()
        """)
        if txt:
            print()
            print("  ── 页面内容 ──")
            for line in txt.splitlines()[:12]:
                if line.strip():
                    print(f"    {line.strip()[:86]}")

    finally:
        cdp.close()

    print()
    print("=" * 80)
    n_ok = sum(1 for _n, ok, _d in results if ok)
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"    {detail}" if detail else ""))
    print("-" * 80)
    print(f"  通过 {n_ok}/{len(results)}")
    print(f"  截图 {OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
