"""验证 app.py 在根目录和 deploy/ 子目录两种位置都能运行。

============================== 为什么两种都要测 ==============================

魔搭要求入口 `app.py` 必须在**根目录**，而本地开发习惯放在 `deploy/` 下。
为了两边共用同一份代码，`app.py` 里的 `_find_root()` 会自己往上找根目录。

但如果只测一个位置，另一个位置坏了也不知道。所以两个都跑一遍。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKG = Path(r"E:\deepseek work\rag-knowledge-base\deploy_pkg")

# 两个位置
CANDIDATES = [
    ("根目录", PKG / "app.py"),
    ("deploy/ 子目录", PKG / "deploy" / "app.py"),
]

from streamlit.testing.v1 import AppTest

ok_all = True
for label, path in CANDIDATES:
    print("=" * 76)
    print(f" 测试位置：{label}  ({path.relative_to(PKG)})")
    print("=" * 76)

    if not path.exists():
        print(f"  [FAIL] 文件不存在")
        ok_all = False
        continue

    try:
        at = AppTest.from_file(str(path), default_timeout=300)
        at.run()

        if at.exception:
            print(f"  [FAIL] 启动异常：{str(at.exception[0].value)[:300]}")
            ok_all = False
            continue

        titles = [t.value for t in at.title]
        print(f"  OK   启动成功，标题 {titles}")

        # 检查是否有初始化错误
        if at.error:
            print(f"  [FAIL] 页面报错：{str(at.error[0].value)[:300]}")
            ok_all = False
            continue

        # 交互测试
        at.chat_input[0].set_value("Pod 的生命周期有哪几个阶段").run(timeout=300)
        if at.exception:
            print(f"  [FAIL] 交互异常：{str(at.exception[0].value)[:300]}")
            ok_all = False
            continue

        n = len(at.get("chat_message"))
        print(f"  OK   交互产生 {n} 条消息")

        mds = [str(m.value) for m in at.markdown]
        has_jie = any("节" in m for m in mds)
        has_ans = any("Pending" in m for m in mds)
        print(f"  {'OK  ' if has_jie else 'FAIL'} 来源显示「节」")
        print(f"  {'OK  ' if has_ans else 'FAIL'} 答案含 Pending")
        ok_all &= (n >= 2 and has_jie and has_ans)

    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:300]}")
        ok_all = False

    print()

print("=" * 76)
print(f" 结论：{'两种位置都可用' if ok_all else '有问题，见上面 FAIL'}")
print("=" * 76)
sys.exit(0 if ok_all else 1)
