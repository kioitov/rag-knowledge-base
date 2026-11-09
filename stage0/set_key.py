"""一条命令填 Key：把百炼 API Key 写进 .env，不需要手动编辑文件。

为什么写这个脚本：
    手动改 .env 有三个常见翻车点 ——
    1. 复制时带了首尾空格，key 变成 " sk-xxx"，鉴权失败但报错看不懂
    2. 把 key 贴到了注释行后面（`# DASHSCOPE_API_KEY=` 那一行）
    3. 编辑后忘了保存，或者存成了别的编码
    脚本能统一处理，且填完立即验证。

用法：
    python stage0\\set_key.py sk-xxxxxxxx
    python stage0\\set_key.py            # 不带参数，交互式粘贴（不会回显）
    python stage0\\set_key.py --show     # 只看当前配了什么
"""
from __future__ import annotations

import argparse
import getpass
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

ENV = config.ENV_FILE


def read_env() -> str:
    if not ENV.exists():
        example = ENV.parent / ".env.example"
        if example.exists():
            return example.read_text(encoding="utf-8")
        return ""
    return ENV.read_text(encoding="utf-8")


def write_key(var: str, value: str) -> None:
    """把 `VAR=` 那一行替换掉。只动这一行，注释和排版原样保留。"""
    text = read_env()
    # 匹配行首的 VAR=...（只匹配未被注释的行）
    pattern = re.compile(rf"^{re.escape(var)}\s*=.*$", re.MULTILINE)

    if pattern.search(text):
        new_text = pattern.sub(f"{var}={value}", text, count=1)
    else:
        # 变量不存在（比如从旧模板来），追加到文件末尾
        new_text = text.rstrip("\n") + f"\n\n{var}={value}\n"

    # 用 utf-8 无 BOM 写。PS7 + Python 都是这个默认，显式写死更稳。
    with io.open(ENV, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)


def mask(k: str) -> str:
    if not k:
        return "(空)"
    if len(k) <= 12:
        return k[:3] + "***"
    return f"{k[:8]}...{k[-4:]}（{len(k)} 位）"


def main() -> None:
    ap = argparse.ArgumentParser(description="把 API Key 写入 .env")
    ap.add_argument("key", nargs="?", help="API Key；省略则交互式输入")
    ap.add_argument("--show", action="store_true", help="只显示当前配置")
    ap.add_argument("--var", default=None, help="变量名，默认按当前服务商自动判断")
    args = ap.parse_args()

    var = args.var or config.api_key_var_name()

    if args.show:
        print(config.summary())
        return

    raw = args.key
    if not raw:
        print(f"服务商：{config.provider()}")
        print(f"变量名：{var}")
        print(f"目标文件：{ENV}")
        print("粘贴 Key 后回车（输入不回显）：")
        raw = getpass.getpass("")

    key = (raw or "").strip().strip('"').strip("'")
    if not key:
        print("[FAIL] 空值，什么都没改。")
        sys.exit(1)

    # ---- 格式预检：这些错误提前拦住，比调用 API 报 401 再排查省事 ----
    warnings = []
    if key != raw:
        warnings.append("去掉了首尾空格/引号")
    if any(c.isspace() for c in key):
        print(f"[FAIL] Key 中间含空白字符，多半是复制时带进了换行：{mask(key)}")
        sys.exit(1)
    if len(key) < 20:
        warnings.append(f"长度只有 {len(key)} 位，看着不像完整 key")
    if var == "DASHSCOPE_API_KEY" and not key.startswith("sk-"):
        warnings.append("百炼 key 通常以 sk- 开头，确认没复制错东西")

    write_key(var, key)
    print(f"[OK] 已写入 {ENV.name}：{var}={mask(key)}")
    for w in warnings:
        print(f"     WARN {w}")

    # ---- 写完后直接**从文件**回读，确认真的落盘了 ----
    # 不要用 importlib.reload(config) 来验证：load_dotenv(override=False) 不会
    # 覆盖进程里已存在的环境变量，reload 后读到的还是旧值（甚至空值），
    # 会误报「回读不一致」。验证文件就必须直接读文件。
    text = ENV.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(var)}\s*=(.*)$", text, re.MULTILINE)
    on_disk = (m.group(1).strip().strip('"').strip("'") if m else "")

    if on_disk == key:
        print(f"[OK] 磁盘回读一致，{var} 已落盘")
    else:
        print(f"[FAIL] 磁盘回读不一致：拿到 {mask(on_disk)}")
        sys.exit(1)

    # ---- 再确认没有重复行（重复时 dotenv 取第一个，容易踩坑）----
    dup = len(re.findall(rf"^{re.escape(var)}\s*=", text, re.MULTILINE))
    if dup > 1:
        print(f"[WARN] .env 里有 {dup} 行 {var}，dotenv 只会用第一行。删掉多余的。")

    print()
    print("下一步验证连通性：")
    print(f'  python "{ENV.parent / "stage0" / "hello_llm.py"}"')


if __name__ == "__main__":
    main()
