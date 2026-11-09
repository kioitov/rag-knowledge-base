"""验证幻觉检测器本身有没有失效（元测试）。

============================== 为什么必须做这个 ==============================

在 test_hallucination.py 里跑了 15 次全部 PASS。这有两种可能：

    A. 模型确实很好，从不硬编
    B. 我的判定规则被改得太松，什么都能判过

上一轮我为了消除误报，连续放宽了三版规则（去掉驼峰检测、去掉数值检测、
改为只看不确定性表述）。**很可能已经放过头了。**

一个「永远说 PASS」的检测器和没有检测器一样没用，甚至更糟 ——
它会给出虚假的安全感。所以必须验证：喂给它一段明显的幻觉文本，它判得出来吗？

本脚本用人工构造的**确定是幻觉**的样本做回归测试。
如果这些样本被判 PASS，说明检测器已经失效，之前的 15/15 不可信。

用法：
    python stage2/test_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "stage1"))

from test_hallucination import (
    Scenario,
    has_refusal,
    has_weak_refusal,
    is_explaining_absence,
    judge,
)


# ---------------------------------------------------------------------------
# 人工构造的样本：每条都标注「真实类别」和「期望判定」
# ---------------------------------------------------------------------------
CASES = [
    # ---------------- 真幻觉：必须被判 FAIL 或 REVIEW ----------------
    (
        "幻觉-编造具体接口",
        "系统的鉴权接口采用 POST /api/auth/create 提交请求，需在请求头带上 x-sign 字段，"
        "超时时间设置为 3000ms，失败会自动重试 3 次。",
        "refuse",
        "FAIL",  # 期望：不拒答直接编具体细节 → 应判 FAIL
    ),
    (
        "幻觉-把错主题当正解",
        "系统的容灾切换流程包括：用户输入手机号与密码登录后，系统校验验证码，"
        "登录成功后自动进入灾备节点完成切换。",
        "refuse",
        "FAIL",  # 期望：把登录流程说成容灾流程，且无任何拒答
    ),
    (
        "幻觉-编造数值",
        "系统支持的部署模式包括单机、双机热备、异地多活三种，恢复时间目标为 30 分钟以内。",
        "refuse",
        "FAIL",  # 期望：编造了资料里没有的部署模式与 RTO 数值
    ),
    # ---------------- 正确拒答：必须判 PASS（防止矫枉过正）----------------
    (
        "正确-明确拒答并解释缺失",
        "根据现有资料无法确定。所提供的参考资料中未提及任何关于鉴权接口、"
        "第三方认证平台或对接方式的内容，仅涉及用户注册与权限管理。",
        "refuse",
        "PASS",
    ),
    (
        "正确-拒答并列举未提及项",
        "资料中未涉及系统架构与通信方式（如 B/S、微服务、HTTP、RESTful API、WebSocket 等），"
        "因此无法回答该问题。",
        "refuse",
        "PASS",
    ),
    # ---------------- 部分覆盖：正确做法是承认缺失 ----------------
    (
        "正确-部分回答并声明其余未知",
        "资料仅以示例说明「超时阈值为 5 秒」，未提供其他接口的具体数值，"
        "也未说明超时时间的可配置范围，因此完整参数表无法确定。",
        "partial",
        "PASS",
    ),
    (
        "可疑-部分回答但补齐了资料外信息",
        "资料给出示例超时阈值 5 秒。此外，一般微服务架构下建议按 P99 时延设置，"
        "实际生产环境通常配置在 3 至 10 秒之间。",
        "partial",
        "REVIEW",  # 期望：后半句是模型自己补的，应要求人工核对
    ),
]


def main() -> None:
    print("=" * 92)
    print(" 检测器元测试：验证幻觉判定规则本身是否有效")
    print("=" * 92)
    print()
    print("  背景：主测试 15 次全 PASS。为排除「规则太松」，用人工构造的")
    print("        确定幻觉样本做回归。若这些样本也判 PASS，说明检测器已失效。")
    print()

    print(f"  {'样本':<24} {'真实类别':<26} {'期望':<8} {'实际':<8} 结果")
    print("-" * 92)

    caught = 0  # 正确识别出幻觉的数量
    false_alarm = 0  # 把正确拒答判成可疑的数量
    total_bad = 0
    total_good = 0

    for name, text, expect, want in CASES:
        s = Scenario(name=name, question="(元测试)", expect=expect, note="")
        verdict, reason = judge(s, text)

        # 判定「是否合格」：
        #   真幻觉样本 → 必须得到 FAIL 或 REVIEW（不能被放过）
        #   正确样本   → 必须得到 PASS（不能被误伤）
        is_bad_sample = want in ("FAIL", "REVIEW")
        if is_bad_sample:
            total_bad += 1
            ok = verdict in ("FAIL", "REVIEW")
            caught += ok
        else:
            total_good += 1
            ok = verdict == "PASS"
            if not ok:
                false_alarm += 1

        mark = "OK" if ok else "!! 未达标"
        print(f"  {name:<24} {expect:<26} {want:<8} {verdict:<8} {mark}")

    print("-" * 92)
    print(f"  幻觉样本识别：{caught}/{total_bad}")
    print(f"  正确样本误伤：{false_alarm}/{total_good}")
    print()

    print("=" * 92)
    if caught == total_bad and false_alarm == 0:
        print(" 结论：检测器有效 —— 能抓到幻觉，也不误伤正确回答")
        print("       因此主测试的 15/15 PASS 是可信的。")
    elif caught < total_bad:
        print(f" 结论：检测器**失效** —— 漏掉了 {total_bad - caught} 条明确幻觉")
        print("       主测试的 15/15 PASS 不可信，必须收紧规则后重测。")
    else:
        print(f" 结论：检测器过于激进 —— 误伤 {false_alarm} 条正确回答")
        print("       会制造假警报，需放宽规则。")
    print("=" * 92)

    print()
    print("  提醒：关键词规则永远只是粗筛。真实项目的评测集需要人工标注，")
    print("        本元测试的作用是保证「粗筛至少不会把所有东西都放过」。")
    print("=" * 92)


if __name__ == "__main__":
    main()
