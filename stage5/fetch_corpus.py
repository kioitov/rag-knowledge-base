"""拉取 Kubernetes 中文文档（CC BY 4.0），转成纯文本语料。

============================== 为什么选这份 ==============================

用户要中文语料、且希望能合法公开演示。候选：

    Rust 中文书        Apache-2.0/MIT   叙述性强，事实密度低
    K8s 中文文档       CC BY 4.0         概念+任务+参考，事实密度中等偏高
    PostgreSQL 中文    ——               没有官方中文版

选 K8s 中文文档，理由：
    1. 许可证明确：文档分发于 CC BY 4.0，可自由分发，只需署名
    2. 有官方中文翻译，质量由 CNCF 社区维护，不是机翻
    3. 内含大量可精确匹配的事实：参数名、默认值、组件职责、命令用法
    4. 源文件是 Markdown（GitHub kubernetes/website），比 PDF 干净得多

============================== 一个必须处理的问题 ==============================

K8s 文档的中文版里，**英文原文被保留在 HTML 注释里**，形如：

    <!--
    ## Why you need Kubernetes
    -->
    ## 为什么需要 Kubernetes

这是他们维护双语对照的方式。直接抽文本会把英文全带进来，
导致中英语料混杂、检索时中文 query 命中英文段落。

所以必须做两件事：
    1. 删除 `<!-- ... -->` 注释块（里面全是英文原文）
    2. 删除 Hugo 短代码 `{{% ... %}}` 和 front matter

============================== 抓取策略 ==============================

不用 GitHub API（匿名限流 60 次/小时，实测已触发 403），
直接走 raw.githubusercontent.com，没有限流。

按目录抓，只取 docs 下的核心章节（concepts / tasks / reference），
跳过 contribute 这类维护者向的文档（对问答无价值）。

用法：
    python stage5/fetch_corpus.py            # 抓取并清洗
    python stage5/fetch_corpus.py --limit 60 # 只抓 60 篇
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "docs" / "k8s-zh"
RAW = "https://raw.githubusercontent.com/kubernetes/website/main/content/zh-cn/docs"

# 要抓的目录（相对 content/zh-cn/docs/）
SECTIONS = [
    "concepts/workloads",
    "concepts/overview",
    "concepts/architecture",
    "concepts/configuration",
    "concepts/services-networking",
    "concepts/storage",
    "concepts/security",
    "concepts/scheduling-eviction",
    "tasks/configure-pod-container",
    "tasks/run-application",
    "reference/glossary",
]

UA = {"User-Agent": "Mozilla/5.0 (corpus-fetch; educational)"}


def fetch(url: str, retry: int = 3) -> str | None:
    """抓一个 URL，返回文本。404 返回 None，其他错误重试。"""
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retry - 1:
                print(f"      [HTTP {exc.code}] {url}")
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            if attempt == retry - 1:
                print(f"      [{type(exc).__name__}] {url}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def list_dir(url: str) -> list[dict]:
    """列目录内容。

    raw.githubusercontent.com 不支持列目录，所以这里用 GitHub 的
    HTML 页面解析 -- 但那太脆弱。改用另一条路：直接尝试常见文件名。

    实际上更可靠的办法是用 codeload 下载整个 tarball，但那是 100+ MB。
    这里采取折中：维护一份已知文件名清单（从文档站导航结构生成），
    加上按目录猜测常见命名。
    """
    return []


# ---------------------------------------------------------------------------
# 已知文件清单：从 K8s 文档站导航结构整理出的核心页面
# ---------------------------------------------------------------------------
KNOWN_PAGES = [
    # 概述与架构
    "concepts/overview/_index",
    "concepts/overview/components",
    "concepts/overview/kubernetes-api",
    "concepts/overview/working-with-objects/_index",
    "concepts/overview/working-with-objects/names",
    "concepts/overview/working-with-objects/namespaces",
    "concepts/overview/working-with-objects/labels",
    "concepts/overview/working-with-objects/annotations",
    "concepts/overview/working-with-objects/field-selectors",
    "concepts/overview/working-with-objects/object-management",
    "concepts/architecture/_index",
    "concepts/architecture/nodes",
    "concepts/architecture/control-plane-node-communication",
    "concepts/architecture/cloud-controller",
    # 工作负载
    "concepts/workloads/_index",
    "concepts/workloads/pods/_index",
    "concepts/workloads/pods/pod-lifecycle",
    "concepts/workloads/pods/init-containers",
    "concepts/workloads/pods/sidecar-containers",
    "concepts/workloads/pods/ephemeral-containers",
    "concepts/workloads/controllers/deployment",
    "concepts/workloads/controllers/replicaset",
    "concepts/workloads/controllers/statefulset",
    "concepts/workloads/controllers/daemonset",
    "concepts/workloads/controllers/job",
    "concepts/workloads/controllers/cron-jobs",
    "concepts/workloads/management",
    # 配置
    "concepts/configuration/_index",
    "concepts/configuration/configmap",
    "concepts/configuration/secret",
    "concepts/configuration/manage-resources-containers",
    "concepts/configuration/environment-variables",
    "concepts/configuration/liveness-readiness-startup-probes",
    # 网络
    "concepts/services-networking/_index",
    "concepts/services-networking/service",
    "concepts/services-networking/ingress",
    "concepts/services-networking/network-policies",
    "concepts/services-networking/dns-pod-service",
    # 存储
    "concepts/storage/_index",
    "concepts/storage/volumes",
    "concepts/storage/persistent-volumes",
    "concepts/storage/storage-classes",
    # 安全
    "concepts/security/_index",
    "concepts/security/service-accounts",
    "concepts/security/secrets-good-practices",
    "concepts/security/rbac-good-practices",
    # 调度
    "concepts/scheduling-eviction/_index",
    "concepts/scheduling-eviction/kube-scheduler",
    "concepts/scheduling-eviction/assign-pod-node",
    "concepts/scheduling-eviction/taint-and-toleration",
    "concepts/scheduling-eviction/resource-bin-packing",
    # 任务
    "tasks/configure-pod-container/_index",
    "tasks/configure-pod-container/configure-service-account",
    "tasks/configure-pod-container/assign-cpu-resource",
    "tasks/configure-pod-container/assign-memory-resource",
    "tasks/configure-pod-container/configure-liveness-readiness-startup-probes",
    "tasks/run-application/_index",
    "tasks/run-application/run-stateless-application-deployment",
    "tasks/run-application/scale-stateful-set",
    # ---------------------------------------------------------------
    # 集群管理与运维
    # 补这批是因为实测发现覆盖缺口：问「etcd 是干什么的」两个检索器都答不出，
    # 全库 etcd 只出现 15 次、且散落在「良好实践」类文档里，没有专门章节。
    # 这不是检索算法的问题，是语料没覆盖到 —— 评测集必须建立在
    # 「答案确实在语料里」的前提下，否则测的是「语料缺失」而不是「检索能力」。
    # ---------------------------------------------------------------
    "concepts/cluster-administration/_index",
    "concepts/cluster-administration/flow-control",
    "concepts/cluster-administration/logging",
    "concepts/cluster-administration/manage-deployment",
    "concepts/cluster-administration/networking",
    "tasks/administer-cluster/configure-upgrade-etcd",
    "tasks/administer-cluster/kubeadm/_index",
    "tasks/administer-cluster/kubeadm/kubeadm-certs",
    "tasks/administer-cluster/kubelet-config-file",
    "tasks/administer-cluster/manage-resources/_index",
    "tasks/tools/_index",
    "tasks/tools/install-kubectl-linux",
    # 参考：术语表
    "reference/glossary/_index",
]


def clean_markdown(md: str, title: str) -> str:
    """把 K8s 的 Markdown 清洗成适合 RAG 的纯文本。

    ---------------------------- 清洗过度的教训 ----------------------------
    第一版用 `re.sub(r"\{\{[<%].*?[%>]\}\}", "")` 把整个 Hugo 短代码删掉。
    结果这句被毁：

        原文  ：例如当不满足 Deployment 的 {{< glossary_tooltip text="期望副本数" >}} 字段时
        清洗后：例如当不满足 Deployment 的  字段时，要启动新的 ）。

    `glossary_tooltip` 里 `text="..."` 的内容是有价值的正文术语，
    整体删除让句子直接残缺。

    正确做法：**只删短代码标记，保留内容**。有 text="..." 就取出来，
    没有（如 `{{% heading %}}`）才整体删掉。
    """
    # 1. 删 HTML 注释（英文原文全在这里面）
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)

    # 2. 删 front matter（--- ... ---）
    md = re.sub(r"^---\s*\n.*?\n---\s*\n", "", md, flags=re.DOTALL)

    # 3. Hugo 短代码：只删标记，保留内容
    def _shortcode(m: re.Match) -> str:
        inner = m.group(1)
        for attr in ("text", "term", "title", "name"):
            tm = re.search(rf'{attr}\s*=\s*"([^"]*)"', inner)
            if tm:
                return tm.group(1)
        return ""

    md = re.sub(r"\{\{[<%]\s*(.*?)\s*[%>]\}\}", _shortcode, md, flags=re.DOTALL)

    # 4. 删图片
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

    # 5. 链接只保留文字：[文字](url) → 文字
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)

    # 6. 删 HTML 标签（<br>, <details> 之类）
    md = re.sub(r"<[^>]+>", "", md)

    # 7. 代码块保留内容，但删掉语言标注的围栏噪音
    md = re.sub(r"^```[a-zA-Z]*\s*$", "", md, flags=re.MULTILINE)

    # 8. 压缩多余空行
    md = re.sub(r"\n{3,}", "\n\n", md)

    # 9. 去掉每行尾部空白
    md = "\n".join(line.rstrip() for line in md.splitlines())

    # 10. 开头插入标题（方便检索时定位）
    body = md.strip()
    if title and not body.startswith(title):
        body = f"# {title}\n\n{body}"

    return body.strip()


def page_title(md: str) -> str:
    """从 front matter 或第一个标题取标题。"""
    m = re.search(r'^title:\s*"?(.*?)"?\s*$', md, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^#\s+(.+)$", md, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取 K8s 中文文档")
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 篇（调试用）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = KNOWN_PAGES[: args.limit] if args.limit else KNOWN_PAGES

    print("=" * 84)
    print(" 拉取 Kubernetes 中文文档（CC BY 4.0）")
    print("=" * 84)
    print(f"  目标目录：{OUT_DIR}")
    print(f"  页面数：{len(pages)}")
    print()

    ok = 0
    skipped = 0
    total_chars = 0
    manifest = []

    for i, rel in enumerate(pages, 1):
        url = f"{RAW}/{rel}.md"
        md = fetch(url)
        if md is None:
            skipped += 1
            print(f"  [{i:2d}/{len(pages)}] MISS  {rel}")
            continue

        title = page_title(md)
        clean = clean_markdown(md, title)
        if len(clean) < 300:
            skipped += 1
            print(f"  [{i:2d}/{len(pages)}] SKIP  内容过短 {rel}")
            continue

        # 文件名：目录层级展平，避免重名
        fname = rel.replace("/", "__") + ".md"
        out = OUT_DIR / fname
        with io.open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean)

        ok += 1
        total_chars += len(clean)
        manifest.append({
            "file": fname,
            "path": rel,
            "title": title,
            "chars": len(clean),
            "source_url": f"https://kubernetes.io/zh-cn/docs/{rel}/",
        })
        print(f"  [{i:2d}/{len(pages)}] OK    {len(clean):>6,} 字  {title[:40]}")

    # 写清单（记录来源，满足 CC BY 4.0 的署名要求）
    manifest_path = OUT_DIR / "_MANIFEST.json"
    with io.open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "corpus": "Kubernetes 中文文档",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "© The Kubernetes Authors | Documentation Distributed under CC BY 4.0",
            "source_repo": "https://github.com/kubernetes/website",
            "fetched_from": RAW,
            "page_count": ok,
            "total_chars": total_chars,
            "pages": manifest,
        }, f, ensure_ascii=False, indent=2)

    # 写署名文件（CC BY 4.0 要求保留）
    attrib = OUT_DIR / "_ATTRIBUTION.md"
    with io.open(attrib, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# 语料来源与许可\n\n"
            "本目录下的文档来自 Kubernetes 官方中文文档。\n\n"
            "- **版权**：© The Kubernetes Authors\n"
            "- **许可**：Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
            "- **许可全文**：https://creativecommons.org/licenses/by/4.0/\n"
            "- **源仓库**：https://github.com/kubernetes/website\n"
            "- **在线文档**：https://kubernetes.io/zh-cn/docs/\n\n"
            "按 CC BY 4.0 要求保留署名。本项目仅用于检索增强生成的技术演示，\n"
            "未对原文内容做实质性修改，仅去除了双语对照用的 HTML 注释与站点标记。\n"
        )

    print()
    print("=" * 84)
    print(f"  成功 {ok} 篇 / 跳过 {skipped} 篇")
    print(f"  总字数 {total_chars:,}")
    print(f"  输出 {OUT_DIR}")
    print(f"  清单 {manifest_path.name}，署名 {attrib.name}")
    print("=" * 84)


if __name__ == "__main__":
    main()
