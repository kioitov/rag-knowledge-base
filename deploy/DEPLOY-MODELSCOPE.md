# 部署到魔搭创空间（ModelScope Studio）

## 为什么换成魔搭

原本计划用 Streamlit Community Cloud，但实测网络不通：

| 站点 | 无 VPN | 有 VPN |
|---|---|---|
| github.com | ✗ 超时 | ✓ |
| share.streamlit.io | ✓ 可访问 | ✓ 但登录被风控拦截（「访问被阻止」） |

**Streamlit Cloud 必须用 GitHub 授权登录**，而 GitHub 在国内直连不通。
开着 VPN 又会被 Streamlit 的 Snowflake 风控判定为高风险 IP。

这是一个死锁。所以换平台。

### 魔搭创空间的优势

实测连通性：**modelscope.cn 直连 246ms，不需要任何代理**。

| 项 | 魔搭创空间 | Streamlit Cloud |
|---|---|---|
| 国内直连 | ✅ 246ms | ❌ 需 VPN |
| 登录方式 | 阿里云账号 / 手机号 | 必须 GitHub |
| 免费 CPU 资源 | **2 核 16GB** | 约 1–2.7GB |
| 支持 Streamlit | ✅ 原生 `sdk_type: streamlit` | ✅ |
| 与百炼 API | 同一生态（都是阿里） | 跨云 |
| 公网链接 | ✅ 自动生成 | ✅ |

---

## 已准备好的部署包

```
deploy_pkg/          16.9 MB / 20 个文件
├── ms_deploy.json          部署配置（sdk_type: streamlit）
├── requirements.txt        依赖清单
├── config.py              配置层（已支持 st.secrets 和纯环境变量）
├── paths.py
├── deploy/app.py           应用入口
├── stage1/                 解析、切块、BM25
├── stage2/                 生成
├── stage3/                 向量、混合检索
└── data/deploy/
    ├── chunks.jsonl        1290 块（1.25 MB）
    └── chroma/             向量索引（15.5 MB）
```

**已验证可独立运行**（`python deploy/verify_package.py` 全项通过）：
- 10 个模块全部从包内加载，无外部依赖
- 1290 块 + 1290 个向量，数量一致
- 问答链路完整，页码正确显示「第 N 节」
- Streamlit UI 交互正常

---

## 部署步骤

### 1. 注册 / 登录魔搭

访问 https://modelscope.cn/ ，用**阿里云账号**或手机号登录。

> 你已经用百炼的 key，同一个阿里云账号即可，不需要另外注册。

### 2. 创建创空间

访问 https://modelscope.cn/studios → 点 **创建创空间**

选择 **编程式创空间**，然后切到 **「快速部署并创建」** 模式。

### 3. 填写基础信息

| 字段 | 填什么 |
|---|---|
| 名称 | `rag-knowledge-base` |
| 中文名 | 知识库问答（Kubernetes 中文文档） |
| 描述 | 基于 RAG 的文档问答，回答带来源定位 |
| 可见性 | 公开（面试官要能打开） |

### 4. 上传项目

把 `deploy_pkg/` 目录**整个**上传（20 个文件，16.9 MB）。

⚠️ **必须保持目录结构**。不能让 `stage1/chunker.py` 变成根目录的 `chunker.py`，
否则 `from chunker import Chunk` 这类 import 会全断。

### 5. 配置环境变量（重要）

在创空间的 **环境变量 / 密钥** 设置里添加：

```
DASHSCOPE_API_KEY = sk-你的百炼key
LLM_BASE_URL      = https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL         = qwen-plus
```

**不要**把 key 写进代码。`config.py` 已支持从环境变量读取（也兼容 Streamlit secrets）。

### 6. 部署

点「确认创建并部署」，等构建完成。

首次构建要装依赖，`chromadb` 比较大，预计 3-8 分钟。

---

## 验证清单

部署完成后逐条确认：

- [ ] 页面能打开，标题是「📚 知识库问答」
- [ ] 侧边栏显示「语料：Kubernetes 中文文档 63 篇」
- [ ] 问「Pod 的生命周期有哪几个阶段」→ 答出 Pending / Running / Succeeded / Failed
- [ ] 来源显示**「第 N 节」而不是「第 N 页」** ← 关键，证明没有伪造页码
- [ ] 问「Kubernetes 的许可证编号是多少」→ 回答「无法确定」（拒答正常）
- [ ] 打开「显示调试信息」能看到延迟和 token 数

---

## 常见问题

### 构建失败：`ModuleNotFoundError`

检查上传时目录结构是否被压平。`stage1/`、`stage2/`、`stage3/` 三个目录必须存在。

### 运行时：`没有配置 API Key`

环境变量没设或键名错了。必须是 `DASHSCOPE_API_KEY`（全大写、下划线）。

### 运行时：`FileNotFoundError: chunks.jsonl`

`data/deploy/` 没上传完整。检查 `data/deploy/chunks.jsonl` 和 `data/deploy/chroma/` 是否都在。

### 显示「第 N 页」而不是「第 N 节」

`meta["paged"]` 丢了。检查 `stage1/cache.py` 的序列化和反序列化是否都带 `meta`。
（这个 bug 真实发生过，是在预构建时才发现的。）

### 回答很慢（5-8 秒）

正常。混合检索要调一次 embedding（约 1 秒），生成约 3-6 秒。
如果想快，侧边栏把检索方式切到 `bm25`，能省掉 embedding 那一步。

---

## 如果魔搭也不行

备选方案，按可行性排序：

| 方案 | 网络 | 免费额度 | 门槛 |
|---|---|---|---|
| **魔搭创空间** | ✅ 直连 | 2核16G | 阿里云账号 |
| Gitee Pages + 静态前端 | ✅ 直连 | 静态托管 | 需改造前端 |
| Railway | ✅ 直连（实测通） | $5 试用额 | 需 GitHub 或邮箱 |
| Fly.io | ✅ 直连（实测通） | 有限免费额度 | 需绑卡 |
| Vercel | ✅ 直连（实测 239ms） | 免费 | 需 GitHub |

**注意**：Railway / Fly.io / Vercel 部署通常也需要连 GitHub 拉代码，
所以在没有 VPN 的环境下，魔搭是唯一完全绕开 GitHub 的免费方案。

---

## 本地验证（不用等云端）

```powershell
cd <项目根目录>

# 1. 打包
python deploy/build_package.py

# 2. 验证包能独立跑（需要 key）
$env:DASHSCOPE_API_KEY="sk-你的key"
python deploy/verify_package.py

# 3. 本地起服务看界面
cd deploy_pkg
python -m streamlit run deploy/app.py --server.port 8503
```

第 2 步通过后再上传，能省掉一轮「传上去才发现挂了」的折腾。
