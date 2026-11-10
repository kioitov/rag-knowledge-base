# 部署说明

## 目标平台：Streamlit Community Cloud

**为什么不用 Hugging Face Spaces**：免费层现在只支持 Static Space 和带 ZeroGPU 的 Gradio Space，
Docker / Streamlit Space 需要 PRO 订阅（$9/月）。而且 HF 已经废弃了原生 Streamlit SDK。

Streamlit Community Cloud 是官方的、免费的、公网可访问的。

---

## 免费层的实际限制

| 项 | 限制 | 对项目的影响 |
|---|---|---|
| 休眠 | 12 小时无访问会睡，首次访问需 30-60 秒唤醒 | 面试前先自己点开一次预热 |
| 内存 | 约 1–2.7 GB | 本项目的索引只占 ~17 MB，无压力 |
| 磁盘 | 临时文件系统，重启即丢 | 所以必须预构建索引（见下） |
| 私有应用 | 只允许 1 个 | 项目仓库是 public，无影响 |
| 自定义域名 | 不支持 | 地址是 `<你的名字>-rag-knowledge-base.streamlit.app` |
| 私有仓库 | 需要额外授权 | 本项目 public，不需要 |

---

## 架构：单进程

本地开发时前端通过 HTTP 调 FastAPI（`stage3/api.py`），那是合理的解耦。
但 Streamlit Cloud **只能跑一个进程**，起不了第二个服务。

所以 `deploy/app.py` 把检索和生成直接 import 在同进程里：

```
本地开发                          线上部署
┌──────────┐  HTTP  ┌──────────┐   ┌─────────────────┐
│stage4/app│ ─────► │stage3/api│   │  deploy/app.py  │
└──────────┘        └──────────┘   │  ├ BM25         │
                                   │  ├ Chroma       │
                                   │  └ 生成          │
                                   └─────────────────┘
```

两种模式共存，各自服务不同场景。阶段 4 的 `app.py` 保留不动。

---

## 关键：预构建索引

Streamlit Cloud 的容器文件系统是临时的。如果启动时才解析语料 + 调 embedding：

```
解析 63 篇 Markdown      0.1 秒
构建向量索引（1290 块）   129 次 API 调用 → 约 50 秒
加上网络抖动             实际 1-2 分钟
```

每次重启都重来一遍，体验很差且白烧 token。

**所以本地构建一次，产物提交到仓库，线上直接加载。**

```powershell
python deploy/prebuild_index.py
```

产物：

```
data/deploy/
├── chunks.jsonl     1290 块，1.25 MB
└── chroma/          向量索引，约 15 MB
```

总计 **16.8 MB**，提交到 git 可接受（没到需要 LFS 的程度）。

启动流程变成：

```
读 chunks.jsonl    → 建 BM25（0.1 秒，纯内存）
加载 chroma/       → 直接可用
```

实测冷启动约 3-5 秒。

---

## 部署步骤

### 1. 本地预构建并提交

```powershell
cd <项目根目录>

# 确认 .gitignore 没有排除 data/deploy/
python deploy/prebuild_index.py

git add data/deploy/
git commit -m "预构建部署产物"
git push
```

### 2. 在 Streamlit Cloud 创建应用

1. 访问 https://share.streamlit.io/
2. 用 GitHub 账号登录并授权（需要对仓库有 **admin 权限**）
3. 点 **New app**
4. 填写：
   - Repository：`kioitov/rag-knowledge-base`
   - Branch：`main`
   - Main file path：**`deploy/app.py`**
5. 点 **Advanced settings**：
   - Python version：**3.11 或 3.12**（不能部署后改，选错要删了重来）
   - Secrets：粘贴下面这段

### 3. 配置 Secrets

```toml
DASHSCOPE_API_KEY = "sk-你的百炼key"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"
```

**不要**把 key 写进代码或提交到仓库。Streamlit Cloud 会把 Secrets 注入到 `st.secrets`，
代码里通过 `load_config()` 读取（已实现，兼容本地 `.env` 和云端 secrets）。

### 4. 部署

点 **Deploy**，等 2-5 分钟。首次构建会装依赖（chromadb 较大，可能慢一点）。

成功后地址形如：
```
https://<你的github用户名>-rag-knowledge-base.streamlit.app
```

---

## 验证清单

部署完成后逐条确认：

- [ ] 页面能打开，标题显示「📚 知识库问答」
- [ ] 侧边栏显示「语料：Kubernetes 中文文档 63 篇」
- [ ] 问「Pod 的生命周期有哪几个阶段」能答出 Pending/Running/Succeeded/Failed
- [ ] 来源显示**「第 N 节」而不是「第 N 页」** ← 关键，证明没有伪造页码
- [ ] 问「Kubernetes 的许可证编号是多少」应该回答「无法确定」
- [ ] 点「显示调试信息」能看到延迟和 token 数

---

## 常见问题

### 部署失败：`ModuleNotFoundError`

`requirements.txt` 缺失依赖。检查：
- 是不是漏了某个 import 对应的包
- `requirements.txt` 里不能有中文注释以外的 `##` 标题行（第一版就是这个问题，`pip` 会报错）

### 部署失败：`FileNotFoundError: chunks.jsonl`

`data/deploy/` 没有提交。检查 `.gitignore` 是否误排除了。

### 页面报「没有配置 API Key」

Secrets 没设或设错了键名。必须是 `DASHSCOPE_API_KEY`。

### 唤醒很慢

免费层 12 小时无访问会睡。面试前自己先打开一次预热。
或者用 UptimeRobot 之类的免费监控定时 ping（注意：这可能违反 Streamlit 的公平使用条款，
不建议长期这么干）。

### 显示「第 N 页」而不是「第 N 节」

说明 `meta["paged"]` 丢了。检查 `stage1/cache.py` 里序列化/反序列化是否都带上了 `meta`。
（这个 bug 真实发生过：缓存往返后 paged 标记丢失，我是在部署预构建时才发现的。）

---

## 本地验证部署版

不用等云端，本地就能测：

```powershell
# 1. 预构建
python deploy/prebuild_index.py

# 2. 启动（用 8502 端口，避免和开发版冲突）
python -m streamlit run deploy/app.py --server.port 8502

# 3. 浏览器打开 http://127.0.0.1:8502
```

或者用自动化测试：

```powershell
# 需要先起带调试端口的浏览器
msedge.exe --remote-debugging-port=9223 --user-data-dir=<临时目录> http://127.0.0.1:8502
python deploy/test_deploy.py
```

实测结果：**8/8 通过**。

---

## 一个教训：CDP 测试别用回车提交

写自动化测试时，我用 `Input.dispatchKeyEvent` 发 Enter 键想触发提交，
结果输入框被清空但消息没发出去。更糟的是**我的等待条件太宽**
（匹配「页面包含某段文字」），被侧边栏文字误匹配，误判为成功。

正确做法：
1. **点发送按钮**（`[data-testid="stChatInputSubmitButton"]`），不要靠回车
2. **等待条件要具体**——等 `stChatMessage` 节点数 >= 2，不要等文本出现

这个坑在阶段 4 也踩过一次。两次的共同点是：**假成功比失败更危险**，
因为不看截图就会以为跑通了。
