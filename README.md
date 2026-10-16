# 面向企业工程资料的检索增强问答智能体（HydroRAG）

HydroRAG 是一个面向企业工程资料的检索增强问答智能体。它将 PDF、图片、DOCX、XLSX、CSV 解析，来源位置级检索、任务路由、分层记忆、证据校验和回答生成组织成一条完整的 RAG 工作流，并通过 LangServe 暴露本地 API。

它不只是一个 ChatPDF：系统会先识别用户意图，再选择资料问答、总结、多文档对比、表格提取或安全计算路径，并在回答中保留资料标题与原文位置。

> 当前项目是个人项目和本地原型，适合展示 RAG、LangGraph、LangChain、Chroma、PDF 解析与 LangServe 的工程闭环，不是面向生产环境的多用户服务。

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 技术架构](#2-技术架构)
- [3. 前期准备](#3-前期准备)
- [4. 部署与启动](#4-部署与启动)
- [5. 对话与功能使用](#5-对话与功能使用)
- [6. 搭建过程中遇到的问题](#6-搭建过程中遇到的问题)
- [7. 常见问题及解决方案](#7-常见问题及解决方案)
- [8. 测试与项目边界](#8-测试与项目边界)
- [9. 后续改进方向](#9-后续改进方向)
- [10. GitHub 发布清单](#10-github-发布清单)
- [11. 许可证](#11-许可证)

---

## 1. 项目简介

### 1.1 已实现能力

| 能力 | 当前实现 | 输出 |
| --- | --- | --- |
| 多格式资料入库 | PDF、图片 OCR、DOCX、XLSX、CSV 统一分块 | 保留页码、段落、表格、工作表或行号位置 |
| 资料问答 | Chroma 本地向量检索 + 来源位置元数据 | 基于资料证据回答并展示来源位置 |
| 资料总结 | LangGraph 检索节点 + DeepSeek 生成节点 | 主题、方法、数据和结论 |
| 多文档对比 | 使用多个 `document_id` 限定检索范围 | Markdown 对比内容 |
| 表格提取 | PyMuPDF `find_tables` + 表格工具 | 可供模型整理的结构化表格文本 |
| 工具调用 | `search_documents`、`extract_tables`、`calculate` | 检索、表格和计算结果 |
| 任务路由 | LangGraph 条件边，DeepSeek 可辅助分类 | `meta / qa / summary / compare / table / calculate` |
| Web 服务 | FastAPI + LangServe | `/api/chat/invoke`、`/api/chat/batch`、`/api/chat/stream` |
| 本地前端 | 原生 HTML/CSS/JavaScript | 上传、选文、对话和引用展示 |
| 单机运维加固 | SQLite 任务账本、后台线程、进程内限流、滚动 JSON 审计日志、指标快照、ZIP 备份 | 上传任务可追踪、429 限流、`/api/metrics`、备份恢复命令 |
| 短期记忆 | SQLite 会话消息 + 滚动摘要 | 多轮指代解析与会话隔离 |
| 长期记忆 | 用户显式“请记住/忘记”指令 | 持久化回答偏好，可查看和删除 |
| 幻觉治理 | 距离阈值、证据不足拒答、引用编号校验 | 阻止无有效引用的模型回答 |

### 1.2 技术选型

- **LangGraph**：使用状态图表达“路由 → 检索或工具 → 生成”的多分支流程。
- **LangChain**：提供 Runnable、StructuredTool 和 OpenAI 兼容模型接口。
- **Chroma**：使用 PersistentClient 在本地持久化索引，无需额外启动数据库。
- **PyMuPDF**：提取 PDF 文本、页码和基础表格。
- **DeepSeek**：通过 OpenAI 兼容接口完成回答生成和可选的任务分类。
- **FastAPI + LangServe**：提供本地 HTTP 和 SSE 接口，供浏览器及其他客户端调用。

默认向量方案是本地 ONNX Runtime 加载的 `BAAI/bge-small-zh-v1.5`，并结合 BM25 和标题/文件名/工作表元数据融合排序；前置候选还可使用本地 `BAAI/bge-reranker-base` 重排。Hash Embedding 仅保留为测试和无模型环境的显式兼容后端。

### 1.3 项目结构

```text
HydroRAG/
├─ hydrorag/
│  ├─ agent.py          # LangGraph 状态图、任务路由和节点
│  ├─ documents.py      # PDF/图片/DOCX/XLSX/CSV 统一解析
│  ├─ memory.py         # SQLite 长短期记忆
│  ├─ cli.py            # ingest / ask / list / health / demo / reset
│  ├─ config.py         # .env 配置读取
│  ├─ llm.py            # DeepSeek OpenAI 兼容客户端
│  ├─ models.py         # 文档 chunk 和检索结果模型
│  ├─ pdfs.py           # PDF 文本、页码和表格抽取
│  ├─ tools.py          # RAG、表格和计算工具
│  ├─ vectorstore.py    # Chroma、语义向量、BM25 和本地重排
│  ├─ server.py         # FastAPI、LangServe 和静态文件服务
│  └─ static/index.html # 本地资料对话界面
├─ data/
│  ├─ documents/        # CLI 导入时可放置工程资料
│  └─ chroma/           # Chroma 持久化索引
├─ tests/test_smoke.py
├─ .env.example         # 配置模板，不存放真实密钥
├─ requirements.txt
└─ README.md
```

---

## 2. 技术架构

### 2.1 数据处理和问答流程

```text
上传 PDF / 图片 / DOCX / XLSX / CSV
  ↓
按文件类型提取文本 / 表格 / OCR，并保留来源位置
  ↓
文本切分 + 本地 Embedding
  ↓
Chroma PersistentClient

用户问题
  ↓
LangServe / FastAPI
  ↓
LangGraph route 节点
  ├─ meta                   → 直接回答 HydroRAG 身份和使用方式
  ├─ qa / summary / compare → 检索 → DeepSeek 生成
  ├─ table                  → 表格工具 → 可选 DeepSeek 整理
  └─ calculate              → 安全数学表达式计算
  ↓
引用编号校验 / 证据不足拒答
  ↓
回答 + 资料标题 + 原文位置
```

### 2.2 请求状态

每个请求可包含：

| 字段 | 含义 |
| --- | --- |
| `query` | 用户问题，必填 |
| `document_ids` | 限定检索范围；为空时检索全部资料 |
| `top_k` | 返回的证据片段数量，范围为 1 至 20 |
| `thread_id` | 进程内会话标识 |
| `user_id` | 用户标识，用于隔离会话和长期记忆 |

身份、能力和使用方式问题会进入 `meta` 路由，不会检索资料或附带资料引用。例如“你是谁”“你能做什么”。

当前使用 SQLite 保存最近会话、滚动摘要和用户显式确认的长期偏好。Web 端为每个浏览器生成独立 `user_id`，每个页面会话生成独立 `thread_id`；这仍是本地原型标识，不等同于生产级身份认证。

### 2.3 数据与隐私

原始资料、Chroma 索引和 SQLite 记忆默认保存在本机。配置 DeepSeek API Key 后，检索到的证据片段、必要的对话上下文和用户偏好会随生成请求发送到 DeepSeek API。请勿在未完成数据分级、脱敏和公司审批的情况下处理敏感内部资料。

---

## 3. 前期准备

以下命令以 Windows PowerShell 和 `D:\Codes\HydroRAG` 为例。换到其他电脑时，请替换为项目的实际路径。

### 3.1 软件要求

- Windows 10/11 或其他可运行 Python 的系统；
- Python 3.13；当前锁定环境使用并验证于 Python 3.13.13；
- 图片 OCR 需要额外安装 Tesseract 5.5.3，以及 `chi_sim` 与 `eng` 语言包；
- 可访问 PyPI 的网络；
- 可选：DeepSeek 开放平台账号及 API Key。

检查 Python：

```powershell
py --version
```

如果没有 `py` 命令：

```powershell
python --version
```

### 3.2 获取项目

项目发布到 GitHub 后，可以使用：

```powershell
git clone <你的仓库地址>
cd HydroRAG
```

也可以直接下载源码压缩包并解压，然后在 PowerShell 中进入项目目录。

### 3.3 创建虚拟环境并安装 requirements.txt

`requirements.txt` 记录了当前虚拟环境中全部直接与间接 Python 第三方依赖的精确版本。它不包含 Tesseract，因为 Tesseract 是系统程序而不是 Python 包。

```powershell
cd D:\Codes\HydroRAG
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止执行 `Activate.ps1`，可以只对当前窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以不激活虚拟环境，直接使用项目解释器：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

安装后检查 Python 依赖是否完整且相互兼容：

```powershell
& ".\.venv\Scripts\python.exe" -m pip check
```

只有在有意升级依赖并完成全量测试后，才重新生成版本清单：

```powershell
& ".\.venv\Scripts\python.exe" -m pip freeze | Set-Content -Encoding utf8 requirements.txt
```

### 3.4 安装 Tesseract 和中英文语言包（Windows）

通过 winget 安装已验证的 Tesseract 5.5.3：

```powershell
winget install --id tesseract-ocr.tesseract --exact --version 5.5.3 `
  --accept-package-agreements --accept-source-agreements
```

安装程序自带英文 `eng`。下面将 `eng` 与官方 `tessdata_fast` 4.1.0 中的简体中文 `chi_sim` 安装到当前用户目录，因此不需要管理员权限修改 `Program Files`：

```powershell
$tesseractBin = "C:\Program Files\Tesseract-OCR"
$userTessData = Join-Path $env:LOCALAPPDATA "Tesseract-OCR\tessdata"
New-Item -ItemType Directory -Path $userTessData -Force | Out-Null
Copy-Item -LiteralPath "$tesseractBin\tessdata\eng.traineddata" `
  -Destination "$userTessData\eng.traineddata" -Force
Invoke-WebRequest `
  -Uri "https://github.com/tesseract-ocr/tessdata_fast/raw/refs/tags/4.1.0/chi_sim.traineddata" `
  -OutFile "$userTessData\chi_sim.traineddata"
$chiSimHash = (Get-FileHash "$userTessData\chi_sim.traineddata" -Algorithm SHA256).Hash
if ($chiSimHash -ne "A5FCB6F0DB1E1D6D8522F39DB4E848F05984669172E584E8D76B6B3141E1F730") {
  throw "chi_sim.traineddata 校验失败"
}

[Environment]::SetEnvironmentVariable("TESSDATA_PREFIX", $userTessData, "User")
$userPathValue = [Environment]::GetEnvironmentVariable("Path", "User")
$userPathParts = @($userPathValue -split ";" | Where-Object { $_ })
if ($userPathParts -notcontains $tesseractBin) {
  [Environment]::SetEnvironmentVariable(
    "Path", (($userPathParts + $tesseractBin) -join ";"), "User"
  )
}
```

关闭并重新打开 PowerShell（正在运行的 Codex/IDE 也需要重启），然后验证：

```powershell
tesseract --version
tesseract --list-langs
& ".\.venv\Scripts\python.exe" -m hydrorag health
```

语言列表应同时包含 `chi_sim` 和 `eng`，健康检查中的 `ocr.available` 应为 `true`。

### 3.5 申请 DeepSeek API Key

1. 打开 [DeepSeek 开放平台](https://platform.deepseek.com/)并注册或登录。
2. 进入 API Keys 页面，创建新的 API Key。
3. 根据平台当前要求完成余额或计费设置。
4. 创建后立即复制 Key。不要把 Key 提交到 GitHub、截图或简历中。

### 3.6 配置环境变量

复制配置模板：

```powershell
Copy-Item .env.example .env
notepad .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

HYDRORAG_TOP_K=6
HYDRORAG_CHUNK_CHARS=1200
HYDRORAG_CHUNK_OVERLAP=180
HYDRORAG_EMBEDDING_BACKEND=fastembed
HYDRORAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
HYDRORAG_EMBEDDING_DIM=512
HYDRORAG_RERANKER_ENABLED=true
HYDRORAG_RERANKER_MODEL=BAAI/bge-reranker-base
HYDRORAG_RETRIEVAL_CANDIDATES=40
HYDRORAG_RERANKER_CANDIDATES=24
HYDRORAG_VECTOR_WEIGHT=0.50
HYDRORAG_BM25_WEIGHT=0.35
HYDRORAG_METADATA_WEIGHT=0.15
HYDRORAG_TEMPERATURE=0.1
HYDRORAG_DISTANCE_THRESHOLD=1.15
HYDRORAG_SHORT_TERM_TURNS=8
HYDRORAG_OCR_LANGUAGE=chi_sim+eng
HYDRORAG_TESSERACT_CMD=
HYDRORAG_TESSDATA_DIR=
HYDRORAG_MAX_UPLOAD_MB=25
HYDRORAG_RATE_LIMIT_PER_MINUTE=60
```

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API 密钥；为空时进入离线检索模式 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 生成和可选路由模型 |
| `HYDRORAG_API_TOKEN` | 空 | 设置后保护 API；请求需携带 `Authorization: Bearer <token>` |
| `HYDRORAG_REQUIRE_API_TOKEN` | `false` | 设为 `true` 后，若未配置 API Token，服务启动失败；用于部署时 fail-closed |
| `HYDRORAG_TOP_K` | `6` | 默认证据片段数量 |
| `HYDRORAG_CHUNK_CHARS` | `1200` | 文本分块长度 |
| `HYDRORAG_CHUNK_OVERLAP` | `180` | 相邻分块重叠长度 |
| `HYDRORAG_EMBEDDING_BACKEND` | `fastembed` | 向量后端；正式检索使用本地 FastEmbed，测试可设为 `hash` |
| `HYDRORAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文语义 Embedding 模型 |
| `HYDRORAG_EMBEDDING_DIM` | `512` | Hash 后端维度；FastEmbed 使用模型自身维度 |
| `HYDRORAG_RERANKER_ENABLED` | `true` | 是否启用候选片段重排序 |
| `HYDRORAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | 本地 Cross-Encoder 重排模型 |
| `HYDRORAG_RETRIEVAL_CANDIDATES` | `40` | 向量/BM25 融合候选数 |
| `HYDRORAG_RERANKER_CANDIDATES` | `24` | 送入 Reranker 的候选数 |
| `HYDRORAG_VECTOR_WEIGHT` | `0.50` | 向量检索融合权重 |
| `HYDRORAG_BM25_WEIGHT` | `0.35` | BM25 融合权重 |
| `HYDRORAG_METADATA_WEIGHT` | `0.15` | 标题、文件名和结构元数据权重 |
| `HYDRORAG_TEMPERATURE` | `0.1` | 模型生成温度 |
| `HYDRORAG_DISTANCE_THRESHOLD` | `1.15` | Chroma 余弦距离上限，超出后拒绝作为证据 |
| `HYDRORAG_SHORT_TERM_TURNS` | `8` | 保留的完整对话轮数，更早内容滚动摘要 |
| `HYDRORAG_OCR_LANGUAGE` | `chi_sim+eng` | Tesseract OCR 语言 |
| `HYDRORAG_TESSERACT_CMD` | 空 | Tesseract 可执行文件路径；系统 `PATH` 已配置时可留空 |
| `HYDRORAG_TESSDATA_DIR` | 空 | `chi_sim`、`eng` 等 Tesseract 语言包所在目录 |
| `HYDRORAG_MAX_UPLOAD_MB` | `25` | 单文件上传大小上限 |
| `HYDRORAG_RATE_LIMIT_PER_MINUTE` | `60` | 单进程每个 Token/IP 每分钟允许的 API 请求数 |

真实 Key 必须写入 `.env`，不能写入 `.env.example`。`.env` 已被 `.gitignore` 忽略，`.env.example` 只用于展示配置格式。

部署环境建议通过密钥管理服务注入 Token，并设置 `HYDRORAG_REQUIRE_API_TOKEN=true`；这样缺少 Token 时服务会拒绝启动。此设置只提供共享入口保护，不会建立终端用户身份。

### 3.7 准备资料

可以通过两种方式导入：

- 将 PDF、图片、DOCX、XLSX 或 CSV 放到 `data/documents/`，随后执行 CLI 导入；
- 启动 Web 服务后，从页面左侧上传受支持的文件。

扫描版 PDF 会在没有文本层时逐页 OCR；复杂 PDF 版面、低清晰度扫描件和复杂合并单元格仍需人工核验。首次使用本地 Embedding/Reranker 时需要下载模型文件。

---

## 4. 部署与启动

### 4.1 检查环境

```powershell
python -m hydrorag health
```

未激活虚拟环境时：

```powershell
& ".\.venv\Scripts\python.exe" -m hydrorag health
```

健康检查会显示项目目录、模型名、API Key 是否已配置以及核心模块能否导入，但不会打印真实 Key。

### 4.2 命令行导入资料

导入 `data/documents/` 中支持的全部资料文件：

```powershell
python -m hydrorag ingest data/documents
```

导入单个文件：

```powershell
python -m hydrorag ingest "D:\documents\report.pdf"
```

查看已入库资料及其 `document_id`：

```powershell
python -m hydrorag list
```

重复导入同一个文件时，程序会先删除该资料的旧 chunk，再写入新索引。

### 4.3 启动 LangServe 和 Web 前端

推荐使用虚拟环境中的解释器：

```powershell
cd D:\Codes\HydroRAG
& ".\.venv\Scripts\python.exe" -m uvicorn hydrorag.server:app --host 127.0.0.1 --port 8000 --reload
```

启动成功后打开：

```text
http://127.0.0.1:8000/
```

本地接口：

| 地址 | 作用 |
| --- | --- |
| `GET /` | 资料对话前端 |
| `GET /api/health` | 服务、模型和索引状态 |
| `GET /api/documents` | 已入库资料列表 |
| `POST /api/documents` | 上传并创建后台索引任务，返回 `202` 和 `job_id` |
| `GET /api/jobs/{job_id}` | 查询解析/索引任务状态 |
| `GET /api/metrics` | 查看当前进程指标、任务计数和延迟摘要 |
| `GET /api/memories` | 查看当前用户长期记忆 |
| `DELETE /api/memories/{id}` | 删除指定长期记忆 |
| `DELETE /api/conversations/{thread_id}` | 清除指定短期会话 |
| `POST /api/chat/invoke` | LangServe 单次调用 |
| `POST /api/chat/batch` | LangServe 批量调用 |
| `POST /api/chat/stream` | LangServe SSE 流式接口 |
| `GET /docs` | FastAPI 接口文档 |

完整地址示例：

```text
前端：      http://127.0.0.1:8000/
健康检查：  http://127.0.0.1:8000/api/health
API 文档：  http://127.0.0.1:8000/docs
```

服务默认只监听 `127.0.0.1`，不会对局域网开放。关闭服务时，在运行 uvicorn 的终端按 `Ctrl+C`。

### 4.4 验证 LangServe

PowerShell 示例：

```powershell
$body = @{
  input = @{
    query = "总结这份工程报告的主要结论"
    document_ids = @()
    thread_id = "powershell"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/chat/invoke `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

`/api/chat/stream` 返回 Server-Sent Events（SSE），可用于实现逐步显示回答的前端。

---

## 5. 对话与功能使用

### 5.1 Web 操作流程

1. 打开 `http://127.0.0.1:8000/`。
2. 在左侧上传一个或多个 PDF、图片、DOCX、XLSX 或 CSV 文件。
3. 上传接口立即返回任务；后台执行解析、OCR、Embedding 和入库，页面轮询任务状态直到显示“已索引”。
4. 在“检索范围”中勾选资料；不勾选时检索全部资料。
5. 在底部输入问题，按 Enter 发送，Shift+Enter 换行。
6. 查看回答以及下方的资料标题和原文位置引用。

“刷新资料库”只刷新列表；“清空对话”会清除当前页面消息及对应的服务端短期会话，不会删除原始资料、长期记忆或 Chroma 索引。

### 5.2 推荐问题

```text
这份技术报告要解决什么工程问题？核心方案是什么？
```

```text
请按工程问题、技术方案、关键参数、验证结果和主要结论梳理这份资料。
```

```text
总结这份资料，并标出主要结论对应的原文位置。
```

```text
比较选中资料的适用场景、技术方案、关键参数、验证结果和局限性。
```

```text
提取资料中的关键参数表格，保留表头、数值和原文位置。
```

```text
计算 (12.5 - 3.2) / 2
```

### 5.3 命令行问答

```powershell
python -m hydrorag ask "这份报告的核心方案是什么？"
python -m hydrorag ask "总结资料中的技术方案、关键参数和验证结果"
python -m hydrorag ask "提取资料中的关键参数表格"
python -m hydrorag ask "计算 (12.5 - 3.2) / 2"
```

多文档对比时，重复传入 `--document-id`：

```powershell
python -m hydrorag ask "比较这两份资料的技术方案和验证结果" `
  --document-id document-a-id `
  --document-id document-b-id
```

未配置 API Key 时仍可执行本地解析和检索。程序会返回召回的证据片段及页码，而不会伪造模型回答。

### 5.4 单机备份与恢复

备份包含原始资料、Chroma 索引、SQLite 记忆/任务账本和审计日志，并写入 SHA-256 清单：

```powershell
python -m hydrorag backup
python -m hydrorag backup D:\backup\hydrorag-before-upgrade.zip
```

恢复是覆盖操作，必须显式确认。程序会校验 ZIP 路径、文件校验和及 SQLite 完整性，并在覆盖前自动生成安全备份：

```powershell
python -m hydrorag restore D:\backup\hydrorag-before-upgrade.zip --confirm
```

恢复期间应停止服务，完成后重新启动并检查 `/api/health`。这是单机本地备份，不等于异地容灾；仍需定期复制到受控的独立存储。

---

## 6. 搭建过程中遇到的问题

这一节记录项目从 CLI 原型扩展到 LangServe + Web 前端时实际遇到的问题。

### 6.1 API Key 应写入 `.env`，不是 `.env.example`

- `.env.example` 是可以提交到 GitHub 的公开模板，应保留空值；
- `.env` 才是项目启动时读取的本地配置；
- `.env` 已加入 `.gitignore`；
- Key 写在 `.env.example` 时，程序不会自动读取，健康检查会显示未配置。

### 6.2 PowerShell 无法激活虚拟环境

Windows 可能因为执行策略阻止 `Activate.ps1`。可以对当前 PowerShell 窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以完全不激活，直接调用 `.venv\Scripts\python.exe`。

### 6.3 常用命令分别做什么

```powershell
python -m venv .venv
```

在项目目录创建隔离的 Python 环境，避免污染系统 Python。

```powershell
python -m pip install -r requirements.txt
```

按依赖清单安装 LangGraph、LangChain、Chroma、PyMuPDF、FastAPI 和 LangServe。

```powershell
python -m hydrorag ingest data/documents
```

解析 PDF、切分文本、生成本地向量并写入 Chroma。

```powershell
python -m hydrorag ask "问题"
```

通过命令行调用 LangGraph Agent。

```powershell
python -m uvicorn hydrorag.server:app --host 127.0.0.1 --port 8000 --reload
```

启动本地 FastAPI 服务。`hydrorag.server:app` 表示加载 `hydrorag/server.py` 中名为 `app` 的对象，`--reload` 表示代码变动后自动重启。

### 6.4 LangServe Playground 地址返回 404

当前项目只显式启用了 `/api/chat/invoke`、`/api/chat/batch` 和 `/api/chat/stream`，并没有启用 `/api/chat/playground/`。正确入口是：

- 自定义前端：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`
- 单次调用：`POST http://127.0.0.1:8000/api/chat/invoke`

### 6.5 为什么推荐使用虚拟环境中的 Python

如果本机存在多个 Python，直接运行 `python -m uvicorn` 可能找不到已安装的 LangServe 或 Chroma。遇到 `ModuleNotFoundError` 时使用：

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn hydrorag.server:app --host 127.0.0.1 --port 8000 --reload
```

### 6.6 8000 端口已被占用

出现 `address already in use` 时，说明已有进程占用端口。可以使用另一个端口：

```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn hydrorag.server:app --host 127.0.0.1 --port 8001 --reload
```

然后访问 `http://127.0.0.1:8001/`。

### 6.7 API Key 已配置但无法生成回答

按以下顺序检查：

1. Key 位于项目根目录的 `.env`，不是 `.env.example`；
2. 执行 `python -m hydrorag health`，确认 `deepseek_key_configured` 为 `true`；
3. 核对 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`；
4. 检查 Key 是否失效、账户余额及网络连接；
5. 查看 uvicorn 日志：`401` 通常表示鉴权失败，`429` 通常表示频率或额度限制。

### 6.8 页面提示“索引中没有找到相关证据”

常见原因：

- 尚未导入资料；
- Web 上传还未完成；
- 选择了错误的资料范围；
- 问题与资料内容关系较弱；
- PDF 是低清晰度扫描件，OCR 没有识别出有效文字；
- 修改了分块或 Embedding 配置但没有重新建库。

先执行下面的命令确认资料和 chunk 是否存在：

```powershell
python -m hydrorag list
```

### 6.9 表格提取为空或不完整

当前表格能力依赖 PyMuPDF `find_tables`，更适合带文本层且边界清晰的表格。扫描版 PDF、图片表格和复杂合并单元格可能无法完整识别，需要人工核验。

### 6.10 修改向量参数后检索异常

修改 Embedding 模型、`HYDRORAG_EMBEDDING_DIM`、分块长度或重叠长度后，应重新建立索引：

```powershell
python -m hydrorag reset
python -m hydrorag ingest data/documents
```

`reset` 会删除 HydroRAG 的 Chroma 集合，但不会删除 `data/documents/` 中的原始资料。

---

## 7. 常见问题及解决方案

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `ModuleNotFoundError` | 依赖安装到了另一个 Python | 使用 `.venv\Scripts\python.exe` 执行命令 |
| 上传接口返回 400 | 文件格式不受支持 | 使用 PDF、图片、DOCX、XLSX 或 CSV 文件 |
| 上传接口返回 422 | 文件损坏、无法解析或 OCR 不可用 | 检查文件，并用 `health` 核对 Tesseract 与语言包 |
| `deepseek_configured: false` | 没读取到 Key | 检查 `.env` 的位置和变量名 |
| `401 Unauthorized` | Key 无效或 Base URL 错误 | 重新生成 Key并核对配置 |
| `429 Too Many Requests` | 额度或频率限制 | 检查余额、等待或降低频率 |
| 页面无法打开 | uvicorn 未启动或端口错误 | 查看终端日志并核对 URL |
| 资料列表为空 | 尚未索引 | 执行 `ingest` 或通过页面上传 |
| 引用页码存在偏差 | PDF 版面或页眉页脚影响分块 | 回到原 PDF 核验关键结论 |
| 修改代码后页面未变化 | 缓存或服务未重载 | 刷新浏览器并使用 `--reload` |

提交到公开仓库前，不要包含 API Key、企业内部资料、Chroma 索引或其他私有数据。`.gitignore` 已忽略 `.env`、`data/chroma/` 和 `data/documents/`，但仍应检查：

```powershell
git status --short
```

如果真实 Key 曾经被提交到 Git 历史，仅删除文件是不够的，应立即在 DeepSeek 平台撤销该 Key 并生成新 Key。

---

## 8. 测试与项目边界

### 8.1 运行测试

```powershell
python -m pytest -q
```

当前自动化测试覆盖：

- 中文任务路由；
- 文本切分是否保留首尾内容；
- 基础数学表达式的安全计算。
- DOCX、XLSX、CSV 与图片 OCR 的分块及来源位置；
- SQLite 长短期记忆的用户/会话隔离与滚动压缩；
- 回答引用编号校验及未引用来源过滤；
- 多格式文档从解析、Chroma 入库到检索的元数据保留。

真实 PDF、DeepSeek API 和浏览器交互依赖外部文件、网络与模型输出，还需要进行人工验收。

### 8.2 答案级评测

答案级评测使用 `work/answer_eval_cases.jsonl` 中的人工标注问题，实际调用当前配置的模型，结果写入 `work/answer_eval_results.json`：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
& ".\.venv\Scripts\python.exe" work\evaluate_answer_quality.py
```

运行前应确认 `.env` 中已配置 `DEEPSEEK_API_KEY`；每条问题会产生模型调用和相应 API 成本。标注字段包括：

| 字段 | 作用 |
| --- | --- |
| `question`、`answerable` | 问题文本及资料是否足以回答 |
| `relevant_sources` | Top-1 文档和检索命中率的期望来源 |
| `required_points` | 最终答案必须覆盖的事实点及匹配模式 |
| `numeric_facts` | 数值、单位和相对误差容忍度 |
| `required_citation_documents` | 引用准确率的期望来源（可选） |

当前 20 条开发集问题（17 条可回答、3 条证据不足）在 2026-10-12 的最近一次完整运行结果如下。之后增加了跨文档漏引的程序化修复与拒答校验，但因模型 API 返回余额不足（HTTP 402），变更后的完整复测未完成；下表是变更前基线，不应视为当前版本的最终验收结果：

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| Top-1 文档准确率 | 100.00% | 首个检索文档命中标注来源的比例 |
| 最终答案事实准确率 | 100.00% | `required_points` 的平均覆盖率 |
| 引用 Precision | 97.06% | 回答引用的文档中，属于期望来源的比例 |
| 引用 Recall | 97.06% | 期望来源被回答引用的比例 |
| 数值准确率 | 100.00% | 期望数值被正确回答的比例 |
| 单位准确率 | 100.00% | 期望单位被正确回答的比例 |
| 数值+单位联合准确率 | 100.00% | 数值和单位同时正确的比例 |
| 证据不足拒答召回率 | 100.00% | 不可回答问题被正确拒答的比例 |
| 可回答题误拒答率 | 0.00% | 可回答问题被错误拒答的比例 |

该开发集仍只有 20 条，不能代表生产流量。当前两项引用偏差分别是多引无关来源和跨文档比较漏引来源；检索与回答指标均需用独立、扩充后的人工标注集复核。逐题模型输出和证据保存在被 `.gitignore` 排除的 `work/answer_eval_results.json`，不要直接对外发布。

### 8.3 当前限制

- 默认使用本地 BGE 中文 Embedding，并融合 BM25 与结构元数据；
- 候选片段默认使用本地 BGE Cross-Encoder 重排，模型文件不可用时回退到融合排序；
- 章节识别采用启发式规则，不是完整的 PDF 版面分析；
- 扫描版 PDF 使用 Tesseract OCR，但低清晰度页面和复杂表格仍可能需要人工校验；
- 上传解析、OCR、Embedding 和入库已改为单进程后台线程，任务状态持久化在 `data/jobs.sqlite3`；服务重启会恢复未完成任务，但不支持多实例共享队列；
- API Token 可选；`HYDRORAG_REQUIRE_API_TOKEN=true` 可阻止无 Token 的部署启动，但当前仍是共享 Token，不提供用户身份、文档级 RBAC、SSO 或部门/项目数据隔离；`/api/health` 保持公开；
- API 已提供进程内滑动窗口限流；指标和审计日志只在当前实例内统计，多实例部署仍需要网关级限流和集中式观测；
- 审计日志使用滚动 JSON 文件，避免记录原文、完整回答和密钥；当前身份仍来自共享 Token 或客户端标识，不是企业 SSO 身份；
- CLI 已提供带校验和的本地备份/恢复；备份仍需人工或脚本复制到独立位置，尚未实现自动异地备份；
- Chroma 与 SQLite 适用于单实例原型，不适合多实例共享部署；
- 回答属于基于检索证据的模型生成结果，关键结论仍应回到原始资料核验。

---

## 9. 后续改进方向

当前 64 条人工标注查询的文档级检索评测结果（41 份资料、9,194 个分块）如下；这是检索指标，不代表最终答案正确率，也不与上一节的答案级指标混用：

| 指标 | 结果 |
| --- | ---: |
| MRR | 74.97% |
| Recall@5 | 94.01% |
| Recall@10 | 94.79% |
| NDCG@5 | 79.51% |
| NDCG@10 | 79.94% |
| Hit@10 | 95.31% |

原 Hash Embedding 基线为 `MRR 43.86%`、`Recall@10 65.62%`、`NDCG@10 48.96%`。升级后的评测脚本位于 `work/evaluate_hydrorag_metrics.py`，结果保存于 `work/metrics_after2.json`。

以下仍是生产化工作项，不应描述成已具备的能力；按依赖关系建议依次推进：

1. 明确企业身份提供方和授权策略，再接入 OIDC/SSO、用户/部门身份映射及文档级 RBAC；共享 API Token 只能作为服务入口保护，不能替代用户授权。
2. 按数据规模和并发要求迁移单机 SQLite/Chroma，确定 PostgreSQL/pgvector 或专用检索服务，并设计租户隔离、迁移、自动备份、恢复与索引回滚。
3. 将当前单进程后台线程替换为共享任务队列，补充跨实例幂等、失败重试、死信和索引版本切换。
4. 将进程内限流、日志和指标接入网关、集中式日志、Prometheus/Grafana 和告警系统，审计身份必须来自已验证的认证上下文，不能信任请求体中的 `user_id`。
5. 将答案级评测集扩充至至少 100 条经业务人员复核的案例，并划分调参集与独立验收集；覆盖权限越权、冲突证据、数值单位、拒答和文档更新。
6. 对扫描表格等高风险资料单独评估 OCR/表格识别，并加入人工复核流程和可追溯的数据治理记录。

项目介绍和简历中应严格区分“已经完成并可演示的能力”与“计划能力”，尤其不要把低清晰度 OCR 的完整识别和生产级部署写成无条件保证。

---

## 10. GitHub 发布清单

在首次公开仓库前，建议按以下顺序检查：

```powershell
cd D:\Codes\HydroRAG

git init
git branch -M main
git add .
git status --short
git diff --cached --name-only
```

暂存文件列表中不应出现：

```text
.env
.venv/
data/chroma/
data/documents/*
__pycache__/
.pytest_cache/
```

检查密钥和敏感配置：

```powershell
rg -n --hidden `
  -g '!.git/**' `
  -g '!.venv/**' `
  -g '!data/chroma/**' `
  -g '!*.pdf' `
  -g '!.env' `
  '(sk-[A-Za-z0-9_-]{20,}|DEEPSEEK_API_KEY\s*=\s*[^<\s])' .
```

如果确认暂存内容正确，再提交：

```powershell
git commit -m "Initial commit: HydroRAG engineering document QA agent"
git remote add origin https://github.com/<你的用户名>/<你的仓库名>.git
git push -u origin main
```

GitHub 仓库建议开启 Secret Scanning 和 Push Protection。真实 API Key 只能放在本地 `.env` 或 GitHub Actions Secrets 中，不能写入源码、README、截图或前端代码。如果密钥曾经进入 Git 历史，应先在 DeepSeek 平台撤销并重新生成。

发布前还应确认：

- README 中的仓库地址占位符已替换；
- `.env.example` 中没有真实密钥；
- 项目可以按 README 从零安装和启动；
- `python -m pytest -q` 通过；
- 工程资料和 Chroma 索引没有进入仓库；
- 第三方资料仅在拥有再分发许可时才随仓库发布。

---

## 11. 许可证

本项目源代码采用 [MIT License](LICENSE)。MIT License 仅适用于本仓库中由项目作者贡献的源代码和文档，不自动授予第三方资料、数据集、模型、API 或其他依赖的再分发权。使用企业内部或受版权保护的资料时，请遵守所属组织的数据安全制度和版权持有者的许可要求。
