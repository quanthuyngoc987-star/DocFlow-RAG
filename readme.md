<div align="center">


# DocFlow-RAG 中文文档问答系统 📚

基于 RAG + FAISS + BM25 + Reranker + DeepSeek 的中文文档智能问答项目

---

## 使用必看

请务必先完成相关环境配置，并正确填写项目根目录下的 `.env` 文件。

其中：

- `DEEPSEEK_API_KEY` 为核心配置项，用于调用大模型生成答案
- `SERPAPI_KEY` 为可选配置项，用于启用联网搜索增强能力
- 若不启用联网搜索，可不配置 `SERPAPI_KEY`

建议在首次运行前，先准备好测试文档，用于验证上传、切分、检索和问答链路是否正常。

---

## 📖 项目简介

DocFlow-RAG 是一个面向中文场景的文档问答系统，聚焦于构建一个**完整、清晰、可扩展**的 RAG（Retrieval-Augmented Generation，检索增强生成）工程流程。

项目支持从文档解析、文本切分、向量化建库、混合检索、重排序，到大模型回答生成的完整链路，既适合用作课程设计、科研展示和简历项目，也适合作为后续扩展知识库系统的基础工程。

系统具备以下核心能力：

- **多格式文档解析**：支持 PDF、TXT、MD、DOCX、XLS、XLSX、PPTX 等常见文档格式
- **RAG 增强检索**：围绕用户问题优先检索本地知识库内容，降低大模型幻觉
- **混合召回机制**：结合 FAISS 向量检索与 BM25 关键词检索，提高召回覆盖率
- **二阶段重排序**：对候选检索结果进一步精排，提升最终输入上下文质量
- **递归检索能力**：支持围绕复杂问题进行扩展查询与补充检索
- **联网搜索增强**：可选接入外部搜索能力，适配时效性更强的问题
- **双入口运行模式**：同时支持 Gradio Web 交互界面与 FastAPI 接口调用
- **中文问答友好**：适合中文论文、技术文档、课程资料、项目手册等知识库场景

---

## ✨ 核心特性

| 特性       | 说明                                         |
| ---------- | -------------------------------------------- |
| LLM        | DeepSeek 大模型（默认 `deepseek-chat`）    |
| Embedding  | 文本向量化模块，用于构建语义检索索引         |
| 向量数据库 | FAISS，本地向量检索                          |
| 稀疏检索   | BM25，用于关键词召回                         |
| 重排序     | Cross-Encoder / Reranker 二阶段精排          |
| 前端       | Gradio Web 交互界面                          |
| 后端       | FastAPI 服务接口                             |
| 联网增强   | SerpAPI（可选）                              |
| 检索策略   | 向量检索 + 关键词检索 + 重排序 + 递归检索    |
| 适用场景   | 中文文档问答、课程设计、项目展示、知识库原型 |

---

## 🏗 系统架构

```text
┌──────────────────────────────────────────────────────┐
│                Gradio 前端 (rag_demo.py)              │
│     - 文档上传  - 问答交互  - 检索结果展示             │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                  RAG 主流程 (core/)                   │
│  ┌────────────────────────────────────────────────┐  │
│  │ document_loader.py   文档解析                   │  │
│  │ text_splitter.py     文本切分                   │  │
│  │ embeddings.py        文本向量化                 │  │
│  │ vector_store.py      FAISS 向量索引             │  │
│  │ bm25_index.py        BM25 关键词索引            │  │
│  │ retriever.py         混合召回 / 递归检索         │  │
│  │ reranker.py          二阶段重排序                │  │
│  │ generator.py         Prompt 构造与答案生成       │  │
│  └────────────────────────────────────────────────┘  │
└───────────────┬──────────────────────┬──────────────┘
                │                      │
                ▼                      ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│      本地文档知识库         │   │      扩展功能 (features/)   │
│ PDF / TXT / MD / DOCX ... │   │ web_search.py            │
│ 文档解析 + 文本切分 + 索引  │   │ conflict_detector.py     │
└───────────────────────────┘   │ thinking_chain.py        │
                                └───────────────────────────┘

最终输出：Answer + Retrieved Context + Source References
```

---

## 📂 目录结构

```text
docflow-rag/
├── rag_demo.py                  # Gradio Web 界面入口
├── api_router.py                # FastAPI 服务入口
├── config.py                    # 项目配置
├── example.env                  # 环境变量示例
├── requirements.txt             # 依赖列表
├── core/
│   ├── document_loader.py       # 多格式文档解析
│   ├── text_splitter.py         # 文本切分
│   ├── embeddings.py            # 文本向量化
│   ├── vector_store.py          # FAISS 向量索引管理
│   ├── bm25_index.py            # BM25 检索模块
│   ├── retriever.py             # 混合检索 / 递归检索
│   ├── reranker.py              # 重排序模块
│   └── generator.py             # 大模型答案生成
├── features/
│   ├── web_search.py            # 联网搜索增强
│   ├── conflict_detector.py     # 多来源冲突检测
│   └── thinking_chain.py        # 推理链相关处理
├── utils/
│   └── network.py               # 网络和端口工具
├── images/                      # README 配图、界面截图
├── backups/                     # 备份文件
└── README.md                    # 项目说明文档
```

---

## 📦 环境依赖

### Python 版本

建议使用 **Python 3.10+**

### 主要依赖包

| 包名                                      | 用途                     |
| ----------------------------------------- | ------------------------ |
| gradio                                    | 构建 Web 前端界面        |
| fastapi                                   | 提供后端 API 服务        |
| uvicorn                                   | 启动 FastAPI 服务        |
| faiss-cpu                                 | 本地向量索引与相似度搜索 |
| jieba                                     | 中文分词，支持 BM25 检索 |
| rank-bm25                                 | 关键词检索               |
| pypdf                                     | PDF 文档解析             |
| python-docx                               | Word 文档解析            |
| pandas                                    | Excel / 表格数据处理     |
| openpyxl                                  | 读取 xlsx 文件           |
| requests / httpx                          | 网络请求                 |
| python-dotenv                             | 加载 `.env` 配置       |
| sentence-transformers / reranker 相关依赖 | 向量模型与重排序         |

### 安装依赖

```bash
pip install -r requirements.txt
```

如果某些环境下未包含 Excel 依赖，可额外安装：

```bash
pip install pandas openpyxl
```

---

## ⚙️ 配置说明

### 1. DeepSeek API Key

本项目默认使用 DeepSeek 模型完成答案生成，请在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL_NAME=deepseek-chat
```

### 2. 联网搜索配置（可选）

若希望启用联网搜索增强能力，请额外配置：

```env
SERPAPI_KEY=your_serpapi_key
```

若未配置该项，联网搜索功能将自动关闭，但不会影响本地文档问答功能。

### 3. 环境变量示例

你可以复制示例文件后再修改：

```bash
cp example.env .env
```

Windows PowerShell：

```powershell
Copy-Item example.env .env
```

### 4. 配置文件说明

项目配置主要位于以下文件：

```text
config.py
example.env
.env
```

可在这些文件中调整：

- 默认模型名称
- 服务端口
- 检索参数
- 是否启用联网搜索
- 前端/后端启动配置

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/quanthuyngoc987-star/docflow-rag.git
cd docflow-rag
```

### 2. 创建虚拟环境

Windows：

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制示例配置文件：

Windows PowerShell：

```powershell
Copy-Item example.env .env
```

macOS / Linux：

```bash
cp example.env .env
```

然后编辑 `.env` 文件，填写你的 API Key。

### 5. 启动 Web UI

```bash
python rag_demo.py
```

启动后，在浏览器中访问前端页面即可进行文档上传和问答交互。

### 6. 启动 API 服务

```bash
python api_router.py
```

或者使用 uvicorn 启动：

```bash
uvicorn api_router:app --host 0.0.0.0 --port 8000 --reload
```

---

## 💬 使用方式

启动后，你可以通过 Gradio 页面或 FastAPI 接口完成以下操作：

### 文档上传

上传本地文档，系统会自动执行：

- 文档解析
- 文本切分
- 向量化
- 构建 FAISS 索引
- 构建 BM25 索引

### 文档问答

上传文档后，直接对知识库进行提问，例如：

```text
用户：这份文档主要讲了什么？
用户：论文中提到的方法分为哪几个步骤？
用户：资料里关于故障处理的建议有哪些？
用户：这个技术方案相比传统方法有什么优点？
```

### 混合检索增强

系统会同时结合：

- 向量检索：处理语义相似问题
- BM25 检索：处理关键词、术语、编号类问题
- 重排序：提升最终上下文质量

### 联网搜索增强（可选）

如果配置了 `SERPAPI_KEY`，系统还可对时效性问题进行补充搜索，例如：

```text
用户：这个方向最近有没有新的公开进展？
用户：相关技术最近有什么新闻或趋势？
```

---

## 🛠 核心模块说明

项目核心由以下几个模块组成：

| 模块                     | 描述                        |
| ------------------------ | --------------------------- |
| `document_loader.py`   | 负责多格式文档解析          |
| `text_splitter.py`     | 负责长文本切分              |
| `embeddings.py`        | 负责文本向量化              |
| `vector_store.py`      | 负责构建与查询 FAISS 向量库 |
| `bm25_index.py`        | 负责构建 BM25 检索索引      |
| `retriever.py`         | 负责混合召回与递归检索      |
| `reranker.py`          | 负责候选结果重排序          |
| `generator.py`         | 负责 Prompt 组织与答案生成  |
| `web_search.py`        | 负责联网搜索增强            |
| `conflict_detector.py` | 负责多来源信息冲突分析      |

---

## 🔄 系统流程说明

```text
用户上传文档
   ↓
文档解析（PDF/TXT/MD/DOCX/XLSX/PPTX）
   ↓
文本切分（chunk）
   ↓
向量化建库（FAISS）+ 关键词建库（BM25）
   ↓
用户输入问题
   ↓
混合召回（Vector + BM25）
   ↓
候选结果重排序（Reranker）
   ↓
构造 Prompt
   ↓
调用 DeepSeek 生成答案
   ↓
返回答案与参考上下文
```

---

## 📡 API 使用示例

如果项目开放了 API 接口，可以通过 FastAPI 方式调用。
以下示例可作为 README 展示参考，你可根据自己实际接口名进行调整。

### 检查

```bash
curl http://127.0.0.1:8000/health
```

### 提交问题

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"这份资料主要讲了什么？\"}"
```

### 上传文件

如果你已实现上传接口，可参考如下格式：

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@example.pdf"
```

> 注：如果你当前项目中的接口路径与这里不同，直接把接口路径替换成你实际实现的版本即可。

---

## 🖼 界面展示

你可以在 `images/` 目录中放置界面截图，并在 README 中展示：

```markdown
## Web 界面示意

![demo](images/demo.png)
```

建议至少展示以下内容：

- 首页界面
- 上传文档后的知识库状态
- 问答示例
- 检索片段展示

---

## 📚 支持的知识库文档类型

当前项目适合处理以下类型资料：

- 学术论文
- 技术文档
- 课程讲义
- 项目手册
- 产品说明书
- 常见问题汇总
- 规范文档
- 实验报告

支持格式包括但不限于：

- `.pdf`
- `.txt`
- `.md`
- `.docx`
- `.xls`
- `.xlsx`
- `.pptx`

---

## 📋 项目亮点

你可以在简历或项目介绍中突出以下几点：

- 从 0 到 1 完成 RAG 系统全链路搭建
- 支持多格式文档接入与中文场景问答
- 引入 FAISS + BM25 混合召回机制
- 使用 Reranker 提升检索精度
- 同时提供 Gradio 前端与 FastAPI 接口
- 支持可选联网搜索增强
- 工程结构清晰，便于扩展与部署

---

## 🔮 后续优化方向

- 增加持久化知识库与增量索引更新能力
- 支持更细粒度的来源引用展示
- 增加文档权限隔离与多用户支持
- 加入检索效果评测脚本
- 支持 Docker 一键部署
- 支持多轮对话记忆与会话管理
- 支持更多国产大模型或本地模型切换
- 引入更完整的日志、监控与异常追踪体系

---

## 📄 许可证

本项目采用 **MIT License**。

---

如果这个项目对你有帮助，欢迎点个 Star ⭐

<style>#mermaid-1776417391248{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1776417391248 .error-icon{fill:#552222;}#mermaid-1776417391248 .error-text{fill:#552222;stroke:#552222;}#mermaid-1776417391248 .edge-thickness-normal{stroke-width:2px;}#mermaid-1776417391248 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1776417391248 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1776417391248 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1776417391248 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1776417391248 .marker{fill:#333333;}#mermaid-1776417391248 .marker.cross{stroke:#333333;}#mermaid-1776417391248 svg{font-family:sans-serif;font-size:16px;}#mermaid-1776417391248 .label{font-family:sans-serif;color:#333;}#mermaid-1776417391248 .label text{fill:#333;}#mermaid-1776417391248 .node rect,#mermaid-1776417391248 .node circle,#mermaid-1776417391248 .node ellipse,#mermaid-1776417391248 .node polygon,#mermaid-1776417391248 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1776417391248 .node .label{text-align:center;}#mermaid-1776417391248 .node.clickable{cursor:pointer;}#mermaid-1776417391248 .arrowheadPath{fill:#333333;}#mermaid-1776417391248 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1776417391248 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1776417391248 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1776417391248 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1776417391248 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1776417391248 .cluster text{fill:#333;}#mermaid-1776417391248 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1776417391248:root{--mermaid-font-family:sans-serif;}#mermaid-1776417391248:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1776417391248 flowchart{fill:apa;}</style>
