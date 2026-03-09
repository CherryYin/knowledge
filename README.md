# Knowledgebase - RAG 知识库文档摄入服务

支持多格式文档解析、TOC 驱动分块、多模态 Embedding 的向量知识库服务。

## 🚀 快速开始

### 前置要求

- Python 3.10 - 3.12
- PostgreSQL 14+ (带 pgvector 扩展)
- Poetry 包管理器

### 安装

```bash
# 克隆仓库
git clone git@github.com:CherryYin/knowledge.git
cd knowledge

# 安装 Poetry (如未安装)
curl -sSL https://install.python-poetry.org | python3 -

# 安装依赖
poetry install

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 启动服务

```bash
# 开发模式
poetry run uvicorn main:app --reload --port 8000

# 生产模式
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 健康检查

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}
```

---

## 📋 功能特性

### 支持的文档格式

| 格式 | 扩展名 | 解析器 |
|------|--------|--------|
| PDF | `.pdf` | 内置 / MinerU (可选) |
| Word | `.docx` | python-docx |
| PowerPoint | `.pptx` | python-pptx |
| Excel | `.xlsx`, `.xls` | pandas + openpyxl |
| CSV | `.csv` | pandas |
| Markdown | `.md`, `.markdown` | 内置 |
| HTML | `.html`, `.htm` | BeautifulSoup4 |
| 纯文本 | `.txt`, `.text` | 内置 |

### 核心能力

- **TOC 驱动分块** - 基于文档目录结构智能分块
- **多模态 Embedding** - 支持 vLLM 服务或本地 Transformers
- **增量摄入** - 自动检测重复文档，避免重复索引
- **FAQ 管理** - 独立的 FAQ 向量存储与 CRUD API
- **外部同步** - 可选将分块同步到外部系统

---

## 🔧 配置

### 环境变量

详见 `.env.example`。核心配置：

```bash
# 数据库
DATABASE_URL=postgresql://user:pass@localhost:5432/knowledge

# Embedding 服务
EMBEDDING_BASE_URL=http://localhost:8100
EMBEDDING_MODEL=Qwen/Qwen2.5-VL-Embedding

# 可选：本地 Embedding
USE_LOCAL_EMBEDDING=false
```

### 可选依赖

```bash
# PDF 高级解析 (MinerU)
poetry install --extras pdf

# 本地 Embedding (Transformers)
poetry install --extras local-embed

# S3 / MinIO 存储
poetry install --extras blob-s3

# Azure Blob 存储
poetry install --extras blob-azure

# 全部功能
poetry install --extras all
```

---

## 📡 API 概览

### 文件摄入

```bash
# 上传文件
curl -X POST "http://localhost:8000/file-ingestion/upload" \
  -H "X-User-Id: user-001" \
  -F "files=@document.pdf"

# 删除文档
curl -X DELETE "http://localhost:8000/file-ingestion/delete" \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "abc123..."}'
```

### 文本摄入

```bash
curl -X POST "http://localhost:8000/text-ingestion/upsert" \
  -H "X-User-Id: user-001" \
  -H "Content-Type: application/json" \
  -d '{"text": "内容...", "sourcefile": "my-text", "category": "general"}'
```

### FAQ 管理

```bash
# 添加/更新 FAQ
curl -X POST "http://localhost:8000/faq-ingestion/upsert" \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "question": "问题?", "answer": "答案", "markets": ""}'

# 删除 FAQ
curl -X DELETE "http://localhost:8000/faq-ingestion/123"
```

### API 文档

- Swagger UI: http://localhost:8000/swagger
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

---

## 🧪 测试

```bash
poetry run pytest tests/ -v
```

---

## 📦 Docker 部署

```bash
# 构建镜像
docker build -t knowledge-base .

# 启动容器
docker-compose up -d
```

---

## 📝 开发

### 代码规范

```bash
# 格式化
poetry run ruff format .

# Lint
poetry run ruff check .
```

### 导出 OpenAPI

```bash
poetry run python scripts/export_api.py
```

---

## 📄 许可证

MIT

---

## 🙏 致谢

- FastAPI
- pgvector
- vLLM
- Qwen Embedding
