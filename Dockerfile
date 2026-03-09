# =============================================
# Knowledgebase - Dockerfile
# =============================================
# 多阶段构建，优化镜像体积

# ── 阶段 1: 依赖构建 ──────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /app

# 安装 Poetry
RUN pip install --no-cache-dir poetry==1.8.0

# 复制依赖定义
COPY pyproject.toml poetry.lock ./

# 安装生产依赖 (不含可选依赖)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root \
    && poetry install --no-interaction --no-ansi --extras all

# ── 阶段 2: 运行镜像 ──────────────────────────────────────────────────────────
FROM python:3.11-slim as runtime

WORKDIR /app

# 安装系统依赖 (PDF 解析等需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data && chmod 777 /app/data

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz')" || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
