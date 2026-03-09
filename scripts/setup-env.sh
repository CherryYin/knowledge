#!/bin/bash
# 快速环境配置脚本
# 用法：./scripts/setup-env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$WORKSPACE/.env"

echo "🔧 Knowledgebase 环境配置向导"
echo "================================"
echo ""

# 检查是否已存在 .env 文件
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  检测到 .env 文件已存在"
    read -p "是否覆盖？(y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "已取消操作"
        exit 0
    fi
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d%H%M%S)"
    echo "✓ 已备份现有配置"
fi

# 复制模板
cp "$WORKSPACE/.env.example" "$ENV_FILE"
echo "✓ 已创建 .env 文件"
echo ""

# 交互式配置
echo "📋 开始配置（直接回车使用默认值）"
echo ""

# 数据库配置
read -p "PostgreSQL 主机 (默认 localhost): " DB_HOST
DB_HOST=${DB_HOST:-localhost}

read -p "PostgreSQL 端口 (默认 5432): " DB_PORT
DB_PORT=${DB_PORT:-5432}

read -p "PostgreSQL 用户名 (默认 postgres): " DB_USER
DB_USER=${DB_USER:-postgres}

read -sp "PostgreSQL 密码 (默认 postgres): " DB_PASS
DB_PASS=${DB_PASS:-postgres}
echo ""

read -p "数据库名称 (默认 knowledge): " DB_NAME
DB_NAME=${DB_NAME:-knowledge}

# 更新 DATABASE_URL
sed -i "s|^DATABASE_URL=.*|DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}|" "$ENV_FILE"
sed -i "s|^PGVECTOR_DSN=.*|PGVECTOR_DSN=postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}|" "$ENV_FILE"

echo "✓ 数据库配置完成"
echo ""

# Redis 配置
read -p "Redis 主机 (默认 localhost): " REDIS_HOST
REDIS_HOST=${REDIS_HOST:-localhost}

read -p "Redis 端口 (默认 6379): " REDIS_PORT
REDIS_PORT=${REDIS_PORT:-6379}

sed -i "s|^REDIS_URL=.*|REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/0|" "$ENV_FILE"

echo "✓ Redis 配置完成"
echo ""

# Embedding 服务
read -p "Embedding 服务地址 (默认 http://localhost:8100): " EMBED_URL
EMBED_URL=${EMBED_URL:-http://localhost:8100}

sed -i "s|^EMBEDDING_BASE_URL=.*|EMBEDDING_BASE_URL=${EMBED_URL}|" "$ENV_FILE"

echo "✓ Embedding 服务配置完成"
echo ""

# TOC LLM 服务
read -p "TOC LLM 服务地址 (默认 http://localhost:8000/v1): " LLM_URL
LLM_URL=${LLM_URL:-http://localhost:8000/v1}

sed -i "s|^TOC_LLM_BASE_URL=.*|TOC_LLM_BASE_URL=${LLM_URL}|" "$ENV_FILE"

echo "✓ TOC LLM 配置完成"
echo ""

# 端口配置
read -p "API 服务端口 (默认 8000): " API_PORT
API_PORT=${API_PORT:-8000}

sed -i "s|^PORT=.*|PORT=${API_PORT}|" "$ENV_FILE"

echo "✓ 端口配置完成"
echo ""

# 显示配置摘要
echo ""
echo "📊 配置摘要"
echo "================================"
grep -E "^DATABASE_URL=|^PGVECTOR_DSN=|^REDIS_URL=|^EMBEDDING_BASE_URL=|^TOC_LLM_BASE_URL=|^PORT=" "$ENV_FILE"
echo ""

# 检查依赖服务
echo "🔍 检查服务状态"
echo "================================"

# 检查 PostgreSQL
if command -v pg_isready &> /dev/null; then
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -q 2>/dev/null; then
        echo "✓ PostgreSQL: 运行中"
    else
        echo "⚠️  PostgreSQL: 未运行或无法连接"
    fi
else
    echo "ℹ️  PostgreSQL: 跳过检查（未安装 pg_isready）"
fi

# 检查 Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        echo "✓ Redis: 运行中"
    else
        echo "⚠️  Redis: 未运行或无法连接"
    fi
else
    echo "ℹ️  Redis: 跳过检查（未安装 redis-cli）"
fi

echo ""
echo "✅ 配置完成！"
echo ""
echo "下一步："
echo "  1. 检查并修改配置：vim .env"
echo "  2. 启动 PostgreSQL: docker-compose up -d postgres"
echo "  3. 启动 Redis: docker run -d --name knowledge-redis -p 6379:6379 redis:7-alpine"
echo "  4. 启动 Celery Worker: celery -A celery_app worker --loglevel=info"
echo "  5. 启动 API 服务：uvicorn main:app --host 0.0.0.0 --port ${API_PORT}"
echo ""
