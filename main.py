"""
Knowledge Base - FastAPI 应用入口

完整 API 接口层，支持：
- 文档上传 & 异步摄入
- 向量检索
- 文档/TOC 管理
- 任务状态查询

启动服务:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

启动异步 Worker:
    celery -A celery_app worker --loglevel=info --concurrency=4
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    BackgroundTasks,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api_models import HealthResponse, build_pydantic_json_schemas
from config import settings
from file_ingestion import router as file_ingestion_router
from faq_ingestion import router as faq_ingestion_router
from text_ingestion import router as text_ingestion_router

# 异步任务
try:
    from celery_app import ingest_document_task, get_task_status
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    ingest_document_task = None
    get_task_status = None

# Pipeline
from ingestion_pipeline import DocumentIngestionPipeline

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  FastAPI 应用
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Base API",
        description="""
## 知识库文档摄入与检索 API

支持多种文档格式（PDF, DOCX, PPTX, XLSX, MD, HTML, TXT）的解析、TOC 抽取、智能分块、多模态 Embedding 和向量检索。

### 核心功能

- **文档摄入**: 上传文档 → 自动解析 → TOC 抽取 → 智能分块 → Embedding → 向量存储
- **向量检索**: 支持文本/多模态检索，按章节/文档类型过滤
- **文档管理**: 查看文档 TOC 结构、删除文档、查询摄入状态
- **异步任务**: 大文件支持异步处理，可查询任务进度

### 技术栈

- **解析引擎**: MinerU (PDF), python-docx, python-pptx
- **TOC 抽取**: Qwen2.5-72B-Instruct
- **Embedding**: Qwen2.5-VL-Embedding (多模态)
- **向量库**: PostgreSQL + pgvector
        """,
        version="2.0.0",
        docs_url="/swagger",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS 配置
    if settings.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allow_origins,
            allow_credentials=settings.allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # 默认允许所有（开发环境）
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 注册路由
    app.include_router(file_ingestion_router)
    app.include_router(faq_ingestion_router)
    app.include_router(text_ingestion_router)

    # 注册 API 文档路由
    @app.get("/api-docs/openapi.json", tags=["api-docs"])
    async def api_openapi_json():
        return app.openapi()

    @app.get("/api-docs/json-schema", tags=["api-docs"])
    async def api_json_schema():
        return {
            "openapi": app.openapi(),
            "pydantic_schemas": build_pydantic_json_schemas(),
        }

    # 健康检查
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "celery_available": CELERY_AVAILABLE,
        }

    return app


app = create_app()

# ─────────────────────────────────────────────
#  请求/响应模型
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    """文档摄入请求"""
    kb_id: str = Field(default="default", description="知识库 ID")
    user_id: str = Field(default="default", description="用户 ID")
    async_mode: bool = Field(default=True, description="是否使用异步模式（大文件推荐）")


class IngestResponse(BaseModel):
    """文档摄入响应"""
    task_id: Optional[str] = Field(None, description="异步任务 ID（async_mode=true 时返回）")
    doc_id: str = Field(..., description="文档 ID")
    status: str = Field(..., description="状态：submitted | success | skipped")
    message: str = Field("", description="附加信息")
    
    # 同步模式返回的详细信息
    toc_nodes: Optional[int] = None
    toc_depth: Optional[int] = None
    total_chunks: Optional[int] = None
    chunking_mode: Optional[str] = None
    toc_outline: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
    ready: bool
    successful: Optional[bool] = None
    result: Optional[dict] = None


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="检索文本")
    kb_id: str = Field(default="default", description="知识库 ID")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数")
    doc_id: Optional[str] = Field(None, description="限定文档 ID")
    chunk_type: Optional[str] = Field(None, description="限定 chunk 类型：text | table | image | formula")
    section_path: Optional[str] = Field(None, description="按章节路径前缀过滤")
    min_score: float = Field(default=0.0, ge=0, le=1, description="最小相似度阈值")


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    doc_id: str
    text: str
    section_path: str
    section_title: str
    chunk_type: str
    page: Optional[int] = None
    score: float
    source: str


class SearchResponse(BaseModel):
    """检索响应"""
    query: str
    total: int
    results: list[SearchResult]


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    filename: str
    format: str
    title: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: int
    toc_nodes: int
    toc_depth: int
    created_at: str
    toc_outline: str


class TOCNodeResponse(BaseModel):
    """TOC 节点"""
    id: str
    title: str
    level: int
    node_type: str
    children: list["TOCNodeResponse"] = []


class TOCResponse(BaseModel):
    """TOC 响应"""
    doc_id: str
    total_nodes: int
    max_depth: int
    outline: str
    tree: list[TOCNodeResponse] = []


class DeleteResponse(BaseModel):
    """删除响应"""
    doc_id: str
    status: str
    message: str


# ─────────────────────────────────────────────
#  API 路由
# ─────────────────────────────────────────────

@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["文档摄入"])
async def ingest_document(
    file: UploadFile = File(..., description="要上传的文档文件"),
    kb_id: str = Query(default="default", description="知识库 ID"),
    user_id: str = Query(default="default", description="用户 ID"),
    async_mode: bool = Query(default=True, description="是否使用异步模式"),
):
    """
    ## 上传并摄入文档
    
    支持格式：PDF, DOCX, PPTX, XLSX, MD, HTML, TXT
    
    ### 流程
    
    1. 上传文件到临时目录
    2. 同步模式：直接处理并返回结果
    3. 异步模式：提交 Celery 任务，返回 task_id 供查询进度
    
    ### 异步模式优势
    
    - 不阻塞 API 请求
    - 支持大文件处理
    - 可查询任务进度
    - 失败自动重试
    """
    # 保存上传文件
    temp_dir = tempfile.mkdtemp(prefix="knowledge_upload_")
    temp_path = Path(temp_dir) / file.filename
    
    try:
        content = await file.read()
        temp_path.write_bytes(content)
        
        logger.info(f"[API] 收到文件：{file.filename}, size={len(content)} bytes")
        
        # 异步模式
        if async_mode and CELERY_AVAILABLE:
            task = ingest_document_task.delay(
                str(temp_path),
                kb_id,
                user_id,
            )
            logger.info(f"[API] 提交异步任务：{task.id}")
            
            return IngestResponse(
                task_id=task.id,
                doc_id="pending",
                status="submitted",
                message="任务已提交，使用 task_id 查询进度",
            )
        
        # 同步模式
        else:
            pipeline = DocumentIngestionPipeline()
            result = await pipeline.ingest(str(temp_path), kb_id, user_id)
            
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return IngestResponse(
                doc_id=result.get("doc_id", "unknown"),
                status=result.get("status", "success"),
                message="文档摄入完成",
                toc_nodes=result.get("toc_nodes"),
                toc_depth=result.get("toc_depth"),
                total_chunks=result.get("total_chunks"),
                chunking_mode=result.get("chunking_mode"),
                toc_outline=result.get("toc_outline"),
            )
            
    except Exception as e:
        logger.exception(f"[API] 摄入失败：{e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["任务管理"])
async def get_task(task_id: str):
    """
    ## 查询异步任务状态
    
    返回任务当前状态和结果（如果已完成）
    """
    if not CELERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Celery 异步任务不可用")
    
    status = get_task_status(task_id)
    
    return TaskStatusResponse(
        task_id=task_id,
        status=status["status"],
        ready=status["ready"],
        successful=status["successful"],
        result=status["result"],
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["检索"])
async def search(request: SearchRequest):
    """
    ## 向量检索
    
    支持：
    - 纯文本检索
    - 按文档 ID 过滤
    - 按 chunk 类型过滤（text/table/image/formula）
    - 按章节路径过滤
    """
    from ingestion_pipeline import DocumentIngestionPipeline
    
    pipeline = DocumentIngestionPipeline()
    
    # 生成 query embedding
    embedder = pipeline.embedder
    embedding = await embedder.embed_text([request.query])
    
    # 检索
    results = await pipeline.vector_store.search(
        query_embedding=embedding[0],
        top_k=request.top_k,
        filters={
            "kb_id": request.kb_id,
            "doc_id": request.doc_id,
            "chunk_type": request.chunk_type,
            "section_path_prefix": request.section_path,
        } if any([request.doc_id, request.chunk_type, request.section_path]) else None,
    )
    
    # 格式化结果
    formatted_results = []
    for r in results:
        if r.get("score", 1.0) < request.min_score:
            continue
            
        formatted_results.append(SearchResult(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            text=r["text_content"],
            section_path=r.get("section_path", ""),
            section_title=r.get("section_title", ""),
            chunk_type=r.get("chunk_type", "text"),
            page=r.get("page"),
            score=r.get("score", 0),
            source=f"[{r.get('section_path', 'Unknown')}] (第{r.get('page', '?')}页)" if r.get('section_path') else f"第{r.get('page', '?')}页",
        ))
    
    return SearchResponse(
        query=request.query,
        total=len(formatted_results),
        results=formatted_results,
    )


@app.get("/api/v1/documents/{doc_id}", response_model=DocumentInfo, tags=["文档管理"])
async def get_document(doc_id: str):
    """
    ## 获取文档信息
    
    返回文档元数据、TOC 统计等
    """
    from ingestion_pipeline import DocumentIngestionPipeline
    
    pipeline = DocumentIngestionPipeline()
    doc_info = await pipeline.vector_store.get_document(doc_id)
    
    if not doc_info:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")
    
    return DocumentInfo(**doc_info)


@app.get("/api/v1/documents/{doc_id}/toc", response_model=TOCResponse, tags=["文档管理"])
async def get_document_toc(doc_id: str):
    """
    ## 获取文档 TOC 结构
    
    返回完整的目录树结构
    """
    from ingestion_pipeline import DocumentIngestionPipeline
    
    pipeline = DocumentIngestionPipeline()
    toc_data = await pipeline.vector_store.get_toc(doc_id)
    
    if not toc_data:
        raise HTTPException(status_code=404, detail=f"文档不存在或无 TOC：{doc_id}")
    
    return TOCResponse(**toc_data)


@app.delete("/api/v1/documents/{doc_id}", response_model=DeleteResponse, tags=["文档管理"])
async def delete_document(doc_id: str):
    """
    ## 删除文档
    
    同时删除：
    - 向量数据库中的 chunks
    - 元数据记录
    - Blob Storage 中的文件（如果配置）
    """
    from ingestion_pipeline import DocumentIngestionPipeline
    
    pipeline = DocumentIngestionPipeline()
    
    try:
        await pipeline.vector_store.delete_doc(doc_id)
        return DeleteResponse(
            doc_id=doc_id,
            status="deleted",
            message="文档及其所有 chunks 已删除",
        )
    except Exception as e:
        logger.exception(f"[API] 删除失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents", response_model=list[DocumentInfo], tags=["文档管理"])
async def list_documents(
    kb_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    ## 列出知识库中的所有文档
    """
    from ingestion_pipeline import DocumentIngestionPipeline
    
    pipeline = DocumentIngestionPipeline()
    docs = await pipeline.vector_store.list_documents(kb_id, limit, offset)
    
    return [DocumentInfo(**d) for d in docs]


# ─────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting Knowledge Base API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
