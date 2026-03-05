"""
文档摄入 Pipeline

完整流程:
  ① 文件上传 & 哈希去重
  ② 文档解析（FormatRouter → MinerU / python-docx / ...）
  ③ LLM 提取 TOC（Qwen 大模型）
  ④ TOC 驱动分块（min_tokens 自适应合并 + 降级滑动窗口）
  ⑤ 多模态 Embedding（vLLM / 本地 Transformers）
  ⑥ 写入向量数据库 + 元数据

配置方式（优先 kwargs，其次环境变量，最后默认值）:
  CHUNK_MIN_TOKENS      min_chunk_tokens（默认 200）
  CHUNK_MAX_TOKENS      max_chunk_tokens（默认 512）
  CHUNK_OVERLAP_TOKENS  overlap_tokens（默认 64）
  TOC_LLM_MODEL         TOC 抽取用的 LLM 模型名
  TOC_LLM_BASE_URL      LLM 接口地址
  TOC_LLM_API_KEY       LLM API Key
  EMBEDDING_BASE_URL    vLLM Embedding 接口地址
  EMBEDDING_MODEL       Embedding 模型名
  PGVECTOR_DSN          PostgreSQL 连接串，如:
                        postgresql://user:pass@localhost:5432/ragdb

  # Blob Storage（可选，不设则跳过）
  BLOB_BACKEND          存储后端: s3 | azure | none（默认 none）

  # S3 / MinIO
  BLOB_S3_ENDPOINT      S3 兼容端点，如 http://minio:9000（留空则用 AWS 默认）
  BLOB_S3_BUCKET        桶名
  BLOB_S3_ACCESS_KEY    Access Key
  BLOB_S3_SECRET_KEY    Secret Key
  BLOB_S3_REGION        区域（默认 us-east-1）
  BLOB_S3_PREFIX        对象键前缀（默认 documents/）

  # Azure Blob Storage
  BLOB_AZURE_CONN_STR   连接字符串
  BLOB_AZURE_CONTAINER  容器名
  BLOB_AZURE_PREFIX     Blob 前缀（默认 documents/）

    # Document Segments（可选，将 chunking 后文本同步到外部接口）
    DOCUMENT_SEGMENTS_BASE_URL        例如 http://127.0.0.1:8001
    DOCUMENT_SEGMENTS_PATH            默认 /api/v1/document_segments
    DOCUMENT_SEGMENTS_TOKEN           Bearer Token（不带 Bearer 前缀）
    DOCUMENT_SEGMENTS_TIMEOUT_SECONDS 请求超时秒数（默认 8）

依赖: pip install asyncpg pgvector
可选: pip install aiobotocore   # S3 后端
可选: pip install azure-storage-blob azure-identity  # Azure 后端
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from chunker import Chunk, TOCDrivenChunker
from db_schema import INIT_SQL, UPSERT_DOC_CHUNKS_SQL
from doc_parser import FormatRouter, ParsedDocument
from embedding_service import VLLMEmbeddingClient, create_embedding_service
from toc_extractor import LLMTOCExtractor, TOCTree

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def build_doc_id(user_id: str, file_name: str) -> str:
    """使用 user_id + file_name 生成稳定 doc_id。"""
    raw = f"{user_id}|{file_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _DocumentSegmentsClient:
    """将 chunking 后文本同步到外部 document_segments 接口（可选）。"""

    def __init__(
        self,
        base_url: str,
        path: str = "/api/v1/document_segments",
        bearer_token: str = "",
        timeout_seconds: float = 8.0,
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.path = (path.strip() or "/api/v1/document_segments")
        self.bearer_token = bearer_token.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.bearer_token)

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url}/{self.path.lstrip('/')}"

    async def push_chunks(
        self,
        *,
        chunks: list[Chunk],
        document_id: str,
        source_type: str,
        extra_keywords: Optional[str] = None,
    ) -> dict[str, int]:
        if not self.enabled:
            return {"pushed": 0, "failed": 0}

        import httpx

        pushed = 0
        failed = 0
        doc_id_payload: int | str
        doc_id_payload = int(document_id) if str(document_id).isdigit() else document_id

        headers = {"Authorization": f"Bearer {self.bearer_token}"}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for idx, chunk in enumerate(chunks):
                payload = {
                    "source_type": source_type,
                    "document_id": doc_id_payload,
                    "content": chunk.text_content,
                    "chunk_number": idx,
                    "start_page": chunk.start_page if chunk.start_page is not None else (chunk.page or 0),
                    "end_page": chunk.end_page if chunk.end_page is not None else (chunk.page or 0),
                    "keywords": (
                        chunk.section_title
                        or chunk.section_path
                        or (extra_keywords or "")
                    ),
                }

                try:
                    resp = await client.post(self.endpoint_url, json=payload, headers=headers)
                    if 200 <= resp.status_code < 300:
                        pushed += 1
                    else:
                        failed += 1
                        logger.warning(
                            "[segment_sync] failed status=%s chunk=%s detail=%s",
                            resp.status_code,
                            idx,
                            resp.text[:300],
                        )
                except Exception as exc:
                    failed += 1
                    logger.warning("[segment_sync] request error chunk=%s err=%s", idx, exc)

        return {"pushed": pushed, "failed": failed}


# ─────────────────────────────────────────────
#  向量存储（PostgreSQL + pgvector）
# ─────────────────────────────────────────────


class PGVectorStore:
    """
    基于 PostgreSQL + pgvector 的向量存储层

    表结构（自动建表）:
      doc_chunks(chunk_id PK, doc_id, chunk_type, text_content,
                 section_path, section_title, toc_node_id,
                 page, embedding vector(1536), metadata_json JSONB)

    索引:
      - HNSW cosine 索引（向量检索）
      - B-tree doc_id 索引（按文档删除/过滤）
      - text_pattern_ops section_path 索引（前缀过滤）

    依赖: pip install asyncpg pgvector
    """

    def __init__(
        self,
        dsn: str = "postgresql://postgres:postgres@localhost:5432/ragdb",
        embedding_dim: int = 1536,
    ):
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            try:
                import asyncpg
                from pgvector.asyncpg import register_vector
            except ImportError as exc:
                raise ImportError(
                    "asyncpg and pgvector are required. "
                    "Run: pip install asyncpg pgvector"
                ) from exc

            async def init_conn(conn):
                await register_vector(conn)

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=1,
                max_size=10,
                init=init_conn,
            )
            async with self._pool.acquire() as conn:
                await conn.execute(INIT_SQL)
        return self._pool

    async def upsert_chunks(self, records: list[dict]) -> None:
        """批量 upsert chunks，使用 executemany 提高性能"""
        import json
        import numpy as np

        pool = await self._get_pool()
        rows = [
            (
                r["chunk_id"],
                r["doc_id"],
                r["chunk_type"],
                r["text_content"],
                r["section_path"],
                r["section_title"],
                r["toc_node_id"],
                int(r["page"]),
                int(r["start_page"]),
                int(r["end_page"]),
                r["file_name"],
                r["blob_url"],
                r["blob_key"],
                np.array(r["embedding"], dtype=np.float32),
                json.dumps(r["metadata_json"], ensure_ascii=False),
            )
            for r in records
        ]
        async with pool.acquire() as conn:
            await conn.executemany(UPSERT_DOC_CHUNKS_SQL, rows)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        余弦相似度向量检索（越小越相似，pgvector <=> 返回距离）

        filters 支持:
          - doc_id:              精确匹配
          - chunk_type:          精确匹配
          - section_path_prefix: LIKE 前缀匹配
        """
        import json
        import numpy as np

        pool = await self._get_pool()

        where_clauses: list[str] = []
        params: list = [np.array(query_embedding, dtype=np.float32), top_k]
        idx = 3  # $1=embedding, $2=limit, $3+ for filters

        if filters:
            if "doc_id" in filters:
                where_clauses.append(f"doc_id = ${idx}")
                params.append(filters["doc_id"])
                idx += 1
            if "chunk_type" in filters:
                where_clauses.append(f"chunk_type = ${idx}")
                params.append(filters["chunk_type"])
                idx += 1
            if "section_path_prefix" in filters:
                where_clauses.append(f"section_path LIKE ${idx}")
                params.append(filters["section_path_prefix"] + "%")
                idx += 1

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            SELECT
                chunk_id, doc_id, text_content, chunk_type,
                section_path, section_title, toc_node_id, page,
                start_page, end_page, file_name, blob_url, blob_key,
                metadata_json,
                1 - (embedding <=> $1) AS score
            FROM doc_chunks
            {where_sql}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "chunk_id":      row["chunk_id"],
                "doc_id":        row["doc_id"],
                "text_content":  row["text_content"],
                "chunk_type":    row["chunk_type"],
                "section_path":  row["section_path"],
                "section_title": row["section_title"],
                "toc_node_id":   row["toc_node_id"],
                "page":          row["page"],
                "start_page":    row["start_page"],
                "end_page":      row["end_page"],
                "file_name":     row["file_name"],
                "blob_url":      row["blob_url"],
                "blob_key":      row["blob_key"],
                "metadata_json": json.loads(row["metadata_json"]) if isinstance(row["metadata_json"], str) else row["metadata_json"],
                "score":         float(row["score"]),
            }
            for row in rows
        ]

    async def delete_doc(self, doc_id: str) -> None:
        """删除某文档的所有 chunks"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM doc_chunks WHERE doc_id = $1", doc_id)

    async def doc_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT EXISTS (SELECT 1 FROM doc_chunks WHERE doc_id = $1) AS exists",
                doc_id,
            )
        return bool(row["exists"]) if row else False

    async def close(self) -> None:
        """关闭连接池（应用退出时调用）"""
        if self._pool:
            await self._pool.close()
            self._pool = None


# 保留别名，方便外部代码不感知改动
VectorStore = PGVectorStore


# ─────────────────────────────────────────────
#  Blob 对象存储
# ─────────────────────────────────────────────

class _BlobStorageBase:
    """Blob 存储抽象基类"""

    async def upload(self, file_path: str, object_key: str) -> str:
        """上传文件，返回可访问的 URL 或对象路径"""
        raise NotImplementedError

    async def delete(self, object_key: str) -> None:
        """删除对象"""
        raise NotImplementedError


class _NullBlobStorage(_BlobStorageBase):
    """无操作后端（未配置 blob 存储时使用）"""

    async def upload(self, file_path: str, object_key: str) -> str:
        return ""

    async def delete(self, object_key: str) -> None:
        pass


class S3BlobStorage(_BlobStorageBase):
    """
    S3 兼容对象存储（AWS S3 / MinIO / 阿里云 OSS 等）

    依赖: pip install aiobotocore
    """

    def __init__(
        self,
        bucket: str,
        access_key: str,
        secret_key: str,
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        prefix: str = "documents/",
    ):
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url or None
        self.region = region
        self.prefix = prefix.rstrip("/") + "/"

    def _object_url(self, object_key: str) -> str:
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{object_key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{object_key}"

    async def upload(self, file_path: str, object_key: str) -> str:
        try:
            import aiobotocore.session
        except ImportError as exc:
            raise ImportError("aiobotocore is required for S3 backend. Run: pip install aiobotocore") from exc

        session = aiobotocore.session.get_session()
        async with session.create_client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as client:
            with open(file_path, "rb") as f:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=f,
                )
        return self._object_url(object_key)

    async def delete(self, object_key: str) -> None:
        try:
            import aiobotocore.session
        except ImportError as exc:
            raise ImportError("aiobotocore is required for S3 backend. Run: pip install aiobotocore") from exc

        session = aiobotocore.session.get_session()
        async with session.create_client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as client:
            await client.delete_object(Bucket=self.bucket, Key=object_key)


class AzureBlobStorage(_BlobStorageBase):
    """
    Azure Blob Storage 后端

    依赖: pip install azure-storage-blob
    """

    def __init__(
        self,
        connection_string: str,
        container: str,
        prefix: str = "documents/",
    ):
        self.connection_string = connection_string
        self.container = container
        self.prefix = prefix.rstrip("/") + "/"

    async def upload(self, file_path: str, object_key: str) -> str:
        try:
            from azure.storage.blob.aio import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "azure-storage-blob is required for Azure backend. "
                "Run: pip install azure-storage-blob"
            ) from exc

        async with BlobServiceClient.from_connection_string(self.connection_string) as svc:
            container_client = svc.get_container_client(self.container)
            with open(file_path, "rb") as f:
                await container_client.upload_blob(name=object_key, data=f, overwrite=True)
            blob_client = container_client.get_blob_client(object_key)
            return blob_client.url

    async def delete(self, object_key: str) -> None:
        try:
            from azure.storage.blob.aio import BlobServiceClient
        except ImportError as exc:
            raise ImportError(
                "azure-storage-blob is required for Azure backend. "
                "Run: pip install azure-storage-blob"
            ) from exc

        async with BlobServiceClient.from_connection_string(self.connection_string) as svc:
            container_client = svc.get_container_client(self.container)
            await container_client.delete_blob(object_key, delete_snapshots="include")


def create_blob_storage(cfg: dict) -> _BlobStorageBase:
    """
    根据配置/环境变量创建 Blob 存储实例

    backend 优先级: cfg["blob_backend"] > 环境变量 BLOB_BACKEND > "none"
    """
    backend = (cfg.get("blob_backend") or _env("BLOB_BACKEND", "none")).lower()

    if backend == "s3":
        return S3BlobStorage(
            bucket=cfg.get("blob_s3_bucket") or _env("BLOB_S3_BUCKET"),
            access_key=cfg.get("blob_s3_access_key") or _env("BLOB_S3_ACCESS_KEY"),
            secret_key=cfg.get("blob_s3_secret_key") or _env("BLOB_S3_SECRET_KEY"),
            endpoint_url=cfg.get("blob_s3_endpoint") or _env("BLOB_S3_ENDPOINT") or None,
            region=cfg.get("blob_s3_region") or _env("BLOB_S3_REGION", "us-east-1"),
            prefix=cfg.get("blob_s3_prefix") or _env("BLOB_S3_PREFIX", "documents/"),
        )

    if backend == "azure":
        return AzureBlobStorage(
            connection_string=cfg.get("blob_azure_conn_str") or _env("BLOB_AZURE_CONN_STR"),
            container=cfg.get("blob_azure_container") or _env("BLOB_AZURE_CONTAINER"),
            prefix=cfg.get("blob_azure_prefix") or _env("BLOB_AZURE_PREFIX", "documents/"),
        )

    return _NullBlobStorage()


def create_image_blob_storage(cfg: dict) -> _BlobStorageBase:
    """
    创建用于存储抽取图片的 Blob 存储实例（与文档使用不同的 container/bucket）

    与 create_blob_storage() 共享同一 backend 类型，但读取独立的 container/prefix 配置:

    S3 环境变量:
        BLOB_S3_IMAGE_BUCKET   图片 bucket（默认与文档 bucket 相同）
        BLOB_S3_IMAGE_PREFIX   图片 key 前缀（默认 "images/"）

    Azure 环境变量:
        BLOB_AZURE_IMAGE_CONTAINER   图片 container（默认 "images"）
        BLOB_AZURE_IMAGE_PREFIX      图片 blob 前缀（默认 "images/"）
    """
    backend = (cfg.get("blob_backend") or _env("BLOB_BACKEND", "none")).lower()

    if backend == "s3":
        # image bucket 默认 fallback 到文档 bucket
        doc_bucket = cfg.get("blob_s3_bucket") or _env("BLOB_S3_BUCKET", "")
        return S3BlobStorage(
            bucket=cfg.get("blob_s3_image_bucket") or _env("BLOB_S3_IMAGE_BUCKET") or doc_bucket,
            access_key=cfg.get("blob_s3_access_key") or _env("BLOB_S3_ACCESS_KEY"),
            secret_key=cfg.get("blob_s3_secret_key") or _env("BLOB_S3_SECRET_KEY"),
            endpoint_url=cfg.get("blob_s3_endpoint") or _env("BLOB_S3_ENDPOINT") or None,
            region=cfg.get("blob_s3_region") or _env("BLOB_S3_REGION", "us-east-1"),
            prefix=cfg.get("blob_s3_image_prefix") or _env("BLOB_S3_IMAGE_PREFIX", "images/"),
        )

    if backend == "azure":
        return AzureBlobStorage(
            connection_string=cfg.get("blob_azure_conn_str") or _env("BLOB_AZURE_CONN_STR"),
            container=cfg.get("blob_azure_image_container") or _env("BLOB_AZURE_IMAGE_CONTAINER", "images"),
            prefix=cfg.get("blob_azure_image_prefix") or _env("BLOB_AZURE_IMAGE_PREFIX", "images/"),
        )

    return _NullBlobStorage()


# ─────────────────────────────────────────────
#  文档摄入 Pipeline
# ─────────────────────────────────────────────

class DocumentIngestionPipeline:
    """
    端到端文档摄入 Pipeline（TOC 驱动版）

    使用方式:
        pipeline = DocumentIngestionPipeline()
        result = await pipeline.ingest("/path/to/doc.pdf", kb_id="my_kb")
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # ① 文档解析
        self.format_router = FormatRouter(
            use_gpu=cfg.get("use_gpu", True)
        )

        # ② TOC 抽取
        self.toc_extractor = LLMTOCExtractor(
            model=cfg.get("toc_model") or _env("TOC_LLM_MODEL", "qwen2.5-72b-instruct"),
            base_url=cfg.get("llm_base_url") or _env("TOC_LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=cfg.get("llm_api_key") or _env("TOC_LLM_API_KEY", "not-needed"),
            max_segment_tokens=cfg.get("max_segment_tokens", 6000),
        )

        # ③ 分块
        self.chunker = TOCDrivenChunker(
            min_chunk_tokens=int(cfg.get("min_chunk_tokens") or _env("CHUNK_MIN_TOKENS", "200")),
            max_chunk_tokens=int(cfg.get("max_chunk_tokens") or _env("CHUNK_MAX_TOKENS", "512")),
            overlap_tokens=int(cfg.get("overlap_tokens") or _env("CHUNK_OVERLAP_TOKENS", "64")),
        )

        # ④ Embedding
        self.embedder = create_embedding_service(
            vllm_base_url=cfg.get("embedding_url"),
            vllm_model=cfg.get("embedding_model"),
        )

        # ⑤ 向量存储（PostgreSQL + pgvector）
        self.vector_store = PGVectorStore(
            dsn=cfg.get("pgvector_dsn") or _env(
                "PGVECTOR_DSN", "postgresql://postgres:postgres@localhost:5432/ragdb"
            ),
            embedding_dim=int(cfg.get("embedding_dim") or _env("EMBEDDING_DIM", "1536")),
        )

        # ⑥ Blob 对象存储（可选）
        self.blob_storage = create_blob_storage(cfg)
        # ⑦ 图片专用 Blob 存储（独立 container，避免与文档混存）
        self.image_blob_storage = create_image_blob_storage(cfg)
        # ⑧ 外部分块文本同步（可选）
        self.segment_client = _DocumentSegmentsClient(
            base_url=cfg.get("document_segments_base_url") or _env("DOCUMENT_SEGMENTS_BASE_URL"),
            path=cfg.get("document_segments_path") or _env("DOCUMENT_SEGMENTS_PATH", "/api/v1/document_segments"),
            bearer_token=cfg.get("document_segments_token") or _env("DOCUMENT_SEGMENTS_TOKEN"),
            timeout_seconds=float(
                cfg.get("document_segments_timeout_seconds")
                or _env("DOCUMENT_SEGMENTS_TIMEOUT_SECONDS", "8")
            ),
        )

    # ── 主入口 ────────────────────────────────────────────────

    async def ingest(
        self,
        file_path: str,
        kb_id: str = "default",
        doc_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """
        摄入单个文件，返回摄入结果摘要

        Returns:
            {
                "doc_id": str,
                "status": "success" | "skipped",
                "toc_nodes": int,
                "toc_depth": int,
                "total_chunks": int,
                "chunking_mode": "toc_driven" | "fallback_sliding_window",
                "toc_outline": str,
            }
        """
        doc_id = doc_id or str(uuid.uuid4())
        file_hash = self._compute_hash(file_path)

        logger.info(f"[ingest] doc_id={doc_id} file={file_path}")

        try:
            if await self.vector_store.doc_exists(doc_id):
                logger.info(f"[ingest] duplicate doc_id={doc_id}, skip upsert")
                return {
                    "doc_id": doc_id,
                    "status": "duplicate",
                    "total_chunks": 0,
                }

            # ① 上传原始文件到 Blob Storage
            filename = Path(file_path).name
            blob_key = f"{getattr(self.blob_storage, 'prefix', '')}" \
                       f"{doc_id}/{filename}"
            blob_url = ""
            if not isinstance(self.blob_storage, _NullBlobStorage):
                logger.info(f"[ingest] step=blob_upload key={blob_key}")
                try:
                    blob_url = await self.blob_storage.upload(file_path, blob_key)
                    logger.info(f"[ingest] blob_url={blob_url}")
                except Exception as exc:
                    logger.warning(f"[ingest] blob upload failed (non-fatal): {exc}")

            # ② 解析文档
            logger.info("[ingest] step=parsing")
            output_dir = f"/tmp/parse_{doc_id}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            parsed_doc = self.format_router.parse(file_path, output_dir)
            parsed_doc.doc_id = doc_id
            full_text = parsed_doc.raw_text

            # ② 提取 TOC
            logger.info("[ingest] step=toc_extraction")
            toc_tree: Optional[TOCTree] = None
            toc_valid = False
            try:
                toc_tree = self.toc_extractor.extract_toc(
                    full_text,
                    doc_id,
                    parsed_doc=parsed_doc,
                )
                toc_valid = toc_tree.total_nodes >= 2
                logger.info(
                    f"[ingest] toc_nodes={toc_tree.total_nodes} depth={toc_tree.max_depth}"
                )
            except Exception as exc:
                logger.warning(f"[ingest] TOC extraction failed: {exc}, falling back to sliding window")

            # ③ 分块
            logger.info(f"[ingest] step=chunking mode={'toc' if toc_valid else 'fallback'}")
            chunks: list[Chunk]
            if toc_valid and toc_tree is not None:
                chunks = self.chunker.chunk_document(
                    full_text=full_text,
                    toc_tree=toc_tree,
                    parsed_doc=parsed_doc,
                )
            else:
                chunks = self.chunker.fallback_chunk(full_text, doc_id)

            logger.info(f"[ingest] total_chunks={len(chunks)}")

            # ③.5 上传抽取图片到 Image Blob Storage（独立 container）
            if not isinstance(self.image_blob_storage, _NullBlobStorage):
                logger.info("[ingest] step=image_blob_upload")
                img_prefix = getattr(self.image_blob_storage, "prefix", "images/")
                for chunk in chunks:
                    if chunk.image_path and Path(chunk.image_path).exists():
                        img_name = Path(chunk.image_path).name
                        img_key = f"{img_prefix}{doc_id}/{img_name}"
                        try:
                            img_url = await self.image_blob_storage.upload(chunk.image_path, img_key)
                            chunk.metadata["image_blob_url"] = img_url
                            chunk.metadata["image_blob_key"] = img_key
                            logger.debug(f"[ingest] image uploaded: {img_key}")
                        except Exception as exc:
                            logger.warning(
                                f"[ingest] image blob upload failed (non-fatal) "
                                f"img={chunk.image_path}: {exc}"
                            )

            # ③.6 将 chunking 后文本同步到外部 document_segments（可选，失败不阻断主流程）
            if self.segment_client.enabled:
                stats = await self.segment_client.push_chunks(
                    chunks=chunks,
                    document_id=doc_id,
                    source_type="file",
                    extra_keywords=kb_id,
                )
                logger.info(
                    "[ingest] step=segment_sync pushed=%s failed=%s",
                    stats["pushed"],
                    stats["failed"],
                )

            # ④ Embedding + 构建存储记录
            logger.info("[ingest] step=embedding")
            records: list[dict] = []

            for chunk in chunks:
                inputs = chunk.embedding_inputs
                embedding: list[float]

                if isinstance(self.embedder, VLLMEmbeddingClient):
                    if inputs.get("image"):
                        embedding = await self.embedder.embed_multimodal(
                            text=inputs["text"],
                            image_path=inputs.get("image"),
                        )
                    else:
                        embeddings = await self.embedder.embed_text([inputs["text"]])
                        embedding = embeddings[0]
                else:
                    # 本地 QwenMultiModalEmbedding（同步，在 executor 中运行）
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if inputs.get("image"):
                        emb_arr = await loop.run_in_executor(
                            None,
                            lambda: self.embedder.embed_multimodal(  # type: ignore[union-attr]
                                [inputs["text"]], [inputs.get("image")]
                            ),
                        )
                    else:
                        emb_arr = await loop.run_in_executor(
                            None,
                            lambda: self.embedder.embed_text([inputs["text"]]),  # type: ignore[union-attr]
                        )
                    embedding = emb_arr[0].tolist()

                records.append({
                    "chunk_id":      chunk.chunk_id,
                    "doc_id":        doc_id,
                    "chunk_type":    chunk.chunk_type.value,
                    "text_content":  chunk.text_content,
                    "section_path":  chunk.section_path,
                    "section_title": chunk.section_title,
                    "toc_node_id":   chunk.toc_node_id,
                    "page":          chunk.page or 0,
                    "start_page":    chunk.start_page if chunk.start_page is not None else (chunk.page or 0),
                    "end_page":      chunk.end_page if chunk.end_page is not None else (chunk.page or 0),
                    "file_name":     filename,
                    "blob_url":      blob_url,
                    "blob_key":      blob_key,
                    "embedding":     embedding,
                    "metadata_json": {
                        "depth":       chunk.depth,
                        "token_count": chunk.token_count,
                        "has_image":   chunk.image_path is not None,
                        "kb_id":       kb_id,
                        "user_id":     user_id,
                        "file_hash":   file_hash,
                        "blob_url":    blob_url,
                        "blob_key":    blob_key,
                        **chunk.metadata,
                    },
                })

            # ⑥ 写入向量数据库
            logger.info("[ingest] step=upsert_vectors")
            await self.vector_store.upsert_chunks(records)

            toc_outline = toc_tree.to_outline_str() if toc_tree else ""

            return {
                "doc_id":        doc_id,
                "status":        "success",
                "blob_url":      blob_url,
                "blob_key":      blob_key,
                "toc_nodes":     toc_tree.total_nodes if toc_tree else 0,
                "toc_depth":     toc_tree.max_depth if toc_tree else 0,
                "total_chunks":  len(chunks),
                "chunking_mode": "toc_driven" if toc_valid else "fallback_sliding_window",
                "toc_outline":   toc_outline,
            }

        except Exception as exc:
            logger.exception(f"[ingest] failed: {exc}")
            raise

    async def ingest_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        kb_id: str = "default",
        sourcefile: str = "text.txt",
        category: Optional[str] = None,
        url: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        """
        直接摄入纯文本（适用于 text_ingestion 接口）

        流程: 文本 → 分块（降级滑动窗口）→ Embedding → 写入向量库
        """
        doc_id = doc_id or str(uuid.uuid4())
        if await self.vector_store.doc_exists(doc_id):
            logger.info(f"[ingest_text] duplicate doc_id={doc_id}, skip upsert")
            return {
                "doc_id": doc_id,
                "status": "duplicate",
                "total_chunks": 0,
            }

        chunks = self.chunker.fallback_chunk(text, doc_id)

        logger.info(f"[ingest_text] doc_id={doc_id} chunks={len(chunks)}")

        # 将 chunking 后文本同步到外部 document_segments（可选，失败不阻断主流程）
        if self.segment_client.enabled:
            stats = await self.segment_client.push_chunks(
                chunks=chunks,
                document_id=doc_id,
                source_type="text",
                extra_keywords=category,
            )
            logger.info(
                "[ingest_text] step=segment_sync pushed=%s failed=%s",
                stats["pushed"],
                stats["failed"],
            )

        records: list[dict] = []
        for chunk in chunks:
            inputs = chunk.embedding_inputs
            if isinstance(self.embedder, VLLMEmbeddingClient):
                embeddings = await self.embedder.embed_text([inputs["text"]])
                embedding = embeddings[0]
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                emb_arr = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.embed_text([inputs["text"]]),  # type: ignore[union-attr]
                )
                embedding = emb_arr[0].tolist()

            records.append({
                "chunk_id":      chunk.chunk_id,
                "doc_id":        doc_id,
                "chunk_type":    chunk.chunk_type.value,
                "text_content":  chunk.text_content,
                "section_path":  chunk.section_path,
                "section_title": chunk.section_title,
                "toc_node_id":   chunk.toc_node_id,
                "page":          chunk.page or 0,
                "start_page":    chunk.start_page if chunk.start_page is not None else (chunk.page or 0),
                "end_page":      chunk.end_page if chunk.end_page is not None else (chunk.page or 0),
                "file_name":     sourcefile,
                "blob_url":      "",
                "blob_key":      "",
                "embedding":     embedding,
                "metadata_json": {
                    "sourcefile": sourcefile,
                    "category":   category,
                    "url":        url,
                    "kb_id":      kb_id,
                    "user_id":    user_id,
                    "token_count": chunk.token_count,
                },
            })

        await self.vector_store.upsert_chunks(records)

        return {
            "doc_id":       doc_id,
            "status":       "success",
            "total_chunks": len(chunks),
        }

    async def delete_doc(self, doc_id: str) -> None:
        """从向量库中删除文档的所有 chunks"""
        await self.vector_store.delete_doc(doc_id)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        image_path: Optional[str] = None,
    ) -> list[dict]:
        """从向量库中检索相关 chunks"""
        if isinstance(self.embedder, VLLMEmbeddingClient):
            if image_path:
                embedding = await self.embedder.embed_multimodal(query, image_path)
            else:
                embeddings = await self.embedder.embed_text([query])
                embedding = embeddings[0]
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            if image_path:
                emb_arr = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.embed_multimodal([query], [image_path]),  # type: ignore[union-attr]
                )
            else:
                emb_arr = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.embed_text([query]),  # type: ignore[union-attr]
                )
            embedding = emb_arr[0].tolist()

        return await self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
            filters=filters,
        )

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()
