"""
数据库表结构定义（统一维护）

当前包含:
  - doc_chunks: 文档切片向量表
"""

DOC_CHUNKS_TABLE = "doc_chunks"

ALL_TABLES = [
    DOC_CHUNKS_TABLE,
]

INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT        NOT NULL,
    chunk_type    TEXT        NOT NULL,
    text_content  TEXT        NOT NULL DEFAULT '',
    section_path  TEXT        NOT NULL DEFAULT '',
    section_title TEXT        NOT NULL DEFAULT '',
    toc_node_id   TEXT        NOT NULL DEFAULT '',
    page          INTEGER     NOT NULL DEFAULT 0,
    start_page    INTEGER     NOT NULL DEFAULT 0,
    end_page      INTEGER     NOT NULL DEFAULT 0,
    file_name     TEXT        NOT NULL DEFAULT '',
    blob_url      TEXT        NOT NULL DEFAULT '',
    blob_key      TEXT        NOT NULL DEFAULT '',
    embedding     vector(1536),
    metadata_json JSONB       NOT NULL DEFAULT '{}'
);

ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS start_page INTEGER NOT NULL DEFAULT 0;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS end_page INTEGER NOT NULL DEFAULT 0;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS file_name TEXT NOT NULL DEFAULT '';
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS blob_url TEXT NOT NULL DEFAULT '';
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS blob_key TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id
    ON doc_chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_section_path
    ON doc_chunks (section_path text_pattern_ops);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON doc_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 256);
"""

UPSERT_DOC_CHUNKS_SQL = """
INSERT INTO doc_chunks
       (chunk_id, doc_id, chunk_type, text_content,
        section_path, section_title, toc_node_id,
        page, start_page, end_page, file_name, blob_url, blob_key,
        embedding, metadata_json)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id        = EXCLUDED.doc_id,
    chunk_type    = EXCLUDED.chunk_type,
    text_content  = EXCLUDED.text_content,
    section_path  = EXCLUDED.section_path,
    section_title = EXCLUDED.section_title,
    toc_node_id   = EXCLUDED.toc_node_id,
    page          = EXCLUDED.page,
    start_page    = EXCLUDED.start_page,
    end_page      = EXCLUDED.end_page,
    file_name     = EXCLUDED.file_name,
    blob_url      = EXCLUDED.blob_url,
    blob_key      = EXCLUDED.blob_key,
    embedding     = EXCLUDED.embedding,
    metadata_json = EXCLUDED.metadata_json;
"""
