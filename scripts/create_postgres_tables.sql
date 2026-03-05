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