from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class FileDeleteRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)


class FileUploadResultItem(BaseModel):
    filename: str
    doc_id: Optional[str] = None
    status: Optional[str] = None
    blob_url: Optional[str] = None
    blob_key: Optional[str] = None
    toc_nodes: Optional[int] = None
    toc_depth: Optional[int] = None
    total_chunks: Optional[int] = None
    chunking_mode: Optional[str] = None
    toc_outline: Optional[str] = None


class FileUploadResponse(BaseModel):
    count: int
    results: list[FileUploadResultItem] = Field(default_factory=list)


class FileDeleteResponse(BaseModel):
    status: str
    doc_id: str


class TextUpsertRequest(BaseModel):
    text: str
    sourcefile: Optional[str] = "text.txt"
    category: Optional[str] = None
    url: Optional[str] = None


class TextUpsertResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    total_chunks: Optional[int] = None


class TextDeleteResponse(BaseModel):
    status: str
    doc_id: str


class FAQUpsertRequest(BaseModel):
    id: int
    question: str
    answer: str
    markets: Optional[str] = ""


class FAQReindexRequest(BaseModel):
    items: list[FAQUpsertRequest]


class FAQIdResponse(BaseModel):
    id: int


class FAQReindexResponse(BaseModel):
    count: int


class ErrorResponse(BaseModel):
    detail: str


def build_pydantic_json_schemas() -> dict[str, dict[str, Any]]:
    """导出核心 API Pydantic 模型的 JSON Schema。"""
    models: list[type[BaseModel]] = [
        HealthResponse,
        FileDeleteRequest,
        FileUploadResultItem,
        FileUploadResponse,
        FileDeleteResponse,
        TextUpsertRequest,
        TextUpsertResponse,
        TextDeleteResponse,
        FAQUpsertRequest,
        FAQReindexRequest,
        FAQIdResponse,
        FAQReindexResponse,
        ErrorResponse,
    ]
    return {m.__name__: m.model_json_schema() for m in models}
