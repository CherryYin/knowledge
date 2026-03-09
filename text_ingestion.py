from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from api_models import TextDeleteResponse, TextUpsertRequest, TextUpsertResponse


router = APIRouter(prefix="/text-ingestion", tags=["text-ingestion"])


@lru_cache(maxsize=1)
def _get_local_pipeline():
    from ingestion_pipeline import DocumentIngestionPipeline
    return DocumentIngestionPipeline()


def _require_user_id(x_user_id: Optional[str]) -> str:
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")
    return user_id


@router.post(
    "/upsert",
    response_model=TextUpsertResponse,
    responses={500: {"description": "Text ingestion failed"}},
)
async def upsert(
    body: TextUpsertRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    user_id = _require_user_id(x_user_id)
    sourcefile = body.sourcefile or "text.txt"

    from ingestion_pipeline import build_doc_id

    doc_id = build_doc_id(user_id=user_id, file_name=sourcefile)
    pipeline = _get_local_pipeline()
    try:
        result = await pipeline.ingest_text(
            text=body.text,
            doc_id=doc_id,
            sourcefile=sourcefile,
            category=body.category,
            url=body.url,
            user_id=user_id,
        )
        return {"status": result.get("status", "ok"), **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {e}")


@router.put(
    "/update",
    response_model=TextUpsertResponse,
    responses={500: {"description": "Text update failed"}},
)
async def upsert_put(
    body: TextUpsertRequest,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not body.sourcefile or not body.sourcefile.strip():
        raise HTTPException(status_code=400, detail="sourcefile is required")

    user_id = _require_user_id(x_user_id)
    from ingestion_pipeline import build_doc_id

    doc_id = build_doc_id(user_id=user_id, file_name=body.sourcefile)
    pipeline = _get_local_pipeline()
    try:
        await pipeline.delete_doc(doc_id)
        result = await pipeline.ingest_text(
            text=body.text,
            doc_id=doc_id,
            sourcefile=body.sourcefile,
            category=body.category,
            url=body.url,
            user_id=user_id,
        )
        return {"status": result.get("status", "ok"), **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text update failed: {e}")


@router.delete(
    "/{doc_id}",
    response_model=TextDeleteResponse,
    responses={500: {"description": "Delete failed"}},
)
async def delete(doc_id: str):
    if not doc_id or not doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")

    try:
        await _get_local_pipeline().delete_doc(doc_id)
        return {"status": "ok", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
