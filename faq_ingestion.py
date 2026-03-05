import os
from functools import lru_cache
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from api_models import FAQIdResponse, FAQReindexRequest, FAQReindexResponse, FAQUpsertRequest


router = APIRouter(prefix="/faq-ingestion", tags=["faq-ingestion"])

@lru_cache
def _get_vector_db() -> Any:
    from kblib.vector_db import FAQVectorDB
    db_path = (os.getenv("FAQ_DB_PATH") or "./data/faq.db").strip()
    return FAQVectorDB(db_path=db_path)


@router.post(
    "/upsert",
    response_model=FAQIdResponse,
    responses={500: {"description": "FAQ upsert failed"}},
)
def upsert(body: FAQUpsertRequest):
    try:
        _get_vector_db().upsert_feedback(body.id, body.question, body.answer, markets=body.markets or "")
        return {"id": body.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ upsert failed: {e}")


@router.put(
    "/update",
    response_model=FAQIdResponse,
    responses={500: {"description": "FAQ update failed"}},
)
def upsert_put(body: FAQUpsertRequest):
    try:
        # Update semantics: delete old then insert new
        _get_vector_db().update_feedback(body.id, body.question, body.answer, markets=body.markets or "")
        return {"id": body.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ update failed: {e}")


@router.delete(
    "/{faq_id}",
    response_model=FAQIdResponse,
    responses={500: {"description": "FAQ delete failed"}},
)
def delete_faq(faq_id: int):
    try:
        _get_vector_db().delete_feedback(faq_id)
        return {"id": faq_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ delete failed: {e}")


@router.post(
    "/reindex",
    response_model=FAQReindexResponse,
    responses={500: {"description": "FAQ reindex failed"}},
)
def reindex(body: FAQReindexRequest):
    try:
        db = _get_vector_db()
        db.clear_all()
        db.reindex([it.model_dump() for it in body.items])
        return {"count": len(body.items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAQ reindex failed: {e}")
