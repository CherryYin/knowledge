import os
import tempfile
import uuid
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, UploadFile, File as FastAPIFile

from api_models import FileDeleteRequest, FileDeleteResponse, FileUploadResponse


router = APIRouter(prefix="/file-ingestion", tags=["file-ingestion"])

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
    "/upload",
    response_model=FileUploadResponse,
    responses={500: {"description": "Ingestion failed"}},
)
async def upload(
    files: List[UploadFile] = FastAPIFile(...),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    user_id = _require_user_id(x_user_id)
    from ingestion_pipeline import build_doc_id

    pipeline = _get_local_pipeline()
    results = []
    for upload_file in files:
        filename = upload_file.filename or f"upload_{uuid.uuid4().hex}.bin"
        data = await upload_file.read()
        suffix = os.path.splitext(filename)[-1] or ".bin"
        doc_id = build_doc_id(user_id=user_id, file_name=filename)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            result = await pipeline.ingest(
                tmp_path,
                doc_id=doc_id,
                user_id=user_id,
            )
            results.append({"filename": filename, **result})
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Local pipeline ingestion failed for {filename!r}: {e}",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return {"count": len(files), "results": results}


@router.delete(
    "/delete",
    response_model=FileDeleteResponse,
    responses={500: {"description": "Delete failed"}},
)
async def delete_file(body: FileDeleteRequest):
    if not body.doc_id or not body.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")

    pipeline = _get_local_pipeline()
    try:
        await pipeline.delete_doc(body.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return {"status": "ok", "doc_id": body.doc_id}
