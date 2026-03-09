import os
import tempfile
from pathlib import Path

from ingestion_pipeline import DocumentIngestionPipeline


class UploadUserFileStrategy:
    def __init__(self, search_info, file_processors, embeddings):
        self.search_info = search_info
        self.file_processors = file_processors
        self.embeddings = embeddings
        self.pipeline = DocumentIngestionPipeline()

    async def add_file(self, file_obj):
        content = file_obj.content
        filename = getattr(content, "name", None) or "upload.bin"
        suffix = Path(filename).suffix or ".bin"

        if hasattr(content, "read"):
            payload = content.read()
        else:
            payload = bytes(content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            doc_id = filename
            return await self.pipeline.ingest(tmp_path, doc_id=doc_id)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def remove_file(self, filename: str, _unused: str = ""):
        await self.pipeline.delete_doc(filename)
        return True
