from ingestion_pipeline import DocumentIngestionPipeline


class TextStrategy:
    def __init__(self, search_info, embeddings):
        self.search_info = search_info
        self.embeddings = embeddings
        self.pipeline = DocumentIngestionPipeline()

    async def add_text(self, text: str, sourcefile: str = "text.txt", category=None, url=None):
        await self.pipeline.ingest_text(
            text=text,
            doc_id=sourcefile,
            sourcefile=sourcefile,
            category=category,
            url=url,
        )
        return True

    async def remove_text_document(self, sourcefile: str):
        await self.pipeline.delete_doc(sourcefile)
        return True
