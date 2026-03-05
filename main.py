import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_models import HealthResponse, build_pydantic_json_schemas
from config import settings
from file_ingestion import router as file_ingestion_router
from faq_ingestion import router as faq_ingestion_router
from text_ingestion import router as text_ingestion_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="knowledgebase",
        description="知识库文档摄入与检索 API",
        version="1.0.0",
        docs_url="/swagger",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    if settings.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allow_origins,
            allow_credentials=settings.allow_credentials,
            allow_methods=["*"] ,
            allow_headers=["*"],
        )

    app.include_router(file_ingestion_router)
    app.include_router(faq_ingestion_router)
    app.include_router(text_ingestion_router)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return {"status": "ok"}

    @app.get("/api-docs/openapi.json", tags=["api-docs"])
    async def api_openapi_json():
        return app.openapi()

    @app.get("/api-docs/json-schema", tags=["api-docs"])
    async def api_json_schema():
        return {
            "openapi": app.openapi(),
            "pydantic_schemas": build_pydantic_json_schemas(),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
