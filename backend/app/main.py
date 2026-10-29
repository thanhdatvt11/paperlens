"""Application entry point for PaperLens."""

from fastapi import FastAPI

from app.controllers.document_controller import router as document_router
from app.controllers.health_controller import router as health_router

app = FastAPI(
    title="PaperLens API",
    description="Document review system with an AI agent harness.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(document_router)

# Restored version and review endpoints.
from app.controllers.restored_document_controller import router as _restored_document_router
app.include_router(_restored_document_router)
