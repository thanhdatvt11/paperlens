"""Application entry point for PaperLens."""

from fastapi import FastAPI

from app.controllers.harness_run_controller import (
    router as harness_run_router,
)
from app.controllers.version_controller import (
    router as version_router,
)
from app.controllers.document_controller import router as document_router
from app.controllers.health_controller import router as health_router
from app.controllers.harness_ui_controller import (
    router as harness_ui_router,
)

app = FastAPI(
    title="PaperLens API",
    description="Document review system with an AI agent harness.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(document_router)
app.include_router(harness_run_router)
app.include_router(harness_ui_router)
app.include_router(version_router)

from app.controllers.harness_agent_controller import (
    router as harness_agent_router,
)
app.include_router(harness_agent_router)