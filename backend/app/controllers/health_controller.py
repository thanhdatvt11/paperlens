"""Endpoints for checking API availability."""
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1",
    tags=["Health"],
)


@router.get("/health")
def check_health() -> dict[str, str]:
    """Return the API availability status."""
    return {
        "status": "ok",
        "service": "PaperLens API",
    }
