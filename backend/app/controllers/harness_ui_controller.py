"""Serve the local harness console and its assets."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter()

STATIC_DIR = (
    Path(__file__).resolve().parents[1] / "static"
).resolve()


@router.get("/harness", include_in_schema=False)
def harness_console():
    """Serve the harness interface."""
    path = STATIC_DIR / "harness.html"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="UI file not found.",
        )

    return FileResponse(path, media_type="text/html")


@router.get(
    "/harness-assets/{asset_path:path}",
    include_in_schema=False,
)
def harness_asset(asset_path: str):
    """Serve assets restricted to the static directory."""
    path = (STATIC_DIR / asset_path).resolve()

    if not path.is_relative_to(STATIC_DIR):
        raise HTTPException(
            status_code=404,
            detail="Asset not found.",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Asset not found.",
        )

    return FileResponse(path)