"""Version responses rebuilt for the restored upload/list API."""
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class VersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    document_id: int
    original_filename: str
    sha256: str
    page_count: int
    created_at: datetime


class UploadVersionResponse(VersionResponse):
    warning_pages: list[int] = Field(default_factory=list)
    chunk_count: int


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chunk_index: int
    page_number: int
    text: str
