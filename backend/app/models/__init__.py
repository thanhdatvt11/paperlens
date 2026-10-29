"""Expose database models and their shared metadata."""

from app.core.database import Base
from app.models.document import Document

__all__ = ["Base", "Document"]