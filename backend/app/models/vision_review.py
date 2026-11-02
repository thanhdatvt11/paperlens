"""Page checkpoints for vision jobs; existing review tables stay unchanged."""

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class VisionReviewRun(Base):
    __tablename__ = "vision_review_runs"

    run_id: Mapped[int] = mapped_column(
        ForeignKey("review_runs.id"), primary_key=True
    )
    storage_key: Mapped[str] = mapped_column(String(255))
    source_sha256: Mapped[str] = mapped_column(String(64))
    source_page_count: Mapped[int] = mapped_column()
    model: Mapped[str] = mapped_column(String(255))
    selected_pages_json: Mapped[str] = mapped_column(Text)
    contract_version: Mapped[str] = mapped_column(String(80))
    summary_cache_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class VisionReviewPage(Base):
    __tablename__ = "vision_review_pages"
    __table_args__ = (
        UniqueConstraint("run_id", "page_number", name="uq_vision_run_page"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("vision_review_runs.run_id"), index=True
    )
    page_number: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String(20), default="QUEUED")
    review_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    check_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

