"""Database models for harness runs and execution events."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class HarnessRun(Base):
    """Store one harness execution and its input snapshots."""

    __tablename__ = "harness_runs"

    __table_args__ = (
        CheckConstraint(
            "status IN "
            "('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')",
            name="ck_harness_runs_status",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "harness_agents.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("document_versions.id"),
        nullable=False,
        index=True,
    )

    instruction: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    pages_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    agent_config_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="QUEUED",
    )

    step_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
    )

    result_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )


class HarnessRunEvent(Base):
    """Store one event without image bytes."""

    __tablename__ = "harness_run_events"

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "sequence",
            name="uq_harness_event_run_sequence",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[int] = mapped_column(
        ForeignKey("harness_runs.id"),
        nullable=False,
        index=True,
    )

    sequence: Mapped[int] = mapped_column(
        nullable=False,
    )

    event: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    payload_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        nullable=False,
    )