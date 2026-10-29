"""add durable review runs

Revision ID: c1a2b3c4d5e6
Revises: 7c0d8f5c5a1e
Create Date: 2026-10-27

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "7c0d8f5c5a1e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the durable FIFO review queue table."""
    op.create_table(
        "review_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.id"]),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_runs_document_id", "review_runs", ["document_id"]
    )
    op.create_index("ix_review_runs_version_id", "review_runs", ["version_id"])
    op.create_index("ix_review_runs_chunk_id", "review_runs", ["chunk_id"])
    op.create_index("ix_review_runs_status", "review_runs", ["status"])


def downgrade() -> None:
    """Drop the durable FIFO review queue table."""
    op.drop_index("ix_review_runs_status", table_name="review_runs")
    op.drop_index("ix_review_runs_chunk_id", table_name="review_runs")
    op.drop_index("ix_review_runs_version_id", table_name="review_runs")
    op.drop_index("ix_review_runs_document_id", table_name="review_runs")
    op.drop_table("review_runs")
