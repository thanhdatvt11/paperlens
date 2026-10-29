"""add whole-document review runs and chunk snapshots

Revision ID: d2e3f4a5b6c7
Revises: c1a2b3c4d5e6
Create Date: 2026-10-27

"""

# Alembic requires these module names and dynamically supplies op methods.
# pylint: disable=invalid-name,no-member

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d2e3f4a5b6c7"
down_revision: Union[str, Sequence[str], None] = "c1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Extend runs and create durable chunk snapshots."""
    with op.batch_alter_table("review_runs", recreate="always") as batch:
        batch.alter_column(
            "chunk_id", existing_type=sa.Integer(), nullable=True
        )
        batch.add_column(
            sa.Column(
                "job_type",
                sa.String(length=30),
                server_default="CHUNK_PREVIEW",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "stage",
                sa.String(length=30),
                server_default="QUEUED",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "total_chunks",
                sa.Integer(),
                server_default="1",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "processed_chunks",
                sa.Integer(),
                server_default="0",
                nullable=False,
            )
        )

    op.create_table(
        "review_run_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("review_run_id", sa.Integer(), nullable=False),
        sa.Column("source_chunk_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["review_run_id"], ["review_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "review_run_id",
            "chunk_index",
            name="uq_review_run_chunks_run_index",
        ),
    )
    op.create_index(
        "ix_review_run_chunks_review_run_id",
        "review_run_chunks",
        ["review_run_id"],
    )


def downgrade() -> None:
    """Remove snapshots and whole-document run fields."""
    op.drop_index(
        "ix_review_run_chunks_review_run_id",
        table_name="review_run_chunks",
    )
    op.drop_table("review_run_chunks")
    with op.batch_alter_table("review_runs", recreate="always") as batch:
        batch.drop_column("processed_chunks")
        batch.drop_column("total_chunks")
        batch.drop_column("stage")
        batch.drop_column("job_type")
        batch.alter_column(
            "chunk_id", existing_type=sa.Integer(), nullable=False
        )
