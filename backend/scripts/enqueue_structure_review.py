"""Create a new FIFO run using validated reviews from a completed source run.

Run from backend: python scripts/enqueue_structure_review.py --source-run-id 26
This does not modify the source run, call Ollama, or start another worker.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import SESSION_FACTORY  # noqa: E402
from app.models.document import ReviewJobType  # noqa: E402
from app.repositories import review_run_repository as repository  # noqa: E402
from app.services.document_review_service import (  # noqa: E402
    _load_cached_results,
    load_snapshot_contexts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.source_run_id < 1 or args.timeout_seconds < 1:
        parser.error("IDs and timeout must be positive")

    with SESSION_FACTORY() as db:
        with db.begin():
            source = repository.get_run(db, args.source_run_id)
            if source is None or source.job_type != "DOCUMENT_REVIEW" or source.status != "COMPLETED":
                raise RuntimeError("Source run must be a COMPLETED document review")
            document_id, version_id = source.document_id, source.version_id
            instruction, total = source.instruction, source.total_chunks

    contexts = load_snapshot_contexts(SESSION_FACTORY, args.source_run_id)
    cached = _load_cached_results(SESSION_FACTORY, args.source_run_id, document_id, version_id)
    if not contexts or len(contexts) != total or len(cached) != total:
        raise RuntimeError("Source run must contain a complete validated set of chunk results")

    snapshots = [repository.ChunkSnapshotInput(
        source_chunk_id=context.chunk_id, chunk_index=context.chunk_index,
        page_number=context.page_number, text=context.text,
    ) for context in contexts]

    with SESSION_FACTORY() as db:
        with db.begin():
            # No direct claim or status rewrite: the existing worker claims FIFO.
            run = repository.create_run(
                db=db, document_id=document_id, version_id=version_id,
                instruction=instruction, timeout_seconds=args.timeout_seconds,
                job_type=ReviewJobType.DOCUMENT_REVIEW, chunks=snapshots,
            )
            copies = repository.list_run_chunks(db, run.id)
            if len(copies) != total:
                raise RuntimeError("New run snapshot count differs from source")
            for snapshot in copies:
                repository.complete_chunk(
                    db=db, run_id=run.id, snapshot_id=snapshot.id,
                    result_json=json.dumps(cached[snapshot.source_chunk_id], ensure_ascii=False),
                )
            new_id = run.id
    print(f"New run_id: {new_id}")
    print(f"Reused {total} validated chunk reviews from run {args.source_run_id}.")
    print("Original run unchanged. One existing worker will analyze structure and summarize sections.")
    print(f"GET /api/v1/review-runs/{new_id}")


if __name__ == "__main__":
    main()
