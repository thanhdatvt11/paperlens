"""Single-process FIFO worker for durable review runs.

Run separately from Uvicorn, for example:
``python -m app.review_worker``.
"""
import logging
from collections.abc import Callable
import json
import time
from typing import Any

from app.adapters.ollama_adapter import OllamaAdapter
from app.core.config import get_settings
from app.core.database import SESSION_FACTORY
from app.models.document import ReviewJobType
from app.repositories import review_run_repository
from app.services import ai_service, document_review_service


class ReviewWorker:
    """Process exactly one queued review at a time in FIFO order."""

    def __init__(
        self,
        session_factory: Callable[..., Any] = SESSION_FACTORY,
        adapter_factory: Callable[[], OllamaAdapter] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._session_factory = session_factory
        self._adapter_factory = adapter_factory or self._default_adapter
        self._poll_interval = poll_interval

    @staticmethod
    def _default_adapter() -> OllamaAdapter:
        """Create the configured local Ollama adapter."""
        return OllamaAdapter(get_settings())

    def recover_interrupted_runs(self) -> int:
        """Fail RUNNING jobs left by a stopped worker process."""
        with self._session_factory() as db:
            with db.begin():
                return review_run_repository.recover_running_runs(
                    db,
                    "Worker restarted while this review was RUNNING.",
                )

    def process_one(self) -> bool:
        """Claim and finish one job, returning whether work was found."""
        with self._session_factory() as db:
            with db.begin():
                run = review_run_repository.claim_next_queued(db)
            if run is None:
                return False
            job = {
                "run_id": run.id,
                "document_id": run.document_id,
                "version_id": run.version_id,
                "chunk_id": run.chunk_id,
                "instruction": run.instruction,
                "job_type": run.job_type,
                "timeout_seconds": run.timeout_seconds,
            }

        try:
            started = time.monotonic()
            result = self._run_review(job)
            elapsed = time.monotonic() - started
            if elapsed > job["timeout_seconds"]:
                raise TimeoutError("review job exceeded its timeout")
            self._complete(job["run_id"], result)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._fail(job["run_id"], _error_message(exc))
        return True

    def _run_review(self, job: dict[str, Any]) -> dict[str, Any]:
        """Run the model call with a fresh session.

        The claim transaction is already closed before this method runs.
        """
        adapter = self._adapter_factory()
        if job["job_type"] == ReviewJobType.DOCUMENT_REVIEW.value:
            return document_review_service.review_document_run(
                session_factory=self._session_factory,
                run_id=job["run_id"],
                document_id=job["document_id"],
                version_id=job["version_id"],
                instruction=job["instruction"],
                adapter=adapter,
            )

        contexts = document_review_service.load_snapshot_contexts(
            self._session_factory, job["run_id"]
        )
        if len(contexts) != 1:
            raise RuntimeError("chunk preview run must contain one snapshot")
        context = contexts[0]
        chunk_context = ai_service.ChunkReviewContext(
            document_id=job["document_id"],
            version_id=job["version_id"],
            chunk_id=context.chunk_id,
            page_number=context.page_number,
            text=context.text,
        )
        result = ai_service.review_chunk_context(
            context=chunk_context,
            adapter=adapter,
            instruction=job["instruction"],
        )
        with self._session_factory() as db:
            with db.begin():
                review_run_repository.complete_chunk(
                    db=db,
                    run_id=job["run_id"],
                    snapshot_id=context.snapshot_id,
                    result_json=json.dumps(result, ensure_ascii=False),
                )
        return result

    def _complete(self, run_id: int, result: dict[str, Any]) -> None:
        """Persist a successful result in a short transaction."""
        with self._session_factory() as db:
            with db.begin():
                review_run_repository.complete_run(
                    db, run_id, json.dumps(result, ensure_ascii=False)
                )

    def _fail(self, run_id: int, error_message: str) -> None:
        """Persist a failure in a short transaction."""
        with self._session_factory() as db:
            with db.begin():
                review_run_repository.fail_run(db, run_id, error_message)

    def run_forever(self) -> None:
        """Recover interrupted jobs and poll the durable queue forever."""
        self.recover_interrupted_runs()
        while True:
            if not self.process_one():
                time.sleep(self._poll_interval)


def _error_message(error: Exception) -> str:
    """Return a safe frontend error without a traceback."""
    message = str(error).strip()
    cause = str(error.__cause__).strip() if error.__cause__ else ""
    if cause and cause not in message:
        message = f"{message}: {cause}" if message else cause
    return (
        f"{type(error).__name__}: {message}"
        if message
        else type(error).__name__
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ReviewWorker().run_forever()
