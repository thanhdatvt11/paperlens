"""Enqueue page-image jobs in the same durable FIFO as existing reviews."""

import json

from app.core.config import get_settings
from app.models.document import Document, ReviewRun, ReviewJobType
from app.models.vision_review import VisionReviewRun, VisionReviewPage
from app.repositories import review_run_repository, version_repository
from app.repositories import vision_review_repository as repository
from app.services import vision_pdf_service
from app.services.restored_review_queue_service import ReviewTargetNotFoundError
from app.services.vision_contract import CONTRACT_VERSION


def create_version(db, document_id, upload):
    with db.begin():
        if db.get(Document, document_id) is None:
            raise ReviewTargetNotFoundError("Document was not found")
    processed = vision_pdf_service.store_upload(upload)
    try:
        with db.begin():
            version = version_repository.create_version(
                db, document_id, processed.original_filename, processed.storage_key,
                processed.sha256, processed.page_count, [],
            )
    except Exception:
        vision_pdf_service.pdf_path(processed.storage_key).unlink(missing_ok=True)
        raise
    return version, processed


def enqueue(db, document_id, version_id, payload):
    # Detach source metadata before file I/O; no DB transaction during rendering
    # or provider calls. Existing chunks (including corrupt ones) are not used.
    with db.begin():
        version = version_repository.get_version_for_document(db, document_id, version_id)
        if version is None:
            raise ReviewTargetNotFoundError("Document/version was not found")
        key, digest, count = version.storage_key, version.sha256, version.page_count
    selected = payload.pages or list(range(1, count + 1))
    if any(page > count for page in selected):
        raise ValueError("pages must refer to positions in the PDF file")
    with vision_pdf_service.read_pdf(key, digest, count):
        pass
    with db.begin():
        # Recheck after I/O; reject an altered version rather than mixing sources.
        db.expire_all()
        version = version_repository.get_version_for_document(db, document_id, version_id)
        if version is None or (version.storage_key, version.sha256, version.page_count) != (key, digest, count):
            raise ValueError("Version changed while queuing the review")
        run = review_run_repository.create_run(
            db, document_id, version_id, payload.instruction,
            payload.timeout_seconds, ReviewJobType.VISION_DOCUMENT_REVIEW, [],
        )
        info = VisionReviewRun(
            run_id=run.id, storage_key=key, source_sha256=digest,
            source_page_count=count, model=get_settings().ollama_vision_model,
            selected_pages_json=json.dumps(selected), contract_version=CONTRACT_VERSION,
            summary_cache_json="{}",
        )
        db.add(info)
        db.flush()
        db.add_all(VisionReviewPage(run_id=run.id, page_number=page) for page in selected)
        run_id = run.id
    return {"run_id": run_id, "status": "QUEUED"}


def resume(db, run_id):
    with db.begin():
        run = db.get(ReviewRun, run_id)
        info = repository.get_info(db, run_id)
        if run is None:
            raise ReviewTargetNotFoundError("Review run was not found")
        if info is None:
            raise ValueError("This endpoint resumes vision jobs only")
        if info.contract_version != CONTRACT_VERSION:
            raise ValueError("Saved review contract differs; create a new vision run")
        repository.resume(db, run_id)
    return {"run_id": run_id, "status": "QUEUED"}


def progress(db, run_id):
    info = repository.get_info(db, run_id)
    if info is None:
        return {}
    rows = repository.pages(db, run_id)
    return {
        "total_pages": len(rows),
        "processed_pages": sum(row.status == "COMPLETED" for row in rows),
        "reviewed_pages": sum(bool(row.review_json) for row in rows),
        "checked_pages": sum(bool(row.check_json) for row in rows),
    }


def page_results(db, run_id):
    run = db.get(ReviewRun, run_id)
    if run is None:
        raise ReviewTargetNotFoundError("Review run was not found")
    if repository.get_info(db, run_id) is None:
        raise ValueError("This endpoint returns vision page results only")
    return [{
        "page_number": row.page_number, "status": row.status,
        "review": json.loads(row.review_json) if row.review_json else None,
        "check": json.loads(row.check_json) if row.check_json else None,
        "error": row.error_message, "semantic_correctness_verified": False,
        "evidence_visually_verified": False,
    } for row in repository.pages(db, run_id)]

