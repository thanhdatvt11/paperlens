"""Vision page checkpoints, written only by the claimed worker."""

from datetime import datetime
import json

from sqlalchemy import select, update

from app.models.document import ReviewRun
from app.models.vision_review import VisionReviewPage, VisionReviewRun


def get_info(db, run_id):
    return db.get(VisionReviewRun, run_id)


def pages(db, run_id):
    return list(db.scalars(select(VisionReviewPage).where(
        VisionReviewPage.run_id == run_id
    ).order_by(VisionReviewPage.page_number)))


def assert_running(db, run_id):
    if db.scalar(select(ReviewRun.status).where(ReviewRun.id == run_id)) != "RUNNING":
        raise RuntimeError("Vision run is no longer claimed by a worker")


def save_review(db, run_id, page_number, content):
    assert_running(db, run_id)
    db.execute(update(VisionReviewPage).where(
        VisionReviewPage.run_id == run_id,
        VisionReviewPage.page_number == page_number,
    ).values(review_json=json.dumps(content, ensure_ascii=False),
             check_json=None, status="REVIEWED", error_message=None))


def save_check(db, run_id, page_number, content):
    assert_running(db, run_id)
    page = db.scalar(select(VisionReviewPage).where(
        VisionReviewPage.run_id == run_id,
        VisionReviewPage.page_number == page_number,
    ))
    if page is None or not page.review_json:
        raise RuntimeError("Cannot complete a page before saving its review")
    page.check_json = json.dumps(content, ensure_ascii=False)
    page.status = "COMPLETED"
    page.error_message = None


def fail_page(db, run_id, page_number, error):
    assert_running(db, run_id)
    db.execute(update(VisionReviewPage).where(
        VisionReviewPage.run_id == run_id,
        VisionReviewPage.page_number == page_number,
    ).values(status="FAILED", error_message=error))


def resume(db, run_id):
    result = db.execute(update(ReviewRun).where(
        ReviewRun.id == run_id, ReviewRun.job_type == "VISION_DOCUMENT_REVIEW",
        ReviewRun.status == "FAILED",
    ).values(status="QUEUED", stage="QUEUED", started_at=None,
             finished_at=None, error_message=None, result_json=None))
    if result.rowcount != 1:
        raise ValueError("Only a FAILED vision run can be resumed")
    # Successful page reviews and model checks are kept, including a saved
    # review whose second call failed. The worker resumes only the missing step.

