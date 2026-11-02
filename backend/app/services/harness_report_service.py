"""Prepare report previews and enforce export approval."""

import hashlib
import json

from sqlalchemy.orm import Session

from app.harness.actions import ReviewReport
from app.harness.config import AgentConfig
from app.harness.permissions import check_tool_permission
from app.repositories import harness_run_repository as repository
from app.services.harness_run_service import get_run


def prepare_export(
    db: Session,
    run_id: int,
) -> dict:
    """Prepare a stable preview without exporting a file."""
    run = get_run(
        db=db,
        run_id=run_id,
    )

    if run.status != "COMPLETED" or not run.result_json:
        raise RuntimeError(
            "Run must be COMPLETED before report export."
        )

    config = AgentConfig.model_validate_json(
        run.agent_config_json
    )

    decision = check_tool_permission(
        config,
        "save_report",
    )

    if decision.effect == "DENY":
        raise PermissionError(decision.reason)

    result = json.loads(run.result_json)

    if not isinstance(result, dict) or "report" not in result:
        raise RuntimeError("Saved report is unavailable.")

    report = ReviewReport.model_validate(
        result["report"]
    )

    lines = [
        f"# PaperLens — Run {run.id}",
        "",
        "> Báo cáo do AI tạo; chưa xác minh "
        "tính chính xác của nội dung.",
        "",
        f"Document: {run.document_id} · "
        f"Version: {run.version_id}",
        "",
        "## Tóm tắt",
        "",
        report.summary,
        "",
        "## Nhận xét",
        "",
    ]

    if not report.findings:
        lines.append(
            "Model không trả finding trong báo cáo này."
        )

    for index, finding in enumerate(
        report.findings,
        start=1,
    ):
        lines.extend(
            [
                f"### {index}. Trang PDF "
                f"{finding.page_number} — "
                f"{finding.severity}",
                "",
                f"Vấn đề: {finding.problem}",
                "",
                f"Bằng chứng: {finding.evidence}",
                "",
                f"Đề xuất: {finding.suggestion}",
                "",
            ]
        )

    content = "\n".join(lines) + "\n"

    content_sha256 = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return {
        "run_id": run.id,
        "filename": f"harness-run-{run.id}.md",
        "content": content,
        "content_sha256": content_sha256,
        "permission": decision.effect,
        "semantic_correctness_verified": False,
    }


def decide_export(
    db: Session,
    run_id: int,
    decision: str,
    content_sha256: str,
) -> dict:
    """Record approval or rejection of the exact preview."""
    if decision not in {"approve", "reject"}:
        raise ValueError("Invalid export decision.")

    with db.begin():
        preview = prepare_export(
            db=db,
            run_id=run_id,
        )

        if content_sha256 != preview["content_sha256"]:
            raise RuntimeError(
                "Report changed. Load a fresh preview "
                "before approval."
            )

        repository.append_export_event(
            db=db,
            run_id=run_id,
            event_name="tool_permission_checked",
            data={
                "tool": "save_report",
                "effect": preview["permission"],
                "decision": decision,
                "content_sha256": content_sha256,
            },
        )

        repository.append_export_event(
            db=db,
            run_id=run_id,
            event_name=(
                "report_export_approved"
                if decision == "approve"
                else "report_export_rejected"
            ),
            data={
                "filename": preview["filename"],
                "content_sha256": content_sha256,
            },
        )

    return dict(
        preview,
        status=(
            "APPROVED"
            if decision == "approve"
            else "REJECTED"
        ),
    )