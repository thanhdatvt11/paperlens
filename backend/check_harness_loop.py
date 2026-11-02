"""Run one real-model harness check."""

import json

from app.core.database import SESSION_FACTORY
from app.harness.loop import run_harness
from app.harness.pdf_tools import load_pdf_source
from app.services.harness_agent_service import (
    get_agent_config,
)


def show_event(event) -> None:
    """Print progress without image bytes."""
    print(
        json.dumps(event, ensure_ascii=False),
        flush=True,
    )


def main() -> None:
    """Review one page with the stored agent configuration."""
    with SESSION_FACTORY() as db:
        config = get_agent_config(
            db=db,
            agent_id=1,
        )

        source = load_pdf_source(
            db=db,
            document_id=10,
            version_id=9,
        )

    result = run_harness(
        config=config,
        source=source,
        instruction=(
            "Xem trang PDF 18. "
            "Tóm tắt các bước thao tác được trình bày. "
            "Đánh giá độ rõ ràng và tính nhất quán "
            "của nội dung hiện có. "
            "Không yêu cầu bổ sung công cụ ngoài phạm vi trang. "
            "Chỉ nêu lỗi cụ thể có bằng chứng nhìn thấy; "
            "không đủ bằng chứng thì findings rỗng."
        ),
        pages=[18],
        on_event=show_event,
    )

    print("\nRESULT:")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()