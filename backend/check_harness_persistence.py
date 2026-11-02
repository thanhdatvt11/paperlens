"""Check durable harness execution using real SQLite and Ollama."""

import json

from app.core.database import SESSION_FACTORY
from app.repositories import harness_run_repository
from app.services import harness_run_service


def main() -> None:
    with SESSION_FACTORY() as db:
        run = harness_run_service.create_run(
            db=db,
            agent_id=1,
            document_id=10,
            version_id=9,
            instruction=(
                "Xem trang PDF 18 và tóm tắt nội dung. "
                "Chỉ nêu vấn đề có bằng chứng nhìn thấy; "
                "chưa đủ bằng chứng thì findings rỗng."
            ),
            pages=[18],
        )

        run_id = run.id

    print("CREATED RUN:", run_id, flush=True)

    try:
        result = harness_run_service.execute_run(
            run_id=run_id,
            on_event=lambda event: print(
                json.dumps(event, ensure_ascii=False),
                flush=True,
            ),
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        with SESSION_FACTORY() as db:
            saved = harness_run_repository.get_run(
                db=db,
                run_id=run_id,
            )

            events = harness_run_repository.list_events(
                db=db,
                run_id=run_id,
            )

            print("SAVED STATUS:", saved.status)
            print("SAVED MODEL CALLS:", saved.step_count)
            print("SAVED EVENT COUNT:", len(events))


if __name__ == "__main__":
    main()