"""Back up the existing database and create the agent table."""

from datetime import datetime
from pathlib import Path
import sqlite3

from app.models.harness_run import (
    HarnessRun,
    HarnessRunEvent,
)
from sqlalchemy import inspect

from app.core.database import engine
from app.models.harness_agent import HarnessAgent


def main() -> None:
    """Create only the harness_agents table."""
    database = engine.url.database
    if not database:
        raise RuntimeError("SQLite database path is missing.")

    database_path = Path(database).resolve()

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Existing database not found: {database_path}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = database_path.with_name(
        f"paperlens_before_harness_{timestamp}.db"
    )

    source = sqlite3.connect(
        database_path.as_uri() + "?mode=ro",
        uri=True,
    )
    backup = sqlite3.connect(str(backup_path))

    try:
        source.backup(backup)
    finally:
        backup.close()
        source.close()

    print(f"Database backup: {backup_path}")

    tables = (
        HarnessAgent.__table__,
        HarnessRun.__table__,
        HarnessRunEvent.__table__,
    )

    for table in tables:
        table.create(
            bind=engine,
            checkfirst=True,
        )

        columns = inspect(engine).get_columns(
            table.name
        )

        print(f"Table ready: {table.name}")
        print(
            "Columns:",
            [column["name"] for column in columns],
        )

    columns = inspect(engine).get_columns("harness_agents")

 

    engine.dispose()


if __name__ == "__main__":
    main()