"""Configure the SQLite database and database sessions."""

from pathlib import Path
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BACKEND_DIR / "paperlens.db"

database_url = URL.create(
    drivername="sqlite",
    database=str(DATABASE_PATH),
)

engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False},
)

SESSION_FACTORY = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

def get_db() -> Generator[Session, None, None]:
    """Provide a database session and close it after use."""
    with SESSION_FACTORY() as session:
        yield session

class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Base class for database models."""
