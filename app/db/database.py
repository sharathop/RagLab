import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Generator

# Load .env as early as possible — this module's engine is created at
# import time (see `engine = get_engine()` below), so DATABASE_URL must be
# in the real environment before that line runs. load_dotenv() only sets
# variables that aren't already set, so it's safe to call from multiple
# entry points (uvicorn, alembic, tests) without conflicting.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback-aware engine creation
def get_engine():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("DATABASE_URL is not configured")

    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        connect_args={"connect_timeout": 5}
    )

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator:
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_db_connection() -> bool:
    """Check if the database connection is alive."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception:
        return False
