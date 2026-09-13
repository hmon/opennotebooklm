import atexit
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import DATABASE_URL

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def _configure(conn) -> None:
    """Run once per pooled connection: teach psycopg the pgvector type."""
    register_vector(conn)


# check_connection validates a connection before handing it out, so a database
# restart costs one reconnect instead of failing every request holding a dead
# handle. Queries here are interleaved with slow model calls, so connections sit
# idle long enough for that to matter.
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=8,
    open=False,
    configure=_configure,
    check=ConnectionPool.check_connection,
    kwargs={"row_factory": dict_row},
)


def init() -> None:
    """Open the pool and apply schema.sql (idempotent)."""
    pool.open()
    with pool.connection() as conn:
        conn.execute(SCHEMA_PATH.read_text())
        conn.commit()
    # The vector type only exists after schema.sql ran, so any connection
    # opened before that has to be configured again.
    pool.check()
    atexit.register(_close)


def _close() -> None:
    try:
        pool.close()
    except Exception:  # noqa: BLE001 - best-effort shutdown
        pass


def connection():
    return pool.connection()


def fetchall(sql: str, params: tuple | dict = ()) -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetchone(sql: str, params: tuple | dict = ()) -> dict | None:
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchone()
