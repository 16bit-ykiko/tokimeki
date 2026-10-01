import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from tokimeki.library.schema import MIGRATIONS


class SchemaTooNewError(RuntimeError):
    pass


def open_library(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) a series library and bring its schema up to date."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    row: tuple[int] = conn.execute("PRAGMA user_version").fetchone()
    return row[0]


def migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    if current > len(MIGRATIONS):
        raise SchemaTooNewError(
            f"library schema v{current} is newer than this tokimeki (v{len(MIGRATIONS)})"
        )
    if current == len(MIGRATIONS):
        return
    # Rebuilding a table must not fire ON DELETE actions, and the pragma is ignored inside
    # a transaction, so foreign keys are off for the migration and checked afterwards.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for version, script in enumerate(MIGRATIONS[current:], start=current + 1):
            try:
                conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;")
            except sqlite3.Error:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise sqlite3.IntegrityError("foreign key violations after migrating the library")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
