import sqlite3
import traceback

import pytest


@pytest.fixture
def closed_sqlite_connections(monkeypatch):
    """Retain handles so garbage collection cannot conceal missing close calls."""
    original = sqlite3.connect
    opened = []

    def tracked_connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append((conn, traceback.extract_stack(limit=5)))
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    yield
    leaks = []
    for conn, stack in opened:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue
        else:
            leaks.append("".join(traceback.format_list(stack)))
        finally:
            # Release retained handles after measuring, including on test failure.
            conn.close()
    assert not leaks, "SQLite connections were not explicitly closed:\n" + "\n".join(leaks)
