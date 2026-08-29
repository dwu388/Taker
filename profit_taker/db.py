from __future__ import annotations

import sqlite3
from pathlib import Path

RAW_DB_DEFAULT = "data/axiom_v24_raw.sqlite"
COLLECTOR_SCHEMA_VERSION = "v24_clipboard_raw_v2"

SCHEMA = r'''
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS collection_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    purpose TEXT NOT NULL,
    collector_schema TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS capture_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    captured_at TEXT NOT NULL,
    screenshot_path TEXT,
    clipboard_valid INTEGER NOT NULL DEFAULT 0,
    rows_detected INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS axiom_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    token_key TEXT NOT NULL,
    token_address TEXT,
    name TEXT,
    short_address_hint TEXT,
    snapshot_at TEXT NOT NULL,
    age_minutes INTEGER,
    image_reuse_count INTEGER,
    market_cap_usd REAL,
    volume_usd REAL,
    fees_sol REAL,
    txns INTEGER,
    holders INTEGER,
    pro_traders INTEGER,
    kols INTEGER,
    dev_migrations INTEGER,
    dev_creations INTEGER,
    recent_visitors INTEGER,
    top10_holders_pct REAL,
    tracked_dev_status_raw TEXT,
    funding_time_raw TEXT,
    funding_time_minutes INTEGER,
    sniper_pct REAL,
    insider_pct REAL,
    bundler_pct REAL,
    dex_paid INTEGER,
    field_confidence_json TEXT,
    raw_ocr_json TEXT,
    source_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(token_key, snapshot_at),
    FOREIGN KEY(cycle_id) REFERENCES capture_cycles(cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_axiom_obs_token_time ON axiom_observations(token_key, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_axiom_obs_time ON axiom_observations(snapshot_at);

CREATE TABLE IF NOT EXISTS capture_payloads (
    cycle_id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    compression TEXT NOT NULL,
    payload BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(cycle_id) REFERENCES capture_cycles(cycle_id)
);

CREATE TABLE IF NOT EXISTS capture_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    success INTEGER NOT NULL,
    clipboard_valid INTEGER NOT NULL DEFAULT 0,
    rows_detected INTEGER NOT NULL DEFAULT 0,
    cycle_id INTEGER,
    source TEXT,
    error_type TEXT,
    error_message TEXT,
    raw_payload_sha256 TEXT,
    raw_payload_bytes INTEGER,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(cycle_id) REFERENCES capture_cycles(cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_capture_attempts_time ON capture_attempts(started_at);
CREATE INDEX IF NOT EXISTS idx_capture_attempts_success ON capture_attempts(success, started_at);

CREATE TABLE IF NOT EXISTS capture_attempt_payloads (
    attempt_id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    compression TEXT NOT NULL,
    payload BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(attempt_id) REFERENCES capture_attempts(attempt_id)
);
'''


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=FULL")
    return con


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def _prepare_legacy_name_collisions(con: sqlite3.Connection) -> None:
    for table, identity in (("axiom_observations", "observation_id"), ("capture_cycles", "cycle_id")):
        cols = _table_columns(con, table)
        if cols and identity not in cols:
            base = f"legacy_{table}_pre_v24"
            name = base
            i = 2
            existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            while name in existing:
                name = f"{base}_{i}"
                i += 1
            con.execute(f"ALTER TABLE {table} RENAME TO {name}")


def _ensure_columns(con: sqlite3.Connection) -> None:
    additions = {
        "capture_cycles": {
            "session_id": "TEXT",
            "screenshot_path": "TEXT", "clipboard_valid": "INTEGER NOT NULL DEFAULT 0",
            "rows_detected": "INTEGER NOT NULL DEFAULT 0", "completed": "INTEGER NOT NULL DEFAULT 1",
        },
        "axiom_observations": {
            "cycle_id": "INTEGER", "token_address": "TEXT", "name": "TEXT", "short_address_hint": "TEXT",
            "age_minutes": "INTEGER", "image_reuse_count": "INTEGER", "market_cap_usd": "REAL",
            "volume_usd": "REAL", "fees_sol": "REAL", "txns": "INTEGER", "holders": "INTEGER",
            "pro_traders": "INTEGER", "kols": "INTEGER", "dev_migrations": "INTEGER",
            "dev_creations": "INTEGER", "recent_visitors": "INTEGER", "top10_holders_pct": "REAL",
            "tracked_dev_status_raw": "TEXT", "funding_time_raw": "TEXT", "funding_time_minutes": "INTEGER",
            "sniper_pct": "REAL", "insider_pct": "REAL", "bundler_pct": "REAL", "dex_paid": "INTEGER",
            "field_confidence_json": "TEXT", "raw_ocr_json": "TEXT", "source_json": "TEXT",
        },
        "capture_attempts": {
            "session_id": "TEXT", "source": "TEXT", "error_type": "TEXT", "error_message": "TEXT",
            "raw_payload_sha256": "TEXT", "raw_payload_bytes": "INTEGER", "details_json": "TEXT",
        },
    }
    for table, cols in additions.items():
        existing = _table_columns(con, table)
        if not existing:
            continue
        for name, decl in cols.items():
            if name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def migrate(db_path: str | Path) -> None:
    con = connect(db_path)
    try:
        _prepare_legacy_name_collisions(con)
        con.commit()
        try:
            con.executescript(SCHEMA)
        except sqlite3.OperationalError:
            con.rollback()
            _ensure_columns(con)
            con.commit()
            con.executescript(SCHEMA)
        _ensure_columns(con)
        con.commit()
    finally:
        con.close()
