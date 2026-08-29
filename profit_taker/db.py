from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = r'''
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS capture_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
'''


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
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
