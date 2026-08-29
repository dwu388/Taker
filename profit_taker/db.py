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

CREATE TABLE IF NOT EXISTS axiom_features_v18 (
    observation_id INTEGER PRIMARY KEY,
    token_key TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(observation_id) REFERENCES axiom_observations(observation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_features_token_time ON axiom_features_v18(token_key, snapshot_at);

CREATE TABLE IF NOT EXISTS axiom_visibility (
    token_key TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    capture_count INTEGER NOT NULL DEFAULT 1,
    consecutive_capture_count INTEGER NOT NULL DEFAULT 1,
    max_consecutive_capture_count INTEGER NOT NULL DEFAULT 1,
    reappearance_count INTEGER NOT NULL DEFAULT 0,
    visibility_bonus REAL NOT NULL DEFAULT 0.0,
    last_cycle_id INTEGER
);

CREATE TABLE IF NOT EXISTS axiom_labels_24h_v18 (
    observation_id INTEGER PRIMARY KEY,
    token_key TEXT NOT NULL,
    decision_time TEXT NOT NULL,
    label_status TEXT NOT NULL,
    terminal_reason TEXT,
    terminal_time TEXT,
    target_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(observation_id) REFERENCES axiom_observations(observation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_labels_status ON axiom_labels_24h_v18(label_status);

CREATE TABLE IF NOT EXISTS axiom_api_handoff_queue (
    token_key TEXT PRIMARY KEY,
    token_address TEXT,
    snapshot_at TEXT NOT NULL,
    model_probability REAL,
    visibility_bonus REAL NOT NULL DEFAULT 0.0,
    handoff_priority REAL,
    state TEXT NOT NULL,
    prediction_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallet_history_snapshots_v12 (
    wallet TEXT NOT NULL,
    as_of TEXT NOT NULL,
    mature_exit_count INTEGER NOT NULL DEFAULT 0,
    winning_exit_count INTEGER NOT NULL DEFAULT 0,
    losing_exit_count INTEGER NOT NULL DEFAULT 0,
    win_rate REAL,
    total_realized_pnl_usd REAL,
    median_realized_pnl_usd REAL,
    mean_roi REAL,
    median_roi REAL,
    distinct_tokens INTEGER,
    median_exit_quality REAL,
    median_holding_minutes REAL,
    total_cost_basis_usd REAL,
    total_proceeds_usd REAL,
    PRIMARY KEY(wallet, as_of)
);

CREATE TABLE IF NOT EXISTS helius_token_observations (
    token_address TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    price_usd REAL,
    supply REAL,
    decimals INTEGER,
    mint_authority TEXT,
    freeze_authority TEXT,
    mint_authority_revoked INTEGER,
    freeze_authority_revoked INTEGER,
    top10_token_account_pct REAL,
    raw_json TEXT,
    PRIMARY KEY(token_address, observed_at)
);

CREATE TABLE IF NOT EXISTS normalized_trades (
    trade_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    signature TEXT,
    wallet TEXT NOT NULL,
    token_address TEXT NOT NULL,
    side TEXT NOT NULL,
    token_amount REAL,
    usd_value REAL,
    price_usd REAL,
    source TEXT NOT NULL,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_wallet_time ON normalized_trades(wallet, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_token_time ON normalized_trades(token_address, timestamp);

CREATE TABLE IF NOT EXISTS provider_usage (
    provider TEXT NOT NULL,
    usage_date TEXT NOT NULL,
    units REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(provider, usage_date)
);

CREATE TABLE IF NOT EXISTS wallet_backfill_queue (
    wallet TEXT PRIMARY KEY,
    priority REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '96h',
    state TEXT NOT NULL DEFAULT 'queued',
    reason TEXT,
    queued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shyft_wallet_prescreens (
    wallet TEXT NOT NULL,
    screened_at TEXT NOT NULL,
    transactions_seen INTEGER,
    swap_transactions INTEGER,
    swap_ratio REAL,
    oldest_at TEXT,
    newest_at TEXT,
    requests_used INTEGER,
    prescreen_score REAL,
    promoted INTEGER,
    raw_json TEXT,
    PRIMARY KEY(wallet, screened_at)
);

CREATE TABLE IF NOT EXISTS token_sync_state (
    token_address TEXT PRIMARY KEY,
    last_synced_at TEXT,
    history_backfilled_to TEXT
);
CREATE TABLE IF NOT EXISTS wallet_sync_state (
    wallet TEXT PRIMARY KEY,
    last_synced_at TEXT,
    history_backfilled_to TEXT
);
CREATE TABLE IF NOT EXISTS helius_wallet_sync_state (
    wallet TEXT PRIMARY KEY,
    last_synced_at TEXT
);
CREATE TABLE IF NOT EXISTS helius_token_sync_state (
    token_address TEXT PRIMARY KEY,
    last_synced_at TEXT
);
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
    # If a much older build used the same table name with a different identity column,
    # preserve it under a legacy name rather than destructively altering/replacing it.
    for table, identity in (("axiom_observations", "observation_id"), ("capture_cycles", "cycle_id")):
        cols = _table_columns(con, table)
        if cols and identity not in cols:
            base = f"legacy_{table}_pre_v18"
            name = base
            i = 2
            existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            while name in existing:
                name = f"{base}_{i}"; i += 1
            con.execute(f"ALTER TABLE {table} RENAME TO {name}")


def _ensure_columns(con: sqlite3.Connection) -> None:
    additions = {
        "capture_cycles": {
            "screenshot_path":"TEXT", "clipboard_valid":"INTEGER NOT NULL DEFAULT 0", "rows_detected":"INTEGER NOT NULL DEFAULT 0", "completed":"INTEGER NOT NULL DEFAULT 1"
        },
        "axiom_observations": {
            "cycle_id":"INTEGER", "token_address":"TEXT", "name":"TEXT", "short_address_hint":"TEXT", "age_minutes":"INTEGER",
            "image_reuse_count":"INTEGER", "market_cap_usd":"REAL", "volume_usd":"REAL", "fees_sol":"REAL", "txns":"INTEGER",
            "holders":"INTEGER", "pro_traders":"INTEGER", "kols":"INTEGER", "dev_migrations":"INTEGER", "dev_creations":"INTEGER",
            "recent_visitors":"INTEGER", "top10_holders_pct":"REAL", "tracked_dev_status_raw":"TEXT", "funding_time_raw":"TEXT",
            "funding_time_minutes":"INTEGER", "sniper_pct":"REAL", "insider_pct":"REAL", "bundler_pct":"REAL", "dex_paid":"INTEGER",
            "field_confidence_json":"TEXT", "raw_ocr_json":"TEXT", "source_json":"TEXT"
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
            # A same-named older table may be missing a newer indexed column.
            con.rollback()
            _ensure_columns(con)
            con.commit()
            con.executescript(SCHEMA)
        _ensure_columns(con)
        con.commit()
    finally:
        con.close()
