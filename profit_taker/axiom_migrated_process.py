"""Production-hardened raw persistence facade.

The previously validated implementation is preserved in
:mod:`profit_taker.axiom_migrated_process_base`. This facade makes silent
``INSERT OR IGNORE`` loss impossible and requires a single schema-compatible
collection session whose purpose matches the capture provenance.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from . import axiom_migrated_process_base as _impl
from .db import COLLECTOR_SCHEMA_VERSION

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

PRODUCTION_COLLECTION_PURPOSE = "v24_production_raw_collection"
REPLAY_COLLECTION_PURPOSE = "v24_replay_experiment"


class _StrictObservationConnection(sqlite3.Connection):
    """Convert the legacy observation INSERT OR IGNORE into a hard INSERT."""

    def execute(self, sql: str, parameters: Any = (), /):  # type: ignore[override]
        stripped = sql.lstrip()
        prefix = "INSERT OR IGNORE INTO axiom_observations"
        if stripped.startswith(prefix):
            leading = sql[: len(sql) - len(stripped)]
            stripped = stripped.replace(prefix, "INSERT INTO axiom_observations", 1)
            sql = leading + stripped
        return super().execute(sql, parameters)


def _strict_connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30.0, factory=_StrictObservationConnection)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=FULL")
    return con


def _expected_purpose(attempt_source: str) -> str:
    return REPLAY_COLLECTION_PURPOSE if str(attempt_source) == "replay_file" else PRODUCTION_COLLECTION_PURPOSE


def _require_single_collection_session(db_path: str | Path, attempt_source: str) -> None:
    migrate(db_path)
    con = _strict_connect(db_path)
    try:
        rows = con.execute(
            "SELECT session_id,purpose,collector_schema FROM collection_sessions ORDER BY started_at, created_at"
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError(
                "Capture requires exactly one initialized collection session; "
                f"found {len(rows)}. Start through collection_admin/init or the supported launcher."
            )
        row = rows[0]
        if str(row["collector_schema"]) != COLLECTOR_SCHEMA_VERSION:
            raise RuntimeError(
                "Capture refuses a collection session created by a different collector schema: "
                f"stored={row['collector_schema']!r}, required={COLLECTOR_SCHEMA_VERSION!r}"
            )
        required_purpose = _expected_purpose(attempt_source)
        if str(row["purpose"]) != required_purpose:
            raise RuntimeError(
                "Capture provenance does not match the collection-session purpose: "
                f"source={attempt_source!r}, stored_purpose={row['purpose']!r}, required_purpose={required_purpose!r}"
            )
    finally:
        con.close()


def process_rows(
    db_path: str,
    snapshot_at: str,
    source_path: str | None,
    rows: list[dict[str, Any]],
    clipboard_valid: bool = False,
    output_dir: str | Path = "data/axiom_migrated",
    *,
    screenshot_rows_detected: int | None = None,
    raw_clipboard_text: str | None = None,
    attempt_started_at: str | None = None,
    attempt_source: str = "interactive_clipboard",
) -> dict[str, Any]:
    """Persist one complete capture or roll back the entire capture."""
    detected = len(rows) if screenshot_rows_detected is None else int(screenshot_rows_detected)
    if detected != len(rows):
        raise RuntimeError(
            f"Capture row-count invariant failed before persistence: detected={detected}, parsed={len(rows)}"
        )
    if not clipboard_valid:
        raise RuntimeError("Persistence refuses a capture not marked clipboard-valid")
    if attempt_source not in {"interactive_clipboard", "replay_file"}:
        raise RuntimeError(f"Unsupported capture provenance: {attempt_source!r}")
    bad_identity = [i for i, row in enumerate(rows) if not row.get("token_key")]
    if bad_identity:
        raise RuntimeError(f"Persistence received rows without token identity: {bad_identity}")
    bad_mc = [
        i for i, row in enumerate(rows)
        if row.get("market_cap_usd") is None or float(row.get("market_cap_usd") or 0.0) <= 0.0
    ]
    if bad_mc:
        raise RuntimeError(f"Persistence received rows without positive market cap: {bad_mc}")

    _require_single_collection_session(db_path, attempt_source)
    # The preserved implementation performs one SQLite transaction for cycle,
    # payload, observations and success-attempt. Patch its connection factory so
    # duplicate observation keys raise and roll the whole transaction back.
    _impl.connect = _strict_connect
    result = _impl.process_rows(
        db_path,
        snapshot_at,
        source_path,
        rows,
        clipboard_valid,
        output_dir,
        screenshot_rows_detected=detected,
        raw_clipboard_text=raw_clipboard_text,
        attempt_started_at=attempt_started_at,
        attempt_source=attempt_source,
    )
    if int(result.get("rows_inserted", -1)) != len(rows) or int(result.get("rows_stored", -1)) != len(rows):
        raise RuntimeError(
            "Post-persistence row-count invariant failed; the capture is not trustworthy: "
            f"parsed={len(rows)}, inserted={result.get('rows_inserted')}, stored={result.get('rows_stored')}"
        )
    return result


_impl.connect = _strict_connect
record_failed_capture_attempt = _impl.record_failed_capture_attempt
