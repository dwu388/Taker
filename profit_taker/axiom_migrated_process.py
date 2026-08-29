from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import json_dumps, normalize_token_key
from .db import connect, migrate


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_identity(con, row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve canonical identity without silently merging conflicting mints.

    Full mints are preferred for new tokens. If an older short-address-only key
    already exists, a newly discovered full mint is attached to that same key so
    the lifecycle does not split. Any one-to-many mapping is rejected loudly.
    """
    token_address = _clean_text(row.get("token_address"))
    short_hint = _clean_text(row.get("short_address_hint"))
    proposed = _clean_text(row.get("token_key")) or normalize_token_key(
        row.get("name"), short_hint, token_address
    )
    if not proposed:
        return None, token_address

    if token_address:
        by_mint = con.execute(
            "SELECT DISTINCT token_key FROM axiom_observations WHERE token_address=? AND token_key IS NOT NULL",
            (token_address,),
        ).fetchall()
        mint_keys = {str(r[0]) for r in by_mint if r[0]}
        if len(mint_keys) > 1:
            raise RuntimeError(
                f"Identity collision: full mint {token_address} is already associated with multiple token keys {sorted(mint_keys)}"
            )
        if mint_keys:
            proposed = next(iter(mint_keys))

    if short_hint:
        rows = con.execute(
            """SELECT DISTINCT token_key,token_address FROM axiom_observations
               WHERE short_address_hint=? AND token_key IS NOT NULL""",
            (short_hint,),
        ).fetchall()
        prior_keys = {str(r[0]) for r in rows if r[0]}
        prior_mints = {str(r[1]) for r in rows if r[1]}
        if token_address and prior_mints and token_address not in prior_mints:
            raise RuntimeError(
                f"Identity collision: shortened address {short_hint} maps to existing mint(s) {sorted(prior_mints)} and incoming mint {token_address}"
            )
        if len(prior_mints) > 1:
            raise RuntimeError(
                f"Identity collision: shortened address {short_hint} maps to multiple full mints {sorted(prior_mints)}"
            )
        if len(prior_keys) == 1:
            # Preserve an existing legacy short-key lifecycle when a full mint is
            # learned later. The mint is still persisted and collision-checked.
            proposed = next(iter(prior_keys))
        elif len(prior_keys) > 1 and token_address:
            compatible = con.execute(
                "SELECT DISTINCT token_key FROM axiom_observations WHERE short_address_hint=? AND token_address=?",
                (short_hint, token_address),
            ).fetchall()
            compatible_keys = {str(r[0]) for r in compatible if r[0]}
            if len(compatible_keys) == 1:
                proposed = next(iter(compatible_keys))
            else:
                raise RuntimeError(
                    f"Identity collision: shortened address {short_hint} is already associated with multiple token keys {sorted(prior_keys)}"
                )

    existing_mints = con.execute(
        "SELECT DISTINCT token_address FROM axiom_observations WHERE token_key=? AND token_address IS NOT NULL",
        (proposed,),
    ).fetchall()
    known = {str(r[0]) for r in existing_mints if r[0]}
    if token_address and known and token_address not in known:
        raise RuntimeError(
            f"Identity collision: token key {proposed} maps to existing mint(s) {sorted(known)} and incoming mint {token_address}"
        )
    if len(known) > 1:
        raise RuntimeError(
            f"Identity collision: token key {proposed} already maps to multiple full mints {sorted(known)}"
        )
    return proposed, token_address


def process_rows(
    db_path: str,
    snapshot_at: str,
    source_path: str | None,
    rows: list[dict[str, Any]],
    clipboard_valid: bool = False,
    output_dir: str | Path = "data/axiom_migrated",
    *,
    screenshot_rows_detected: int | None = None,
) -> dict[str, Any]:
    """Persist raw clipboard observations without legacy V18 feature/visibility writes."""
    migrate(db_path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    try:
        detected_count = len(rows) if screenshot_rows_detected is None else int(screenshot_rows_detected)
        cur = con.execute(
            "INSERT INTO capture_cycles(captured_at,screenshot_path,clipboard_valid,rows_detected,completed) VALUES(?,?,?,?,1)",
            (snapshot_at, source_path, 1 if clipboard_valid else 0, detected_count),
        )
        cycle_id = int(cur.lastrowid)
        stored: list[dict[str, Any]] = []
        cols = [
            "cycle_id","token_key","token_address","name","short_address_hint","snapshot_at",
            "age_minutes","image_reuse_count","market_cap_usd","volume_usd","fees_sol","txns",
            "holders","pro_traders","kols","dev_migrations","dev_creations","recent_visitors",
            "top10_holders_pct","tracked_dev_status_raw","funding_time_raw","funding_time_minutes",
            "sniper_pct","insider_pct","bundler_pct","dex_paid","field_confidence_json",
            "raw_ocr_json","source_json",
        ]
        for row in rows:
            token_key, token_address = _resolve_identity(con, row)
            if not token_key:
                continue
            row["token_key"] = token_key
            row["token_address"] = token_address
            row["snapshot_at"] = snapshot_at
            source_payload = dict(row.get("source") or {})
            source_payload.setdefault("data_origin", row.get("data_origin") or "clipboard")
            source_payload.setdefault("identity", {})
            source_payload["identity"].update({
                "token_key": token_key,
                "token_address": token_address,
                "short_address_hint": row.get("short_address_hint"),
                "full_mint_verified": bool(token_address),
            })
            if row.get("training_eligible") is not None:
                source_payload.setdefault("training_eligible", bool(row.get("training_eligible")))
            row["source"] = source_payload
            vals = [
                cycle_id, token_key, token_address, row.get("name"), row.get("short_address_hint"), snapshot_at,
                row.get("age_minutes"), row.get("image_reuse_count"), row.get("market_cap_usd"), row.get("volume_usd"),
                row.get("fees_sol"), row.get("txns"), row.get("holders"), row.get("pro_traders"), row.get("kols"),
                row.get("dev_migrations"), row.get("dev_creations"), row.get("recent_visitors"), row.get("top10_holders_pct"),
                row.get("tracked_dev_status_raw"), row.get("funding_time_raw"), row.get("funding_time_minutes"),
                row.get("sniper_pct"), row.get("insider_pct"), row.get("bundler_pct"),
                None if row.get("dex_paid") is None else (1 if row.get("dex_paid") else 0),
                json_dumps(row.get("field_confidence", {})), json_dumps({}), json_dumps(source_payload),
            ]
            sql = f"INSERT OR IGNORE INTO axiom_observations({','.join(cols)}) VALUES({','.join('?' for _ in cols)})"
            con.execute(sql, vals)
            obs = con.execute(
                "SELECT * FROM axiom_observations WHERE token_key=? AND snapshot_at=?",
                (token_key, snapshot_at),
            ).fetchone()
            if obs:
                stored.append(dict(obs))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    stem = Path(source_path).stem if source_path else snapshot_at.replace(":", "-")
    json_path = outdir / f"{stem}.rows.json"
    csv_path = outdir / f"{stem}.rows.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    flat_fields = [
        "token_key","token_address","name","short_address_hint","data_origin","training_eligible","age_minutes",
        "market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols",
        "dev_migrations","dev_creations","recent_visitors","top10_holders_pct","funding_time_minutes",
        "sniper_pct","insider_pct","bundler_pct","dex_paid",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in flat_fields} for r in rows])
    return {
        "cycle_id": cycle_id,
        "rows_input": len(rows),
        "rows_stored": len(stored),
        "collection_mode": "clipboard_only",
        "json": str(json_path),
        "csv": str(csv_path),
    }
