from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import json_dumps, normalize_token_key
from .db import connect, migrate
from .axiom_features import compute_feature_row
from .visibility import update_visibility


def process_rows(
    db_path: str,
    snapshot_at: str,
    screenshot_path: str | None,
    rows: list[dict[str, Any]],
    clipboard_valid: bool = False,
    output_dir: str | Path = "data/axiom_migrated",
    *,
    screenshot_rows_detected: int | None = None,
) -> dict[str, Any]:
    migrate(db_path); outdir=Path(output_dir); outdir.mkdir(parents=True,exist_ok=True)
    con=connect(db_path)
    try:
        detected_count = len(rows) if screenshot_rows_detected is None else int(screenshot_rows_detected)
        cur=con.execute("INSERT INTO capture_cycles(captured_at,screenshot_path,clipboard_valid,rows_detected,completed) VALUES(?,?,?,?,1)",(snapshot_at,screenshot_path,1 if clipboard_valid else 0,detected_count))
        cycle_id=int(cur.lastrowid)
        stored=[]
        for row in rows:
            token_key=row.get("token_key") or normalize_token_key(row.get("name"),row.get("short_address_hint"),row.get("token_address"))
            if not token_key: continue
            row["token_key"]=token_key; row["snapshot_at"]=snapshot_at
            cols=["cycle_id","token_key","token_address","name","short_address_hint","snapshot_at","age_minutes","image_reuse_count","market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols","dev_migrations","dev_creations","recent_visitors","top10_holders_pct","tracked_dev_status_raw","funding_time_raw","funding_time_minutes","sniper_pct","insider_pct","bundler_pct","dex_paid","field_confidence_json","raw_ocr_json","source_json"]
            source_payload=dict(row.get("source") or {})
            if row.get("data_origin") is not None:
                source_payload.setdefault("data_origin", row.get("data_origin"))
            if row.get("training_eligible") is not None:
                source_payload.setdefault("training_eligible", bool(row.get("training_eligible")))
            row["source"] = source_payload
            vals=[cycle_id,token_key,row.get("token_address"),row.get("name"),row.get("short_address_hint"),snapshot_at,row.get("age_minutes"),row.get("image_reuse_count"),row.get("market_cap_usd"),row.get("volume_usd"),row.get("fees_sol"),row.get("txns"),row.get("holders"),row.get("pro_traders"),row.get("kols"),row.get("dev_migrations"),row.get("dev_creations"),row.get("recent_visitors"),row.get("top10_holders_pct"),row.get("tracked_dev_status_raw"),row.get("funding_time_raw"),row.get("funding_time_minutes"),row.get("sniper_pct"),row.get("insider_pct"),row.get("bundler_pct"),None if row.get("dex_paid") is None else (1 if row.get("dex_paid") else 0),json_dumps(row.get("field_confidence",{})),json_dumps(row.get("ocr_diagnostics",{})),json_dumps(source_payload)]
            sql=f"INSERT OR IGNORE INTO axiom_observations({','.join(cols)}) VALUES({','.join('?' for _ in cols)})"
            con.execute(sql,vals)
            obs=con.execute("SELECT * FROM axiom_observations WHERE token_key=? AND snapshot_at=?",(token_key,snapshot_at)).fetchone()
            if not obs: continue
            history=[dict(r) for r in con.execute("SELECT * FROM axiom_observations WHERE token_key=? AND snapshot_at<=? ORDER BY snapshot_at",(token_key,snapshot_at))]
            feat=compute_feature_row(dict(obs),history)
            con.execute("INSERT OR REPLACE INTO axiom_features_v18(observation_id,token_key,snapshot_at,feature_json) VALUES(?,?,?,?)",(obs["observation_id"],token_key,snapshot_at,json_dumps(feat)))
            stored.append(dict(obs))
        con.commit()
    finally: con.close()
    for row in stored: update_visibility(db_path,cycle_id,row["token_key"],snapshot_at)

    stem=Path(screenshot_path).stem if screenshot_path else snapshot_at.replace(":","-")
    json_path=outdir/f"{stem}.rows.json"; csv_path=outdir/f"{stem}.rows.csv"
    json_path.write_text(json.dumps(rows,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    flat_fields=["token_key","name","short_address_hint","data_origin","training_eligible","age_minutes","market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols","dev_migrations","dev_creations","recent_visitors","top10_holders_pct","funding_time_minutes","sniper_pct","insider_pct","bundler_pct","dex_paid"]
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=flat_fields); w.writeheader(); w.writerows([{k:r.get(k) for k in flat_fields} for r in rows])
    origin_counts: dict[str, int] = {}
    for row in rows:
        origin = str((row.get("source") or {}).get("data_origin") or "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    return {
        "cycle_id": cycle_id,
        "rows_input": len(rows),
        "screenshot_rows_detected": detected_count,
        "rows_stored": len(stored),
        "origin_counts": origin_counts,
        "json": str(json_path),
        "csv": str(csv_path),
    }
