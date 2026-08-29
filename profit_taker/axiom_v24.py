"""Final production-readiness facade for V24.

The policy-cfg fix and full statistical implementation remain preserved in
:mod:`profit_taker.axiom_v24_base` / ``axiom_v24_impl``.  This layer closes
operational-feature, sequence-vintage, sealed-audit and canonical-DB gaps found in
the final repository audit.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import axiom_v24_base as _base
from .db import RAW_DB_DEFAULT

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

_impl = _base._impl
MODEL_DB_DEFAULT = RAW_DB_DEFAULT


# ---------------------------------------------------------------------------
# Keep collection/provenance metadata out of the market model.
# ---------------------------------------------------------------------------
_OPERATIONAL_EXACT = {
    "observation_id", "cycle_id", "capture_id", "attempt_id", "session_id",
    "created_at", "first_ingested_at", "last_corrected_at", "value_version",
    "birth_ordinal", "assigned_at",
}
_OPERATIONAL_FRAGMENTS = (
    "observation_id", "cycle_id", "capture_id", "attempt_id", "session_id",
    "raw_payload", "payload_sha", "collector_schema", "ingestion_provenance",
    "data_vintage_hash", "row_fingerprint",
)


def _safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_fragments = (
        "label", "future", "target", "terminal", "path_end", "learning_updated",
        "fingerprint", "next_substantial_peak", "later_higher_peak", "recurrent_",
        "event_", "interval_", "calendar_cohort", "lifetime_id", "decision_market_cap", "hit_plus",
    )
    blocked_exact = {
        "token_key", "snapshot_at", "decision_at", "schema_version", "config_json",
        "label_status_next_peak", "next_substantial_peak_at", "next_substantial_peak_confirmed_at",
        "later_higher_peak_at", "label_ready_at",
    } | _OPERATIONAL_EXACT
    cols: list[str] = []
    for c in frame.columns:
        lc = str(c).lower()
        if c in blocked_exact or any(x in lc for x in blocked_fragments):
            continue
        if any(x in lc for x in _OPERATIONAL_FRAGMENTS):
            continue
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().sum() >= 5:
            frame[c] = s
            cols.append(c)
    return sorted(set(cols))


_impl._safe_feature_columns = _safe_feature_columns


# ---------------------------------------------------------------------------
# Late historical inserts must invalidate all later sequence states for a token.
# ---------------------------------------------------------------------------
_original_sequence_refresh = _impl.refresh_sequence_fingerprint_cache


def _invalidate_sequence_cache_for_late_insertions(
    conn: sqlite3.Connection, observations: pd.DataFrame
) -> dict[str, int]:
    if observations.empty or not _table_exists(conn, SEQUENCE_CACHE_TABLE):
        return {"tokens": 0, "rows": 0}
    invalidated_tokens = 0
    invalidated_rows = 0
    for token, g in observations.groupby("token_key", sort=False):
        cached = pd.read_sql_query(
            f"SELECT snapshot_at FROM {SEQUENCE_CACHE_TABLE} WHERE token_key=? ORDER BY snapshot_at",
            conn,
            params=(str(token),),
        )
        if cached.empty:
            continue
        cached_times = set(pd.to_datetime(cached.snapshot_at, utc=True, errors="coerce").dropna())
        if not cached_times:
            continue
        last_cached = max(cached_times)
        observed_times = pd.to_datetime(g.snapshot_at, utc=True, errors="coerce").dropna()
        missing_historical = sorted(
            t for t in observed_times if t <= last_cached and t not in cached_times
        )
        if not missing_historical:
            continue
        dirty = missing_historical[0]
        conn.execute(
            f"DELETE FROM {SEQUENCE_CACHE_TABLE} WHERE token_key=? AND snapshot_at>=?",
            (str(token), _iso(dirty)),
        )
        changed = int(conn.execute("SELECT changes()").fetchone()[0])
        invalidated_rows += changed
        invalidated_tokens += 1
    if invalidated_rows:
        conn.commit()
    return {"tokens": invalidated_tokens, "rows": invalidated_rows}


def refresh_sequence_fingerprint_cache(
    conn: sqlite3.Connection,
    observations: pd.DataFrame,
    cfg: V24Config,
    force: bool = False,
) -> dict[str, int]:
    pre = {"tokens": 0, "rows": 0}
    if not force:
        pre = _invalidate_sequence_cache_for_late_insertions(conn, observations)
    out = _original_sequence_refresh(conn, observations, cfg, force=force)
    out["late_insert_tokens_invalidated"] = int(pre["tokens"])
    out["late_insert_rows_invalidated"] = int(pre["rows"])
    return out


_impl.refresh_sequence_fingerprint_cache = refresh_sequence_fingerprint_cache


# ---------------------------------------------------------------------------
# Sealed audit: token-first membership, confirmation-safe truth, token weights.
# ---------------------------------------------------------------------------
def _mature_audit_cohorts(conn: sqlite3.Connection, cfg: V24Config) -> pd.DataFrame:
    latest = _latest_capture(conn)
    if latest is None:
        return pd.DataFrame()
    # A token born near a cohort end can remain visible for ~72h; its late-life
    # decisions then need another 72h outcome horizon. The extra audit delay is
    # applied only after that complete prospective window.
    unlock_before = (
        latest
        - pd.Timedelta(minutes=2 * cfg.horizon_minutes)
        - pd.Timedelta(days=cfg.audit_min_age_days)
    )
    return pd.read_sql_query(
        f"SELECT cohort_id,ordinal,start_at,end_at,status FROM {COHORT_TABLE} "
        "WHERE role='audit' AND end_at<=? ORDER BY ordinal",
        conn,
        params=(unlock_before.isoformat(),),
    )


def audit_manifest(conn: sqlite3.Connection, cfg: V24Config, reveal: bool = False) -> dict[str, object]:
    migrate(conn)
    rows = pd.read_sql_query(
        f"SELECT cohort_id,ordinal,start_at,end_at,status FROM {COHORT_TABLE} WHERE role='audit' ORDER BY ordinal",
        conn,
    )
    if rows.empty:
        return {"sealed_audit_cohorts": 0, "mature_sealed_audit_cohorts": 0, "revealed": False}
    mature = _mature_audit_cohorts(conn, cfg)
    out: dict[str, object] = {
        "sealed_audit_cohorts": int(len(rows)),
        "mature_sealed_audit_cohorts": int(len(mature)),
        "revealed": bool(reveal),
        "maturity_rule": "cohort_end + 2x72h token-lifetime/outcome window + audit delay",
        "note": "Audit-born tokens remain sealed for their entire lifetime and are never development-eligible.",
    }
    if reveal:
        out["cohorts"] = rows.to_dict("records")
    return out


def evaluate_sealed_audit_stream(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, object]:
    migrate(conn)
    mature = _mature_audit_cohorts(conn, cfg)
    if mature.empty:
        return {"available": False, "reason": "no time-unlocked mature audit cohort"}
    mature_ids = set(mature.cohort_id.astype(str))

    assignments = pd.read_sql_query(
        f"SELECT token_key,forecast_cohort_id,forecast_role FROM {TOKEN_ASSIGNMENT_TABLE} "
        "WHERE forecast_role='audit'",
        conn,
    )
    if assignments.empty:
        return {"available": False, "reason": "no audit-born token assignments"}
    audit_tokens = set(
        assignments[assignments.forecast_cohort_id.astype(str).isin(mature_ids)].token_key.astype(str)
    )
    if not audit_tokens:
        return {"available": False, "reason": "no tokens belong to mature audit cohorts"}

    led = pd.read_sql_query(
        f"SELECT prediction_id,token_key,decision_at,generated_at,prediction_json "
        f"FROM {PREDICTION_LEDGER} WHERE provenance='live' "
        "AND ineligibility_reason IN ('sealed_audit_token','sealed_audit_cohort') ORDER BY generated_at",
        conn,
    )
    if led.empty:
        return {"available": False, "reason": "no prospective sealed-audit live predictions recorded"}
    led = led[led.token_key.astype(str).isin(audit_tokens)].copy()
    if led.empty:
        return {"available": False, "reason": "no live predictions for mature audit-born tokens"}
    led["decision_at"] = pd.to_datetime(led.decision_at, utc=True, errors="coerce")
    led["generated_at"] = pd.to_datetime(led.generated_at, utc=True, errors="coerce")
    # A rerun of a predictor at the same decision timestamp is not an independent
    # audit sample. Keep the first genuinely prospective prediction only.
    led = led.sort_values("generated_at").drop_duplicates(["token_key", "decision_at"], keep="first")

    labels = pd.read_sql_query(
        f"SELECT token_key,decision_at,next_substantial_peak_at,next_substantial_peak_confirmed_at,"
        f"terminal_at,terminal_reason,path_end_at,label_finalized FROM {peak.LABEL_TABLE}",
        conn,
    )
    if labels.empty:
        return {"available": False, "reason": "audit labels unavailable"}
    labels["decision_at"] = pd.to_datetime(labels.decision_at, utc=True, errors="coerce")
    data = led.merge(labels, on=["token_key", "decision_at"], how="inner")
    if data.empty:
        return {"available": False, "reason": "audit predictions do not yet have matching labels"}

    horizon = int(max(cfg.survival_bins_minutes))
    horizon_key = f"p_first_peak_by_{horizon}m"
    scored: list[dict[str, object]] = []
    for _, r in data.iterrows():
        event, event_min, known = first_competing_event(r, cfg)
        if not known and event_min < horizon:
            continue
        p = _finite(_loads(r.prediction_json).get(horizon_key))
        if p is None:
            continue
        y = 1.0 if event == EVENT_PEAK and event_min <= horizon else 0.0
        scored.append({"token_key": str(r.token_key), "p": float(np.clip(p, 0.0, 1.0)), "y": y})
    if not scored:
        return {"available": False, "reason": f"no usable {horizon_key} audit predictions"}

    score_df = pd.DataFrame(scored)
    score_df["brier"] = (score_df.p - score_df.y) ** 2
    token_brier = score_df.groupby("token_key").brier.mean()
    counts = score_df.token_key.value_counts()
    weights = np.asarray([1.0 / counts[str(t)] for t in score_df.token_key], dtype=float)
    weighted_log = None
    if score_df.y.nunique() > 1:
        weighted_log = float(
            log_loss(
                score_df.y.astype(int).to_numpy(),
                np.clip(score_df.p.to_numpy(dtype=float), 1e-6, 1 - 1e-6),
                labels=[0, 1],
                sample_weight=weights,
            )
        )
    metrics = {
        "rows": int(len(score_df)),
        "tokens": int(score_df.token_key.nunique()),
        "token_balanced_peak_brier": float(token_brier.mean()),
        "token_balanced_peak_log_loss": weighted_log,
        "horizon_probability": horizon_key,
        "audit_membership": "token_first_seen_lifetime",
    }
    conn.execute(
        f"INSERT INTO {AUDIT_RESULTS_TABLE}(audit_result_id,created_at,model_family,prediction_rows,metric_json,note) "
        "VALUES(?,?,?,?,?,?)",
        (
            str(uuid.uuid4()),
            _now_iso(),
            SCHEMA_VERSION,
            len(score_df),
            _json(metrics),
            "Prospective sealed audit; token-first membership; confirmation-safe truth; never development-eligible",
        ),
    )
    conn.commit()
    return {"available": True, **metrics}


_impl.audit_manifest = audit_manifest
_impl.evaluate_sealed_audit_stream = evaluate_sealed_audit_stream


# ---------------------------------------------------------------------------
# Canonical raw DB for direct CLI use, not only BAT wrappers.
# ---------------------------------------------------------------------------
def _argv_with_canonical_db(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if any(arg == "--db" or arg.startswith("--db=") for arg in argv):
        return argv
    return [argv[0], "--db", RAW_DB_DEFAULT, *argv[1:]]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return _impl.main(_argv_with_canonical_db(args))


# Preserve the already-fixed policy training entry point from the prior facade.
train_distributional_policy = _base.train_distributional_policy
_impl.train_distributional_policy = train_distributional_policy


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
