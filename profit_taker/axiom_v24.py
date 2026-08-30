"""Final production-readiness facade for the 24-hour V24 lifecycle.

The retained implementation remains in :mod:`profit_taker.axiom_v24_base` /
``axiom_v24_impl``. This layer preserves the production-hardening contract while
specializing V24 to a 24-hour forecast lifecycle observed through a 25-hour Axiom
view. The final visible hour is collection buffer only.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import axiom_v24_base as _base
from . import axiom_peak_structure as peak
from .db import RAW_DB_DEFAULT

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

_impl = _base._impl
MODEL_DB_DEFAULT = RAW_DB_DEFAULT

PREDICTION_HORIZON_MINUTES = 24 * 60
AXIOM_VIEW_MINUTES = 25 * 60
AGE_OUT_GUARD_MINUTES = 24 * 60
RECURRENT_HORIZONS_MINUTES = (240, 480, 720, 1440)
SCHEMA_VERSION = "v24_lifetime_purged_adaptive_event_policy_24h_v2"


@dataclass
class V24Config(_impl.V24Config):
    """Production V24 configuration for a complete 24h token lifecycle."""

    horizon_minutes: int = PREDICTION_HORIZON_MINUTES
    operational_gap_minutes: float = 50.0
    age_out_minutes: float = float(AGE_OUT_GUARD_MINUTES)

    # Keep the conservative calendar/holdout machinery. Maturity now requires
    # 24h visible lifetime + 24h target outcome rather than 72h + 72h.
    promotion_purge_hours: int = 24

    survival_bins_minutes: tuple[int, ...] = (
        5, 15, 30, 60, 120, 240, 480, 720, 1440
    )
    probability_horizons_minutes: tuple[int, ...] = (60, 240, 480, 720, 1440)
    sequence_windows_minutes: tuple[int, ...] = (60, 240, 360, 720, 1440)
    promotion_required_horizons_minutes: tuple[int, ...] = (240, 480, 720, 1440)


# Every retained implementation path that creates/configures V24 must resolve the
# active 24h class, including the implementation CLI.
_impl.V24Config = V24Config
_base.V24Config = V24Config
_impl.SCHEMA_VERSION = SCHEMA_VERSION
_base.SCHEMA_VERSION = SCHEMA_VERSION


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
# 24h recurrent/marked-event grids. These replace the retained 72h literals while
# preserving confirmation-time semantics and direct marked higher-peak heads.
# ---------------------------------------------------------------------------
def add_recurrent_targets(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    cfg: V24Config,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    peaks = _impl._future_peak_lists(conn)
    out = frame.copy()
    horizons = tuple(h for h in RECURRENT_HORIZONS_MINUTES if h <= cfg.horizon_minutes)
    for h in horizons:
        out[f"recurrent_peak_count_{h}m"] = np.nan
        out[f"second_peak_by_{h}m"] = np.nan
        for margin in cfg.higher_peak_mark_margins:
            out[f"later_higher_{int(round(margin * 100))}pct_by_{h}m"] = np.nan
    for c in (
        "prior_confirmed_peak_count", "minutes_since_last_confirmed_peak",
        "last_confirmed_peak_multiple_vs_decision", "recurrent_next_gap_minutes",
        "recurrent_next_occurrence_gap_minutes", "recurrent_second_gap_minutes",
        "recurrent_next_peak_multiple", "recurrent_second_peak_relative_to_first",
        "recurrent_second_peak_is_higher",
    ):
        out[c] = np.nan

    for i, r in out.iterrows():
        decision = _utc(r.decision_at)
        known_end = _impl._known_followup_end(r, cfg, as_of)
        events = peaks.get(str(r.token_key), [])
        prior = [e for e in events if e["confirmed_at"] <= decision]
        if prior:
            last = prior[-1]
            out.at[i, "prior_confirmed_peak_count"] = float(len(prior))
            out.at[i, "minutes_since_last_confirmed_peak"] = (
                decision - last["confirmed_at"]
            ).total_seconds() / 60.0
            entry = float(r.decision_market_cap_usd) if _finite(r.get("decision_market_cap_usd")) else np.nan
            if np.isfinite(entry) and entry > 0:
                out.at[i, "last_confirmed_peak_multiple_vs_decision"] = last["peak_price"] / entry
        else:
            out.at[i, "prior_confirmed_peak_count"] = 0.0

        eligible = [
            e for e in events
            if decision < e["peak_at"] <= decision + pd.Timedelta(minutes=cfg.horizon_minutes)
            and decision < e["confirmed_at"] <= known_end
        ]
        eligible.sort(key=lambda e: e["peak_at"])
        if eligible:
            first = eligible[0]
            out.at[i, "recurrent_next_occurrence_gap_minutes"] = (
                first["peak_at"] - decision
            ).total_seconds() / 60.0
            out.at[i, "recurrent_next_gap_minutes"] = (
                first["confirmed_at"] - decision
            ).total_seconds() / 60.0
            entry = float(r.decision_market_cap_usd) if _finite(r.get("decision_market_cap_usd")) else np.nan
            if np.isfinite(entry) and entry > 0:
                out.at[i, "recurrent_next_peak_multiple"] = first["peak_price"] / entry
        if len(eligible) >= 2:
            first, second = eligible[0], eligible[1]
            out.at[i, "recurrent_second_gap_minutes"] = (
                second["confirmed_at"] - first["confirmed_at"]
            ).total_seconds() / 60.0
            out.at[i, "recurrent_second_peak_relative_to_first"] = (
                second["peak_price"] / first["peak_price"] - 1.0
            )
            out.at[i, "recurrent_second_peak_is_higher"] = float(
                second["peak_price"] > first["peak_price"] * (1 + cfg.higher_peak_margin_pct)
            )

        for h in horizons:
            deadline = decision + pd.Timedelta(minutes=h)
            inside = [
                e for e in events
                if decision < e["peak_at"] <= deadline
                and decision < e["confirmed_at"] <= deadline
                and e["confirmed_at"] <= known_end
            ]
            complete = known_end >= deadline or (
                bool(int(r.get("label_finalized", 0) or 0)) and as_of is None
            )
            if complete:
                out.at[i, f"recurrent_peak_count_{h}m"] = float(len(inside))
                out.at[i, f"second_peak_by_{h}m"] = float(len(inside) >= 2)
            elif len(inside) >= 2:
                out.at[i, f"second_peak_by_{h}m"] = 1.0
            if inside:
                first = inside[0]
                for margin in cfg.higher_peak_mark_margins:
                    col = f"later_higher_{int(round(margin * 100))}pct_by_{h}m"
                    hit = any(
                        e["peak_price"] > first["peak_price"] * (1.0 + margin)
                        for e in inside[1:]
                    )
                    if hit:
                        out.at[i, col] = 1.0
                    elif complete:
                        out.at[i, col] = 0.0
            elif complete:
                for margin in cfg.higher_peak_mark_margins:
                    out.at[i, f"later_higher_{int(round(margin * 100))}pct_by_{h}m"] = 0.0
    return out


def _higher_specs(cfg: V24Config) -> list[tuple[str, float, float]]:
    horizons = tuple(h for h in RECURRENT_HORIZONS_MINUTES if h <= cfg.horizon_minutes)
    return [
        (f"later_higher_{int(round(m * 100))}pct_by_{h}m", float(m), float(h))
        for m in cfg.higher_peak_mark_margins for h in horizons
    ]


def _derive_adapter_head_weights(adapter: dict, cfg: V24Config) -> dict[str, float]:
    out: dict[str, float] = {}
    base = adapter.get("weights") or {}
    for h in cfg.probability_horizons_minutes:
        horizon_scale = max(0.35, min(1.20, math.sqrt(480.0 / max(float(h), 60.0))))
        hz = float(base.get("hazard", 0.0)) * horizon_scale
        out[f"p_first_peak_by_{h}m"] = min(cfg.adapter_max_weight, hz)
        out[f"p_death_by_{h}m"] = min(cfg.adapter_max_weight, hz)
        out[f"p_event_free_by_{h}m"] = min(cfg.adapter_max_weight, hz)
        bf = float(base.get("short_barrier" if h <= 480 else "long_barrier", 0.0)) * horizon_scale
        for thr in cfg.upside_thresholds:
            out[f"p_hit_plus{_impl._threshold_tag(thr)}_by_{h}m"] = min(cfg.adapter_max_weight, bf)
    for h in (x for x in RECURRENT_HORIZONS_MINUTES if x <= cfg.horizon_minutes):
        hs = max(0.35, min(1.15, math.sqrt(480.0 / max(float(h), 240.0))))
        for m in cfg.higher_peak_mark_margins:
            out[f"p_later_higher_{int(round(m * 100))}pct_by_{h}m"] = min(
                cfg.adapter_max_weight, float(base.get("marked_higher", 0.0)) * hs
            )
        out[f"recurrent_peak_count_{h}m"] = min(
            cfg.adapter_max_weight, float(base.get("recurrent", 0.0)) * hs
        )
    for k in (
        "next_gap_q25", "next_gap_q50", "next_gap_q75", "next_peak_multiple_q25",
        "next_peak_multiple_q50", "next_peak_multiple_q75", "second_gap_q50",
        "second_peak_relative_q50",
    ):
        out[k] = float(base.get("recurrent", 0.0))
    return out


def project_recurrent_outputs(pred: dict[str, np.ndarray]) -> None:
    counts = [f"recurrent_peak_count_{h}m" for h in RECURRENT_HORIZONS_MINUTES if f"recurrent_peak_count_{h}m" in pred]
    if counts:
        mat = np.column_stack([np.clip(pred[k], 0.0, None) for k in counts])
        mat = np.maximum.accumulate(mat, axis=1)
        for j, k in enumerate(counts):
            pred[k] = mat[:, j]


for _name, _fn in {
    "add_recurrent_targets": add_recurrent_targets,
    "_higher_specs": _higher_specs,
    "_derive_adapter_head_weights": _derive_adapter_head_weights,
    "project_recurrent_outputs": project_recurrent_outputs,
}.items():
    setattr(_impl, _name, _fn)
    setattr(_base, _name, _fn)


# Wrap retained fitters so recurrent regression heads use the active grid rather
# than the old literal 72h grid, while leaving hazard/shared-grid/CPCV logic intact.
_original_fit_batch_bundle = _impl.fit_batch_bundle
_original_fit_online_adapter = _impl.fit_online_adapter


def _strip_out_of_contract_recurrent(bundle: dict, cfg: V24Config) -> dict:
    allowed = {f"recurrent_peak_count_{h}m" for h in RECURRENT_HORIZONS_MINUTES if h <= cfg.horizon_minutes}
    rec = bundle.get("recurrent") or {}
    bundle["recurrent"] = {
        k: v for k, v in rec.items()
        if not k.startswith("recurrent_peak_count_") or k in allowed
    }
    if bundle.get("adapter") and isinstance(bundle["adapter"], dict):
        arec = bundle["adapter"].get("recurrent") or {}
        bundle["adapter"]["recurrent"] = {
            k: v for k, v in arec.items()
            if not k.startswith("recurrent_peak_count_") or k in allowed
        }
    return bundle


def fit_batch_bundle(*args, **kwargs):
    cfg = args[4] if len(args) > 4 else kwargs.get("cfg")
    bundle = _original_fit_batch_bundle(*args, **kwargs)
    return _strip_out_of_contract_recurrent(bundle, cfg)


def fit_online_adapter(*args, **kwargs):
    cfg = args[5] if len(args) > 5 else kwargs.get("cfg")
    adapter = _original_fit_online_adapter(*args, **kwargs)
    wrapper = {"adapter": adapter}
    _strip_out_of_contract_recurrent(wrapper, cfg)
    return wrapper["adapter"]


_impl.fit_batch_bundle = fit_batch_bundle
_base.fit_batch_bundle = fit_batch_bundle
_impl.fit_online_adapter = fit_online_adapter
_base.fit_online_adapter = fit_online_adapter


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
        missing_historical = sorted(t for t in observed_times if t <= last_cached and t not in cached_times)
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
        "maturity_rule": "cohort_end + 2x24h token-lifetime/outcome window + audit delay",
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
        f"SELECT token_key,forecast_cohort_id,forecast_role FROM {TOKEN_ASSIGNMENT_TABLE} WHERE forecast_role='audit'",
        conn,
    )
    if assignments.empty:
        return {"available": False, "reason": "no audit-born token assignments"}
    audit_tokens = set(assignments[assignments.forecast_cohort_id.astype(str).isin(mature_ids)].token_key.astype(str))
    if not audit_tokens:
        return {"available": False, "reason": "no tokens belong to mature audit cohorts"}

    led = pd.read_sql_query(
        f"SELECT prediction_id,token_key,decision_at,generated_at,prediction_json FROM {PREDICTION_LEDGER} "
        "WHERE provenance='live' AND ineligibility_reason IN ('sealed_audit_token','sealed_audit_cohort') ORDER BY generated_at",
        conn,
    )
    if led.empty:
        return {"available": False, "reason": "no prospective sealed-audit live predictions recorded"}
    led = led[led.token_key.astype(str).isin(audit_tokens)].copy()
    if led.empty:
        return {"available": False, "reason": "no live predictions for mature audit-born tokens"}
    led["decision_at"] = pd.to_datetime(led.decision_at, utc=True, errors="coerce")
    led["generated_at"] = pd.to_datetime(led.generated_at, utc=True, errors="coerce")
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
        weighted_log = float(log_loss(
            score_df.y.astype(int).to_numpy(),
            np.clip(score_df.p.to_numpy(dtype=float), 1e-6, 1 - 1e-6),
            labels=[0, 1], sample_weight=weights,
        ))
    metrics = {
        "rows": int(len(score_df)), "tokens": int(score_df.token_key.nunique()),
        "token_balanced_peak_brier": float(token_brier.mean()),
        "token_balanced_peak_log_loss": weighted_log, "horizon_probability": horizon_key,
        "audit_membership": "token_first_seen_lifetime",
    }
    conn.execute(
        f"INSERT INTO {AUDIT_RESULTS_TABLE}(audit_result_id,created_at,model_family,prediction_rows,metric_json,note) VALUES(?,?,?,?,?,?)",
        (str(uuid.uuid4()), _now_iso(), SCHEMA_VERSION, len(score_df), _json(metrics),
         "Prospective sealed audit; token-first membership; confirmation-safe truth; never development-eligible"),
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


# ---------------------------------------------------------------------------
# Preserve policy cfg forwarding through this outer safety facade.
# ---------------------------------------------------------------------------
def train_distributional_policy(db, policy_root, cfg, *, allow_small=False):
    for name in (
        "migrate", "refresh_counterfactual_policy_targets", "refresh_policy_cohorts",
        "refresh_token_assignments", "next_one_use_policy_cohort",
    ):
        setattr(_base, name, globals()[name])
    return _base.train_distributional_policy(db, policy_root, cfg, allow_small=allow_small)


_impl.train_distributional_policy = train_distributional_policy


def lifecycle_contract() -> dict[str, object]:
    cfg = V24Config()
    return {
        "prediction_horizon_minutes": cfg.horizon_minutes,
        "axiom_view_minutes": AXIOM_VIEW_MINUTES,
        "age_out_guard_minutes": int(cfg.age_out_minutes),
        "buffer_minutes": AXIOM_VIEW_MINUTES - cfg.horizon_minutes,
        "probability_horizons_minutes": list(cfg.probability_horizons_minutes),
        "recurrent_horizons_minutes": list(RECURRENT_HORIZONS_MINUTES),
        "sequence_windows_minutes": list(cfg.sequence_windows_minutes),
        "schema_version": SCHEMA_VERSION,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
