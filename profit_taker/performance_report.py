from __future__ import annotations

"""Human-readable V24 model performance reporting.

This module intentionally uses only the Python standard library so it can run in
collector-only environments. It reads the existing V24/Paper/Benchmark ledgers
and writes a compact text report without creating a new performance score.
"""

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .collection_admin import collection_status
from .db import RAW_DB_DEFAULT

REPORT_VERSION = "v24_performance_report_v1"
REPORT_OUTPUT_DEFAULT = "data/CURRENT_MODEL_PERFORMANCE.txt"
REPORT_INTERVAL_MINUTES_DEFAULT = 60
BENCHMARK_DB_DEFAULT = "data/axiom_v24_1000_benchmark.sqlite"
RECENT_EVENTS_DEFAULT = 8
FORECAST_MODEL_DEFAULT = "models/axiom_v24/champion.joblib"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt_num(value: Any, digits: int = 2) -> str:
    x = _finite(value)
    return "N/A" if x is None else f"{x:,.{digits}f}"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct_ratio(value: Any, digits: int = 1) -> str:
    x = _finite(value)
    return "N/A" if x is None else f"{100.0 * x:.{digits}f}%"


def _fmt_return(value: Any, digits: int = 2) -> str:
    x = _finite(value)
    return "N/A" if x is None else f"{100.0 * x:+.{digits}f}%"


def _safe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def _file_sha256(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _flatten_json(value: Any, prefix: str = "", *, limit: int = 80) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    def visit(node: Any, key: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            for child_key in sorted(node):
                visit(node[child_key], f"{key}.{child_key}" if key else str(child_key))
        elif isinstance(node, list):
            if len(node) <= 8 and all(not isinstance(x, (dict, list)) for x in node):
                out.append((key, node))
            else:
                out.append((key, f"[{len(node)} items]"))
        else:
            out.append((key, node))

    visit(value, prefix)
    return out


def _open_query_only(path: str | Path) -> sqlite3.Connection | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    con = sqlite3.connect(str(p), timeout=10.0)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA busy_timeout=10000")
    except sqlite3.DatabaseError:
        pass
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(con, table):
        return set()
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def _scalar(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = con.execute(sql, params).fetchone()
    return row[0] if row else None


def _rows(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(con.execute(sql, params).fetchall())


def _status_counts(con: sqlite3.Connection, table: str, column: str = "status") -> Counter[str]:
    if not _table_exists(con, table) or column not in _columns(con, table):
        return Counter()
    return Counter({str(row[0]): int(row[1]) for row in con.execute(
        f"SELECT COALESCE({column},'NULL'), COUNT(*) FROM {table} GROUP BY {column}"
    )})


def _numeric_stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    clean = [x for value in values if (x := _finite(value)) is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _win_rate(values: Iterable[Any]) -> float | None:
    clean = [x for value in values if (x := _finite(value)) is not None]
    return None if not clean else sum(x > 0.0 for x in clean) / len(clean)


def _max_drawdown(values: Iterable[Any]) -> float | None:
    clean = [x for value in values if (x := _finite(value)) is not None and x > 0.0]
    if not clean:
        return None
    peak = clean[0]
    worst = 0.0
    for value in clean:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _metric_lines(metrics: Any, indent: str = "    ", limit: int = 60) -> list[str]:
    obj = _safe_json(metrics)
    flat = _flatten_json(obj, limit=limit)
    if not flat:
        return [f"{indent}(no metric payload recorded)"]
    lines: list[str] = []
    for key, value in flat:
        if isinstance(value, float):
            rendered = f"{value:.8g}"
        elif isinstance(value, list):
            rendered = ", ".join(str(x) for x in value)
        else:
            rendered = str(value)
        lines.append(f"{indent}{key}: {rendered}")
    return lines


def _top_counts(rows: Iterable[sqlite3.Row], key: str, limit: int = 8) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row[key] if key in row.keys() else None
        label = str(value) if value not in (None, "") else "(none)"
        counts[label] += 1
    return counts.most_common(limit)


def _collection_section(db_path: str) -> tuple[list[str], dict[str, Any]]:
    lines = ["COLLECTION / DATA QUALITY", "-------------------------"]
    try:
        status = collection_status(db_path)
    except Exception as exc:
        lines.append(f"Collection status unavailable: {type(exc).__name__}: {exc}")
        return lines, {"ready": False, "status_error": str(exc)}

    lines.extend([
        f"Production collection ready: {bool(status.get('ready_to_collect'))}",
        f"SQLite integrity: {status.get('sqlite_integrity', 'N/A')}",
        f"Capture cycles: {_fmt_int(status.get('capture_cycles'))}",
        f"Capture attempts: {_fmt_int(status.get('capture_attempts'))} "
        f"({_fmt_int(status.get('successful_attempts'))} successful / {_fmt_int(status.get('failed_attempts'))} failed)",
        f"Attempt success rate: {_fmt_pct_ratio(status.get('attempt_success_rate'))}",
        f"Observations: {_fmt_int(status.get('observations'))}",
        f"Unique tokens: {_fmt_int(status.get('unique_tokens'))}",
        f"Full-mint row fraction: {_fmt_pct_ratio(status.get('full_mint_row_fraction'))}",
        f"First capture: {status.get('first_capture_at') or 'N/A'}",
        f"Last capture: {status.get('last_capture_at') or 'N/A'}",
        f"Rows/capture min-avg-max: {_fmt_num(status.get('rows_detected_min'),0)} / "
        f"{_fmt_num(status.get('rows_detected_avg'),1)} / {_fmt_num(status.get('rows_detected_max'),0)}",
        f"Raw payload coverage complete: {bool(status.get('raw_payload_complete'))}",
        f"Raw payload integrity errors: {_fmt_int(status.get('raw_payload_integrity_errors', 0))}",
        f"Cycle row-count mismatches: {_fmt_int(status.get('cycle_row_count_mismatches', 0))}",
        f"Identity conflicts: {status.get('identity_conflicts', {})}",
        "",
    ])
    return lines, {"ready": bool(status.get("ready_to_collect")), "status": status}


def _forecast_section(
    con: sqlite3.Connection,
    recent: int,
    forecast_model: str = FORECAST_MODEL_DEFAULT,
) -> tuple[list[str], dict[str, Any]]:
    lines = ["FORECAST MODEL", "--------------"]
    summary: dict[str, Any] = {}

    artifact_hash = _file_sha256(forecast_model)
    artifact_exists = artifact_hash is not None
    summary["artifact_exists"] = artifact_exists
    if artifact_exists:
        lines.append(
            f"Champion artifact: PRESENT | path={forecast_model} | sha256={artifact_hash}"
        )
    else:
        lines.append(f"Champion artifact: MISSING | path={forecast_model}")

    registry = "axiom_v24_model_registry"
    if _table_exists(con, registry):
        counts = _status_counts(con, registry)
        summary["registry_counts"] = dict(counts)
        lines.append("Model registry: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"))
        latest = con.execute(f"SELECT * FROM {registry} ORDER BY created_at DESC LIMIT 1").fetchone()
        champion = con.execute(
            f"SELECT * FROM {registry} WHERE status='champion' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if latest:
            lines.append(
                f"Latest registry entry: {latest['created_at']} | status={latest['status']} | "
                f"stable_generation={latest['stable_generation'] if 'stable_generation' in latest.keys() else 'N/A'} | "
                f"adapter_round={latest['adapter_round'] if 'adapter_round' in latest.keys() else 'N/A'}"
            )
            if "metrics_json" in latest.keys():
                lines.append("  Latest model metrics:")
                lines.extend(_metric_lines(latest["metrics_json"], indent="    ", limit=50))
        if champion:
            registered_hash = champion["model_hash"] if "model_hash" in champion.keys() else None
            artifact_matches = bool(artifact_hash and registered_hash == artifact_hash)
            summary["registry_champion"] = True
            summary["artifact_registry_match"] = artifact_matches
            lines.append(
                f"Current champion: {champion['created_at']} | model_hash={champion['model_hash'] or 'N/A'} | "
                f"training_cutoff={champion['stable_training_cutoff'] if 'stable_training_cutoff' in champion.keys() else 'N/A'}"
            )
            source = (
                champion["registration_source"]
                if "registration_source" in champion.keys() and champion["registration_source"]
                else "legacy_registry_entry"
            )
            lines.append(
                f"  Registration source: {source} | artifact hash match: {artifact_matches}"
            )
            if source == "preserved_artifact_link":
                lines.append(
                    "  Evidence note: preserved champion linked after database restart; "
                    "this registration is not a current-database promotion."
                )
        else:
            summary["registry_champion"] = False
            summary["artifact_registry_match"] = False
            lines.append("Current champion: NONE")
    else:
        summary["registry_champion"] = False
        summary["artifact_registry_match"] = False
        lines.append("Model registry: not created yet")
    summary["champion"] = bool(artifact_exists or summary.get("registry_champion"))
    if artifact_exists and not summary.get("registry_champion"):
        lines.append(
            "Artifact/registry state: DETACHED preserved artifact. Maintenance should link it without deleting or retraining it."
        )

    readiness = "axiom_v24_training_readiness"
    if _table_exists(con, readiness):
        latest_readiness = con.execute(
            f"SELECT * FROM {readiness} ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if latest_readiness:
            report = _safe_json(latest_readiness["report_json"])
            lines.append(f"Latest first-training readiness: {bool(report.get('ready'))}")
            gates = report.get("gates") if isinstance(report, dict) else None
            if isinstance(gates, dict):
                for name, gate in sorted(gates.items()):
                    if not isinstance(gate, dict):
                        continue
                    lines.append(
                        f"  Gate {name}: value={gate.get('value', 'N/A')} | "
                        f"minimum={gate.get('minimum', 'N/A')} | pass={bool(gate.get('pass'))}"
                    )

    learning = "axiom_v24_learning_cycles"
    if _table_exists(con, learning):
        cycle = con.execute(
            f"SELECT * FROM {learning} ORDER BY cycle_id DESC LIMIT 1"
        ).fetchone()
        if cycle:
            result = _safe_json(cycle["result_json"])
            lines.append(
                f"Latest learning cycle: {cycle['finished_at']} | "
                f"completed={bool(cycle['completed'])} | productive={bool(cycle['productive'])}"
            )
            for stage in result.get("stages", []) if isinstance(result, dict) else []:
                if isinstance(stage, dict):
                    lines.append(
                        f"  {stage.get('stage')}: {stage.get('semantic_state', 'unknown')}"
                        + (f" | {stage.get('reason')}" if stage.get("reason") else "")
                    )

    promotions = "axiom_v24_promotions"
    promoted = rejected = total = 0
    if _table_exists(con, promotions):
        total = int(_scalar(con, f"SELECT COUNT(*) FROM {promotions}") or 0)
        promoted = int(_scalar(con, f"SELECT COUNT(*) FROM {promotions} WHERE promoted=1") or 0)
        rejected = total - promoted
        lines.append(f"Forecast promotions: {total} total | {promoted} promoted | {rejected} rejected")
        recent_rows = _rows(con, f"SELECT * FROM {promotions} ORDER BY created_at DESC LIMIT ?", (recent,))
        for row in recent_rows:
            verdict = "SUCCESS/PROMOTED" if int(row["promoted"]) else "FAILURE/REJECTED"
            lines.append(f"  {row['created_at']} | {verdict} | cohort={row['cohort_id']} | {row['reason'] or '(no reason)'}")
        if recent_rows:
            lines.append("  Latest promotion metrics:")
            lines.extend(_metric_lines(recent_rows[0]["metrics_json"], indent="    ", limit=50))
    else:
        lines.append("Forecast promotions: no promotion table yet")
    summary.update({"promotion_total": total, "promotion_promoted": promoted, "promotion_rejected": rejected})

    ledger = "axiom_v24_prediction_ledger"
    if _table_exists(con, ledger):
        cols = _columns(con, ledger)
        predictions = int(_scalar(con, f"SELECT COUNT(*) FROM {ledger}") or 0)
        tokens = int(_scalar(con, f"SELECT COUNT(DISTINCT token_key) FROM {ledger}") or 0)
        by_prov = _rows(con, f"SELECT provenance,COUNT(*) AS n FROM {ledger} GROUP BY provenance ORDER BY n DESC")
        lines.append(f"Prediction ledger: {predictions:,} rows across {tokens:,} tokens")
        lines.append("  Scope: training/OOS provenance only; isolated paper-loop predictions are intentionally read-only and excluded")
        lines.append("  Provenance: " + (", ".join(f"{r['provenance']}={r['n']}" for r in by_prov) or "none"))
        if "oos_valid" in cols:
            oos = int(_scalar(con, f"SELECT COUNT(*) FROM {ledger} WHERE oos_valid=1") or 0)
            lines.append(f"  OOS-valid predictions: {oos:,} ({_fmt_pct_ratio(oos / predictions if predictions else None)})")
        if "policy_training_eligible" in cols:
            eligible = int(_scalar(con, f"SELECT COUNT(*) FROM {ledger} WHERE policy_training_eligible=1") or 0)
            lines.append(f"  Policy-training eligible predictions: {eligible:,}")
        sealed = int(_scalar(con, f"SELECT COUNT(*) FROM {ledger} WHERE ineligibility_reason IN ('sealed_audit_token','sealed_audit_cohort')") or 0)
        lines.append(f"  Sealed-audit prediction rows: {sealed:,}")
        summary["prediction_rows"] = predictions
    else:
        lines.append("Prediction ledger: not created yet")
        summary["prediction_rows"] = 0

    audit = "axiom_v24_audit_results"
    audit_rows = 0
    if _table_exists(con, audit):
        audit_rows = int(_scalar(con, f"SELECT COUNT(*) FROM {audit}") or 0)
        latest = con.execute(f"SELECT * FROM {audit} ORDER BY created_at DESC LIMIT 1").fetchone()
        lines.append(f"Sealed prospective audits recorded: {audit_rows}")
        if latest:
            metrics = _safe_json(latest["metric_json"])
            lines.append(f"  Latest audit: {latest['created_at']} | prediction_rows={latest['prediction_rows']}")
            if isinstance(metrics, dict):
                if "token_balanced_peak_brier" in metrics:
                    lines.append(f"  Token-balanced peak Brier: {_fmt_num(metrics.get('token_balanced_peak_brier'), 4)}")
                if "token_balanced_peak_log_loss" in metrics:
                    lines.append(f"  Token-balanced peak log loss: {_fmt_num(metrics.get('token_balanced_peak_log_loss'), 4)}")
                if "tokens" in metrics:
                    lines.append(f"  Audit tokens: {_fmt_int(metrics.get('tokens'))}")
            lines.append("  Full audit metric payload:")
            lines.extend(_metric_lines(latest["metric_json"], indent="    ", limit=50))
    else:
        lines.append("Sealed prospective audits recorded: 0 (audit table not created yet)")
    summary["audit_rows"] = audit_rows

    calibration = "axiom_v24_calibration_state"
    cal_updates = "axiom_v24_calibration_updates"
    if _table_exists(con, calibration):
        rows = _rows(con, f"SELECT * FROM {calibration}")
        biases = [abs(x) for row in rows if (x := _finite(row["bias_logit"] if "bias_logit" in row.keys() else None)) is not None]
        total_updates = sum(int(row["n_updates"] or 0) for row in rows if "n_updates" in row.keys())
        resolved = int(_scalar(con, f"SELECT COUNT(*) FROM {cal_updates}") or 0) if _table_exists(con, cal_updates) else 0
        lines.append(
            f"Calibration: {len(rows)} state keys | {total_updates:,} state updates | {resolved:,} resolved prediction-target updates"
        )
        if biases:
            lines.append(f"  Mean absolute calibration logit bias: {_fmt_num(statistics.fmean(biases), 4)}")
    else:
        lines.append("Calibration: not initialized yet")

    sequence = "axiom_v24_sequence_challengers"
    if _table_exists(con, sequence):
        counts = _status_counts(con, sequence)
        lines.append("Sequence challenger registry: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"))
        latest = con.execute(f"SELECT * FROM {sequence} ORDER BY created_at DESC LIMIT 1").fetchone()
        if latest:
            lines.append(f"  Latest sequence challenger: {latest['created_at']} | status={latest['status']} | tokens={latest['training_tokens']}")
            lines.extend(_metric_lines(latest["metrics_json"], indent="    ", limit=30))
    else:
        lines.append("Sequence challenger registry: not initialized yet")

    lines.append("")
    return lines, summary


def _policy_section(con: sqlite3.Connection, recent: int) -> tuple[list[str], dict[str, Any]]:
    lines = ["POLICY / ENTRY-HOLD DECISIONS", "-----------------------------"]
    summary: dict[str, Any] = {}
    registry = "axiom_v24_policy_registry"
    if _table_exists(con, registry):
        counts = _status_counts(con, registry)
        lines.append("Policy registry: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"))
        latest = con.execute(f"SELECT * FROM {registry} ORDER BY created_at DESC LIMIT 1").fetchone()
        champion = con.execute(f"SELECT * FROM {registry} WHERE status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
        if latest:
            lines.append(
                f"Latest policy entry: {latest['created_at']} | status={latest['status']} | "
                f"entry_rows={latest['training_rows_entry']} | hold_rows={latest['training_rows_hold']} | oos_only={bool(latest['oos_only'])}"
            )
            lines.append("  Latest policy metrics:")
            lines.extend(_metric_lines(latest["metrics_json"], indent="    ", limit=50))
        lines.append("Policy champion: " + ("AVAILABLE" if champion else "NONE"))
        summary["champion"] = bool(champion)
    else:
        lines.append("Policy registry: not created yet")
        summary["champion"] = False

    promotions = "axiom_v24_policy_promotions"
    total = promoted = rejected = 0
    if _table_exists(con, promotions):
        total = int(_scalar(con, f"SELECT COUNT(*) FROM {promotions}") or 0)
        promoted = int(_scalar(con, f"SELECT COUNT(*) FROM {promotions} WHERE promoted=1") or 0)
        rejected = total - promoted
        lines.append(f"Policy promotions: {total} total | {promoted} promoted | {rejected} rejected")
        rows = _rows(con, f"SELECT * FROM {promotions} ORDER BY created_at DESC LIMIT ?", (recent,))
        for row in rows:
            verdict = "SUCCESS/PROMOTED" if int(row["promoted"]) else "FAILURE/REJECTED"
            lines.append(f"  {row['created_at']} | {verdict} | cohort={row['cohort_id']} | {row['reason'] or '(no reason)'}")
        if rows:
            lines.append("  Latest policy promotion metrics:")
            lines.extend(_metric_lines(rows[0]["metrics_json"], indent="    ", limit=50))
    else:
        lines.append("Policy promotions: no promotion table yet")
    summary.update({"promotion_total": total, "promotion_promoted": promoted, "promotion_rejected": rejected})

    counter = "axiom_v24_counterfactual_policy_targets"
    if _table_exists(con, counter):
        total_cf = int(_scalar(con, f"SELECT COUNT(*) FROM {counter}") or 0)
        ready_cf = int(_scalar(con, f"SELECT COUNT(*) FROM {counter} WHERE target_ready_at IS NOT NULL") or 0)
        lines.append(f"Counterfactual targets: {total_cf:,} total | {ready_cf:,} mature/ready")
        by_action = _rows(con, f"SELECT action_kind,horizon_minutes,COUNT(*) AS n FROM {counter} GROUP BY action_kind,horizon_minutes ORDER BY action_kind,horizon_minutes")
        if by_action:
            lines.append("  Action/horizon coverage: " + "; ".join(f"{r['action_kind']}@{r['horizon_minutes']}m={r['n']}" for r in by_action))
    else:
        lines.append("Counterfactual targets: not initialized yet")

    lines.append("")
    return lines, summary


def _paper_section(con: sqlite3.Connection, recent: int) -> tuple[list[str], dict[str, Any]]:
    lines = ["PAPER TRADING", "-------------"]
    summary: dict[str, Any] = {"closed": 0}
    table = "axiom_paper_positions_v20"
    if not _table_exists(con, table):
        lines.extend(["Paper trading ledger: not initialized yet", ""])
        return lines, summary

    rows = _rows(con, f"SELECT * FROM {table} ORDER BY opened_at")
    closed = [row for row in rows if str(row["status"]) == "closed"]
    open_rows = [row for row in rows if str(row["status"]) == "open"]
    summary["closed"] = len(closed)
    lines.append(f"Positions: {len(rows)} total | {len(open_rows)} open | {len(closed)} closed")
    if closed:
        execution = [
            row["execution_net_return_pct"] if "execution_net_return_pct" in row.keys() and row["execution_net_return_pct"] is not None else row["net_return_pct"]
            for row in closed
        ]
        observed = [
            row["observed_net_return_pct"] if "observed_net_return_pct" in row.keys() and row["observed_net_return_pct"] is not None else row["net_return_pct"]
            for row in closed
        ]
        ex_stats = _numeric_stats(execution)
        ob_stats = _numeric_stats(observed)
        summary["execution_mean"] = ex_stats["mean"]
        summary["execution_win_rate"] = _win_rate(execution)
        lines.extend([
            f"Closed-trade execution win rate: {_fmt_pct_ratio(_win_rate(execution))}",
            f"Execution returns mean / median / best / worst: {_fmt_return(ex_stats['mean'])} / {_fmt_return(ex_stats['median'])} / {_fmt_return(ex_stats['max'])} / {_fmt_return(ex_stats['min'])}",
            f"Observed returns mean / median / best / worst: {_fmt_return(ob_stats['mean'])} / {_fmt_return(ob_stats['median'])} / {_fmt_return(ob_stats['max'])} / {_fmt_return(ob_stats['min'])}",
        ])
        rewards = [
            row["execution_reward"] if "execution_reward" in row.keys() and row["execution_reward"] is not None else row["reward"]
            for row in closed
        ]
        reward_stats = _numeric_stats(rewards)
        lines.append(f"Execution reward mean / median: {_fmt_num(reward_stats['mean'],4)} / {_fmt_num(reward_stats['median'],4)}")
        mfe = _numeric_stats(row["mfe_pct"] for row in closed)
        mae = _numeric_stats(row["mae_pct"] for row in closed)
        lines.append(f"MFE mean / MAE mean: {_fmt_return(mfe['mean'])} / {_fmt_return(mae['mean'])}")
        if "execution_peak_capture_ratio" in closed[0].keys():
            pcr = _numeric_stats(row["execution_peak_capture_ratio"] for row in closed)
            lines.append(f"Execution peak-capture ratio mean: {_fmt_pct_ratio(pcr['mean'])}")
        lines.append("Close reasons: " + ", ".join(f"{k}={v}" for k, v in _top_counts(closed, "close_reason", recent)))
        recent_closed = sorted(closed, key=lambda r: str(r["closed_at"] or ""), reverse=True)[:recent]
        lines.append("Recent closed trades:")
        for row in recent_closed:
            ex = row["execution_net_return_pct"] if "execution_net_return_pct" in row.keys() and row["execution_net_return_pct"] is not None else row["net_return_pct"]
            verdict = "WIN" if (_finite(ex) or 0.0) > 0 else "LOSS/NON-POSITIVE"
            lines.append(f"  {row['closed_at']} | {row['token_key']} | {verdict} | execution={_fmt_return(ex)} | reason={row['close_reason'] or '(none)'}")
    else:
        lines.append("No closed paper trades yet; realized policy performance cannot be judged.")

    pending = "axiom_paper_pending_entries_v24"
    if _table_exists(con, pending):
        counts = _status_counts(con, pending)
        lines.append("Pending-entry ledger: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"))
        cancelled = _rows(con, f"SELECT * FROM {pending} WHERE status='cancelled' ORDER BY cancelled_at DESC LIMIT ?", (recent,))
        if cancelled:
            lines.append("  Recent cancellation reasons: " + ", ".join(f"{k}={v}" for k, v in _top_counts(cancelled, "cancel_reason", recent)))

    runs = "axiom_self_teach_runs_v20"
    if _table_exists(con, runs):
        latest = con.execute(f"SELECT * FROM {runs} ORDER BY run_at DESC LIMIT 1").fetchone()
        if latest:
            lines.append(
                f"Latest paper cycle: {latest['run_at']} | current_tokens={latest['current_tokens']} | "
                f"entries={latest['entries']} | exits={latest['exits']} | open_after={latest['open_after']}"
            )

    lines.append("")
    return lines, summary


def _benchmark_section(path: str, recent: int) -> tuple[list[str], dict[str, Any]]:
    lines = ["$1,000 ISOLATED BENCHMARK", "-------------------------"]
    summary: dict[str, Any] = {"available": False}
    con = _open_query_only(path)
    if con is None:
        lines.extend([f"Benchmark database not found: {path}", ""])
        return lines, summary
    try:
        if not _table_exists(con, "benchmark_account_v22"):
            lines.extend(["Benchmark database exists but has not been initialized.", ""])
            return lines, summary
        acct = con.execute("SELECT * FROM benchmark_account_v22 ORDER BY created_at DESC LIMIT 1").fetchone()
        if not acct:
            lines.extend(["Benchmark account not initialized.", ""])
            return lines, summary
        summary["available"] = True
        initial = _finite(acct["initial_cash_usd"])
        lines.append(f"Benchmark id: {acct['benchmark_id']} | status={acct['status']} | initial_cash=${_fmt_num(initial,2)}")

        equity_table = "benchmark_equity_v22"
        if _table_exists(con, equity_table):
            eq_rows = _rows(con, f"SELECT * FROM {equity_table} WHERE benchmark_id=? ORDER BY snapshot_at", (acct["benchmark_id"],))
            if eq_rows:
                latest = eq_rows[-1]
                exec_equity = latest["execution_equity_usd"] if "execution_equity_usd" in latest.keys() and latest["execution_equity_usd"] is not None else latest["equity_usd"]
                obs_equity = latest["observed_equity_usd"] if "observed_equity_usd" in latest.keys() and latest["observed_equity_usd"] is not None else latest["equity_usd"]
                exec_series = [
                    row["execution_equity_usd"] if "execution_equity_usd" in row.keys() and row["execution_equity_usd"] is not None else row["equity_usd"]
                    for row in eq_rows
                ]
                exec_return = (float(exec_equity) / initial - 1.0) if initial and exec_equity is not None else None
                obs_return = (float(obs_equity) / initial - 1.0) if initial and obs_equity is not None else None
                summary["execution_return"] = exec_return
                lines.extend([
                    f"Latest snapshot: {latest['snapshot_at']}",
                    f"Execution equity: ${_fmt_num(exec_equity,2)} | return vs initial: {_fmt_return(exec_return)}",
                    f"Observed equity: ${_fmt_num(obs_equity,2)} | return vs initial: {_fmt_return(obs_return)}",
                    f"Execution max drawdown: {_fmt_return(_max_drawdown(exec_series))}",
                    f"Open positions at latest snapshot: {_fmt_int(latest['open_positions'])}",
                ])
            else:
                lines.append("No benchmark equity snapshots yet.")

        positions = "benchmark_positions_v22"
        if _table_exists(con, positions):
            rows = _rows(con, f"SELECT * FROM {positions} WHERE benchmark_id=? ORDER BY opened_at", (acct["benchmark_id"],))
            closed = [row for row in rows if str(row["status"]) == "closed"]
            open_rows = [row for row in rows if str(row["status"]) == "open"]
            lines.append(f"Benchmark positions: {len(rows)} total | {len(open_rows)} open | {len(closed)} closed")
            if closed:
                execution = [
                    row["execution_realized_return_pct"] if "execution_realized_return_pct" in row.keys() and row["execution_realized_return_pct"] is not None else row["realized_return_pct"]
                    for row in closed
                ]
                observed = [
                    row["observed_realized_return_pct"] if "observed_realized_return_pct" in row.keys() and row["observed_realized_return_pct"] is not None else row["realized_return_pct"]
                    for row in closed
                ]
                ex_stats = _numeric_stats(execution)
                ob_stats = _numeric_stats(observed)
                lines.extend([
                    f"Closed-position execution win rate: {_fmt_pct_ratio(_win_rate(execution))}",
                    f"Execution returns mean / median / best / worst: {_fmt_return(ex_stats['mean'])} / {_fmt_return(ex_stats['median'])} / {_fmt_return(ex_stats['max'])} / {_fmt_return(ex_stats['min'])}",
                    f"Observed returns mean / median / best / worst: {_fmt_return(ob_stats['mean'])} / {_fmt_return(ob_stats['median'])} / {_fmt_return(ob_stats['max'])} / {_fmt_return(ob_stats['min'])}",
                    "Close reasons: " + ", ".join(f"{k}={v}" for k, v in _top_counts(closed, "close_reason", recent)),
                ])

        pending = "benchmark_pending_entries_v24"
        if _table_exists(con, pending):
            counts = _status_counts(con, pending)
            lines.append("Benchmark pending entries: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "empty"))
            cancelled = _rows(con, f"SELECT * FROM {pending} WHERE status='cancelled' ORDER BY cancelled_at DESC LIMIT ?", (recent,))
            if cancelled:
                lines.append("  Cancellation reasons: " + ", ".join(f"{k}={v}" for k, v in _top_counts(cancelled, "cancel_reason", recent)))
    finally:
        con.close()
    lines.append("")
    return lines, summary


def _summary_section(collection: dict[str, Any], forecast: dict[str, Any], policy: dict[str, Any], paper: dict[str, Any], benchmark: dict[str, Any]) -> list[str]:
    successes: list[str] = []
    failures: list[str] = []

    if collection.get("ready"):
        successes.append("Raw collection integrity gate is currently passing.")
    else:
        failures.append("Raw collection integrity gate is not currently passing or could not be evaluated.")

    if forecast.get("promotion_promoted", 0):
        successes.append(f"{forecast['promotion_promoted']} forecast challenger(s) have passed a one-use promotion cohort.")
    if forecast.get("promotion_rejected", 0):
        failures.append(f"{forecast['promotion_rejected']} forecast challenger(s) have been rejected by promotion testing.")
    if forecast.get("audit_rows", 0):
        successes.append(f"{forecast['audit_rows']} sealed prospective audit result(s) are available for review.")
    else:
        failures.append("No sealed prospective audit result is mature yet; long-run generalization is still unproven.")

    if policy.get("promotion_promoted", 0):
        successes.append(f"{policy['promotion_promoted']} policy challenger(s) have passed one-use promotion testing.")
    if policy.get("promotion_rejected", 0):
        failures.append(f"{policy['promotion_rejected']} policy challenger(s) have been rejected.")
    if not policy.get("champion"):
        failures.append("No promoted V24 policy champion is recorded yet; policy decisions may still be bootstrap-driven.")

    closed = int(paper.get("closed") or 0)
    if closed:
        mean_ret = _finite(paper.get("execution_mean"))
        win = _finite(paper.get("execution_win_rate"))
        if mean_ret is not None and mean_ret > 0:
            successes.append(f"Paper execution return is positive on average across {closed} closed trade(s) ({_fmt_return(mean_ret)} mean).")
        else:
            failures.append(f"Paper execution return is not positive on average across {closed} closed trade(s) ({_fmt_return(mean_ret)} mean).")
        if win is not None:
            (successes if win >= 0.5 else failures).append(f"Paper execution win rate is {_fmt_pct_ratio(win)} across closed trades.")
    else:
        failures.append("No closed paper trades yet; realized entry/hold/exit performance cannot be judged.")

    if benchmark.get("available"):
        ret = _finite(benchmark.get("execution_return"))
        if ret is not None:
            (successes if ret > 0 else failures).append(f"$1,000 benchmark execution equity is {_fmt_return(ret)} versus starting capital.")
    else:
        failures.append("The isolated $1,000 benchmark has not produced reviewable results yet.")

    if forecast.get("audit_rows", 0):
        evidence = "PROSPECTIVE_AUDIT_EVIDENCE_AVAILABLE"
    elif forecast.get("promotion_total", 0) or policy.get("promotion_total", 0):
        evidence = "ONE_USE_PROMOTION_EVIDENCE_AVAILABLE"
    elif closed or benchmark.get("available"):
        evidence = "PAPER_EXECUTION_EVIDENCE_AVAILABLE"
    elif forecast.get("champion") or forecast.get("prediction_rows", 0):
        evidence = "MODEL_ACTIVE_BUT_NOT_FULLY_VALIDATED"
    else:
        evidence = "COLLECTION_ONLY_NO_MODEL_PERFORMANCE_EVIDENCE_YET"

    lines = [
        "EXECUTIVE PERFORMANCE SUMMARY",
        "-----------------------------",
        f"Evidence maturity: {evidence}",
        "",
        "Success signals:",
    ]
    lines.extend(f"  + {item}" for item in successes or ["No validated success signal is available yet."])
    lines.append("")
    lines.append("Failures / unresolved evidence:")
    lines.extend(f"  - {item}" for item in failures or ["No current failure signal was detected in the recorded evidence."])
    lines.append("")
    return lines


def build_report(
    db_path: str = RAW_DB_DEFAULT,
    *,
    benchmark_db: str = BENCHMARK_DB_DEFAULT,
    forecast_model: str = FORECAST_MODEL_DEFAULT,
    generated_at: datetime | None = None,
    interval_minutes: int = REPORT_INTERVAL_MINUTES_DEFAULT,
    recent_events: int = RECENT_EVENTS_DEFAULT,
) -> str:
    generated_at = generated_at or _now_utc()
    recent_events = max(1, int(recent_events))
    interval_minutes = max(1, int(interval_minutes))

    collection_lines, collection_summary = _collection_section(db_path)
    con = _open_query_only(db_path)
    if con is None:
        forecast_lines = ["FORECAST MODEL", "--------------", "Raw/model database not found.", ""]
        policy_lines = ["POLICY / ENTRY-HOLD DECISIONS", "-----------------------------", "Raw/model database not found.", ""]
        paper_lines = ["PAPER TRADING", "-------------", "Raw/model database not found.", ""]
        forecast_summary = {"champion": False, "promotion_total": 0, "promotion_promoted": 0, "promotion_rejected": 0, "audit_rows": 0, "prediction_rows": 0}
        policy_summary = {"champion": False, "promotion_total": 0, "promotion_promoted": 0, "promotion_rejected": 0}
        paper_summary = {"closed": 0}
    else:
        try:
            forecast_lines, forecast_summary = _forecast_section(
                con, recent_events, forecast_model
            )
            policy_lines, policy_summary = _policy_section(con, recent_events)
            paper_lines, paper_summary = _paper_section(con, recent_events)
        finally:
            con.close()

    benchmark_lines, benchmark_summary = _benchmark_section(benchmark_db, recent_events)
    summary_lines = _summary_section(collection_summary, forecast_summary, policy_summary, paper_summary, benchmark_summary)

    header = [
        "V24 CURRENT MODEL PERFORMANCE REPORT",
        "====================================",
        f"Report version: {REPORT_VERSION}",
        f"Generated at UTC: {_iso(generated_at)}",
        f"Scheduled refresh interval: {interval_minutes} minutes",
        f"Primary database: {db_path}",
        f"Benchmark database: {benchmark_db}",
        "",
        "This report does not manufacture a single success score. It summarizes the evidence already recorded by V24.",
        "Promotion success = challenger passed its one-use OOS gate. Promotion rejection = challenger failed that gate.",
        "Sealed audit results are kept separate from development. Paper/benchmark results emphasize execution-conservative returns.",
        "A lack of mature evidence is reported as unresolved, not silently treated as success or failure.",
        "",
    ]

    interpretation = [
        "HOW TO READ SUCCESSES AND FAILURES",
        "----------------------------------",
        "1. Forecast SUCCESS: a candidate is promoted only after outperforming the existing champion on a one-use OOS cohort under the configured uncertainty/material-degradation rules.",
        "2. Forecast FAILURE: a candidate is rejected when it does not clear that promotion gate. Rejections are useful evidence and should not be hidden.",
        "3. Sealed audit: Brier/log-loss on audit-born tokens is the cleanest prospective generalization evidence. Lower is better; compare it over time rather than to an arbitrary universal threshold.",
        "4. Policy SUCCESS/FAILURE: policy challengers use their own one-use promotion stream. A policy champion is materially stronger evidence than bootstrap entry/hold scores.",
        "5. Paper execution: judge execution_net_return_pct/reward first. Observed-only returns are shown for diagnosis but can overstate what was executable.",
        "6. $1,000 benchmark: execution equity and drawdown are the most direct end-to-end strategy evidence. Positive returns with unacceptable drawdown are not automatically a success.",
        "7. Collection failures: failed captures, identity conflicts, payload mismatches or row-count mismatches are data-quality failures, not market-model failures, but they can invalidate later model conclusions.",
        "8. Early-stage warning: raw observation count or time elapsed is not evidence of alpha. The report intentionally keeps 'no mature audit/paper evidence yet' visible.",
        "",
    ]
    return "\n".join(header + summary_lines + collection_lines + forecast_lines + policy_lines + paper_lines + benchmark_lines + interpretation).rstrip() + "\n"


def _atomic_write(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, output)


def _next_due_from_mtime(path: Path, interval_minutes: int) -> datetime | None:
    if not path.exists():
        return None
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    return modified + timedelta(minutes=max(1, int(interval_minutes)))


def maybe_generate_report(
    db_path: str = RAW_DB_DEFAULT,
    *,
    output_path: str = REPORT_OUTPUT_DEFAULT,
    benchmark_db: str = BENCHMARK_DB_DEFAULT,
    forecast_model: str = FORECAST_MODEL_DEFAULT,
    interval_minutes: int = REPORT_INTERVAL_MINUTES_DEFAULT,
    recent_events: int = RECENT_EVENTS_DEFAULT,
    force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or _now_utc()).astimezone(timezone.utc)
    output = Path(output_path)
    interval_minutes = max(1, int(interval_minutes))
    next_due = _next_due_from_mtime(output, interval_minutes)
    if not force and next_due is not None and now < next_due:
        return {
            "generated": False,
            "reason": "interval_not_elapsed",
            "output_path": str(output),
            "next_due_at": _iso(next_due),
            "interval_minutes": interval_minutes,
        }

    report = build_report(
        db_path,
        benchmark_db=benchmark_db,
        forecast_model=forecast_model,
        generated_at=now,
        interval_minutes=interval_minutes,
        recent_events=recent_events,
    )
    _atomic_write(output, report)
    next_due = now + timedelta(minutes=interval_minutes)
    return {
        "generated": True,
        "output_path": str(output),
        "generated_at": _iso(now),
        "next_due_at": _iso(next_due),
        "interval_minutes": interval_minutes,
        "bytes": len(report.encode("utf-8")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the human-readable V24 model performance report")
    parser.add_argument("--db", default=RAW_DB_DEFAULT)
    parser.add_argument("--benchmark-db", default=BENCHMARK_DB_DEFAULT)
    parser.add_argument("--forecast-model", default=FORECAST_MODEL_DEFAULT)
    parser.add_argument("--output", default=REPORT_OUTPUT_DEFAULT)
    parser.add_argument("--interval-minutes", type=int, default=REPORT_INTERVAL_MINUTES_DEFAULT)
    parser.add_argument("--recent-events", type=int, default=RECENT_EVENTS_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Generate now even if the scheduled interval has not elapsed")
    args = parser.parse_args(argv)
    result = maybe_generate_report(
        args.db,
        output_path=args.output,
        benchmark_db=args.benchmark_db,
        forecast_model=args.forecast_model,
        interval_minutes=args.interval_minutes,
        recent_events=args.recent_events,
        force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
