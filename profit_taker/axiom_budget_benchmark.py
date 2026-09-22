from __future__ import annotations

"""Hardened public facade for the retained isolated benchmark module."""

from . import axiom_budget_benchmark_impl as _impl
from . import recurrent_swing_policy as swing
from .absence_utils import verified_absence_minutes

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

# One canonical V24 benchmark database is used by both Python defaults and BATs.
DEFAULT_BENCHMARK_DB = "data/axiom_v24_1000_benchmark.sqlite"
_impl.DEFAULT_BENCHMARK_DB = DEFAULT_BENCHMARK_DB


def _cycle_v24(
    source_db: str,
    benchmark_db: str,
    predictions_path: str,
    forecast_model: str,
    policy_model: str,
    config: BenchmarkConfig,
    *, live_decisions: bool = False,
) -> dict[str, Any]:
    _require_v21()
    if live_decisions:
        snapshot, current = _read_current(
            source_db,
            predictions_path,
            exact_prediction_snapshot=True,
        )
    else:
        snapshot, current = _read_current(source_db, predictions_path)
    loaded_policy = _load_bundle(policy_model)
    forecast_bundle = _load_bundle(forecast_model)
    available_policy_hash = _hash_file(policy_model)
    required_policy_schema = getattr(selfteach, "SCHEMA_VERSION", None)
    is_v24_forecast = bool(
        (v24 is not None and forecast_bundle is not None and str(forecast_bundle.get("schema_version", "")) == v24.SCHEMA_VERSION)
        or "v24_model_hash" in current.columns
    )
    policy_compatible = bool(
        loaded_policy is None
        or required_policy_schema is None
        or str(loaded_policy.get("schema_version", "")) == str(required_policy_schema)
    )
    if is_v24_forecast and loaded_policy is not None:
        policy_compatible = bool(
            policy_compatible
            and str(loaded_policy.get("v24_policy_schema", "")).startswith("v24_")
            and bool(loaded_policy.get("oos_only", False))
        )
    policy = loaded_policy if policy_compatible else None
    forecast_hash = _hash_file(forecast_model) or _hash_file(predictions_path)
    policy_hash = available_policy_hash if policy_compatible else None
    policy_version = _policy_version(policy, policy_model) if policy_compatible else "bootstrap_pending_v24_policy_promotion"
    entry_fee_rate, exit_fee_rate = _entry_exit_rates(config)

    def terminal_absence(last_seen: pd.Timestamp) -> tuple[bool, float]:
        elapsed = max(0.0, (snapshot - last_seen).total_seconds() / 60.0)
        if is_v24_forecast and v24 is not None:
            try:
                with sqlite3.connect(source_db, timeout=10.0) as src:
                    src.execute("PRAGMA busy_timeout=10000")
                    src.execute("PRAGMA query_only=ON")
                    run_start, run_end, n = v24._contiguous_capture_absence(
                        src, last_seen, v24.V24Config(), upto=snapshot
                    )
                    if run_start is None or run_end is None:
                        return False, elapsed
                    valid = verified_absence_minutes(run_start, run_end)
                    return bool(
                        valid >= config.missing_close_minutes
                        and n >= v24.V24Config().heartbeat_min_valid_captures_for_death
                    ), valid
            except Exception:
                return False, elapsed
        return elapsed >= config.missing_close_minutes, elapsed

    with sqlite3.connect(benchmark_db, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        migrate(conn)
        swing.migrate(conn)
        acct = _account(conn)
        if acct is None:
            raise RuntimeError("Benchmark is not initialized. Run the init command first.")
        bid = str(acct["benchmark_id"])
        config = _config_from_saved_json(acct["config_json"])
        _validate_munger_config(config)
        entry_fee_rate, exit_fee_rate = _entry_exit_rates(config)
        _backfill_closed_execution_columns(conn, bid, config)
        last_snapshot = _to_ts(acct["last_snapshot_at"]) if acct["last_snapshot_at"] else None
        if last_snapshot is not None and snapshot <= last_snapshot:
            return {
                "processed": False,
                "reason": "no new clipboard snapshot",
                "benchmark_id": bid,
                "snapshot_at": snapshot.isoformat(),
            }
        current_series = {str(r["token_key"]): r for _, r in current.iterrows()}
        cash = float(acct["cash_usd"])
        execution_cash = _execution_cash_from_ledger(conn, bid, float(acct["initial_cash_usd"]))
        entries = []
        exits = []
        cancelled = []

        pending = conn.execute(
            "SELECT * FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending' ORDER BY decision_at",
            (bid,),
        ).fetchall()
        pre_fill_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'",
            (bid,),
        ).fetchall()
        fill_tokens = {str(p["token_key"]) for p in pre_fill_positions} | {
            str(p["token_key"]) for p in pending
        }
        fill_buckets = _correlation_buckets(source_db, fill_tokens, snapshot, config)
        fill_committed_total = sum(float(p["entry_cash_spent_usd"]) for p in pre_fill_positions)
        fill_committed_by_bucket: dict[str, float] = {}
        fill_open_value = 0.0
        for pos in pre_fill_positions:
            pos_token = str(pos["token_key"])
            bucket = fill_buckets.get(pos_token, pos_token)
            fill_committed_by_bucket[bucket] = (
                fill_committed_by_bucket.get(bucket, 0.0) + float(pos["entry_cash_spent_usd"])
            )
            current_row = current_series.get(pos_token)
            if current_row is not None:
                fill_open_value += _liquidation_value(
                    pos, float(current_row["market_cap_usd"]), exit_fee_rate
                )
            else:
                fill_open_value += _liquidation_value(
                    pos,
                    _execution_proxy_mc(pos, float(pos["last_mc"]), config, unavailable=True),
                    exit_fee_rate,
                )
        fill_effect_equity = execution_cash + fill_open_value
        for pen in pending:
            if snapshot <= _to_ts(pen["decision_at"]):
                continue
            token = str(pen["token_key"])
            row = current_series.get(token)
            if row is None:
                terminal, mins = terminal_absence(_to_ts(pen["decision_at"]))
                if terminal:
                    conn.execute(
                        "UPDATE benchmark_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason=? WHERE pending_id=?",
                        (snapshot.isoformat(), "unavailable_before_next_observable_fill", pen["pending_id"]),
                    )
                    cancelled.append({"token_key": token, "missing_minutes": mins})
                    swing.release_watch(conn, pen["swing_watch_id"])
                continue
            other_reserved = float(conn.execute(
                """
                SELECT COALESCE(SUM(reserved_cash_usd),0)
                FROM benchmark_pending_entries_v24
                WHERE benchmark_id=? AND status='pending' AND pending_id<>?
                """,
                (bid, pen["pending_id"]),
            ).fetchone()[0] or 0.0)
            tier = str(pen["conviction_tier"] or "ordinary")
            target_fraction = _finite_float(pen["target_position_fraction"])
            if target_fraction is None:
                target_fraction = _tier_fraction(tier, config)
            target_fraction = min(float(target_fraction), config.exceptional_position_fraction)
            bucket = fill_buckets.get(token, str(pen["risk_bucket"] or token))
            spendable_cash = max(0.0, min(cash, execution_cash) - other_reserved)
            reserved = _allocation_amount(
                target_cash=min(
                    float(pen["reserved_cash_usd"]),
                    fill_effect_equity * target_fraction,
                ),
                available_cash=spendable_cash,
                current_cash=spendable_cash,
                execution_equity=fill_effect_equity,
                committed_total=fill_committed_total,
                committed_bucket=fill_committed_by_bucket.get(bucket, 0.0),
                config=config,
            )
            if reserved < 1.0:
                conn.execute(
                    "UPDATE benchmark_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason='exposure_or_cash_cap_at_fill' WHERE pending_id=?",
                    (snapshot.isoformat(), pen["pending_id"]),
                )
                swing.release_watch(conn, pen["swing_watch_id"])
                continue
            notional = reserved / (1.0 + entry_fee_rate)
            fee = reserved - notional
            mc = float(row["market_cap_usd"])
            units = notional / mc
            pid = str(uuid.uuid4())
            swing_sequence = int(conn.execute(
                "SELECT COUNT(*) FROM benchmark_positions_v22 WHERE benchmark_id=? AND token_key=?",
                (bid, token),
            ).fetchone()[0]) + 1
            conn.execute(
                """INSERT INTO benchmark_positions_v22
                (position_id,benchmark_id,token_key,opened_at,entry_mc,entry_notional_usd,entry_fee_usd,entry_cash_spent_usd,
                 exposure_units,entry_score,entry_score_kind,entry_state_json,forecast_model_hash,policy_model_hash,policy_version,status,
                 last_seen_at,last_mc,last_mark_value_usd,mfe_pct,mae_pct,entry_decision_at,entry_fill_kind,
                 conviction_tier,target_position_fraction,risk_bucket,swing_watch_id,swing_sequence)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,0,0,?,'next_observable',?,?,?,?,?)""",
                (
                    pid, bid, token, snapshot.isoformat(), mc, notional, fee, reserved, units,
                    pen["entry_score"], pen["entry_score_kind"], pen["entry_state_json"],
                    pen["forecast_model_hash"], pen["policy_model_hash"], pen["policy_version"],
                    snapshot.isoformat(), mc, notional * (1 - exit_fee_rate), pen["decision_at"],
                    tier, target_fraction, bucket, pen["swing_watch_id"], swing_sequence,
                ),
            )
            conn.execute(
                """UPDATE benchmark_pending_entries_v24
                   SET status='filled',filled_at=?,fill_mc=?,reserved_cash_usd=?,risk_bucket=?
                   WHERE pending_id=?""",
                (snapshot.isoformat(), mc, reserved, bucket, pen["pending_id"]),
            )
            swing.mark_watch_filled(conn, pen["swing_watch_id"], pid)
            cash -= reserved
            execution_cash -= reserved
            fill_committed_total += reserved
            fill_committed_by_bucket[bucket] = fill_committed_by_bucket.get(bucket, 0.0) + reserved
            entries.append({
                "position_id": pid,
                "token_key": token,
                "decision_mc": float(pen["decision_mc"]),
                "entry_mc": mc,
                "cash_spent_usd": reserved,
                "conviction_tier": tier,
                "target_position_fraction": target_fraction,
                "risk_bucket": bucket,
                "fill_kind": "next_observable",
                "swing_sequence": swing_sequence,
            })

        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open' ORDER BY opened_at",
            (bid,),
        ).fetchall()
        open_tokens = {str(p["token_key"]) for p in open_positions}
        for pos in open_positions:
            token = str(pos["token_key"])
            row = current_series.get(token)
            pending_exit = _to_ts(pos["pending_exit_at"]) if pos["pending_exit_at"] else None
            if row is None:
                terminal, absent = terminal_absence(_to_ts(pos["last_seen_at"]))
                observed_mc = float(pos["last_mc"])
                observed_liq = _liquidation_value(pos, observed_mc, exit_fee_rate)
                proxy = _execution_proxy_mc(pos, observed_mc, config, unavailable=True)
                exliq = _liquidation_value(pos, proxy, exit_fee_rate)
                ret = observed_mc / float(pos["entry_mc"]) - 1
                conn.execute(
                    """INSERT OR REPLACE INTO benchmark_marks_v22
                    (mark_id,benchmark_id,position_id,token_key,snapshot_at,market_cap_usd,liquidation_value_usd,return_pct,mfe_pct,mae_pct,hold_score,action,state_json,forecast_model_hash,policy_model_hash,price_available,mark_kind,execution_liquidation_value_usd)
                    VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,0,?,?)""",
                    (
                        str(uuid.uuid4()), bid, pos["position_id"], token, snapshot.isoformat(), observed_mc,
                        observed_liq, ret, float(pos["mfe_pct"]), float(pos["mae_pct"]),
                        "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD",
                        _json({"missing_minutes": absent}), forecast_hash, policy_hash,
                        "disappearance_terminal" if terminal else "stale_last_observed", exliq,
                    ),
                )
                if terminal:
                    closed = _close_position(
                        conn, pos, snapshot, observed_mc,
                        str(pos["pending_exit_reason"] or "dead_after_valid_capture_absence"),
                        exit_fee_rate, config, price_available=False,
                    )
                    cash += closed["observed_proceeds_usd"]
                    execution_cash += closed["execution_proceeds_usd"]
                    exits.append(closed)
                    open_tokens.discard(token)
                continue
            mc = float(row["market_cap_usd"])
            if pending_exit is not None and snapshot > pending_exit:
                fresh = conn.execute(
                    "SELECT * FROM benchmark_positions_v22 WHERE position_id=?", (pos["position_id"],)
                ).fetchone()
                closed = _close_position(
                    conn, fresh, snapshot, mc,
                    str(pos["pending_exit_reason"] or "policy_exit_next_observable"),
                    exit_fee_rate, config, price_available=True,
                )
                if str(pos["pending_exit_reason"] or "") == "recurrent_swing_peak_boundary":
                    swing.create_watch(
                        conn, bid, fresh, snapshot, mc, _safe_state(row), config
                    )
                cash += closed["observed_proceeds_usd"]
                execution_cash += closed["execution_proceeds_usd"]
                exits.append(closed)
                open_tokens.discard(token)
                continue
            state = _safe_state(row)
            mark_state = _state_with_position(state, pos, mc, snapshot)
            ret = mc / float(pos["entry_mc"]) - 1
            mfe = max(float(pos["mfe_pct"]), ret)
            mae = min(float(pos["mae_pct"]), ret)
            liq = _liquidation_value(pos, mc, exit_fee_rate)
            hold_score, kind = _hold_score(mark_state, policy)
            swing_decision = (
                swing.peak_boundary_decision(state, ret, config)
                if config.recurrent_swing_enabled else None
            )
            if swing_decision is not None:
                mark_state["recurrent_swing_at_peak_boundary"] = float(
                    bool(swing_decision["at_peak_boundary"])
                )
                mark_state["recurrent_swing_hold_through_value"] = float(
                    swing_decision["hold_through_value"]
                )
                mark_state["recurrent_swing_sell_reentry_value"] = float(
                    swing_decision["sell_reentry_value"]
                )
            held = max(0.0, (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0)
            action = "HOLD"
            reason = None
            if held >= config.max_hold_minutes:
                action, reason = "EXIT_DECISION", "max_hold_72h"
            elif held >= config.min_hold_minutes:
                if swing_decision is not None and swing_decision["sell"]:
                    action, reason = "EXIT_DECISION", "recurrent_swing_peak_boundary"
                elif not (
                    swing_decision is not None
                    and swing_decision["reason"] == "hold_strong_near_second_peak"
                ) and hold_score <= 0:
                    action, reason = "EXIT_DECISION", f"{kind}_hold_value_nonpositive"
            conn.execute(
                "UPDATE benchmark_positions_v22 SET last_seen_at=?,last_mc=?,last_mark_value_usd=?,mfe_pct=?,mae_pct=? WHERE position_id=?",
                (snapshot.isoformat(), mc, liq, mfe, mae, pos["position_id"]),
            )
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_marks_v22
                (mark_id,benchmark_id,position_id,token_key,snapshot_at,market_cap_usd,liquidation_value_usd,return_pct,mfe_pct,mae_pct,hold_score,action,state_json,forecast_model_hash,policy_model_hash,price_available,mark_kind,execution_liquidation_value_usd)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'observed',?)""",
                (
                    str(uuid.uuid4()), bid, pos["position_id"], token, snapshot.isoformat(), mc, liq,
                    ret, mfe, mae, hold_score, action, _json(mark_state), forecast_hash, policy_hash, liq,
                ),
            )
            if action == "EXIT_DECISION" and pending_exit is None:
                conn.execute(
                    "UPDATE benchmark_positions_v22 SET pending_exit_at=?,pending_exit_reason=? WHERE position_id=?",
                    (snapshot.isoformat(), reason, pos["position_id"]),
                )

        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'", (bid,)
        ).fetchall()
        reserved = float(conn.execute(
            "SELECT COALESCE(SUM(reserved_cash_usd),0) FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending'",
            (bid,),
        ).fetchone()[0] or 0.0)
        exec_open = 0.0
        for p in open_positions:
            if str(p["token_key"]) in current_series:
                exec_open += float(p["last_mark_value_usd"])
            else:
                exec_open += _liquidation_value(
                    p, _execution_proxy_mc(p, float(p["last_mc"]), config, unavailable=True), exit_fee_rate
                )
        effect_equity = execution_cash + exec_open
        candidates = []
        open_tokens = {str(p["token_key"]) for p in open_positions}
        pending_tokens = {
            str(r[0]) for r in conn.execute(
                "SELECT token_key FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending'", (bid,)
            ).fetchall()
        }
        for _, row in current.iterrows():
            token = str(row["token_key"])
            if token in open_tokens or token in pending_tokens:
                continue
            state = _safe_state(row)
            if not state:
                continue
            score, kind = _entry_score(state, policy)
            if config.recurrent_swing_enabled:
                setup, kind = swing.short_term_setup(state, config, score, kind)
                score = setup.score
                state["recurrent_swing_entry_probability"] = setup.probability
                state["recurrent_swing_entry_net_edge"] = setup.net_edge
                state["recurrent_swing_entry_qualifies"] = float(setup.qualifies)
            else:
                setup = None
            if math.isfinite(score):
                candidates.append({
                    "token_key": token,
                    "market_cap_usd": float(row["market_cap_usd"]),
                    "state": state,
                    "score": score,
                    "kind": kind,
                    "setup": setup,
                })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        # Five positions is the normal diversification limit. Candidates that
        # causally qualify for the exceptional tier may exceed that count, but
        # never the portfolio, cash-reserve, per-position, or risk-bucket caps.
        occupied_positions = len(open_positions) + len(pending_tokens)
        base_slots = max(0, config.max_open_positions - occupied_positions)
        risk_buckets = _correlation_buckets(
            source_db,
            open_tokens | pending_tokens | {str(c["token_key"]) for c in candidates},
            snapshot,
            config,
        )
        threshold_cache: dict[str, tuple[float, float] | None] = {}
        eligible = []
        for c in candidates:
            c["chosen"] = False
            c["conviction_tier"] = "pass"
            c["target_fraction"] = 0.0
            c["target_cash"] = 0.0
            c["reserved_cash"] = 0.0
            c["risk_bucket"] = risk_buckets.get(c["token_key"], c["token_key"])
            c["swing_watch_id"] = None
            if c["setup"] is not None and not c["setup"].qualifies:
                c["selection_reason"] = c["setup"].reason
                continue
            if c["score"] <= config.min_entry_score:
                c["selection_reason"] = "score_not_above_entry_threshold"
                continue
            if config.recurrent_swing_enabled:
                allowed, reentry_reason, watch_id = swing.reentry_eligibility(
                    conn, bid, c["token_key"], snapshot, c["market_cap_usd"],
                    c["setup"], config,
                )
                c["swing_watch_id"] = watch_id
                if not allowed:
                    c["selection_reason"] = reentry_reason
                    continue
            else:
                lc = _last_closed_at(conn, bid, c["token_key"])
                if lc is not None and (snapshot - lc).total_seconds() < config.reentry_cooldown_minutes * 60:
                    c["selection_reason"] = "reentry_cooldown"
                    continue
            if c["kind"] not in threshold_cache:
                threshold_cache[c["kind"]] = _conviction_thresholds(
                    conn, bid, c["kind"], snapshot, config
                )
            tier = _conviction_tier(c["score"], threshold_cache[c["kind"]])
            c["conviction_tier"] = tier
            c["target_fraction"] = _tier_fraction(tier, config)
            c["target_cash"] = effect_equity * c["target_fraction"]
            c["selection_reason"] = "eligible"
            eligible.append(c)

        committed_total = sum(float(p["entry_cash_spent_usd"]) for p in open_positions) + reserved
        committed_by_bucket: dict[str, float] = {}
        for pos in open_positions:
            token = str(pos["token_key"])
            bucket = risk_buckets.get(token, token)
            committed_by_bucket[bucket] = (
                committed_by_bucket.get(bucket, 0.0) + float(pos["entry_cash_spent_usd"])
            )
        for pen in conn.execute(
            """SELECT token_key,reserved_cash_usd FROM benchmark_pending_entries_v24
               WHERE benchmark_id=? AND status='pending'""",
            (bid,),
        ).fetchall():
            token = str(pen["token_key"])
            bucket = risk_buckets.get(token, token)
            committed_by_bucket[bucket] = (
                committed_by_bucket.get(bucket, 0.0) + float(pen["reserved_cash_usd"])
            )

        cash_after_reservations = max(0.0, min(cash, execution_cash) - reserved)
        selected = []
        for c in eligible:
            within_base_limit = len(selected) < base_slots
            exceptional_overflow = (
                not within_base_limit and c["conviction_tier"] == "exceptional"
            )
            if not within_base_limit and not exceptional_overflow:
                c["selection_reason"] = "position_slot_limit"
                continue
            bucket = c["risk_bucket"]
            reserve = _allocation_amount(
                target_cash=c["target_cash"],
                available_cash=cash_after_reservations,
                current_cash=cash_after_reservations,
                execution_equity=effect_equity,
                committed_total=committed_total,
                committed_bucket=committed_by_bucket.get(bucket, 0.0),
                config=config,
            )
            if reserve < 1.0:
                c["selection_reason"] = "exposure_or_cash_cap"
                continue
            c["chosen"] = True
            c["reserved_cash"] = reserve
            c["selection_reason"] = (
                "selected_exceptional_overflow" if exceptional_overflow else "selected"
            )
            selected.append(c)
            cash_after_reservations -= reserve
            committed_total += reserve
            committed_by_bucket[bucket] = committed_by_bucket.get(bucket, 0.0) + reserve

        for c in candidates:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_candidates_v22
                (benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,state_json,
                 forecast_model_hash,policy_model_hash,conviction_tier,target_position_fraction,target_cash_usd,
                 risk_bucket,swing_watch_id,selection_reason)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    bid, snapshot.isoformat(), c["token_key"], c["market_cap_usd"], c["score"], c["kind"],
                    int(c["chosen"]), _json(c["state"]), forecast_hash, policy_hash,
                    c["conviction_tier"], c["target_fraction"], c["target_cash"],
                    c["risk_bucket"], c["swing_watch_id"], c["selection_reason"],
                ),
            )

        pending_created = []
        for c in selected:
            pid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO benchmark_pending_entries_v24
                (pending_id,benchmark_id,token_key,decision_at,decision_mc,reserved_cash_usd,entry_score,
                 entry_score_kind,entry_state_json,forecast_model_hash,policy_model_hash,policy_version,status,
                 conviction_tier,target_position_fraction,risk_bucket,swing_watch_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'pending',?,?,?,?)""",
                (
                    pid, bid, c["token_key"], snapshot.isoformat(), c["market_cap_usd"], c["reserved_cash"],
                    c["score"], c["kind"], _json(c["state"]), forecast_hash, policy_hash, policy_version,
                    c["conviction_tier"], c["target_fraction"], c["risk_bucket"],
                    c["swing_watch_id"],
                ),
            )
            swing.mark_watch_pending(conn, c["swing_watch_id"], snapshot)
            pending_created.append({
                "pending_id": pid,
                "token_key": c["token_key"],
                "reserved_cash_usd": c["reserved_cash"],
                "decision_mc": c["market_cap_usd"],
                "conviction_tier": c["conviction_tier"],
                "target_position_fraction": c["target_fraction"],
                "risk_bucket": c["risk_bucket"],
                "reentry_watch_id": c["swing_watch_id"],
            })

        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'", (bid,)
        ).fetchall()
        open_liq = exec_open_liq = unreal = exunreal = 0.0
        stale = 0
        for p in open_positions:
            obsval = float(p["last_mark_value_usd"])
            open_liq += obsval
            unreal += obsval - float(p["entry_cash_spent_usd"])
            if str(p["token_key"]) in current_series:
                exval = obsval
            else:
                stale += 1
                exval = _liquidation_value(
                    p, _execution_proxy_mc(p, float(p["last_mc"]), config, unavailable=True), exit_fee_rate
                )
            exec_open_liq += exval
            exunreal += exval - float(p["entry_cash_spent_usd"])
        observed_realized = float(conn.execute(
            "SELECT COALESCE(SUM(realized_pnl_usd),0) FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'",
            (bid,),
        ).fetchone()[0] or 0.0)
        execution_realized = float(conn.execute(
            "SELECT COALESCE(SUM(COALESCE(execution_realized_pnl_usd,realized_pnl_usd)),0) FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'",
            (bid,),
        ).fetchone()[0] or 0.0)
        execution_cash = _execution_cash_from_ledger(conn, bid, float(acct["initial_cash_usd"]))
        observed_equity = cash + open_liq
        execution_equity = execution_cash + exec_open_liq
        conn.execute(
            """INSERT OR REPLACE INTO benchmark_equity_v22
            (benchmark_id,snapshot_at,cash_usd,open_liquidation_value_usd,equity_usd,realized_pnl_usd,unrealized_pnl_usd,open_positions,forecast_model_hash,policy_model_hash,policy_version,observed_equity_usd,execution_cash_usd,execution_open_value_usd,execution_equity_usd,execution_realized_pnl_usd,execution_unrealized_pnl_usd,stale_open_positions)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                bid, snapshot.isoformat(), cash, open_liq, observed_equity, observed_realized, unreal,
                len(open_positions), forecast_hash, policy_hash, policy_version, observed_equity,
                execution_cash, exec_open_liq, execution_equity, execution_realized, exunreal, stale,
            ),
        )
        conn.execute(
            "UPDATE benchmark_account_v22 SET cash_usd=?,execution_cash_usd=?,last_snapshot_at=? WHERE benchmark_id=?",
            (cash, execution_cash, snapshot.isoformat(), bid),
        )
        if live_decisions:
            # A queued board may have been copied while prediction was running.
            # Such a board must never fill an order before it actually existed.
            available_at = max(snapshot, pd.Timestamp.now(tz="UTC")).isoformat()
            conn.execute(
                """UPDATE benchmark_pending_entries_v24 SET decision_at=?
                WHERE benchmark_id=? AND status='pending' AND decision_at=?""",
                (available_at, bid, snapshot.isoformat()),
            )
            conn.execute(
                """UPDATE benchmark_positions_v22 SET pending_exit_at=?
                WHERE benchmark_id=? AND status='open' AND pending_exit_at=?""",
                (available_at, bid, snapshot.isoformat()),
            )
        conn.commit()
    return {
        "processed": True,
        "benchmark_id": bid,
        "snapshot_at": snapshot.isoformat(),
        "cash_usd": cash,
        "execution_cash_usd": execution_cash,
        "observed_equity_usd": observed_equity,
        "effectiveness_equity_usd": execution_equity,
        "effectiveness_total_return_pct": execution_equity / config.initial_cash_usd - 1,
        "effectiveness_accounting": "execution_conservative_next_observable",
        "open_positions": len(open_positions),
        "entries": entries,
        "pending_entries": pending_created,
        "cancelled_pending": cancelled,
        "exits": exits,
        "committed_exposure_usd": committed_total,
        "committed_exposure_fraction": (
            committed_total / execution_equity if execution_equity > 0 else None
        ),
        "max_total_exposure_fraction": config.max_total_exposure_fraction,
        "min_cash_reserve_fraction": config.min_cash_reserve_fraction,
        "forecast_model_hash": forecast_hash,
        "policy_model_hash": policy_hash,
        "policy_version": policy_version,
        "policy_accounting_compatible": policy_compatible,
        "training_feedback": "disabled",
    }


_impl._cycle_v24 = _cycle_v24


def main() -> None:
    """Same CLI as the implementation, with the canonical benchmark DB default."""
    ap = argparse.ArgumentParser(
        description="V24 isolated $1,000 benchmark using leakage-hardened 1-minute / 72-hour champion models"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="Start a fresh isolated $1,000 benchmark account")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--reset", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("cycle", help="Run one benchmark decision cycle from the latest V24 predictions")
    p.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    p.add_argument("--forecast-model", default=DEFAULT_PEAK_MODEL)
    p.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--refresh-predictions", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("loop", help="Continuously refresh V24 predictions and run the isolated benchmark")
    p.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    p.add_argument("--forecast-model", default=DEFAULT_PEAK_MODEL)
    p.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--interval-seconds", type=float, default=60.0)
    _add_config_args(p)

    p = sub.add_parser("status", help="Show dollar P&L and effectiveness statistics")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)

    p = sub.add_parser("export", help="Export equity curve, trades, marks, candidates and summary")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--out-dir", default="data/axiom_v24_1000_benchmark")

    args = ap.parse_args()
    if args.cmd == "init":
        result = init_benchmark(args.benchmark_db, _config_from_args(args), reset=args.reset)
    elif args.cmd == "cycle":
        if args.refresh_predictions:
            refresh_predictions(args.source_db, args.forecast_model, args.predictions)
        result = cycle(
            args.source_db, args.benchmark_db, args.predictions,
            args.forecast_model, args.policy_model, _config_from_args(args),
        )
    elif args.cmd == "loop":
        loop(
            args.source_db, args.benchmark_db, args.predictions,
            args.forecast_model, args.policy_model, _config_from_args(args), args.interval_seconds,
        )
        return
    elif args.cmd == "status":
        result = status(args.benchmark_db)
    elif args.cmd == "export":
        result = export(args.benchmark_db, args.out_dir)
    else:  # pragma: no cover
        raise AssertionError(args.cmd)
    print(json.dumps(result, indent=2, default=str))


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":  # pragma: no cover
    main()
