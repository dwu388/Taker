from __future__ import annotations

"""Hardened public facade for the retained self-teaching compatibility module."""

from . import axiom_self_teach_impl as _impl
from .absence_utils import verified_absence_minutes

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def _paper_cycle_v24(
    db: str,
    predictions_path: str,
    policy_dir: str,
    config: PolicyConfig,
    *,
    allow_stale_predictions: bool = False,
    auto_train: bool = True,
    allow_small_policy: bool = False,
) -> dict[str, Any]:
    """Run one paper-policy decision cycle with next-observation execution.

    Missing-token terminal handling measures only a contiguous run of successful
    capture heartbeats. Collector downtime before that run is censored and cannot
    be counted toward token death.
    """
    _require_peak()
    predictions = _load_predictions(predictions_path)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        _backfill_paper_execution_accounting(conn)
        observations, _ = peak.load_observations(conn)
        if observations.empty:
            raise RuntimeError("No Axiom observations are available.")
        snapshot = observations.snapshot_at.max()
        current_obs = observations[observations.snapshot_at == snapshot].copy()
        current_obs = current_obs.sort_values("token_key").drop_duplicates("token_key", keep="last")
        current_obs = current_obs[["token_key", "market_cap_usd"] + (["name"] if "name" in current_obs.columns else [])]

        pred_snapshot = _prediction_snapshot(predictions)
        if pred_snapshot is not None and not allow_stale_predictions:
            if abs((pred_snapshot - snapshot).total_seconds()) > 180:
                raise RuntimeError(
                    f"Prediction CSV is stale relative to the latest clipboard capture: predictions={pred_snapshot}, capture={snapshot}."
                )
        current = current_obs.merge(predictions, on="token_key", how="left", suffixes=("", "__pred"))
        if "market_cap_usd__pred" in current.columns:
            current = current.drop(columns=["market_cap_usd__pred"])
        current = current[current["market_cap_usd"].notna()].copy()
        current_series = {str(r["token_key"]): r for _, r in current.iterrows()}

        is_v24_forecast = bool("v24_model_hash" in predictions.columns and predictions["v24_model_hash"].notna().any())
        is_v23_forecast = bool("v23_model_hash" in predictions.columns and predictions["v23_model_hash"].notna().any())
        v24mod = None
        if is_v24_forecast:
            try:
                from . import axiom_v24 as v24mod
                v24mod.migrate(conn)
                v24mod.record_capture_heartbeat(
                    conn, snapshot, valid_capture=True, row_count=len(current_obs), source="paper_cycle_v24"
                )
            except Exception:
                v24mod = None

        policy_id, policy_path = _policy_champion_path(conn)
        loaded_policy_bundle = _load_policy(policy_path)
        policy_accounting_incompatible = bool(
            loaded_policy_bundle is not None and str(loaded_policy_bundle.get("schema_version", "")) != SCHEMA_VERSION
        )
        policy_generation_incompatible = False
        if is_v24_forecast:
            policy_generation_incompatible = bool(
                loaded_policy_bundle is None
                or not str(loaded_policy_bundle.get("v24_policy_schema", "")).startswith("v24_")
                or not bool(loaded_policy_bundle.get("oos_only", False))
            )
        elif is_v23_forecast:
            policy_generation_incompatible = bool(
                loaded_policy_bundle is None
                or not str(loaded_policy_bundle.get("v23_policy_schema", "")).startswith("v23_")
                or not bool(loaded_policy_bundle.get("oos_only", False))
            )
        policy_bundle = None if (policy_accounting_incompatible or policy_generation_incompatible) else loaded_policy_bundle
        if policy_accounting_incompatible:
            policy_id = "bootstrap_pending_execution_accounting_rebootstrap"
        elif policy_generation_incompatible:
            policy_id = "bootstrap_pending_v24_oos_policy" if is_v24_forecast else "bootstrap_pending_v23_oos_distributional_policy"

        if is_v24_forecast:
            forecast_hash = str(predictions.loc[predictions["v24_model_hash"].notna(), "v24_model_hash"].iloc[-1])
        elif is_v23_forecast:
            forecast_hash = str(predictions.loc[predictions["v23_model_hash"].notna(), "v23_model_hash"].iloc[-1])
        else:
            forecast_hash = _hash_file(PEAK_CHAMPION_DEFAULT) or _hash_file(predictions_path)

        def terminal_absence(last_seen: pd.Timestamp) -> tuple[bool, float, int]:
            elapsed = max(0.0, (snapshot - last_seen).total_seconds() / 60.0)
            if v24mod is not None:
                run_start, run_end, n = v24mod._contiguous_capture_absence(
                    conn, last_seen, v24mod.V24Config(), upto=snapshot
                )
                if run_start is None or run_end is None:
                    return False, elapsed, 0
                valid_minutes = verified_absence_minutes(run_start, run_end)
                return bool(
                    valid_minutes >= config.missing_close_minutes
                    and n >= v24mod.V24Config().heartbeat_min_valid_captures_for_death
                ), valid_minutes, n
            return elapsed >= config.missing_close_minutes, elapsed, 0

        filled_entries: list[dict[str, Any]] = []
        cancelled_pending: list[dict[str, Any]] = []
        pending = conn.execute(
            "SELECT * FROM axiom_paper_pending_entries_v24 WHERE status='pending' ORDER BY decision_at"
        ).fetchall()
        for pen in pending:
            token = str(pen["token_key"])
            decision_at = _to_ts(pen["decision_at"])
            if snapshot <= decision_at:
                continue
            row = current_series.get(token)
            if row is None:
                terminal, absent_minutes, _ = terminal_absence(decision_at)
                if terminal:
                    conn.execute(
                        "UPDATE axiom_paper_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason=? WHERE pending_id=?",
                        (snapshot.isoformat(), "unavailable_before_next_observable_fill", pen["pending_id"]),
                    )
                    cancelled_pending.append({
                        "token_key": token,
                        "reason": "unavailable_before_next_observable_fill",
                        "missing_minutes": absent_minutes,
                    })
                continue
            fill_mc = float(row["market_cap_usd"])
            position_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO axiom_paper_positions_v20
                (position_id, token_key, opened_at, entry_mc, entry_state_json, entry_policy_version,
                 entry_forecast_hash, exploration, status, last_seen_at, last_mc, missed_cycles,
                 mfe_pct, mae_pct, config_json, entry_decision_at, entry_fill_kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, 0, 0, 0, ?, ?, 'next_observable')
                """,
                (
                    position_id, token, snapshot.isoformat(), fill_mc, pen["decision_state_json"], pen["policy_version"],
                    pen["forecast_hash"], int(pen["exploration"]), snapshot.isoformat(), fill_mc,
                    _json(asdict(config)), pen["decision_at"],
                ),
            )
            conn.execute(
                "UPDATE axiom_paper_pending_entries_v24 SET status='filled',filled_at=?,fill_mc=? WHERE pending_id=?",
                (snapshot.isoformat(), fill_mc, pen["pending_id"]),
            )
            filled_entries.append({
                "position_id": position_id,
                "token_key": token,
                "decision_mc": float(pen["decision_mc"]),
                "entry_mc": fill_mc,
                "fill_kind": "next_observable",
            })

        open_positions = conn.execute(
            "SELECT * FROM axiom_paper_positions_v20 WHERE status='open' ORDER BY opened_at"
        ).fetchall()
        open_before = len(open_positions)
        exits: list[dict[str, Any]] = []
        open_tokens = {str(p["token_key"]) for p in open_positions}

        for pos in open_positions:
            token = str(pos["token_key"])
            row = current_series.get(token)
            pending_exit_at = _to_ts(pos["pending_exit_at"]) if pos["pending_exit_at"] else None
            if row is None:
                misses = int(pos["missed_cycles"] or 0) + 1
                conn.execute(
                    "UPDATE axiom_paper_positions_v20 SET missed_cycles=? WHERE position_id=?",
                    (misses, pos["position_id"]),
                )
                terminal, absent_minutes, _ = terminal_absence(_to_ts(pos["last_seen_at"]))
                observed_mc = float(pos["last_mc"])
                execution_mc = _execution_proxy_mc(float(pos["entry_mc"]), observed_mc, config, unavailable=True)
                observed_return = observed_mc / float(pos["entry_mc"]) - 1.0
                execution_return = execution_mc / float(pos["entry_mc"]) - 1.0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO axiom_paper_marks_v20
                    (mark_id, position_id, token_key, snapshot_at, market_cap_usd, return_pct,
                     mfe_pct, mae_pct, state_json, action, action_value, policy_version,
                     price_available, mark_kind, execution_return_pct, training_eligible,
                     action_probability,behavior_policy_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, ?, 0, 1.0, ?)
                    """,
                    (
                        str(uuid.uuid4()), pos["position_id"], token, snapshot.isoformat(), observed_mc,
                        observed_return, float(pos["mfe_pct"]), float(pos["mae_pct"]),
                        _json({"missing_minutes": absent_minutes, "last_observed_mc": observed_mc}),
                        "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD", policy_id,
                        "disappearance_terminal" if terminal else "stale_last_observed", execution_return, policy_id,
                    ),
                )
                if terminal:
                    reason = str(pos["pending_exit_reason"] or "dead_after_valid_capture_absence")
                    exits.append(_close_position(
                        conn, pos, snapshot, observed_mc, reason, config, price_available=False
                    ))
                    open_tokens.discard(token)
                continue

            mc = float(row["market_cap_usd"])
            if pending_exit_at is not None and snapshot > pending_exit_at:
                fresh = conn.execute(
                    "SELECT * FROM axiom_paper_positions_v20 WHERE position_id=?", (pos["position_id"],)
                ).fetchone()
                exits.append(_close_position(
                    conn, fresh, snapshot, mc,
                    str(pos["pending_exit_reason"] or "policy_exit_next_observable"),
                    config, price_available=True,
                ))
                open_tokens.discard(token)
                continue

            pred_state = _safe_prediction_state(row)
            mark_state = _make_mark_state(pred_state, pos, mc, snapshot)
            current_return = mc / float(pos["entry_mc"]) - 1.0
            mfe = max(float(pos["mfe_pct"]), current_return)
            mae = min(float(pos["mae_pct"]), current_return)
            conn.execute(
                "UPDATE axiom_paper_positions_v20 SET last_seen_at=?,last_mc=?,missed_cycles=0,mfe_pct=?,mae_pct=? WHERE position_id=?",
                (snapshot.isoformat(), mc, mfe, mae, pos["position_id"]),
            )
            held = (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0
            if policy_bundle and policy_bundle.get("hold_head"):
                hold_value = float(_predict_policy_head(policy_bundle["hold_head"], [mark_state])[0])
                policy_kind = "learned"
            else:
                hold_value = _bootstrap_hold_value(mark_state)
                policy_kind = "bootstrap"
            action = "HOLD"
            close_reason = None
            if held >= config.max_hold_minutes:
                action = "EXIT_DECISION"
                close_reason = "max_hold"
            elif held >= config.min_hold_minutes and hold_value <= 0.0:
                action = "EXIT_DECISION"
                close_reason = f"{policy_kind}_hold_value_nonpositive"
            conn.execute(
                """
                INSERT OR REPLACE INTO axiom_paper_marks_v20
                (mark_id,position_id,token_key,snapshot_at,market_cap_usd,return_pct,mfe_pct,mae_pct,state_json,
                 action,action_value,policy_version,price_available,mark_kind,execution_return_pct,training_eligible,
                 action_probability,behavior_policy_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,'observed',?,1,1.0,?)
                """,
                (
                    str(uuid.uuid4()), pos["position_id"], token, snapshot.isoformat(), mc, current_return,
                    mfe, mae, _json(mark_state), action, hold_value, policy_id, current_return, policy_id,
                ),
            )
            if action == "EXIT_DECISION":
                conn.execute(
                    "UPDATE axiom_paper_positions_v20 SET pending_exit_at=?,pending_exit_reason=? WHERE position_id=?",
                    (snapshot.isoformat(), close_reason, pos["position_id"]),
                )

        open_tokens = {
            str(r[0]) for r in conn.execute(
                "SELECT token_key FROM axiom_paper_positions_v20 WHERE status='open'"
            ).fetchall()
        }
        pending_tokens = {
            str(r[0]) for r in conn.execute(
                "SELECT token_key FROM axiom_paper_pending_entries_v24 WHERE status='pending'"
            ).fetchall()
        }
        candidate_rows: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            token = str(row["token_key"])
            state = _safe_prediction_state(row)
            if not state:
                continue
            bootstrap = _bootstrap_entry_score(state)
            learned = (
                float(_predict_policy_head(policy_bundle["entry_head"], [state])[0])
                if policy_bundle and policy_bundle.get("entry_head") else None
            )
            score = learned if learned is not None else bootstrap
            candidate_rows.append({
                "token_key": token,
                "market_cap_usd": float(row["market_cap_usd"]),
                "state": state,
                "bootstrap_score": bootstrap,
                "policy_score": learned,
                "rank_score": score,
            })
        candidate_rows.sort(key=lambda r: r["rank_score"], reverse=True)
        slots = max(0, config.max_open_positions - len(open_tokens) - len(pending_tokens))
        eligible = []
        for c in candidate_rows:
            if c["token_key"] in open_tokens or c["token_key"] in pending_tokens:
                continue
            last_close = _last_closed_at(conn, c["token_key"])
            if last_close is not None and (snapshot - last_close).total_seconds() < config.reentry_cooldown_minutes * 60:
                continue
            eligible.append(c)

        seed = int(hashlib.sha256(snapshot.isoformat().encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        e = max(0.0, min(1.0, float(config.exploration_fraction)))
        k = min(slots, len(eligible))
        selected = []
        action_probs = {c["token_key"]: 0.0 for c in candidate_rows}
        exploration_probs = {c["token_key"]: 0.0 for c in candidate_rows}
        if k > 0:
            fixed = max(0, k - 1)
            for c in eligible[:fixed]:
                selected.append((c, False))
                action_probs[c["token_key"]] = 1.0
            pool = eligible[fixed:max(fixed + config.candidate_pool, fixed + 1)]
            if pool:
                for c in pool:
                    exploration_probs[c["token_key"]] = e / len(pool)
                boundary = eligible[fixed] if fixed < len(eligible) else None
                if boundary is not None:
                    action_probs[boundary["token_key"]] = (1.0 - e) + e / len(pool)
                for c in pool:
                    if boundary is None or c["token_key"] != boundary["token_key"]:
                        action_probs[c["token_key"]] = e / len(pool)
                if rng.random() < e:
                    selected.append((rng.choice(pool), True))
                elif boundary is not None:
                    selected.append((boundary, False))

        selected_keys = {c["token_key"] for c, _ in selected}
        exploration_keys = {c["token_key"] for c, ex in selected if ex}
        for rank, c in enumerate(candidate_rows, start=1):
            rank_prob = action_probs.get(c["token_key"], 0.0)
            conn.execute(
                """INSERT OR REPLACE INTO axiom_paper_candidates_v20
                (snapshot_at,token_key,market_cap_usd,state_json,bootstrap_score,policy_score,chosen,exploration,
                 forecast_hash,policy_version,action_probability,rank_probability,exploration_probability,
                 behavior_policy_version,eligible_actions_json,capital_constraint_state_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot.isoformat(), c["token_key"], c["market_cap_usd"], _json(c["state"]),
                    c["bootstrap_score"], c["policy_score"], int(c["token_key"] in selected_keys),
                    int(c["token_key"] in exploration_keys), forecast_hash, policy_id,
                    float(rank_prob), float(rank_prob), float(exploration_probs.get(c["token_key"], 0.0)),
                    policy_id, _json(["PASS", "ENTER"]),
                    _json({"slots": slots, "rank": rank, "open": len(open_tokens), "pending": len(pending_tokens)}),
                ),
            )

        pending_created = []
        for c, exploration in selected:
            pending_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO axiom_paper_pending_entries_v24
                (pending_id,token_key,decision_at,decision_mc,decision_state_json,policy_version,forecast_hash,
                 exploration,action_probability,status)
                VALUES (?,?,?,?,?,?,?,?,?,'pending')""",
                (
                    pending_id, c["token_key"], snapshot.isoformat(), c["market_cap_usd"], _json(c["state"]),
                    policy_id, forecast_hash, int(exploration), float(action_probs.get(c["token_key"], 1.0)),
                ),
            )
            pending_created.append({
                "pending_id": pending_id,
                "token_key": c["token_key"],
                "decision_mc": c["market_cap_usd"],
                "action_probability": action_probs.get(c["token_key"], 1.0),
                "exploration": exploration,
            })

        open_after = int(conn.execute(
            "SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='open'"
        ).fetchone()[0])
        run_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO axiom_self_teach_runs_v20
            (run_id,run_at,snapshot_at,predictions_path,forecast_model_hash,policy_version,current_tokens,open_before,entries,exits,open_after,details_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, _now_iso(), snapshot.isoformat(), predictions_path, forecast_hash, policy_id,
                len(current), open_before, len(filled_entries), len(exits), open_after,
                _json({"config": asdict(config), "pending_entries_created": len(pending_created)}),
            ),
        )
        conn.commit()
        closed_count = _closed_count(conn)
        last_train_closed = _policy_last_training_closed(conn)

    training = None
    if auto_train and not is_v24_forecast and closed_count >= (20 if allow_small_policy else config.policy_min_closed):
        normal_gate = closed_count - last_train_closed >= (5 if allow_small_policy else config.policy_retrain_every_closed)
        if normal_gate or policy_accounting_incompatible:
            try:
                training = train_policy(db, policy_dir, config, allow_small=allow_small_policy)
            except Exception as exc:
                training = {"error": str(exc)}
    return {
        "run_id": run_id,
        "snapshot_at": snapshot.isoformat(),
        "current_tokens": len(current),
        "open_before": open_before,
        "entries": filled_entries,
        "pending_entries": pending_created,
        "cancelled_pending": cancelled_pending,
        "exits": exits,
        "open_after": open_after,
        "active_policy": policy_id or "bootstrap",
        "closed_paper_trades": closed_count,
        "policy_training": training,
        "execution_semantics": "next_observable_fill",
    }


_impl._paper_cycle_v24 = _paper_cycle_v24


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":  # pragma: no cover
    main()
