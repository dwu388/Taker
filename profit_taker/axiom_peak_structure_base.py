from __future__ import annotations

"""Hardened public facade for the retained peak-structure compatibility module.

The implementation is kept byte-for-byte in ``axiom_peak_structure_impl.py``.
This facade pins the durable observation source and validates identity mappings
before any labeling/training code can consume observations.
"""

import json
import sqlite3

import pandas as pd

from . import axiom_peak_structure_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

CANONICAL_OBSERVATION_TABLE = "axiom_observations"
_FALLBACK_REJECT_FRAGMENTS = (
    "liquidity", "paper", "benchmark", "counterfactual", "prediction",
    "label", "feature", "policy",
)


def _canonical_observation_source(conn: sqlite3.Connection) -> dict[str, str] | None:
    if CANONICAL_OBSERVATION_TABLE not in _impl._all_tables(conn):
        return None
    cols = _impl._table_columns(conn, CANONICAL_OBSERVATION_TABLE)
    required = {"token_key", "snapshot_at", "market_cap_usd"}
    if not required.issubset(set(cols)):
        raise RuntimeError(
            f"Canonical observation table {CANONICAL_OBSERVATION_TABLE!r} exists but is missing "
            f"required columns {sorted(required - set(cols))}"
        )
    mapping = {
        "table": CANONICAL_OBSERVATION_TABLE,
        "token": "token_key",
        "time": "snapshot_at",
        "mc": "market_cap_usd",
    }
    if "age_minutes" in cols:
        mapping["age"] = "age_minutes"
    if "name" in cols:
        mapping["name"] = "name"
    return mapping


def discover_observation_source(conn: sqlite3.Connection) -> dict[str, str]:
    """Use the canonical collector table whenever it exists.

    Heuristic discovery remains only for explicit legacy databases that predate
    ``axiom_observations``. Derived/model/execution tables are excluded from that
    fallback so they can never outrank raw observations.
    """
    canonical = _canonical_observation_source(conn)
    if canonical is not None:
        return canonical

    scored: list[tuple[int, str, dict[str, str]]] = []
    for table in _impl._all_tables(conn):
        lname = table.lower()
        if any(fragment in lname for fragment in _FALLBACK_REJECT_FRAGMENTS):
            continue
        cols = _impl._table_columns(conn, table)
        token = _impl._pick(cols, _impl.TOKEN_CANDIDATES)
        time_col = _impl._pick(cols, _impl.TIME_CANDIDATES)
        mc = _impl._pick(cols, _impl.MC_CANDIDATES)
        if not (token and time_col and mc):
            continue
        age = _impl._pick(cols, _impl.AGE_CANDIDATES)
        name = _impl._pick(cols, _impl.NAME_CANDIDATES)
        score = 0
        if "axiom" in lname:
            score += 8
        if "migrated" in lname:
            score += 5
        if "observation" in lname or "observations" in lname:
            score += 6
        if "raw" in lname:
            score += 2
        mapping = {"table": table, "token": token, "time": time_col, "mc": mc}
        if age:
            mapping["age"] = age
        if name:
            mapping["name"] = name
        scored.append((score, table, mapping))

    if not scored:
        details = {t: _impl._table_columns(conn, t) for t in _impl._all_tables(conn)}
        raise RuntimeError(
            "Could not discover a raw Axiom observation table with token, timestamp, and market-cap columns. "
            f"Available schema: {json.dumps(details, default=str)[:8000]}"
        )
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _validate_identity_collisions(df: pd.DataFrame) -> None:
    """Reject known one-to-many token identity mappings before modeling.

    Short-only historical rows remain usable. Once full mints are available, any
    conflicting token-key/short-hint/full-mint mapping becomes a hard failure
    instead of silently merging unrelated lifecycles.
    """
    if df.empty:
        return

    work = df.copy()
    for col in ("token_key", "token_address", "short_address_hint"):
        if col in work:
            work[col] = work[col].map(
                lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else None
            )

    conflicts: list[str] = []
    if "token_address" in work and work["token_address"].notna().any():
        keyed = work.dropna(subset=["token_key", "token_address"])
        if not keyed.empty:
            by_key = keyed.groupby("token_key")["token_address"].nunique()
            bad = by_key[by_key > 1]
            if not bad.empty:
                conflicts.append(f"token_key->multiple_mints: {bad.index.astype(str).tolist()[:10]}")
            by_mint = keyed.groupby("token_address")["token_key"].nunique()
            bad = by_mint[by_mint > 1]
            if not bad.empty:
                conflicts.append(f"mint->multiple_token_keys: {bad.index.astype(str).tolist()[:10]}")

        if "short_address_hint" in work:
            hinted = work.dropna(subset=["short_address_hint", "token_address"])
            if not hinted.empty:
                by_hint = hinted.groupby("short_address_hint")["token_address"].nunique()
                bad = by_hint[by_hint > 1]
                if not bad.empty:
                    conflicts.append(f"short_hint->multiple_mints: {bad.index.astype(str).tolist()[:10]}")

    if conflicts:
        raise RuntimeError(
            "Token identity collision detected in durable observations; refusing labeling/training until resolved. "
            + "; ".join(conflicts)
        )


def load_observations(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict[str, str]]:
    source = discover_observation_source(conn)
    table = source["table"]
    cols = _impl._table_columns(conn, table)
    quoted = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql_query(f'SELECT {quoted} FROM "{table}"', conn)

    rename = {
        source["token"]: "token_key",
        source["time"]: "snapshot_at",
        source["mc"]: "market_cap_usd",
    }
    if "age" in source:
        rename[source["age"]] = "age_minutes"
    if "name" in source:
        rename[source["name"]] = "name"
    df = df.rename(columns=rename)
    _validate_identity_collisions(df)

    df["snapshot_at"] = _impl._to_timestamp(df["snapshot_at"])
    df["market_cap_usd"] = pd.to_numeric(df["market_cap_usd"], errors="coerce")
    df = df[df["token_key"].notna() & df["snapshot_at"].notna() & (df["market_cap_usd"] > 0)].copy()
    df["token_key"] = df["token_key"].astype(str)
    if "age_minutes" in df.columns:
        df["age_minutes"] = pd.to_numeric(df["age_minutes"], errors="coerce")
    df = df.sort_values(["token_key", "snapshot_at"]).drop_duplicates(
        ["token_key", "snapshot_at"], keep="last"
    )
    return df.reset_index(drop=True), source


_impl.discover_observation_source = discover_observation_source
_impl.load_observations = load_observations


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":  # pragma: no cover
    main()
