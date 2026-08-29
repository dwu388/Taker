from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_int(value: Any) -> int | None:
    x = safe_float(value)
    return None if x is None else int(round(x))


MONEY_RE = re.compile(r"[-+]?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KMB]?)", re.I)


def parse_compact_number(text: Any) -> float | None:
    if text is None:
        return None
    s = str(text).strip().replace(" ", "")
    m = MONEY_RE.search(s)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()]
    return val * mult


def parse_percent(text: Any) -> float | None:
    if text is None:
        return None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", str(text))
    if not m:
        return None
    x = float(m.group(1))
    return x if 0 <= x <= 100 else None


def parse_duration_minutes(text: Any) -> int | None:
    if text is None:
        return None
    s = str(text).strip().lower().replace(" ", "")
    s = s.replace("imo", "1mo")
    m = re.fullmatch(r"(\d+)(s|m|h|d|w|mo|y)", s)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    factors = {"s": 1/60, "m": 1, "h": 60, "d": 1440, "w": 10080, "mo": 43200, "y": 525600}
    return int(round(n * factors[unit]))


def normalize_token_key(name: str | None, short_address_hint: str | None, token_address: str | None = None) -> str | None:
    if token_address:
        return token_address.strip()
    if short_address_hint:
        return re.sub(r"\s+", "", short_address_hint).lower()
    if name:
        cleaned = re.sub(r"[^a-z0-9]+", "", name.lower())
        return cleaned or None
    return None


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def chunks(items: Iterable[Any], n: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
