from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ClipboardCard:
    name: str | None = None
    ticker: str | None = None
    short_address_hint: str | None = None

    age_raw: str | None = None
    age_minutes: int | None = None
    age_precision_minutes: int | None = None

    image_reuse_count: int | None = None

    market_cap_usd: float | None = None
    volume_usd: float | None = None
    fees_sol: float | None = None
    txns: int | None = None

    holders: int | None = None
    pro_traders: int | None = None
    kols: int | None = None
    dev_migrations: int | None = None
    dev_creations: int | None = None
    recent_visitors: int | None = None

    top10_holders_pct: float | None = None
    tracked_dev_status_raw: str | None = None
    funding_time_raw: str | None = None
    funding_time_minutes: int | None = None
    sniper_pct: float | None = None
    insider_pct: float | None = None
    bundler_pct: float | None = None
    dex_paid: bool | None = None

    source_block: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ADDRESS_RE = re.compile(
    r"\b("
    r"[1-9A-HJ-NP-Za-km-z]{3,8}"
    r"(?:\.\.\.|…)"
    r"[1-9A-HJ-NP-Za-km-z]{2,10}"
    r")\b"
)

DURATION_RE = re.compile(
    r"^(\d+)\s*(s|m|h|d|w|mo|y)$",
    re.I,
)

PERCENT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)%$"
)

DEV_PAIR_RE = re.compile(
    r"^(\d+)\s*/\s*(\d+)$"
)


CORE_FIELD_NAMES = (
    "name",
    "short_address_hint",
    "age_minutes",
    "image_reuse_count",
    "market_cap_usd",
    "volume_usd",
    "fees_sol",
    "txns",
    "holders",
    "pro_traders",
    "kols",
    "dev_migrations",
    "dev_creations",
    "recent_visitors",
    "top10_holders_pct",
    "tracked_dev_status_raw",
    "funding_time_raw",
    "funding_time_minutes",
    "sniper_pct",
    "insider_pct",
    "bundler_pct",
    "dex_paid",
)


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        value = re.sub(r"\s+", " ", raw).strip()
        if value:
            out.append(value)
    return out


def clipboard_looks_like_axiom(text: str) -> bool:
    """Strictly reject stale/unrelated clipboard contents."""
    if not text or len(text) < 250:
        return False

    lower = text.lower()
    required = (
        "pulse",
        "migrated",
        "mc",
        "tx",
    )
    if sum(term in lower for term in required) < 3:
        return False

    lines = _clean_lines(text)
    mc_count = sum(line.upper() == "MC" for line in lines)
    address_count = sum(bool(ADDRESS_RE.search(line)) for line in lines)

    # A valid Axiom Migrated selection should contain several card-shaped
    # repetitions. Requiring both MC labels and shortened addresses prevents a
    # copied search box, random browser text, or stale clipboard from entering
    # the training database.
    return mc_count >= 2 and address_count >= 2


def _parse_compact_number(value: str) -> float | None:
    s = value.strip().replace("$", "").replace(",", "")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMB]?)", s, re.I)
    if not m:
        return None

    number = float(m.group(1))
    suffix = m.group(2).upper()
    multiplier = {
        "": 1.0,
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
    }[suffix]
    return number * multiplier


def _parse_int(value: str) -> int | None:
    digits = value.replace(",", "").strip()
    if not re.fullmatch(r"\d+", digits):
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    s = value.replace(",", "").strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _duration_minutes(raw: str) -> int | None:
    m = DURATION_RE.fullmatch(raw.strip())
    if not m:
        return None

    count = int(m.group(1))
    unit = m.group(2).lower()
    multipliers = {
        "s": 1 / 60,
        "m": 1,
        "h": 60,
        "d": 1_440,
        "w": 10_080,
        "mo": 43_200,
        "y": 525_600,
    }
    return int(round(count * multipliers[unit]))


def _duration_precision_minutes(raw: str) -> int:
    m = DURATION_RE.fullmatch(raw.strip())
    if not m:
        return 1

    unit = m.group(2).lower()
    return {
        "s": 1,
        "m": 1,
        "h": 60,
        "d": 1_440,
        "w": 10_080,
        "mo": 43_200,
        "y": 525_600,
    }[unit]


def _find_metric(
    lines: list[str],
    label: str,
    *,
    compact: bool,
) -> float | None:
    label_upper = label.upper()

    for i, line in enumerate(lines):
        if line.upper() != label_upper:
            continue

        j = i + 1
        if j < len(lines) and lines[j].upper() == "SOL":
            j += 1

        if j >= len(lines):
            return None

        if compact:
            return _parse_compact_number(lines[j])
        return _parse_float(lines[j])

    return None


def _find_tx(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        m = re.fullmatch(r"TX\s*([0-9][0-9,]*)", line, re.I)
        if m:
            return _parse_int(m.group(1))

        if line.upper() == "TX" and i + 1 < len(lines):
            return _parse_int(lines[i + 1])

    return None


def _find_address_index(lines: list[str]) -> tuple[int | None, str | None]:
    for i, line in enumerate(lines):
        m = ADDRESS_RE.search(line)
        if m:
            return i, m.group(1).replace("…", "...")
    return None, None


def _is_metric_or_ui_line(value: str) -> bool:
    s = value.strip().lower()
    if s in {
        "mc",
        "v",
        "f",
        "sol",
        "pump amm",
        "raydium clmm",
        "paid",
    }:
        return True
    if s.startswith("tx "):
        return True
    if _parse_compact_number(value) is not None and value.strip().startswith("$"):
        return True
    return False


def _parse_identity(lines: list[str], card: ClipboardCard) -> None:
    addr_idx, address = _find_address_index(lines)
    if addr_idx is None:
        return

    card.short_address_hint = address

    # Selected Axiom DOM order around identity is consistently:
    #   [optional image-reuse badge]
    #   display name
    #   short address
    #   ticker
    #   full/display name
    #   age
    if addr_idx + 1 < len(lines):
        candidate = lines[addr_idx + 1]
        if not _is_metric_or_ui_line(candidate) and not DURATION_RE.fullmatch(candidate):
            card.ticker = candidate[:120]

    if addr_idx + 2 < len(lines):
        candidate = lines[addr_idx + 2]
        if not _is_metric_or_ui_line(candidate) and not DURATION_RE.fullmatch(candidate):
            card.name = candidate[:120]

    if not card.name and addr_idx > 0:
        candidate = lines[addr_idx - 1]
        if not _is_metric_or_ui_line(candidate):
            card.name = candidate[:120]


def _parse_image_reuse(lines: list[str], card: ClipboardCard) -> None:
    for i, line in enumerate(lines):
        if line.lower() not in {"pump amm", "raydium clmm"}:
            continue

        if i + 1 >= len(lines):
            return

        # The image reuse badge, when present, is an isolated integer directly
        # after the AMM line. A token name such as "200ms" is deliberately not
        # accepted as an integer.
        value = _parse_int(lines[i + 1])
        if value is not None:
            card.image_reuse_count = value
        return


def _parse_age(lines: list[str], card: ClipboardCard) -> int | None:
    addr_idx, _ = _find_address_index(lines)
    start = 0 if addr_idx is None else addr_idx + 1
    end = len(lines) if addr_idx is None else min(len(lines), addr_idx + 8)

    for i in range(start, end):
        raw = lines[i].strip()
        if DURATION_RE.fullmatch(raw):
            card.age_raw = raw.replace(" ", "")
            card.age_minutes = _duration_minutes(card.age_raw)
            card.age_precision_minutes = _duration_precision_minutes(card.age_raw)
            return i

    return None


def _parse_lifecycle_tail(lines: list[str], card: ClipboardCard, age_idx: int | None) -> None:
    if age_idx is None:
        return

    tail = lines[age_idx + 1:]
    if not tail:
        return

    # The first three isolated integers are holders, pro traders, and KOLs.
    # Stop collecting those once the dev pair is reached so a visitor count
    # cannot be shifted into one of the first three fields.
    prefix_ints: list[int] = []
    dev_pair_index: int | None = None

    for i, value in enumerate(tail):
        pair = DEV_PAIR_RE.fullmatch(value)
        if pair:
            a = int(pair.group(1))
            b = int(pair.group(2))
            if a <= b:
                card.dev_migrations = a
                card.dev_creations = b
            dev_pair_index = i
            break

        parsed_int = _parse_int(value)
        if parsed_int is not None:
            prefix_ints.append(parsed_int)

    if prefix_ints:
        card.holders = prefix_ints[0]
    if len(prefix_ints) >= 2:
        card.pro_traders = prefix_ints[1]
    if len(prefix_ints) >= 3:
        card.kols = prefix_ints[2]

    if dev_pair_index is not None:
        # Recent visitors is the first isolated integer after the dev pair and
        # before the audit/funding tail begins.
        for value in tail[dev_pair_index + 1:]:
            parsed_int = _parse_int(value)
            if parsed_int is not None:
                card.recent_visitors = parsed_int
                break
            if PERCENT_RE.fullmatch(value) or DURATION_RE.fullmatch(value):
                break
            if value.upper() in {"DS", "PAID"}:
                break

    percentages: list[tuple[int, float]] = []
    durations: list[tuple[int, str]] = []

    for i, value in enumerate(tail):
        pct = PERCENT_RE.fullmatch(value)
        if pct:
            percentages.append((i, float(pct.group(1))))
            continue

        if DURATION_RE.fullmatch(value):
            durations.append((i, value.replace(" ", "")))

    # First percentage after the lifecycle counts is Top 10. The final three
    # percentages are the stable sniper / insider / bundler audit fields.
    if percentages:
        card.top10_holders_pct = percentages[0][1]
    if len(percentages) >= 4:
        card.sniper_pct = percentages[-3][1]
        card.insider_pct = percentages[-2][1]
        card.bundler_pct = percentages[-1][1]

    # The first duration after token age is the funding-time field. Token age
    # itself is outside this tail, so there is no need for a "second duration"
    # heuristic here.
    if durations:
        card.funding_time_raw = durations[0][1]
        card.funding_time_minutes = _duration_minutes(card.funding_time_raw)

    if any(value.upper() == "DS" for value in tail):
        card.tracked_dev_status_raw = "DS"

    if any(value.lower() == "paid" for value in tail):
        card.dex_paid = True


def _parse_block(block: str) -> ClipboardCard | None:
    lines = _clean_lines(block)
    if len(lines) < 8:
        return None

    card = ClipboardCard(source_block=block)

    card.market_cap_usd = _find_metric(lines, "MC", compact=True)
    card.volume_usd = _find_metric(lines, "V", compact=True)
    card.fees_sol = _find_metric(lines, "F", compact=False)
    card.txns = _find_tx(lines)

    _parse_identity(lines, card)
    _parse_image_reuse(lines, card)
    age_idx = _parse_age(lines, card)
    _parse_lifecycle_tail(lines, card, age_idx)

    # Identity is mandatory. Without the shortened mint/address the row cannot
    # be linked safely across five-minute captures and therefore must not be
    # invented from a token name.
    if not card.short_address_hint:
        return None

    # Keep cards when the stable identity exists and at least two core market
    # metrics parse. Individual missing features remain NULL and are handled by
    # the existing feature-store missingness logic.
    core_market = (
        card.market_cap_usd,
        card.volume_usd,
        card.fees_sol,
        card.txns,
    )
    if sum(value is not None for value in core_market) < 2:
        return None

    return card


def _split_mc_blocks(text: str) -> list[str]:
    lines = _clean_lines(text)
    starts = [i for i, line in enumerate(lines) if line.upper() == "MC"]

    blocks: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        segment = lines[start:end]
        if len(segment) >= 8:
            blocks.append("\n".join(segment))
    return blocks


def parse_clipboard_cards(text: str) -> list[ClipboardCard]:
    if not clipboard_looks_like_axiom(text):
        return []

    cards: list[ClipboardCard] = []
    seen_addresses: set[str] = set()

    for block in _split_mc_blocks(text):
        card = _parse_block(block)
        if card is None or card.short_address_hint is None:
            continue

        address_key = _normalize_token_key(card.short_address_hint)
        if not address_key or address_key in seen_addresses:
            continue

        seen_addresses.add(address_key)
        cards.append(card)

    return cards


def _normalize_token_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def clipboard_card_to_row(card: ClipboardCard, card_index: int) -> dict[str, Any]:
    """Convert one parsed clipboard card into the canonical observation row."""
    token_key = _normalize_token_key(card.short_address_hint)

    field_values = {
        "name": card.name,
        "short_address_hint": card.short_address_hint,
        "age_minutes": card.age_minutes,
        "image_reuse_count": card.image_reuse_count,
        "market_cap_usd": card.market_cap_usd,
        "volume_usd": card.volume_usd,
        "fees_sol": card.fees_sol,
        "txns": card.txns,
        "holders": card.holders,
        "pro_traders": card.pro_traders,
        "kols": card.kols,
        "dev_migrations": card.dev_migrations,
        "dev_creations": card.dev_creations,
        "recent_visitors": card.recent_visitors,
        "top10_holders_pct": card.top10_holders_pct,
        "tracked_dev_status_raw": card.tracked_dev_status_raw,
        "funding_time_raw": card.funding_time_raw,
        "funding_time_minutes": card.funding_time_minutes,
        "sniper_pct": card.sniper_pct,
        "insider_pct": card.insider_pct,
        "bundler_pct": card.bundler_pct,
        "dex_paid": card.dex_paid,
    }

    confidence = {
        key: (1.0 if value is not None else 0.0)
        for key, value in field_values.items()
    }
    field_source = {
        key: ("clipboard_primary" if value is not None else "missing")
        for key, value in field_values.items()
    }

    confidence["token_key"] = 1.0 if token_key else 0.0
    field_source["token_key"] = "clipboard_primary" if token_key else "missing"

    row: dict[str, Any] = {
        **field_values,
        "token_key": token_key or None,
        "data_origin": "clipboard_only",
        "training_eligible": bool(token_key),
        "field_confidence": confidence,
        "field_source": field_source,
        "ocr_diagnostics": {},
        "source": {
            "data_origin": "clipboard_only",
            "training_eligible": bool(token_key),
            "clipboard_only": {
                "clipboard_card": card_index,
                "identity_method": "short_address",
                "timing": "clipboard_selection",
                "age_raw": card.age_raw,
                "age_precision_minutes": card.age_precision_minutes,
            },
        },
    }

    return row


def rows_from_clipboard(text: str) -> list[dict[str, Any]]:
    cards = parse_clipboard_cards(text)
    return [clipboard_card_to_row(card, i) for i, card in enumerate(cards)]


def clipboard_diagnostics(text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "clipboard_only",
        "clipboard_valid": clipboard_looks_like_axiom(text),
        "clipboard_cards": len(rows),
        "observations": len(rows),
        "unique_token_keys": len({row.get("token_key") for row in rows if row.get("token_key")}),
        "ocr_enabled": False,
        "screenshot_enabled": False,
        "ocr_rows": 0,
        "rows_identity_matched": 0,
        "corrections": [],
        "origin_counts": {"clipboard_only": len(rows)},
    }


def diagnostics_json(diag: dict[str, Any]) -> str:
    return json.dumps(diag, ensure_ascii=False, indent=2, default=str)
