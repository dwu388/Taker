from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .common import parse_compact_number, parse_duration_minutes, parse_percent, safe_float, safe_int


@dataclass
class OCRDecision:
    value: Any
    confidence: float
    agreement_count: int
    engines: list[str]
    variants: list[str]
    reason: str
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rapid_text(result: Any) -> tuple[str, float] | None:
    """Normalize RapidOCR 3.x recognition-only output variants."""
    if result is None:
        return None
    # RapidOCR TextRecOutput object: txts / scores are common in 3.x.
    for text_attr, score_attr in (("txts", "scores"), ("texts", "scores")):
        txts = getattr(result, text_attr, None)
        scores = getattr(result, score_attr, None)
        if txts is not None and len(txts):
            text = str(txts[0])
            score = float(scores[0]) if scores is not None and len(scores) else 0.0
            return text, score
    if isinstance(result, (list, tuple)):
        # Typical recognition-only result can be a pair or nested list.
        if len(result) >= 2 and isinstance(result[0], str):
            try:
                return result[0], float(result[1])
            except Exception:
                return result[0], 0.0
        for item in result:
            parsed = _rapid_text(item)
            if parsed:
                return parsed
    if isinstance(result, dict):
        text = result.get("text") or result.get("txt")
        score = result.get("score") or result.get("confidence") or 0.0
        if text is not None:
            return str(text), float(score)
    return None


def detect_native_decimal(cell_bgr: np.ndarray) -> bool:
    """Detect small punctuation-like foreground near the numeric baseline at native resolution."""
    if cell_bgr is None or cell_bgr.size == 0:
        return False
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY) if cell_bgr.ndim == 3 else cell_bgr.copy()
    # In Axiom dark UI, text is bright. Threshold both ways and use whichever yields fewer foreground pixels.
    _, bw_hi = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    _, bw_ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw_hi if np.count_nonzero(bw_hi) < np.count_nonzero(bw_ot) else bw_ot
    h, w = bw.shape[:2]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        cx, cy = cents[i]
        if area <= max(14, int(h * w * 0.012)) and cw <= max(5, int(w * .10)) and ch <= max(6, int(h * .30)):
            # decimal is generally in lower half but not at the very edge
            if .42 * h <= cy <= .88 * h and .08 * w <= cx <= .92 * w:
                return True
    return False


def preprocess_variants(cell_bgr: np.ndarray, hsv_mask: tuple[tuple[int,int,int], tuple[int,int,int]] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"color": cell_bgr}
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY) if cell_bgr.ndim == 3 else cell_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(gray)
    out["clahe"] = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    if hsv_mask and cell_bgr.ndim == 3:
        hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_mask[0], dtype=np.uint8), np.array(hsv_mask[1], dtype=np.uint8))
        masked = np.zeros_like(cell_bgr)
        masked[mask > 0] = cell_bgr[mask > 0]
        out["masked_color"] = masked
    return out


def _tesseract(cell: np.ndarray, whitelist: str | None = None, psm: int = 7) -> tuple[str, float]:
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(cell, config=config, output_type=pytesseract.Output.DICT)
    texts, confs = [], []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        t = str(text).strip()
        try:
            c = float(conf)
        except Exception:
            c = -1
        if t:
            texts.append(t)
            if c >= 0:
                confs.append(c/100.0)
    return " ".join(texts), (sum(confs)/len(confs) if confs else 0.0)


class HybridOCR:
    def __init__(self, require_rapid: bool = True, tesseract_cmd: str | None = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists():
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.rapid = None
        self.rapid_error: str | None = None
        try:
            from rapidocr import RapidOCR  # type: ignore
            self.rapid = RapidOCR()
        except Exception as exc:
            self.rapid_error = f"{type(exc).__name__}: {exc}"
            if require_rapid:
                raise RuntimeError(
                    "RapidOCR/PP-OCRv6 could not initialize. Run install_v15_ocr.bat. "
                    f"Underlying error: {self.rapid_error}"
                ) from exc

    def _rapid(self, image: np.ndarray) -> tuple[str, float] | None:
        if self.rapid is None:
            return None
        try:
            # Recognition-only mode, as documented by RapidOCR.
            result = self.rapid(image, use_det=False, use_cls=False, use_rec=True)
        except TypeError:
            result = self.rapid(image)
        return _rapid_text(result)

    def recognize(
        self,
        cell_bgr: np.ndarray,
        parser: Callable[[str], Any],
        *,
        whitelist: str | None = None,
        require_agreement: bool = True,
        min_rapid_conf: float = .45,
        hsv_mask: tuple[tuple[int,int,int], tuple[int,int,int]] | None = None,
        decimal_guard: bool = False,
        psm: int = 7,
    ) -> OCRDecision:
        candidates: list[dict[str, Any]] = []
        variants = preprocess_variants(cell_bgr, hsv_mask=hsv_mask)
        for variant_name, image in variants.items():
            if self.rapid is None:
                continue
            rr = self._rapid(image)
            if not rr:
                continue
            text, conf = rr
            parsed = parser(text)
            if parsed is not None and conf >= min_rapid_conf:
                candidates.append({"engine":"rapidocr", "variant":variant_name, "text":text, "confidence":conf, "value":parsed})

        # Group by exact parsed value (floats rounded for OCR consensus).
        def key(v: Any):
            return round(v, 8) if isinstance(v, float) else v
        counts: dict[Any, int] = {}
        for c in candidates:
            counts[key(c["value"])] = counts.get(key(c["value"]), 0) + 1
        winner_key = max(counts, key=counts.get) if counts else None
        agreement = counts.get(winner_key, 0) if winner_key is not None else 0
        need_tess = not candidates or (require_agreement and agreement < 2)
        if need_tess:
            text, conf = _tesseract(variants.get("clahe", cell_bgr), whitelist=whitelist, psm=psm)
            parsed = parser(text)
            if parsed is not None:
                candidates.append({"engine":"tesseract", "variant":"clahe", "text":text, "confidence":conf, "value":parsed})
                counts[key(parsed)] = counts.get(key(parsed), 0) + 1
                winner_key = max(counts, key=counts.get)
                agreement = counts[winner_key]

        if winner_key is None:
            return OCRDecision(None, 0.0, 0, [], [], "no_parseable_candidate", candidates)
        winners = [c for c in candidates if key(c["value"]) == winner_key]
        value = winners[0]["value"]
        engines = sorted({c["engine"] for c in winners})
        used_variants = sorted({c["variant"] for c in winners})
        confidence = max(c["confidence"] for c in winners)

        if require_agreement and agreement < 2:
            return OCRDecision(None, confidence, agreement, engines, used_variants, "insufficient_agreement", candidates)
        if decimal_guard and detect_native_decimal(cell_bgr):
            texts = [str(c["text"]) for c in winners]
            if not any("." in t for t in texts):
                return OCRDecision(None, confidence, agreement, engines, used_variants, "decimal_present_but_not_preserved", candidates)
        return OCRDecision(value, confidence, agreement, engines, used_variants, "accepted_consensus", candidates)


def parse_integer(text: str) -> int | None:
    groups = re.findall(r"\d+", str(text).replace(",", ""))
    if len(groups) != 1:
        return None
    return int(groups[0])


def parse_dev_pair(text: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(text).replace(",", ""))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        return None
    return a, b


def parse_money(text: str) -> float | None:
    return parse_compact_number(text)


def parse_fee(text: str) -> float | None:
    s = str(text).replace(",", "").replace(" ", "")
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", s)
    if not m:
        return None
    x = safe_float(m.group(1))
    return x if x is not None and 0 <= x <= 1_000_000 else None


def parse_pct(text: str) -> float | None:
    return parse_percent(text)


def parse_duration(text: str) -> int | None:
    s = str(text).strip().lower().replace(" ", "")
    # Restrictive duration grammar; do not accept plain numbers as ages.
    m = re.search(r"(\d+)\s*(mo|[smhdwy])\b", s)
    if not m:
        return None
    return parse_duration_minutes(m.group(1) + m.group(2))


def parse_paid(text: str) -> bool | None:
    s = re.sub(r"[^a-z]", "", str(text).lower())
    if "paid" in s:
        return True
    return None
