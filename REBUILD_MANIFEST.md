# Rebuild manifest

The supplied project sources described a patch lineage rather than providing the prior complete source archive. This clean tree was reconstructed from the latest documented behavior, with later behavior taking precedence when versions conflict.

## Precedence used

1. **V18 Maximum-Inference Lifecycle Model** — current training/inference target bank and five-minute momentum representation.
2. **V17 lifecycle relabel** — death is a competing event per `(token, decision_time)`; eventual death does not flatten earlier history into failure.
3. **V17 clipboard-assisted OCR** — Ctrl+A/C before screenshot, safe partial virtualization matching, clipboard/OCR diagnostics.
4. **V15 hybrid OCR** — RapidOCR/PP-OCRv6 primary, Tesseract validator, consensus/fail-closed numeric handling, native decimal guard.
5. **Clean semantic-card revision** — named Axiom fields such as holders/pro traders/KOL/dev pair/visitors/top10/funding/sniper/insider/bundler/dex paid.
6. **V13 visibility behavior** — positive-only bonus, no disappearance penalty, excluded from learned features.
7. **V12 raw-data ML architecture** — collect broadly, neutral derived features, no hand-written bullish/bearish PreScore, equal token weighting, strictly causal data.
8. Earlier provider work — retained as explicit optional modules, not part of the default collection loop.

## Deliberately superseded behavior

- 30-minute collection cadence → **5-minute cadence**.
- Generic `risk_slot_1..4` → **named semantic audit/trader fields**.
- Tesseract-only OCR → **hybrid RapidOCR + Tesseract**.
- Subjective pre-score/reject gates → **raw observation + neutral feature storage**.
- Disappearance as a universally negative feature → **no synthetic disappearance observation**.
- Eventual death rewriting all prior rows → **per-decision competing-event labels**.
- One 2x classifier → **multi-horizon probability + quantile + time + drawdown + death/survival model bank**.

## Security

No real provider credentials are included. `.env` is ignored. Provider commands are opt-in.
