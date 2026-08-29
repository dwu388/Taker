# V15/V17 OCR behavior retained in this rebuild

The OCR path uses recognition on already-isolated semantic cells. RapidOCR/PP-OCRv6 is the primary recognizer. Multiple preprocessing variants are considered; Tesseract is invoked when a dangerous field does not have enough modern-OCR agreement.

Dangerous numeric fields never concatenate disconnected digit groups. Native fee-cell pixels are checked for decimal evidence before upscaling. If source pixels indicate a decimal but accepted OCR text loses it, the fee becomes `NULL`.

Hard integrity checks reject impossible relationships such as `pro_traders > holders`, `kols > holders`, or `dev_migrations > dev_creations`.

Clipboard values, when safely identity-matched, take precedence over OCR and are recorded as corrections in `.selection.json` diagnostics.
