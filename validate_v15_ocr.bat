@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -c "from profit_taker.axiom_ocr_hybrid import HybridOCR; x=HybridOCR(require_rapid=True); print('RapidOCR initialized:', bool(x.rapid))"
.venv\Scripts\python.exe -m pytest tests\test_ocr_parsers.py
endlocal
