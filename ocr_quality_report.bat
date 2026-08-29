@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_ocr_report --db data\live.sqlite
