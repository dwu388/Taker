@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_v24 bootstrap --db data\axiom_v24_raw.sqlite %*
