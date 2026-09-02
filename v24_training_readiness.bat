@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.pretraining_cli readiness --db data\axiom_v24_raw.sqlite %*
