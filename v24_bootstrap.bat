@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.v24_pretraining_bootstrap --db data\axiom_v24_raw.sqlite %*
