@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.v24_contract_runtime_v4 maintain --db data\axiom_v24_raw.sqlite %*
