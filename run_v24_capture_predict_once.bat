@echo off
setlocal
cd /d "%~dp0"
call run_axiom_once.bat || exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_v24 predict --db data\axiom_v24_raw.sqlite %*
