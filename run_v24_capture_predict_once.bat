@echo off
call run_axiom_once.bat || exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_v24 predict --db data\live.sqlite %*
