@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite %*
