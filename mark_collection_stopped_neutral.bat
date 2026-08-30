@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_manual_stop stop --db data\live.sqlite
endlocal
