@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_v24 rebuild-sequence-cache --db data\live.sqlite %*
