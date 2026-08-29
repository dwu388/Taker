@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.collection_admin status --db data\axiom_v24_raw.sqlite
