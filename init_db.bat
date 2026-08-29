@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.migrate_v4 --db data\live.sqlite
