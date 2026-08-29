@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.legacy_import --db data\live.sqlite --auto
