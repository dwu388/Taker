@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_24h train --db data\live.sqlite --output-dir models\axiom24 --allow-small
