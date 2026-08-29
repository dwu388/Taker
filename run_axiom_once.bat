@echo off
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite --once %*
