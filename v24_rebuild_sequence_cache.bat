@echo off
.venv\Scripts\python.exe -m profit_taker.axiom_v24 rebuild-sequence-cache --db data\live.sqlite %*
