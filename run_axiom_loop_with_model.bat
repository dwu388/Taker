@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite --with-model --model models\axiom24\latest.joblib %*
