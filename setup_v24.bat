@echo off
setlocal
cd /d "%~dp0"

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" || (echo Python 3.11 or newer is required.& exit /b 1)

if not exist .venv\Scripts\python.exe python -m venv .venv

.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python.exe -m pytest || exit /b 1

echo V24 setup and regression tests completed successfully.
echo Start or resume the marked production raw dataset with run_axiom_loop.bat.