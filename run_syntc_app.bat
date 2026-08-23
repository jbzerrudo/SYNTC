@echo off
rem ---------------------------------------------------------------
rem  Starts the SynTC genesis tool in your browser. Works offline.
rem  Requires Python 3 on this machine. Flask installs on first run.
rem ---------------------------------------------------------------
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo   Python was not found. Install Python 3 and tick "Add to PATH".
  pause
  exit /b 1
)

python -c "import flask" >nul 2>&1
if errorlevel 1 (
  echo   Installing Flask, one moment...
  python -m pip install --quiet flask
  if errorlevel 1 (
    echo   Could not install Flask. If this machine has no internet, run
    echo     python -m pip install flask
    echo   on a connected machine and copy the package over.
    pause
    exit /b 1
  )
)

start "" http://127.0.0.1:5000
python syntc_app.py
pause
