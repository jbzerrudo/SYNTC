@echo off
rem ===================================================================
rem  Builds SynTC.exe from the SynTC repository. Run this once.
rem
rem  It takes roughly 10 to 25 minutes, needs about 6 GB of free disk
rem  while it works, and produces dist\SynTC\ of about 1 to 2 GB. The
rem  size is torch and GDAL, not the model: model.pkl is 220 KB.
rem
rem  To move the tool to another machine afterwards, copy the whole
rem  dist\SynTC folder. Nothing needs to be installed there.
rem ===================================================================
setlocal
cd /d "%~dp0"

if "%SYNTC_REPO%"==""  set "SYNTC_REPO=D:\2026\SYNTC\SYNTC"
rem  Defaults to the standard model, the one the manuscript describes. Set
rem  SYNTC_MODEL to build
rem  from a different catalogue.
if "%SYNTC_MODEL%"=="" set "SYNTC_MODEL=D:\2026\SYNTC\SYNTC-AI\run07\model.pkl"

echo.
echo   repo   %SYNTC_REPO%
echo   model  %SYNTC_MODEL%
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo   [stop] Python was not found on PATH.
  goto :fail
)

if not exist "%SYNTC_REPO%\genesis_forecast.py" (
  echo   [stop] genesis_forecast.py is not in %SYNTC_REPO%
  echo          Edit SYNTC_REPO at the top of this file.
  goto :fail
)
if not exist "%SYNTC_REPO%\dtm_phil_1km.tif" (
  echo   [stop] dtm_phil_1km.tif is not in %SYNTC_REPO%
  goto :fail
)
if not exist "%SYNTC_MODEL%" (
  echo   [stop] model.pkl is not at %SYNTC_MODEL%
  echo          Edit SYNTC_MODEL at the top of this file.
  goto :fail
)
if not exist "syntc_gui.py" (
  echo   [stop] syntc_gui.py must sit in this folder, beside SynTC.spec.
  goto :fail
)

echo   checking that the runtime packages import...
python -c "import numpy,pandas,scipy,torch,rasterio,matplotlib,PIL,tkinter"
if errorlevel 1 (
  echo.
  echo   [stop] One of the packages above will not import, so PyInstaller
  echo          cannot freeze it either. Fix that first:
  echo            python -m pip install -r "%SYNTC_REPO%\requirements.txt" pillow
  goto :fail
)

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
  echo   installing PyInstaller...
  python -m pip install --quiet "pyinstaller>=6.0"
  if errorlevel 1 (
    echo   [stop] PyInstaller would not install.
    goto :fail
  )
)

echo.
echo   Close SynTC.exe and any Explorer window inside dist\SynTC first.
echo   PyInstaller clears that folder and Windows will block it otherwise.
echo.
echo   building. This is the long part, leave it alone.
echo.
python -m PyInstaller --noconfirm --clean SynTC.spec
if errorlevel 1 goto :fail

if not exist "dist\SynTC\SynTC.exe" (
  echo   [stop] The build reported success but dist\SynTC\SynTC.exe is missing.
  goto :fail
)

echo.
echo   ================================================================
echo    Done.  %~dp0dist\SynTC\SynTC.exe
echo.
echo    Double-click that. Runs are written to %~dp0forecast, which is
echo    OUTSIDE dist on purpose: PyInstaller clears dist\SynTC on every
echo    build, so anything kept in there is destroyed by the next one.
echo    Set SYNTC_OUT to send runs somewhere else.
echo.
echo    To use a retrained model later, drop the new model.pkl next to
echo    SynTC.exe; the tool reads that copy first and only falls back
echo    to the one baked into the bundle.
echo   ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo   Build did not complete.
pause
exit /b 1
