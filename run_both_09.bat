@echo off
REM ===========================================================================
REM  run_both_09.bat  --  the memory-order-3 pair.
REM
REM    run09  --mpi-trend 0.0 --memory-order 3   stationary control
REM    run10  --mpi-trend 4.0 --memory-order 3   warming experiment
REM
REM  Identical to run_both_07.bat in every other respect, so run09 against
REM  run07 isolates the one change: the track propagator sees three past
REM  displacement steps instead of one. Measured worth on held-out data:
REM  +0.0439 nats, 95%% CI +0.0324 to +0.0561, controlled against both a
REM  noise placebo and a wider network.
REM
REM  run07 and run08 are untouched. Nothing here writes outside its own --out.
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "SYNTC=D:\2026\SYNTC"
set "OUT=%SYNTC%\SYNTC-AI"
set "IBTRACS=%SYNTC%\SYNTCGEN\data\IBTrACS.WP.list.v04r01.points.csv"
set "DTM=%SYNTC%\SYNTC-main\dtm_phil_1km.tif"

REM Half the cores each. Two processes at half threads beat one at full threads
REM here because the per-step generation loop does not scale across cores, so
REM the second process uses capacity that would otherwise idle. Expect roughly
REM 1.5x, not 2x.
set /a THREADS=%NUMBER_OF_PROCESSORS%/2
if %THREADS% LSS 1 set THREADS=1

REM A half-finished run left behind is the dangerous case: the next attempt
REM overwrites ensembles one at a time, so an interrupted rerun leaves a folder
REM holding some fresh members and some stale ones, and every script downstream
REM reads all twenty without complaint. Move any existing folder aside rather
REM than deleting it, so nothing is ever destroyed on your behalf.
call :archive "%OUT%\run09" || goto :fail
call :archive "%OUT%\run10" || goto :fail

echo Cores detected: %NUMBER_OF_PROCESSORS%   threads per run: %THREADS%
echo.

start "SynTC run09 stationary, memory 3" powershell -NoLogo -NoExit -Command "$env:OMP_NUM_THREADS='%THREADS%'; $env:MKL_NUM_THREADS='%THREADS%'; Set-Location '%~dp0'; python -u syntc_ai.py --years 2026 2125 --ensembles 20 --ibtracs '%IBTRACS%' --dtm '%DTM%' --out '%OUT%\run09' --mpi-trend 0.0 --memory-order 3 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath '%OUT%\run09.log'"

start "SynTC run10 warming, memory 3" powershell -NoLogo -NoExit -Command "$env:OMP_NUM_THREADS='%THREADS%'; $env:MKL_NUM_THREADS='%THREADS%'; Set-Location '%~dp0'; python -u syntc_ai.py --years 2026 2125 --ensembles 20 --ibtracs '%IBTRACS%' --dtm '%DTM%' --out '%OUT%\run10' --mpi-trend 4.0 --memory-order 3 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath '%OUT%\run10.log'"

echo Two windows launched, one per run, showing live progress.
echo Logs written at the same time to:
echo    %OUT%\run09.log
echo    %OUT%\run10.log
echo.
echo Confirm the banner reads "PAR-entering ... mean 16.2" in BOTH windows
echo before you walk away.
echo.
echo The windows stay open when the runs finish, so the summaries remain
echo readable. Close them yourself.
goto :eof

:archive
if not exist "%~1\" exit /b 0
set "DEST=%~1_aborted_%RANDOM%%RANDOM%"
if exist "%DEST%\" echo   ERROR: %DEST% exists. Move %~nx1 aside yourself. & exit /b 1
move "%~1" "%DEST%" >nul || (echo   ERROR: could not move %~1 & exit /b 1)
echo   existing %~nx1 moved aside to %DEST%
exit /b 0

:fail
echo.
echo Nothing was launched.
exit /b 1
