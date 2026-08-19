@echo off
REM ===========================================================================
REM run_both_07.bat  -  the corrected-physics pair.
REM Identical to run_both.bat except that it writes run07 and run08 instead of
REM run05 and run06, so the published catalogues are left untouched for the
REM before/after comparison. Requires the patched syntc_ai.py, intensity.py,
REM data.py and windconv.py.
REM ===========================================================================
REM ---------------------------------------------------------------------------
REM Launches the two SynTC catalogues side by side.
REM
REM   run07  --mpi-trend 0.0   stationary control, the validation catalogue
REM   run08  --mpi-trend 4.0   warming experiment
REM
REM Safe to run in parallel: nothing in the code writes outside its own --out
REM folder, IBTrACS and the DTM are opened read-only, and the potential
REM intensity table is built in memory rather than cached to disk. The only
REM shared resources are CPU and RAM.
REM
REM CHECK THE BANNER. Each window prints, before generating:
REM
REM     historical PAR-entering storms/year: mean 16.2, range 10-23
REM
REM If it says "basin" or a mean near 25, stop immediately: the run is filling
REM PAR with a basin-wide storm count and every hazard number downstream will be
REM about 55% too high.
REM
REM Why python -u
REM -------------
REM Python block-buffers stdout whenever it is not writing to a console. Send it
REM to a file or a pipe and nothing appears until an eight kilobyte block fills,
REM so a run that is working perfectly looks dead for its first several minutes.
REM -u turns that off and output arrives line by line. Only stderr is unbuffered
REM by default, which is why a torch warning can show up in an otherwise empty
REM log.
REM
REM Why the ForEach-Object in the pipeline
REM --------------------------------------
REM cmd has no tee, so a plain > sends output to the log and leaves the window
REM blank. Tee-Object writes both at once. But PowerShell wraps anything a
REM native command sends to stderr into an ErrorRecord, so 2>&1 alone renders a
REM harmless torch UserWarning as a red NativeCommandError block complete with a
REM fake "At line:1 char:112" pointer. Casting each item to [string] first
REM strips that decoration and the warning prints as the plain text it is.
REM
REM Why both get the SAME thread count
REM ----------------------------------
REM Config.seed is 42 in both runs, so the two model fits are identical and each
REM ensemble's genesis draws come off the same stream. The catalogues start from
REM common ground, so the run07 against run08 comparison carries less sampling
REM noise than two independently seeded runs would.
REM
REM They do not stay exactly paired. As soon as a storm in one run lives longer
REM than its counterpart in the other, the two consume different numbers of
REM random draws and the streams desynchronise. Read run07 against run08 as a
REM low-noise comparison, not a matched-pair one.
REM
REM The identical model fit is what depends on the thread count. Threaded
REM reductions in torch and numpy can sum in a different order at different
REM thread counts, which perturbs the trained weights in the last few decimals.
REM Same THREADS in both keeps the two fits identical. Do not tune one and not
REM the other.
REM ---------------------------------------------------------------------------
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
call :archive "%OUT%\run07" || goto :fail
call :archive "%OUT%\run08" || goto :fail

echo Cores detected: %NUMBER_OF_PROCESSORS%   threads per run: %THREADS%
echo.

start "SynTC run07 stationary" powershell -NoLogo -NoExit -Command "$env:OMP_NUM_THREADS='%THREADS%'; $env:MKL_NUM_THREADS='%THREADS%'; Set-Location '%~dp0'; python -u syntc_ai.py --years 2026 2125 --ensembles 20 --ibtracs '%IBTRACS%' --dtm '%DTM%' --out '%OUT%\run07' --mpi-trend 0.0 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath '%OUT%\run07.log'"

start "SynTC run08 warming" powershell -NoLogo -NoExit -Command "$env:OMP_NUM_THREADS='%THREADS%'; $env:MKL_NUM_THREADS='%THREADS%'; Set-Location '%~dp0'; python -u syntc_ai.py --years 2026 2125 --ensembles 20 --ibtracs '%IBTRACS%' --dtm '%DTM%' --out '%OUT%\run08' --mpi-trend 4.0 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath '%OUT%\run08.log'"

echo Two windows launched, one per run, showing live progress.
echo Logs written at the same time to:
echo    %OUT%\run07.log
echo    %OUT%\run08.log
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
