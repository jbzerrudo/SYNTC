@echo off
REM ---------------------------------------------------------------------
REM Every figure in the paper, from a finished run, in one command.
REM
REM     make_figures.bat D:\2026\SYNTC\SYNTC-AI\run03
REM
REM Edit the two paths below once. Nothing else needs changing between runs.
REM
REM Titles are deliberately NOT drawn into the images: the LaTeX \caption is
REM the only caption. Add --titles to any line below if you want a labelled
REM version for browsing.
REM ---------------------------------------------------------------------
setlocal

set "RUN=%~1"
if "%RUN%"=="" (
  echo usage: make_figures.bat ^<run folder^>
  exit /b 1
)

set "IBTRACS=D:\2026\SYNTC\SYNTCGEN\data\IBTrACS.WP.list.v04r01.points.csv"
set "DTM=D:\2026\SYNTC\SYNTC-main\dtm_phil_1km.tif"
set "COMMON=--run "%RUN%" --ibtracs "%IBTRACS%" --dtm "%DTM%""

if not exist "%RUN%\synthetic_storms_ens01.csv" (
  echo no ensembles in %RUN%
  exit /b 1
)

echo.
echo === acceptance test =================================================
python -u check_run.py %COMMON%
if errorlevel 1 (
  echo.
  echo check_run.py FAILED. Fix the run before building figures on it.
  exit /b 1
)

echo.
echo === spatial validation ==============================================
REM Must run before plot_results.py: it writes spatial_validation.csv, which
REM supplies the r and skill printed on each hotspot panel.
python -u validate_hotspots.py %COMMON%

echo.
echo === hotspots, skill, intensity ======================================
python -u plot_results.py %COMMON% --grid 1

echo.
echo === return levels ===================================================
python -u plot_return_levels.py --ibtracs "%IBTRACS%" --dtm "%DTM%" --run "%RUN%" --compare

echo.
echo === archipelago filtering ===========================================
python -u filtering_effect.py %COMMON%
python -u plot_filtering.py %COMMON%

echo.
echo === seasonality and tracks ==========================================
python -u plot_seasonality.py %COMMON%
python -u plot_tracks.py %COMMON%

echo.
echo === the tool: genesis-conditioned forecasts =========================
REM Three genesis points chosen to span the regimes the paper discusses:
REM a Philippine Sea October storm, an early-season low-latitude storm, and
REM a late-season South China Sea storm.
python -u genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 13 --lon 132 --month 10 --n 2000 --out "%RUN%"
python -u genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 9  --lon 137 --month 7  --n 2000 --out "%RUN%"
python -u genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 16 --lon 127 --month 11 --n 2000 --out "%RUN%"

echo.
echo === done ============================================================
dir /b "%RUN%\*.pdf"
endlocal
