@echo off
REM ===========================================================================
REM Regenerate every figure in the manuscript on this machine.
REM Run from D:\2026\SYNTC\SYNTC  (the code folder, where terrain.py lives).
REM Nothing here contacts a network. All output is produced locally.
REM The window stays open at the end so you can read the console.
REM
REM Step 3 reads the saturation-exponent scout from scout_k_data\scoutk_06 .. _12.
REM If that folder is missing, step 3 stops and fig_saturation_tradeoff is not
REM produced. Change --scout below if you unzipped it somewhere else.
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "SYNTC=D:\2026\SYNTC"
set "IB=%SYNTC%\SYNTCGEN\data\IBTrACS.WP.list.v04r01.points.csv"
set "DTM=%SYNTC%\SYNTC-main\dtm_phil_1km.tif"
set "RUN=%SYNTC%\SYNTC-AI\run07"
set "OUT=%SYNTC%\LaTeX"

echo [1/6] spatial validation  (supplies r and skill for the hotspot panels)
python validate_hotspots.py --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%"

echo [2/6] main figure set
python plot_results.py       --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%" --grid 1
python plot_filtering.py     --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%"
python plot_seasonality.py   --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%"
python plot_tracks.py        --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%"
python plot_return_levels.py --run "%RUN%" --ibtracs "%IB%" --dtm "%DTM%" --compare

echo [3/6] annual-maximum and saturation-exponent figures
python make_new_figs.py --ibtracs "%IB%" --run "%RUN%" --dtm "%DTM%" --scout scout_k_data

echo [3b/6] landfall by island group
python make_island_fig.py --ibtracs "%IB%" --run "%RUN%" --dtm "%DTM%" --out fig_island_landfall

echo [4/6] genesis plume, all four cases  (about 20 minutes each at n=10000)
REM The first two feed the two-panel figure. The second two are text only:
REM they supply the July and November cases quoted in Section 4, so that
REM every number in that section comes out of this script.
python genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 13 --lon 132 --month 10 --n 10000 --out gen07
python genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 10 --lon 140 --month 11 --n 10000 --out gen07
python genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 9  --lon 137 --month 7  --n 10000 --out gen07
python genesis_forecast.py --model "%RUN%\model.pkl" --dtm "%DTM%" --lat 16 --lon 127 --month 11 --n 10000 --out gen07

echo [5/6] two-panel plume
python plume_pair.py --gen gen07 --dtm "%DTM%" --keep 30 --out genesis_plume_pair

echo [6/6] copying the PDFs the manuscript includes into %OUT%
for %%F in (intensity_distribution hotspots_by_category hotspots_by_month skill_summary ^
            filtering_effect seasonality seasonal_shift return_levels) do (
  copy /Y "%RUN%\%%F.pdf" "%OUT%\" >nul
)
copy /Y fig_annual_maxima.pdf      "%OUT%\" >nul
copy /Y fig_saturation_tradeoff.pdf "%OUT%\" >nul
copy /Y genesis_plume_pair.pdf     "%OUT%\" >nul
copy /Y fig_island_landfall.pdf    "%OUT%\" >nul

echo.
echo Done. Every figure in %OUT% was produced on this machine.
echo.
echo If any step above printed a Traceback, that figure was not rebuilt.
pause
endlocal
