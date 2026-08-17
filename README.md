# SynTC

A synthetic tropical cyclone generator for the Philippine Area of Responsibility,
and the analysis code for every figure and table in

> Zerrudo, J.B., Arruejo, S.J., Abdon, S.J., David, S., Aggasid, V.G.
> *Philippine Tropical Cyclone Extreme-Value Analysis and Intensity Hotspots
> from Historical Data and Synthetic Storm Modeling.*
> Submitted to Tropical Cyclone Research and Review.

Track propagation and intensity evolution are learned as conditional densities
from the IBTrACS Western North Pacific record, 1977–2023, using mixture density
networks. Terrain-dependent wind decay, central pressure, radius of maximum wind
and the radial wind profile are taken from published relations and used
unchanged. No post-hoc correction is applied to the output: where the catalogue
disagrees with the record, the disagreement is reported rather than removed.

## What is here

Everything in this repository was used to produce the paper. Nothing else is
included.

**The model**

| file | what it does |
|---|---|
| `syntc_ai.py` | the generator, the CLI, and `simulate_from_genesis` |
| `data.py` | IBTrACS loading, quality control, dequantisation |
| `models.py` | the two mixture density networks |
| `intensity.py` | intensity evolution and the potential-intensity ceiling |
| `terrain.py` | digital terrain model, footprint elevation and land fraction |
| `figstyle.py` | the shared `--titles` switch for figure output |

**Checks — run these before trusting a catalogue**

| file | what it does |
|---|---|
| `check_run.py` | acceptance test; exits non-zero on failure |
| `validate_hotspots.py` | spatial correlation and Murphy skill against a bootstrap noise floor |
| `return_levels.py` | extreme-value fits and bootstrap confidence intervals |
| `filtering_effect.py` | land-crossing statistics |
| `compare_runs.py` | the stationary catalogue against the warming one |

**Figures and tables**

| file | produces |
|---|---|
| `plot_results.py` | hotspots by class and by month, skill summary, intensity distribution |
| `plot_return_levels.py` | the return-level plot |
| `plot_filtering.py` | the archipelago filtering measurement |
| `plot_seasonality.py` | the seasonal cycle of PAR entry |
| `plot_tracks.py` | the seasonal migration of the track corridor |
| `genesis_forecast.py` | the tool: probability of passage from one genesis point |
| `make_table_return_periods.py` | Table 2, as LaTeX, straight from the CSV |
| `to_arcgis.py` | reshapes a run for the ArcGIS scripts below |
| `arcgis_csv2pts2segments.py`, `arcgis_hotspot_batch_final.py` | the ArcGIS side, used for the return-period track maps |

## Reproducing the paper

Two catalogues are needed: a stationary control and a warming experiment. They
differ only in the potential-intensity trend, including the random seed, so any
difference between them is attributable to that trend alone.

```
python syntc_ai.py --ibtracs IBTrACS.WP.list.v04r01.points.csv \
                   --dtm dtm_phil_1km.tif --out run03 \
                   --ensembles 20 --years 100 --mpi-trend 0.0
python syntc_ai.py --ibtracs ... --dtm ... --out run04 \
                   --ensembles 20 --years 100 --mpi-trend 4.0
```

`run_both.bat` runs the pair on Windows. Then, for each run:

```
make_figures.bat run03
```

which runs the acceptance test first and stops if it fails, then
`validate_hotspots.py` (whose output supplies the *r* and skill printed on the
hotspot panels), then every plotting script in order.

### Figure by figure

| paper figure | command |
|---|---|
| 3 intensity distribution | `python plot_results.py --run run03 --ibtracs IB --dtm DTM --grid 1` |
| 4 spatial skill | `python validate_hotspots.py --run run03 --ibtracs IB --dtm DTM` then as above |
| 5 return-period tracks | `python to_arcgis.py --run run03`, then the ArcGIS scripts |
| 6 return levels | `python plot_return_levels.py --ibtracs IB --dtm DTM --run run03 --compare` |
| 7 hotspots by class | `plot_results.py`, as figure 3 |
| 8 seasonality | `python plot_seasonality.py --run run03 --ibtracs IB --dtm DTM` |
| 9 seasonal shift | `python plot_tracks.py --run run03 --ibtracs IB --dtm DTM` |
| 10 genesis plume | `python genesis_forecast.py --model run03/model.pkl --dtm DTM --lat 13 --lon 132 --month 10 --n 2000` |
| 11 filtering effect | `python plot_filtering.py --run run03 --ibtracs IB --dtm DTM` |
| 12 hotspots by month | `plot_results.py`, as figure 3 |
| Table 2 | `python make_table_return_periods.py --run run03 > tab_return_periods.tex` |

Figure titles are **off** by default, because a journal figure carries its
caption in LaTeX and burning the same words into the image duplicates them. Pass
`--titles` to any plotting script when browsing a run folder.

## Using the fitted model on its own

A generation run serialises its fitted model. The tool loads that file and
re-queries it without refitting anything, so a forecast and the paper's
catalogue provably come from the same fit:

```
python genesis_forecast.py --model run03/model.pkl --dtm dtm_phil_1km.tif \
       --lat 13 --lon 132 --month 10 --n 2000
```

It reports the chance of entering PAR, of a Philippine landfall and where, the
time to PAR entry, and the peak intensity distribution, and it writes the
probability-of-passage field, every simulated track, and the figure.

These are probabilities **conditional** on a storm forming at that point in that
month. They are not the probability that one forms; multiply by your own genesis
rate for an absolute risk. This is not a forecast of a particular storm: it
carries no information about the synoptic state or the vortex on the day, so its
spread is climatological.

## Data

The terrain model is included, at `dtm_phil_1km.tif` with its `.tfw`, because
the decay relation of the paper depends on it and a reader reproducing that
result should be using the same raster rather than one they sourced themselves.
It is 1.6 MB.

IBTrACS is not included. It is 25 MB, it is versioned and updated at source, and
shipping a frozen copy invites someone to analyse a stale archive:

- **IBTrACS v04r01**, Western North Pacific points file, from NOAA NCEI:
  https://www.ncei.noaa.gov/products/international-best-track-archive

Intensity is the RSMC Tokyo 10-minute sustained wind (`TOK_WIND`), consistent
with the PAGASA operational convention. Seasons through 2009 are used for
fitting, 2010–2014 for early stopping, and 2015–2023 are held out entirely.

## Known limitations

Quantified in the paper rather than asserted. The intensity tail is thin: 2.4%
of synthetic PAR track points reach super typhoon against 6.2% observed. No
extratropical transition is modelled, so 36.4% of synthetic track points lie
north of 25°N against 14.7% observed, and recurved storms stay alive too long.
The potential-intensity ceiling is built on dequantised observed winds and so
tops out at 142.19 kt against a basin record of 140. Terrain is resolved only
inside the Philippine digital terrain model; storms crossing Taiwan and other
land receive a default elevation.

## Licence and citation

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`. If you use this
code, please cite both the software (Zenodo DOI, below) and the paper.
