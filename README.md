# SynTC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21985553.svg)](https://doi.org/10.5281/zenodo.21985553)

A synthetic tropical cyclone generator for the Philippine Area of Responsibility,
and the analysis code for every figure and table in

> Zerrudo, J.B., Abdon, S.J., Arruejo, S.J., David, S., Aggasid, V.G.
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

Everything in this repository was used to produce the paper, together with the
plotting variants the figures were checked against. Nothing else is included.

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
| `replot_plume.py` | redraws the plume on a PAR-focused window, from the CSVs the tool already wrote |
| `make_table_return_periods.py` | Table 1, as LaTeX, straight from the CSV |
| `make_table_spatial.py` | Table 2, as LaTeX, straight from the CSV |
| `make_new_figs.py` | the annual-maximum comparison and the saturation-exponent sensitivity |
| `make_island_fig.py` | landfall share by island group, and the observed trend |
| `plume_pair.py` | the two-panel genesis plume, see below |
| `to_arcgis.py` | reshapes a run for the ArcGIS scripts below |
| `arcgis_csv2pts2segments.py`, `arcgis_hotspot_batch_final.py` | the ArcGIS side, used for the return-period track maps |
| `betaparams_check_finall.py` | beta fits per intensity class, the imputation basis for Figure 2 |
| `genesis_trend_analysis.py` | TOK_GRADE wind recovery and genesis trend test |
| `SORTMAXWIND.py`, `ExtremeValCalcBootStrapper_v3.py` | annual-maximum extraction and bootstrap EVA behind Table 1 |

### The genesis plume

`genesis_forecast.py` writes two files per query, a per-cell passage field and
every simulated track. It needs `run07/model.pkl`, which is not in this
repository; see *Getting the fitted model* under *Reproducing the paper*.

```
python genesis_forecast.py --model run07/model.pkl --dtm dtm_phil_1km.tif \
       --lat 13 --lon 132 --month 10 --n 10000 --out gen07
python genesis_forecast.py --model run07/model.pkl --dtm dtm_phil_1km.tif \
       --lat 10 --lon 140 --month 11 --n 10000 --out gen07
```

`plume_pair.py` draws the paper's two-panel figure from those CSVs without
re-simulating anything:

```
python plume_pair.py --gen gen07 --dtm dtm_phil_1km.tif --keep 30
```

**Choosing `--keep`.** The colour field is the full 10,000 realisations. The
tracks drawn on top are the N that follow the high-probability corridor most
closely, ranked by the 10th percentile of passage probability along the track.
The first 24 h are dropped from that ranking because every realisation starts in
the genesis cell, where the probability is 1 by construction, and a track that
dissipates early would otherwise outrank one that runs the corridor end to end.
A low quantile rather than a mean is used so that a realisation cannot recurve
out of the corridor and still rank well on the leg it spent inside.

| `--keep` | how it reads |
|---|---|
| 5 | too sparse; five lines do not establish a corridor, and the November panel looks like noise |
| 10 | corridor is legible in the October panel, still thin through the Visayas in the November one |
| **15** | **the figure in the paper.** Both panels read as a corridor and individual tracks are still followable |
| 20 | fine, slightly denser |
| 25 and above | tracks begin to obscure the field they are drawn on |

Other options: `--left` and `--right` select which genesis pair goes in which
panel, `--left-pt`/`--right-pt` set the marker coordinates, `--left-label`/
`--right-label` set the panel headings, and `--out` sets the output stem. The
window is 112–145°E and 2–28°N and holds 49.5% of the October case's passage
probability; the remainder lies northeast of it, where storms recurve out of PAR.

`replot_plume.py` still draws the older single-panel version, `--style field`
for the probability field and `--style spaghetti` for tracks alone.


## Reproducing the paper

**Getting the fitted model.** `.gitignore` excludes `*.pkl` and `run0*/`, so
neither the fitted model nor the catalogues are in this repository or in the
Zenodo source archive. There are two routes. Rebuild them with the commands
below, about 23 minutes for the pair, or download `model.pkl` from the release
assets at https://github.com/jbzerrudo/SYNTC/releases. Prefer the download if
you want the published numbers to the decimal: a rebuild is seeded and
reproduces the results within sampling noise, but threaded reductions sum in a
different order on different hardware, so the refitted weights are
statistically equivalent rather than bit-identical.

Two catalogues are needed: a stationary control and a warming experiment. They
differ only in the potential-intensity trend, including the random seed, so any
difference between them is attributable to that trend alone.

```
python syntc_ai.py --ibtracs IBTrACS.WP.list.v04r01.points.csv \
                   --dtm dtm_phil_1km.tif --out run07 \
                   --ensembles 20 --years 100 --mpi-trend 0.0
python syntc_ai.py --ibtracs ... --dtm ... --out run08 \
                   --ensembles 20 --years 100 --mpi-trend 4.0
```

`run_both_07.bat` runs the pair on Windows. Then, for each run:

```
make_figures.bat run07
```

`REGENERATE_FIGURES.bat` rebuilds every figure in the paper in one pass, from the
spatial validation through to the plume. It expects the saturation-exponent scout
in `scout_k_data/scoutk_06` .. `scoutk_12`; without it every figure but
`fig_saturation_tradeoff` is still produced.

which runs the acceptance test first and stops if it fails, then
`validate_hotspots.py` (whose output supplies the *r* and skill printed on the
hotspot panels), then every plotting script in order.

### Figure by figure

| paper figure | command |
|---|---|
| 1 area of interest | ArcGIS, not scripted |
| 2 pipeline schematic | drawio, not scripted |
| 3 wind imputation | `python betaparams_check_finall.py` |
| 4 intensity distribution | `python plot_results.py --run run07 --ibtracs IB --dtm DTM --grid 1` |
| 5 spatial skill | `python validate_hotspots.py --run run07 --ibtracs IB --dtm DTM` then as above |
| 6 annual maxima | `python make_new_figs.py --run run07 --ibtracs IB --dtm DTM` |
| 7 return-period tracks | `python to_arcgis.py --run run07`, then `arcgis_csv2pts2segments.py` |
| 8 return levels | `python plot_return_levels.py --ibtracs IB --dtm DTM --run run07 --compare` |
| 9 hotspots by class | `arcgis_hotspot_batch_final.py` |
| 10 seasonality | `python plot_seasonality.py --run run07 --ibtracs IB --dtm DTM` |
| 11 island landfall | `python make_island_fig.py --run run07 --ibtracs IB --dtm DTM` |
| 12 seasonal shift | `python plot_tracks.py --run run07 --ibtracs IB --dtm DTM` |
| 13 genesis plume | `python genesis_forecast.py --model run07/model.pkl --dtm DTM --lat 13 --lon 132 --month 10 --n 10000`, then `python plume_pair.py --keep 30` |
| 14 filtering effect | `python plot_filtering.py --run run07 --ibtracs IB --dtm DTM` |
| 15 hotspots by month | `arcgis_hotspot_batch_final.py` |
| 16 saturation tradeoff | `python make_new_figs.py --scout scout_k_data` |
| Table 1 | `python make_table_return_periods.py --run run07 > tab_return_periods.tex` |
| Table 2 | `python make_table_spatial.py --run run07 > tab_spatial.tex` |

Figure titles are **off** by default, because a journal figure carries its
caption in LaTeX and burning the same words into the image duplicates them. Pass
`--titles` to any plotting script when browsing a run folder.

## Model configurations: what the paper used, and what came after

Two configurations exist in this repository. They differ in exactly one
parameter and are kept separate on purpose.

| | `run07` / `run08` | `run09` / `run10` |
|---|---|---|
| track memory | 1 past displacement step | 3 past displacement steps |
| built by | `run_both_07.bat` | `run_both_09.bat` |
| status | **the published configuration** | later work, not in the paper |

**Everything reported in the manuscript comes from `run07`**, the stationary
control, with `run08` as its warming twin. Every number, table and figure in the
paper is reproducible from that pair alone, and `REGENERATE_FIGURES.bat` is
pointed at it. Do not mix `run09` output into a comparison with published
values.

`run09` extends the track propagator's memory from one past displacement to
three, leaving the mixture output, the intensity model, the terrain decay and
the validation framework untouched. Fitted on the same 30,997 transitions with
the same seed and scored on the same held-out 5,910 transitions of 2015--2023,
it raises the mean log likelihood by 0.044, with a bootstrap 95% interval of
0.032 to 0.056. That gain survives controls against a wider network at memory 1
and against six random noise features, so it comes from the lag information
rather than from added parameters.

What it does not change is the archipelago filtering result: median loss for
storms crossing at 100 kt or more is 24% under `run07` and 23% under `run09`,
every such crosser emerges below threshold in both, and the aggregate spatial
skill is the same to two decimals. What it does change is delivery frequency,
since straighter tracks bring more storms to the coast: land crossings rise
from 4.83 to 5.06 per season and intense landfalls from 10.7 to 13.8 per 100
seasons. Both sit inside the sampling range of the five such storms the
47-season record contains, so the record cannot separate the two
configurations on that quantity.

`data.py` and `syntc_ai.py` support both. `track_memory_order` defaults to 1,
so the published behaviour is what you get unless `--memory-order` is passed.

```
python syntc_ai.py --ibtracs IB --dtm DTM --out run09 \
                   --ensembles 20 --years 100 --mpi-trend 0.0 --memory-order 3
```

## Using the fitted model on its own

A generation run serialises its fitted model. The tool loads that file and
re-queries it without refitting anything, so a forecast and the paper's
catalogue provably come from the same fit:

```
python genesis_forecast.py --model run07/model.pkl --dtm dtm_phil_1km.tif \
       --lat 13 --lon 132 --month 10 --n 10000
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
with the PAGASA operational convention. The transition densities are fitted to
dequantised winds, which is correct for a density; the potential-intensity
ceiling is built from the raw reported winds, because dequantising an extremum
lifts it above anything on record. The ceiling therefore tops out at the basin
record of 140 kt. Seasons through 2009 are used for
fitting, 2010–2014 for early stopping, and 2015–2023 are held out entirely.

## Known limitations

Quantified in the paper rather than asserted. The intensity tail is thin: 2.5%
of synthetic PAR track points reach super typhoon against 6.2% observed, and
8.3% of storms peak at super typhoon strength inside PAR against 16.3% observed,
so attainment runs at about half the observed rate. No extratropical transition
is modelled, so 36.1% of synthetic track points lie north of 25°N against 14.7%
observed, and recurved storms stay alive too long. The overland extreme is
overstated: four of twenty ensembles exceed the 95th percentile of a matched
100-season benchmark where one is expected (binomial p = 0.016), so
`check_run.py` fails on that criterion by design rather than being tuned to
pass. Terrain is resolved only inside the Philippine digital terrain model, which
spans 4.5-21.2°N and 116.9-127.3°E. A storm outside that raster is treated as
being over open ocean: mean elevation is zero and the over-land flag is false, so
the decay relation is never evaluated and the storm crosses unweakened. Taiwan
Island and Sabah are the two landmasses this affects inside PAR, carrying 1.1% and
0.1% of synthetic PAR track points. Philippine landfall statistics are unaffected.

## Licence and citation

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`.

If you use this code, cite both the software and the paper. For the software
use the concept DOI, **10.5281/zenodo.21985553**, which always resolves to the
current version. The release accompanying the paper is v1.0.0,
10.5281/zenodo.21985554.
