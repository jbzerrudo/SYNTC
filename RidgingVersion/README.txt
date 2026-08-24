RidgingVersion
==============

A post-submission visualization of genesis_forecast.py. It lives in this
separate folder, deliberately apart from the paper's method files at the repo
root. It is NOT the method described in the submitted manuscript, and it does
not change any result in it. The manuscript's genesis_forecast.py is the copy
at the repo root, and that copy is unchanged.

What it adds to the genesis map
-------------------------------
  --central ridge   a blue line tracing the crest of the passage-probability
                    field, from the genesis point outward. This is the plume's
                    core, and it is what a per-step median cannot follow when
                    the plume splits into a westward branch and a recurving one.
  --pick ridge      draws the realisations that stay within --corridor km of
                    the ridge, and states in the legend how many that is.
  --top N           draws only the N realisations closest to the ridge.
  --underlay        overlays all realisations faintly, the check that the ridge
                    sits on the real density rather than being a drawn-in line.
  auto-crop frame   the map is cropped to the plume instead of the whole basin.

The corridor width is a chosen scale, not a calibration
-------------------------------------------------------
--corridor defaults to 150 km. That is about the radius of an average tropical
cyclone (a 200-500 km diameter is a 100-250 km radius) and it sits inside the
plume's own measured spread of roughly 50-185 km. It is NOT fitted to observed
tracks. The percentage the legend reports ("N of M realisations within 150 km")
depends on that width and is a property of the tube, not a forecast
probability. A reliability test against 1977-2023 observed storms would be
needed before that percentage is treated as calibrated.

The blue line follows the model, not the other way round
--------------------------------------------------------
SynTC generates realisations with no knowledge of any ridge. The ridge is
computed afterwards from the field those realisations produce. It marks where
the model concentrates; it does not steer the model.

What it does NOT change
-----------------------
Nothing in the paper. The storm generation, the passage CSVs, and every number
in Section 4 are produced by code paths this file does not touch. Only the
drawing of the single genesis map changes. plume_pair.py (manuscript Figure 13)
is unchanged and still uses the committed, no-ridge version.

Relationship to run09
---------------------
Separate. run09 is a model change (three-step displacement memory). The ridge
is a visualization change. They are not mixed here.

How SynTC.exe uses this file
----------------------------
SynTC.spec bundles RidgingVersion/genesis_forecast.py as the exe's genesis-map
engine, sourced from here so the repo-root genesis_forecast.py (the paper's
method) is left alone. Override with the SYNTC_GENESIS environment variable;
the build falls back to the repo root if this folder is absent.
