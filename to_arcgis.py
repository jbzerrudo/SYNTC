"""
Reshape a SynTC run into the folder layout the existing ArcGIS batch scripts
expect, without regenerating anything.

    python to_arcgis.py --run ./run02

`hotspot_batch_final.py` globs `ensemble_member_*` folders under ENSEMBLE_DIR
and then `synthetic_storms_*_*.csv` inside each one. SynTC writes a flat
`synthetic_storms_ens01.csv` per ensemble, which that pattern does not match.
This copies the files into the expected shape so the ArcGIS side needs no
edits: point ENSEMBLE_DIR at <run>/ENSEMBLE and run it unchanged.

Column names already line up. The batch scripts select on WIND and MONTH and
build points from LAT, LON and SID, and SynTC writes all of those. The extra
columns SynTC adds (IN_PAR, OVER_LAND, LAND_FRAC, R34/R50/R64) are carried
through and ignored by ArcGIS, which is harmless and useful to keep.

One difference worth knowing: the batch scripts classify by SQL on WIND
(`WIND >= 64 AND WIND < 100` for TY, and so on) rather than by SynTC's CATEGORY
column. So the hotspot classes come from the wind values directly and do not
depend on how SynTC labelled them.
"""

import argparse
import glob
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a SynTC run folder")
    ap.add_argument("--out", default=None,
                    help="defaults to <run>/ENSEMBLE")
    ap.add_argument("--stamp", default="00000000_000000",
                    help="timestamp suffix on the member folders; any value "
                         "works, the scripts only glob on the prefix")
    ap.add_argument("--combined", action="store_true",
                    help="also write combined_all_ensembles.csv, which "
                         "hotspot_batch_final.py uses in PROCESS_COMBINED mode")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {a.run}")

    out = a.out or os.path.join(a.run, "ENSEMBLE")
    os.makedirs(out, exist_ok=True)

    years = None
    parts = []
    for i, src in enumerate(files, start=1):
        df = pd.read_csv(src)
        if years is None:
            years = (int(df.YEAR.min()), int(df.YEAR.max()))
        member = os.path.join(out, f"ensemble_member_{i}_{a.stamp}")
        os.makedirs(member, exist_ok=True)
        dst = os.path.join(member, f"synthetic_storms_{years[0]}_{years[1]}.csv")
        shutil.copyfile(src, dst)
        print(f"  {os.path.basename(src)} -> "
              f"ensemble_member_{i}_{a.stamp}/{os.path.basename(dst)}")
        if a.combined:
            parts.append(df)

    if a.combined:
        allpts = pd.concat(parts, ignore_index=True)
        path = os.path.join(out, "combined_all_ensembles.csv")
        allpts.to_csv(path, index=False)
        print(f"  combined: {len(allpts):,} points -> {path}")

    print(f"\nSet ENSEMBLE_DIR in hotspot_batch_final.py to:\n  {out}")


if __name__ == "__main__":
    main()
