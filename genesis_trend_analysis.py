"""
genesis_trend_analysis.py
=========================
Computes two sets of results from big_perimeter_1977_2023.csv:

  1. Corrected genesis-region centroids (month × category), incorporating
     TOK_GRADE recovery for track points where TOK_WIND == 0.

  2. Spatial-trend analysis (linear regression of peak-intensity position
     and genesis-point position vs. year, per category).

Author : Jef Zerrudo  (with Claude AI assistance)
Usage  : python genesis_trend_analysis.py
Input  : big_perimeter_1977_2023.csv  (IBTrACS PAR extract)
Output : console (copy-paste into manuscript / genesis_regions dict)
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import os

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "big_perimeter_1977_2023.csv"
GRADE_MIDPOINTS = {2: 27.5, 3: 40.5, 4: 55.5, 5: 81.5}
CATS = ['TD', 'TS', 'STS', 'TY', 'STY']
YEARS_SPAN = 47  # 1977–2023


def assign_category(w):
    """PAGASA TC category from 10-min sustained wind (kt)."""
    if w >= 100: return 'STY'
    if w >= 64:  return 'TY'
    if w >= 48:  return 'STS'
    if w >= 34:  return 'TS'
    if w >= 22:  return 'TD'
    return None


def load_and_recover(path):
    """Load IBTrACS CSV and recover effective wind from TOK_GRADE."""
    df = pd.read_csv(path, low_memory=False)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['month'] = df['ISO_TIME'].dt.month

    eff = df['TOK_WIND'].astype(float).copy()
    mask = (eff == 0) & df['TOK_GRADE'].isin(GRADE_MIDPOINTS.keys())
    eff.loc[mask] = df.loc[mask, 'TOK_GRADE'].map(GRADE_MIDPOINTS)
    df['eff_wind'] = eff
    df['cat'] = df['eff_wind'].apply(assign_category)

    n_recovered = mask.sum()
    print(f"Loaded {len(df)} rows; recovered {n_recovered} wind values from TOK_GRADE")
    return df[df['cat'].notna()].copy()


# ── PART 1: Genesis-region centroids ──────────────────────────────────────

def compute_genesis_centroids(valid):
    """Print corrected genesis_regions dict entries."""
    first_pts = (valid.sort_values('ISO_TIME')
                 .groupby(['SID', 'cat']).first().reset_index())

    print("\n" + "=" * 80)
    print("CORRECTED genesis_regions ENTRIES  (first-appearance centroids)")
    print("Format: (center_lat, center_lon, lat_std, lon_std, weight)")
    print("=" * 80)

    for m in range(1, 13):
        print(f"\n    # Month {m}")
        print(f"    {m}: {{")
        for c in CATS:
            sub = first_pts[(first_pts['month'] == m) & (first_pts['cat'] == c)]
            if len(sub) < 3:
                print(f"        # '{c}': n={len(sub)} — keep existing (too few)")
                continue

            scs   = sub[sub['LON'] < 119]
            ocean = sub[sub['LON'] >= 119]
            parts = []

            if len(ocean) >= 3:
                parts.append(
                    f"({ocean.LAT.mean():.1f}, {ocean.LON.mean():.1f}, "
                    f"{max(1.0, ocean.LAT.std():.1f)}, "
                    f"{max(1.5, ocean.LON.std():.1f)}, "
                    f"{len(ocean)/len(sub):.2f})"
                )
            if len(scs) >= 2:
                parts.append(
                    f"({scs.LAT.mean():.1f}, {scs.LON.mean():.1f}, "
                    f"{max(1.0, scs.LAT.std():.1f)}, "
                    f"{max(1.5, scs.LON.std():.1f)}, "
                    f"{len(scs)/len(sub):.2f})"
                )

            if parts:
                print(f"        '{c}': [{', '.join(parts)}],  # n={len(sub)}")
        print(f"    }},")


# ── PART 2: Spatial-trend analysis ────────────────────────────────────────

def compute_trends(valid):
    """Print linear-regression trend tables."""

    # Per-storm peak-intensity position
    peak_rows = []
    for sid, grp in valid.groupby('SID'):
        idx = grp['eff_wind'].idxmax()
        r = grp.loc[idx]
        c = assign_category(r['eff_wind'])
        if c:
            peak_rows.append(dict(SID=sid, SEASON=r['SEASON'],
                                  LAT=r['LAT'], LON=r['LON'], cat=c))
    sdf = pd.DataFrame(peak_rows)

    for label, data in [
        ("PEAK-INTENSITY LOCATION vs YEAR", sdf),
        ("GENESIS-POINT LOCATION vs YEAR",
         _genesis_frame(valid, dict(zip(sdf['SID'], sdf['cat']))))
    ]:
        print(f"\n{'=' * 80}")
        print(f"SPATIAL TREND: {label}")
        print(f"{'=' * 80}")
        hdr = (f"{'Cat':<5} {'n':>4}  {'LAT slope':>10} {'LAT p':>8} "
               f"{'LON slope':>10} {'LON p':>8}  "
               f"{f'{YEARS_SPAN}yr LAT':>10} {f'{YEARS_SPAN}yr LON':>10}")
        print(hdr)
        print("-" * 80)
        for c in CATS:
            sub = data[data['cat'] == c] if 'cat' in data.columns else data[data['peak_cat'] == c]
            col = 'cat' if 'cat' in sub.columns else 'peak_cat'
            if len(sub) < 10:
                print(f"{c:<5} {len(sub):>4}  insufficient data")
                continue
            lr_lat = linregress(sub['SEASON'], sub['LAT'])
            lr_lon = linregress(sub['SEASON'], sub['LON'])
            print(f"{c:<5} {len(sub):>4}  {lr_lat.slope:>+10.4f} "
                  f"{lr_lat.pvalue:>8.4f} {lr_lon.slope:>+10.4f} "
                  f"{lr_lon.pvalue:>8.4f}  "
                  f"{lr_lat.slope * YEARS_SPAN:>+10.2f} "
                  f"{lr_lon.slope * YEARS_SPAN:>+10.2f}")


def _genesis_frame(valid, sid_to_cat):
    genesis = valid.sort_values('ISO_TIME').groupby('SID').first().reset_index()
    genesis['peak_cat'] = genesis['SID'].map(sid_to_cat)
    genesis['cat'] = genesis['peak_cat']
    return genesis[genesis['cat'].notna()]


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        # Try alternate location
        DATA_PATH = os.path.join("data", DATA_PATH)

    valid = load_and_recover(DATA_PATH)
    compute_genesis_centroids(valid)
    compute_trends(valid)
