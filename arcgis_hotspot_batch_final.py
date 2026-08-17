"""
hotspot_batch.py — Full Batch Hotspot Pipeline for SynTC Ensembles

Three sequential stages per category (TD, TS, STS, TY, STY):

  STAGE 1 — Merger + Integrator:
    Select by category → Copy → Clip to PAR → Project UTM 51N →
    Recopy → Merge with BIG_PERL_PTS → Integrate (35 km)

  STAGE 2 — Hot Spot Maker:
    Collect Events → Incremental Spatial Autocorrelation
    (ICOUNT, 10 bands) → Optimized Hot Spot Analysis →
    Add Latitude → Add Longitude

  STAGE 3 — IDW Renderer:
    IDW (GiZScore, 350 m cell) → Clip to PAR

Usage: cd to the directory
e.g.: "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" hotspot_batch_final.py

Author: Jef Zerrudo (DOST-PAGASA), with Claude AI optimization.
"""

import os
import sys
import time
import glob
import logging
import arcpy
from arcpy.sa import *   # brings in Idw, RadiusVariable

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ══════════════════════════════════════════════════════════════════
# SETTINGS — Edit these before running
# ══════════════════════════════════════════════════════════════════

ENSEMBLE_DIR = r"D:\2026\SYNTC\SYNTC-main\outputs\ENSEMBLE"
OUTPUT_DIR   = r"D:\2026\SYNTC\arcmap\HOTSPOTS_TEST"
PAR_SHP_WGS84 = r"D:\2025\SYNTC\PreJUNE2025\GEV2025B\SHAP\PAR_WGS84.shp"
PAR_SHP      = r"D:\2025\SYNTC\PreJUNE2025\GEV2025B\SHAP\PARUTMZN51N.shp"
BIG_PERL_PTS = r"D:\2026\SYNTC\arcmap\Shapes\PAR_vertices.shp"

CSV_PATTERN  = "synthetic_storms_*_*.csv"

SR_WGS84  = arcpy.SpatialReference(4326)
SR_UTM51N = arcpy.SpatialReference(32651)

INTEGRATE_DIST = "35000 Meters"
IDW_CELL_SIZE  = 350          # meters (UTM)
IDW_POWER      = 2
ISA_NUM_BANDS  = 10

# ── Run mode: "category" or "monthly" or "both" ──
RUN_MODE = "both"

CATEGORIES = {
    "TD":      "WIND >= 22 AND WIND < 34",
    "TS":      "WIND >= 34 AND WIND < 48",
    "STS":     "WIND >= 48 AND WIND < 64",
    "TY":      "WIND >= 64 AND WIND < 100",
    "STY":     "WIND >= 100",
    "All_TCs": "WIND >= 22",
}

MONTHLY = {
    "Jan": "MONTH = 1",
    "Feb": "MONTH = 2",
    "Mar": "MONTH = 3",
    "Apr": "MONTH = 4",
    "May": "MONTH = 5",
    "Jun": "MONTH = 6",
    "Jul": "MONTH = 7",
    "Aug": "MONTH = 8",
    "Sep": "MONTH = 9",
    "Oct": "MONTH = 10",
    "Nov": "MONTH = 11",
    "Dec": "MONTH = 12",
}

PROCESS_PER_ENSEMBLE = False   # Set True to hotspot each ensemble individually
PROCESS_COMBINED   = True
COMBINED_CSV_NAME  = "combined_all_ensembles.csv"

# ── ISA output field name (ArcGIS Pro: "zScore"; legacy ArcMap: "ZScore") ──
ISA_ZSCORE_FIELD = "zScore"

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "hotspot_batch.log"))
    ]
)


def _pick_csv(folder):
    """Pick the newest matching CSV in a folder by mtime."""
    csvs = glob.glob(os.path.join(folder, CSV_PATTERN))
    if not csvs:
        return None
    return max(csvs, key=os.path.getmtime)


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — MERGER + INTEGRATOR
# ══════════════════════════════════════════════════════════════════

def stage1_merge_integrate(fc_all, out_gdb, tag, expression):
    """
    Select → Copy → Clip to PAR → Project UTM 51N → Recopy →
    Merge with BIG_PERL_PTS → Integrate (35 km)
    Returns: path to Integrated Features
    """
    logging.info(f"[{tag}] -- STAGE 1: Merger + Integrator --")

    # Select by attribute
    lyr = f"{tag}_lyr"
    arcpy.management.MakeFeatureLayer(fc_all, lyr)
    arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", expression)
    n = int(arcpy.management.GetCount(lyr)[0])
    logging.info(f"[{tag}] Selected {n} points ({expression})")
    if n == 0:
        logging.warning(f"[{tag}] No points — skipping")
        return None

    # Copy Features
    fc_copied = os.path.join(out_gdb, f"{tag}_Copied")
    arcpy.management.CopyFeatures(lyr, fc_copied)

    # Clip Features (WGS84)
    fc_clipped = os.path.join(out_gdb, f"{tag}_Clipped")
    arcpy.analysis.Clip(fc_copied, PAR_SHP_WGS84, fc_clipped)

    # Project to UTM 51N
    fc_proj = os.path.join(out_gdb, f"{tag}_Projected")
    arcpy.management.Project(fc_clipped, fc_proj, SR_UTM51N)

    # Recopy
    fc_recopy = os.path.join(out_gdb, f"{tag}_Recopy")
    arcpy.management.CopyFeatures(fc_proj, fc_recopy)

    # Merge with BIG_PERL_PTS (project once per GDB)
    fc_merged = os.path.join(out_gdb, f"{tag}_Merged")
    if arcpy.Exists(BIG_PERL_PTS):
        big_utm = os.path.join(out_gdb, "BIG_PERL_UTM51N")
        if not arcpy.Exists(big_utm):
            arcpy.management.Project(BIG_PERL_PTS, big_utm, SR_UTM51N)
            logging.info(f"[{tag}] Projected BIG_PERL_PTS to UTM 51N")
        arcpy.management.Merge([fc_recopy, big_utm], fc_merged)
    else:
        logging.warning(f"[{tag}] BIG_PERL_PTS not found — synthetic only")
        arcpy.management.CopyFeatures(fc_recopy, fc_merged)

    n_merged = int(arcpy.management.GetCount(fc_merged)[0])
    logging.info(f"[{tag}] Merged: {n_merged} points")

    # Integrate (35 km) — modifies fc_merged in place
    arcpy.management.Integrate(fc_merged, INTEGRATE_DIST)
    logging.info(f"[{tag}] Integrated at {INTEGRATE_DIST}")

    return fc_merged


# ══════════════════════════════════════════════════════════════════
# STAGE 2 — HOT SPOT MAKER
# ══════════════════════════════════════════════════════════════════

def stage2_hotspot(fc_integrated, out_gdb, tag):
    """
    Collect Events → ISA (ICOUNT, 10 bands) →
    Optimized Hot Spot Analysis → Add Lat/Lon
    Returns: path to Hot Spot feature class
    """
    logging.info(f"[{tag}] -- STAGE 2: Hot Spot Maker --")

    # Collect Events
    fc_collected = os.path.join(out_gdb, f"{tag}_Collected")
    arcpy.stats.CollectEvents(fc_integrated, fc_collected)
    n = int(arcpy.management.GetCount(fc_collected)[0])
    logging.info(f"[{tag}] Collected Events: {n} unique locations")

    # Incremental Spatial Autocorrelation
    isa_table = os.path.join(out_gdb, f"{tag}_ISA")
    isa_report = os.path.join(OUTPUT_DIR, f"{tag}_ISA_report.pdf")

    peak_distance = None
    try:
        arcpy.stats.IncrementalSpatialAutocorrelation(
            fc_collected,          # Input_Features
            "ICOUNT",              # Input_Field
            ISA_NUM_BANDS,         # Number_of_Distance_Increments
            "",                    # Beginning_Distance
            "",                    # Distance_Increment
            "EUCLIDEAN",           # Distance_Method
            "ROW_STANDARDIZATION", # Row_Standardization
            isa_table,             # Output_Table
            isa_report             # Output_Report_File
        )

        # Resolve z-score field name across ArcGIS versions
        isa_fields = {f.name for f in arcpy.ListFields(isa_table)}
        z_field = next(
            (f for f in (ISA_ZSCORE_FIELD, "zScore", "ZScore", "z_score") if f in isa_fields),
            None
        )
        if z_field is None:
            raise RuntimeError(f"No z-score field in ISA table. Fields: {sorted(isa_fields)}")

        max_z = float("-inf")
        with arcpy.da.SearchCursor(isa_table, ["Distance", z_field]) as cur:
            for row in cur:
                if row[1] is not None and row[1] > max_z:
                    max_z = row[1]
                    peak_distance = row[0]
        if peak_distance is not None:
            logging.info(f"[{tag}] ISA peak distance: {peak_distance:.0f} m (Z={max_z:.2f})")
        else:
            logging.warning(f"[{tag}] ISA table empty — no peak distance")
    except Exception as e:
        logging.warning(f"[{tag}] ISA failed: {e}")

    # Optimized Hot Spot Analysis
    # NOTE: Incident_Data_Aggregation_Method is ignored when Analysis_Field is given —
    # output geometry matches input (points → points), which IDW (stage 3) requires.
    fc_hotspot = os.path.join(out_gdb, f"{tag}_HotSpots")
    try:
        arcpy.stats.OptimizedHotSpotAnalysis(
            Input_Features=fc_collected,
            Output_Features=fc_hotspot,
            Analysis_Field="ICOUNT"
        )
    except Exception as e:
        logging.warning(f"[{tag}] Optimized Hot Spot failed ({e}), trying standard")
        arcpy.stats.HotSpots(
            Input_Feature_Class=fc_collected,
            Input_Field="ICOUNT",
            Output_Feature_Class=fc_hotspot,
            Conceptualization_of_Spatial_Relationships="FIXED_DISTANCE_BAND",
            Distance_Band_or_Threshold_Distance=peak_distance if peak_distance else "",
            Standardization="ROW"
        )

    n_hs = int(arcpy.management.GetCount(fc_hotspot)[0])
    logging.info(f"[{tag}] Hot Spots: {n_hs} features")

    # Add Latitude and Longitude (WGS84) — for inspection only; IDW uses GiZScore
    try:
        arcpy.management.CalculateGeometryAttributes(
            in_features=fc_hotspot,
            geometry_property=[["POINT_X", "POINT_X"], ["POINT_Y", "POINT_Y"]],
            coordinate_system=SR_WGS84
        )
        logging.info(f"[{tag}] Added Lat/Lon fields")
    except Exception as e:
        logging.warning(f"[{tag}] Add Lat/Lon failed: {e}")

    return fc_hotspot


# ══════════════════════════════════════════════════════════════════
# STAGE 3 — IDW RENDERER
# ══════════════════════════════════════════════════════════════════

def stage3_idw(fc_hotspot, out_gdb, tag):
    """
    IDW (GiZScore, 350 m cell) → Clip to PAR
    Returns: path to clipped raster
    """
    logging.info(f"[{tag}] -- STAGE 3: IDW Renderer --")
    arcpy.CheckOutExtension("Spatial")
    try:
        fields = [f.name for f in arcpy.ListFields(fc_hotspot)]
        if "GiZScore" in fields:
            z_field = "GiZScore"
        elif "Gi_Bin" in fields:
            z_field = "Gi_Bin"
            logging.warning(f"[{tag}] GiZScore not found, using Gi_Bin")
        else:
            logging.error(f"[{tag}] Neither GiZScore nor Gi_Bin found! Fields: {fields}")
            return None

        idw_raster = os.path.join(out_gdb, f"{tag}_IDW")
        idw_out = Idw(
            in_point_features=fc_hotspot,
            z_field=z_field,
            cell_size=IDW_CELL_SIZE,
            power=IDW_POWER,
            search_radius=RadiusVariable(12)
        )
        idw_out.save(idw_raster)
        logging.info(f"[{tag}] IDW complete (field={z_field}, cell={IDW_CELL_SIZE}m, power={IDW_POWER})")

        clipped = os.path.join(out_gdb, f"{tag}_IDW_Clipped")
        if arcpy.Exists(PAR_SHP):
            arcpy.management.Clip(
                in_raster=idw_raster,
                out_raster=clipped,
                in_template_dataset=PAR_SHP,
                clipping_geometry="ClippingGeometry"
            )
            logging.info(f"[{tag}] Clipped to PAR")
        else:
            logging.warning(f"[{tag}] PAR shapefile not found — no clip")
            arcpy.management.CopyRaster(idw_raster, clipped)
        return clipped
    finally:
        arcpy.CheckInExtension("Spatial")


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def process_csv(csv_path, gdb_path, label):
    """Run all 3 stages for all categories/months on one CSV."""
    gdb_dir = os.path.dirname(gdb_path)
    gdb_name = os.path.basename(gdb_path)
    if not arcpy.Exists(gdb_path):
        arcpy.management.CreateFileGDB(gdb_dir, gdb_name)

    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = gdb_path

    # CSV → Feature Class (WGS84)
    xy_lyr = f"{label}_xy"
    fc_all = os.path.join(gdb_path, f"{label}_AllPoints")
    arcpy.management.MakeXYEventLayer(csv_path, "LON", "LAT", xy_lyr, SR_WGS84)
    arcpy.management.CopyFeatures(xy_lyr, fc_all)
    logging.info(f"[{label}] Loaded {int(arcpy.management.GetCount(fc_all)[0])} points")

    # Build run list based on mode
    runs = {}
    if RUN_MODE in ("category", "both"):
        runs.update(CATEGORIES)
    if RUN_MODE in ("monthly", "both"):
        runs.update(MONTHLY)

    # Loop all runs
    for name, expr in runs.items():
        tag = f"{label}_{name}"
        t0 = time.time()
        try:
            fc_integrated = stage1_merge_integrate(fc_all, gdb_path, tag, expr)
            if fc_integrated is None:
                continue
            fc_hotspot = stage2_hotspot(fc_integrated, gdb_path, tag)
            stage3_idw(fc_hotspot, gdb_path, tag)
            elapsed = (time.time() - t0) / 60
            logging.info(f"[{tag}] Done in {elapsed:.1f} min")
        except Exception as e:
            logging.error(f"[{tag}] FAILED: {e}")
            import traceback
            traceback.print_exc()


def combine_csvs(ensemble_dir, output_path):
    """Merge all ensemble CSVs into one."""
    import pandas as pd
    dfs = []
    folders = sorted(glob.glob(os.path.join(ensemble_dir, "ensemble_member_*")))
    for i, folder in enumerate(folders, 1):
        csv_path = _pick_csv(folder)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        df["ENSEMBLE_ID"] = i
        df["GLOBAL_SID"] = f"E{i:02d}_" + df["SID"].astype(str)
        dfs.append(df)
        logging.info(f"Read ensemble {i}: {df['SID'].nunique()} storms ({os.path.basename(csv_path)})")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_path, index=False)
        logging.info(f"Combined -> {combined['GLOBAL_SID'].nunique()} storms, {len(combined)} pts")
        return output_path
    return None


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    folders = sorted(glob.glob(os.path.join(ENSEMBLE_DIR, "ensemble_member_*")))
    logging.info(f"Found {len(folders)} ensemble folders")
    if not folders:
        sys.exit("No ensemble folders found")

    if PROCESS_PER_ENSEMBLE:
        for i, folder in enumerate(folders, 1):
            csv_path = _pick_csv(folder)
            if csv_path is None:
                continue
            gdb = os.path.join(OUTPUT_DIR, f"Ensemble_{i:02d}.gdb")
            logging.info(f"{'='*60}")
            logging.info(f"ENSEMBLE {i}/{len(folders)}  ({os.path.basename(csv_path)})")
            logging.info(f"{'='*60}")
            process_csv(csv_path, gdb, f"E{i:02d}")

    if PROCESS_COMBINED:
        logging.info(f"{'='*60}")
        logging.info(f"COMBINED ALL ENSEMBLES")
        logging.info(f"{'='*60}")
        combined = os.path.join(OUTPUT_DIR, COMBINED_CSV_NAME)
        if combine_csvs(ENSEMBLE_DIR, combined):
            process_csv(combined, os.path.join(OUTPUT_DIR, "Combined_All.gdb"), "COMBINED")

    hrs = (time.time() - t0) / 3600
    logging.info(f"{'='*60}")
    logging.info(f"ALL DONE — {hrs:.1f} hours")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()