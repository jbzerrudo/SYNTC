"""
Combined SYNTC CSV Pipeline (2026-03 Update)
1. Converts master CSV directly to a Point Feature Class in a Geodatabase.
2. Generates Segmented Track Lines from the points.
3. Calculates both Segment-level (attenuated) and Storm-level Return Periods.

Run as "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" csv2pts2segments.py
"""

import os
import arcpy
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
# --- INPUT ---
INPUT_CSV = r"D:\2026\SYNTC\SYNTC-AI\run07\synthetic_storms_ens09.csv"

# --- OUTPUT ---
OUTPUT_MAIN_FOLDER = r"D:\2026\SYNTC\SYNTC-AI\arcmap\ens09"

OUTPUT_POINTS_GDB = os.path.join(OUTPUT_MAIN_FOLDER, "StormPoints.gdb")
OUTPUT_POINTS_FC = os.path.join(OUTPUT_POINTS_GDB, "all_points_combined")

OUTPUT_TRACKS_FOLDER = os.path.join(OUTPUT_MAIN_FOLDER, "Segmented")
OUTPUT_TRACKS_GDB = os.path.join(OUTPUT_TRACKS_FOLDER, "Storm_Tracks.gdb")
OUTPUT_TRACKS_FC = os.path.join(OUTPUT_TRACKS_GDB, "all_tracks_segmented")

# CSV/Feature Class Field Mapping
SID_COL = "SID"
LAT_COL = "LAT"
LON_COL = "LON"
WIND_COL = "WIND"   
YEAR_COL = "YEAR"
MONTH_COL = "MONTH"
CATEGORY_COL = "CATEGORY"
TIME_COL = "ISO_TIME"

spatial_ref = arcpy.SpatialReference(4326) # WGS 1984
arcpy.env.overwriteOutput = True

# ==========================================
# 2. HELPER FUNCTION
# ==========================================
def classify_return_period(wind):
    """Classifies wind speed into manuscript Weibull return periods."""
    if wind is None: return "Typical"
    wind = float(wind)
    if wind > 124: return "> 50"
    elif wind >= 119: return "10 - 50"
    elif wind >= 106: return "2 - 10"
    else: return "Typical"

def classify_category(wind):
    """Derive category from wind speed (matches attenuated WIND, not stale CSV CATEGORY)."""
    if wind is None: return "Remnant Low"
    wind = float(wind)
    if wind >= 100:  return "Super Typhoon"
    elif wind >= 64: return "Typhoon"
    elif wind >= 48: return "Severe Tropical Storm"
    elif wind >= 34: return "Tropical Storm"
    elif wind >= 22: return "Tropical Depression"
    else:            return "Remnant Low"

# ==========================================
# 3. WORKSPACE PREPARATION
# ==========================================
print("Setting up workspaces...")

for folder in [OUTPUT_MAIN_FOLDER, OUTPUT_TRACKS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Clean/Create GDBs
for gdb_path in [OUTPUT_POINTS_GDB, OUTPUT_TRACKS_GDB]:
    if not arcpy.Exists(gdb_path):
        if os.path.exists(gdb_path): # Handle corrupted/empty folders
            shutil.rmtree(gdb_path)
        arcpy.management.CreateFileGDB(os.path.dirname(gdb_path), os.path.basename(gdb_path))

# ==========================================
# 4. CSV TO POINTS (FAST NATIVE METHOD)
# ==========================================
print(f"\nConverting CSV to Points...")
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: Cannot find CSV at {INPUT_CSV}")
    exit(1)

# XYTableToPoint is much faster than pandas + InsertCursor
arcpy.management.XYTableToPoint(
    in_table=INPUT_CSV,
    out_feature_class=OUTPUT_POINTS_FC,
    x_field=LON_COL,
    y_field=LAT_COL,
    coordinate_system=spatial_ref
)
print(f"  Created point feature class: {OUTPUT_POINTS_FC}")

# ==========================================
# 5. LOAD AND SORT POINTS
# ==========================================
print("\nReading points into memory to calculate tracks...")

storms = {}
search_fields = ["SHAPE@X", "SHAPE@Y", SID_COL, WIND_COL, YEAR_COL, MONTH_COL, CATEGORY_COL, TIME_COL]

with arcpy.da.SearchCursor(OUTPUT_POINTS_FC, search_fields) as cursor:
    for row in cursor:
        sid = row[2] if row[2] else "Unknown"
        if sid not in storms:
            storms[sid] = []
        
        storms[sid].append({
            "x": row[0],
            "y": row[1],
            "wind": float(row[3]) if row[3] is not None else 0.0,
            "year": row[4],
            "month": row[5],
            "category": row[6],
            "time": row[7]
        })

# Sort points chronologically (ISO_TIME format sorts correctly as a string)
for sid in storms:
    storms[sid].sort(key=lambda p: str(p["time"]) if p["time"] is not None else "")

# Calculate lifetime max wind per storm
storm_max_wind = {}
for sid, points in storms.items():
    storm_max_wind[sid] = max([p["wind"] for p in points])

# ==========================================
# 6. CREATE SEGMENTED TRACKS
# ==========================================
print("\nCreating segmented track lines schema...")

arcpy.management.CreateFeatureclass(
    OUTPUT_TRACKS_GDB, os.path.basename(OUTPUT_TRACKS_FC), "POLYLINE", spatial_reference=spatial_ref
)

# Batch add fields
arcpy.management.AddFields(OUTPUT_TRACKS_FC, [
    ["SID", "TEXT", "SID", 50],
    ["Year", "LONG", "Year"],
    ["Month", "LONG", "Month"],
    ["SegWind", "DOUBLE", "Segment Wind"],
    ["MaxWind", "DOUBLE", "Lifetime Max Wind"],
    ["Category", "TEXT", "Category", 50],
    ["SegRP", "TEXT", "Segment Return Period", 30],
    ["StormRP", "TEXT", "Storm Return Period", 30]
])

print(f"Writing line segments to {OUTPUT_TRACKS_FC}...")

segment_count = 0
attenuation_count = 0
seg_rp_counts = {"Typical": 0, "2 - 10": 0, "10 - 50": 0, "> 50": 0}

insert_fields = ["SHAPE@", "SID", "Year", "Month", "SegWind", "MaxWind", "Category", "SegRP", "StormRP"]

with arcpy.da.InsertCursor(OUTPUT_TRACKS_FC, insert_fields) as cursor:
    for sid, points in storms.items():
        if len(points) < 2:
            continue

        max_wind = storm_max_wind[sid]
        storm_rp = classify_return_period(max_wind)

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            # Skip segment if coordinates are missing/null
            if None in (p1["x"], p1["y"], p2["x"], p2["y"]):
                continue

            # Geometry
            arr = arcpy.Array([arcpy.Point(p1["x"], p1["y"]), arcpy.Point(p2["x"], p2["y"])])
            line = arcpy.Polyline(arr, spatial_ref)

            # Segment metrics (driven by the start point of the segment)
            seg_wind = p1["wind"]
            seg_rp = classify_return_period(seg_wind)
            
            # Diagnostics tracking
            seg_rp_counts[seg_rp] += 1
            if seg_rp != storm_rp:
                attenuation_count += 1

            cursor.insertRow([
                line, str(sid), p1["year"], p1["month"],
                seg_wind, max_wind, classify_category(seg_wind),  # ← derived from wind, not CSV
                seg_rp, storm_rp
            ])
            segment_count += 1

        if segment_count % 50000 == 0 and segment_count > 0:
            print(f"  ...{segment_count} segments written")

# ==========================================
# 7. DIAGNOSTICS
# ==========================================
print(f"\nDone. Successfully created {segment_count} track segments.")
print("\nSegment-level RP classification (by segment wind):")
print(f"  Typical (< 106 kt):      {seg_rp_counts['Typical']}")
print(f"  2 - 10 yr (106-118 kt):  {seg_rp_counts['2 - 10']}")
print(f"  10 - 50 yr (119-124 kt): {seg_rp_counts['10 - 50']}")
print(f"  > 50 yr (> 124 kt):      {seg_rp_counts['> 50']}")
print(f"\n  Attenuated segments (Segment RP != Storm RP): {attenuation_count}")
