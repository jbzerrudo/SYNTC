# -*- mode: python ; coding: utf-8 -*-
r"""
PyInstaller recipe for the SynTC genesis tool.

Build it with build_exe.bat, or by hand:

    set SYNTC_REPO=D:\2026\SYNTC\SYNTC
    set SYNTC_MODEL=D:\2026\SYNTC\SYNTC-AI\run07\model.pkl
    python -m PyInstaller --noconfirm --clean SynTC.spec

onedir, not onefile. A onefile build of a torch application unpacks well over a
gigabyte into a temporary folder on every single launch, which costs about a
minute each time and buys nothing when the program lives on your own disk.

genesis_forecast.py and plume_pair.py travel as source, because the GUI runs
them with runpy rather than launching python.exe, which a frozen build does not
have. The modules they import are listed as hidden imports instead, so that
PyInstaller walks into torch, rasterio and scipy and freezes those properly.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO = os.environ.get("SYNTC_REPO", r"D:\2026\SYNTC\SYNTC")
MODEL = os.environ.get("SYNTC_MODEL", r"D:\2026\SYNTC\SYNTC-AI\run07\model.pkl")
CONSOLE = os.environ.get("SYNTC_CONSOLE", "1") != "0"

# Executed at runtime by runpy, so the source has to be in the bundle.
RUNTIME_SCRIPTS = ("plume_multi.py",)
# The genesis map is produced by the ridging tool, sourced from RidgingVersion/
# so the committed genesis_forecast.py at the repo root (the paper's method) is
# left untouched. It is bundled under its own basename, genesis_forecast.py, so
# the GUI calls it exactly as before. Override the source with SYNTC_GENESIS;
# it falls back to the repo root if the RidgingVersion copy is absent.
def _ridging(name):
    """Prefer the RidgingVersion copy, fall back to the paper's version.

    Both are bundled under their own basename, so the GUI calls them exactly as
    before. The repo-root copies, which produce the manuscript figures, are
    never modified; the build simply sources these two from elsewhere.
    """
    cand = os.path.join(REPO, "RidgingVersion", name)
    return cand if os.path.exists(cand) else os.path.join(REPO, name)


RIDGE_GENESIS = os.environ.get("SYNTC_GENESIS", _ridging("genesis_forecast.py"))
RIDGE_PAIR = os.environ.get("SYNTC_PAIR", _ridging("plume_pair.py"))
# Imported by those two.
LIBRARY_MODULES = ("syntc_ai", "terrain", "figstyle", "models", "intensity",
                   "windconv", "data")

missing = [p for p in
           [RIDGE_GENESIS, RIDGE_PAIR]
           + [os.path.join(REPO, s) for s in RUNTIME_SCRIPTS]
           + [os.path.join(REPO, m + ".py") for m in LIBRARY_MODULES]
           + [MODEL, os.path.join(REPO, "dtm_phil_1km.tif")]
           if not os.path.exists(p)]
if missing:
    raise SystemExit("SynTC.spec cannot find:\n  " + "\n  ".join(missing)
                     + "\n\nSet SYNTC_REPO and SYNTC_MODEL and try again.")

# both bundled under their own basenames, so the GUI calls them unchanged
datas = [(RIDGE_GENESIS, "."), (RIDGE_PAIR, ".")]
datas += [(os.path.join(REPO, s), ".") for s in RUNTIME_SCRIPTS]
datas += [(os.path.join(REPO, m + ".py"), ".") for m in LIBRARY_MODULES]
datas += [(MODEL, ".")]
for extra in ("dtm_phil_1km.tif", "dtm_phil_1km.tfw"):
    p = os.path.join(REPO, extra)
    if os.path.exists(p):
        datas.append((p, "."))
# A plain-language description of what this build's track model does, derived
# from the catalogue's own config.json so it cannot drift from the model that
# is actually bundled. Line 1 is shown in the window header; line 2 records
# which catalogue it came from, for traceability in saved output.
import json
_cfgp = os.path.join(os.path.dirname(MODEL), "config.json")
_order = 1
if os.path.exists(_cfgp):
    with open(_cfgp, "r", encoding="utf-8") as _f:
        _order = int(json.load(_f).get("track_memory_order", 1) or 1)
if _order == 1:
    _desc = ("Track model: each 6-hourly step is conditioned on the storm's "
             "previous displacement, along with its position, intensity, "
             "season and age.")
else:
    _desc = ("Track model: each 6-hourly step is conditioned on the storm's "
             "previous %d displacements, along with its position, intensity, "
             "season and age." % _order)
# Stamped into saved summaries so a file is still interpretable away from this
# program. Deliberately carries no internal run or catalogue label: those mean
# nothing outside the development tree and only invite misreading.
import datetime as _dt
_prov = "SynTC, track memory %d step%s, model built %s" % (
    _order, "" if _order == 1 else "s",
    _dt.date.fromtimestamp(os.path.getmtime(MODEL)).isoformat())
_infop = os.path.join(SPECPATH, "model_info.txt")
with open(_infop, "w", encoding="utf-8") as _f:
    _f.write(_desc + "\n" + _prov + "\n")
datas += [(_infop, ".")]

# ---- second, optional track model -----------------------------------------
# Bundled as model_alt.pkl and offered in the GUI as an experimental choice.
# Set SYNTC_MODEL_ALT to a different catalogue, or to a path that does not
# exist, to build without it; the GUI then shows no model selector at all.
import shutil
ALT = os.environ.get("SYNTC_MODEL_ALT",
                     r"D:\2026\SYNTC\SYNTC-AI\run09\model.pkl")
if os.path.exists(ALT) and os.path.abspath(ALT) != os.path.abspath(MODEL):
    _altdst = os.path.join(SPECPATH, "model_alt.pkl")
    shutil.copyfile(ALT, _altdst)
    datas += [(_altdst, ".")]
    _acfgp = os.path.join(os.path.dirname(ALT), "config.json")
    _aorder = 1
    if os.path.exists(_acfgp):
        with open(_acfgp, "r", encoding="utf-8") as _f:
            _aorder = int(json.load(_f).get("track_memory_order", 1) or 1)
    _ainfop = os.path.join(SPECPATH, "model_alt_info.txt")
    with open(_ainfop, "w", encoding="utf-8") as _f:
        _f.write("EXPERIMENTAL. Not the standard model.\n")
        _f.write("SynTC, track memory %d step%s, model built %s\n" % (
            _aorder, "" if _aorder == 1 else "s",
            _dt.date.fromtimestamp(os.path.getmtime(ALT)).isoformat()))
    datas += [(_ainfop, ".")]

# rasterio ships its own GDAL and PROJ data directories inside the wheel.
datas += collect_data_files("rasterio")

hidden = list(LIBRARY_MODULES)
hidden += collect_submodules("rasterio")   # rasterio._shim and friends
hidden += ["PIL.ImageTk", "PIL._tkinter_finder",
           "scipy.signal", "scipy.stats", "scipy.special",
           "matplotlib.backends.backend_agg",
           "matplotlib.backends.backend_pdf"]

EXCLUDES = ["PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "IPython", "jupyter",
            "notebook", "pytest", "sphinx", "torchvision", "torchaudio",
            "flask"]

a = Analysis(
    ["syntc_gui.py"],
    pathex=[REPO],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# PyInstaller 5 carried zipped_data and zipfiles; 6 dropped them. Ask rather
# than assume, so this spec builds on either.
_pyz_extra = [a.zipped_data] if hasattr(a, "zipped_data") else []
pyz = PYZ(a.pure, *_pyz_extra)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SynTC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # upx corrupts torch and GDAL DLLs often enough to ban it
    console=CONSOLE,    # set SYNTC_CONSOLE=0 to hide the black window
)

_coll_extra = [a.zipfiles] if hasattr(a, "zipfiles") else []
coll = COLLECT(
    exe,
    a.binaries,
    *_coll_extra,
    a.datas,
    strip=False,
    upx=False,
    name="SynTC",
)
