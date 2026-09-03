"""
SynTC genesis tool, desktop front end.

Runs the real genesis_forecast.py and plume_pair.py against the real fitted
model. Nothing is precomputed and nothing is approximated, so the numbers this
window shows are the numbers the command-line tool prints.

Two ways to run it:

    python syntc_gui.py      needs the SynTC repo and its packages installed
    SynTC.exe                frozen bundle, needs nothing installed

Frozen, the scripts and their imports live inside the bundle. model.pkl and
dtm_phil_1km.tif are looked for beside the .exe FIRST, so a retrained model can
be dropped in without rebuilding; the copies baked into the bundle are only the
fallback.

The scripts are executed in-process with runpy rather than launched with
subprocess, because a frozen bundle has no python.exe to launch.
"""

import contextlib
import io
import json
import math
import os
import queue
import runpy
import subprocess
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# ---- where things are ----------------------------------------------------
FROZEN = getattr(sys, "frozen", False)
BUNDLE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
HOME = (os.path.dirname(os.path.abspath(sys.executable)) if FROZEN
        else os.path.dirname(os.path.abspath(__file__)))


def _scripts_dir():
    """Where genesis_forecast.py and plume_pair.py live.

    Frozen, that is inside the bundle. Unfrozen, look beside this file, then at
    SYNTC_REPO, then at a SYNTC folder one level up, which is the layout when
    this file sits in RUNBAT beside the cloned repo.
    """
    if FROZEN:
        return BUNDLE
    here = os.path.dirname(os.path.abspath(__file__))
    up = os.path.dirname(here)
    for cand in (here, os.environ.get("SYNTC_REPO", ""),
                 os.path.join(up, "SYNTC"), os.path.join(up, "SYNTC-main")):
        if cand and os.path.exists(os.path.join(cand, "genesis_forecast.py")):
            return cand
    return here


SCRIPTS = _scripts_dir()


def _outdir():
    """Where runs are written.

    Never inside dist\\<name>. PyInstaller clears that whole directory on every
    build, so an output folder beside the exe is destroyed by the next rebuild.
    When the exe is sitting in a dist folder, outputs go one level above it
    instead. Set SYNTC_OUT to override.
    """
    override = os.environ.get("SYNTC_OUT")
    if override:
        return override
    parent = os.path.dirname(HOME)
    if os.path.basename(parent).lower() == "dist":
        return os.path.join(os.path.dirname(parent), "forecast")
    return os.path.join(HOME, "forecast")


OUTDIR = _outdir()

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
KEEPS = tuple(range(0, 55, 5))     # 0, 5, 10 ... 50
# The blue line through the middle. "ridge" is the crest of the passage field
# and drives the single genesis map; the two-panel pair figure (plume_pair.py)
# has no ridge mode, so work() sends it "off" when ridge is chosen.
# Track models available to this build. The standard model is always first
# and is the default. The extended-memory model appears only if the build
# bundled it; it is marked experimental because a run made with it is not
# comparable with results from the standard model.
STANDARD_LABEL = "standard"
ALT_LABEL = "extended memory (experimental)"


def _model_labels():
    out = [STANDARD_LABEL]
    if resolve("model_alt.pkl"):
        out.append(ALT_LABEL)
    return out


CENTRALS = (("ridge", "ridge (core of the plume)"),
            ("off", "none"),
            ("median", "middle of the plume"),
            ("medoid", "representative track"))
CENTRAL_LABEL = {b: a for a, b in CENTRALS}
WINDOW = (112.0, 145.0, 2.0, 28.0)   # lon_min, lon_max, lat_min, lat_max
# How the drawn realisations are chosen. "core" ranks by the lowest passage
# probability along each path, which selects against divergence and therefore
# understates the spread; it is kept because it is the manuscript figure.
# The track sample dropdown is gone. Selection in the ridging build is always
# "closest to the ridge": the old core mode ranked by the probability of the
# cells a track visited, which in a bimodal plume rewards the broad recurving
# branch, and measured 5 of 60 inside the corridor against 7 for a blind
# stride. Keeping it as an option would have offered forecasters two answers
# that disagree, with the wrong one labelled "most likely corridor".
PICK_LABEL = {"ridge": "closest to ridge"}

INK, ACCENT, LINE, BG = "#2b2b2b", "#8a6f4e", "#d8cfbe", "#faf7f1"
HEAD, BTN, BTN_HOT = "#6b4f3a", "#b07d62", "#96654c"


def resolve(name):
    """Beside the exe first, then inside the bundle. None if neither has it."""
    for base in (HOME, SCRIPTS, BUNDLE):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def _describe_order(k):
    """Plain description of the bundled track model.

    States what the model is conditioned on, and nothing about how it compares
    with any other configuration. A comparative claim here would be author
    knowledge the user cannot check, and would need revisiting every time the
    manuscript's status changed.
    """
    if k == 1:
        return ("Track model: each 6-hourly step is conditioned on the "
                "storm's previous displacement, along with its position, "
                "intensity, season and age.")
    return ("Track model: each 6-hourly step is conditioned on the storm's "
            "previous %d displacements, along with its position, intensity, "
            "season and age." % k)


def _override_dir():
    """Directory of a model.pkl that sits outside the bundle, or None.

    resolve() prefers a copy beside the exe, so dropping one there silently
    replaces the bundled model. The build-time note would then describe a
    model that is no longer in use, which is worse than saying nothing.
    """
    m = resolve("model.pkl")
    if m and os.path.dirname(os.path.abspath(m)) != os.path.abspath(BUNDLE):
        return os.path.dirname(os.path.abspath(m))
    return None


# Evaluated here, after resolve() exists, because _model_labels() calls it.
MODEL_LABELS = _model_labels()


def model_note():
    """One plain-language line describing the track model actually in use."""
    d = _override_dir()
    if d:
        cfg = os.path.join(d, "config.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    return _describe_order(
                        int(json.load(f).get("track_memory_order", 1) or 1))
            except (OSError, ValueError, TypeError):
                pass
        return ("Track model: replaced by a model.pkl beside this program. "
                "Copy its config.json here as well to describe it.")
    p = resolve("model_info.txt")
    if p:
        try:
            with open(p, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            if line:
                return line
        except OSError:
            pass
    return "Track model: configuration not recorded in this build."


def model_provenance(alt=False):
    """Second line of the build note: which model produced these numbers.

    alt=True describes the extended-memory model instead of the standard one,
    so a summary saved from an experimental run cannot be mistaken for a
    standard result.
    """
    if alt:
        p = resolve("model_alt_info.txt")
        if p:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return " ".join(x.strip() for x in f.readlines() if x.strip())
            except OSError:
                pass
        return "EXPERIMENTAL extended-memory model, not the standard one"
    d = _override_dir()
    if d:
        return "model.pkl supplied from %s" % d
    p = resolve("model_info.txt")
    if p:
        try:
            with open(p, "r", encoding="utf-8") as f:
                f.readline()
                line = f.readline().strip()
            if line:
                return line
        except OSError:
            pass
    return "catalogue unknown"


def run_script(name, argv):
    """Execute one of the bundled scripts as if it were __main__.

    Returns (ok, captured_output). stdout and stderr are captured because the
    frozen build has no console to print into.
    """
    path = os.path.join(SCRIPTS, name)
    if not os.path.exists(path):
        return False, "missing from this build: %s" % name
    old_argv, old_cwd = sys.argv, os.getcwd()
    buf = io.StringIO()
    sys.argv = [path] + [str(a) for a in argv]
    try:
        os.chdir(SCRIPTS)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            runpy.run_path(path, run_name="__main__")
        return True, buf.getvalue()
    except SystemExit as e:
        return (e.code in (None, 0)), buf.getvalue()
    except Exception:
        return False, buf.getvalue() + "\n" + traceback.format_exc()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)




def fit_window(lat, lon, margin=4.0, snap=5.0):
    """Map window for the two track panels.

    Defaults to the 112-145E, 2-28N frame the manuscript figure uses, and
    widens it only when the genesis point falls outside. plume_pair masks every
    track point beyond its frame, so a point east of 145E would otherwise draw
    two empty maps. Edges snap to whole 5 degrees to keep the ticks tidy.
    """
    lo_min, lo_max, la_min, la_max = WINDOW
    return (min(lo_min, math.floor((lon - margin) / snap) * snap),
            max(lo_max, math.ceil((lon + margin) / snap) * snap),
            min(la_min, math.floor((lat - margin) / snap) * snap),
            max(la_max, math.ceil((lat + margin) / snap) * snap))


def headline(summary, lat, lon, month, n):
    """A compact stats block for the pair figure.

    plume_pair draws this once above both panels, not per panel, because the
    numbers describe every realisation and not the handful of tracks drawn.
    Lines are joined with a literal backslash-n, which is what that script
    expands.
    """
    want = ("enters PAR", "centre crosses Philippine land",
            "peak wind while inside PAR", "reaches TY", "reaches STY")
    picked = [ln.strip() for ln in summary.splitlines()
              if any(w in ln for w in want)]
    if not picked:
        return ""
    head = ("%g\u00b0N %g\u00b0E, %s, %s realisations "
            "(statistics cover all realisations, not the drawn tracks)"
            % (lat, lon, MONTHS[month - 1], "{:,}".format(n)))
    return "\\n".join([head] + picked)


class ImagePanel(ttk.Frame):
    """A figure, scaled to the width it is given, keeping its aspect ratio."""

    def __init__(self, master, placeholder):
        super().__init__(master)
        self.label = ttk.Label(self, text=placeholder, anchor="center",
                               justify="center", foreground="#7a6a58")
        self.label.pack(fill="both", expand=True)
        self.path = None
        self._photo = None
        self._drawn_at = 0
        self._job = None
        self.bind("<Configure>", self._on_configure)

    def show(self, path):
        self.path = path
        self._drawn_at = 0
        self._render()

    def clear(self, text):
        self.path = None
        self._photo = None
        self.label.configure(image="", text=text)

    def _on_configure(self, _event):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(180, self._render)

    def _render(self):
        self._job = None
        if not self.path or not os.path.exists(self.path):
            return
        w = max(self.winfo_width() - 12, 320)
        if abs(w - self._drawn_at) < 24:
            return
        try:
            if HAVE_PIL:
                im = Image.open(self.path)
                scale = min(1.0, w / im.width)
                im = im.resize((max(1, int(im.width * scale)),
                                max(1, int(im.height * scale))),
                               Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(im)
            else:
                # Tk 8.6 reads PNG natively but can only shrink by whole
                # factors, so the fit is coarse without Pillow.
                ph = tk.PhotoImage(file=self.path)
                k = max(1, int(round(ph.width() / float(w))))
                self._photo = ph.subsample(k, k) if k > 1 else ph
            self.label.configure(image=self._photo, text="")
            self._drawn_at = w
        except Exception as exc:
            self.label.configure(image="", text="could not draw %s\n%s"
                                 % (os.path.basename(self.path), exc))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SynTC genesis tool")
        self.geometry("1240x860")
        self.minsize(940, 640)
        self.configure(bg=BG)
        self.queue = queue.Queue()
        self.busy = False
        self.model = resolve("model.pkl")
        self.model_alt = resolve("model_alt.pkl")
        self.dtm = resolve("dtm_phil_1km.tif")
        self._style()
        self._build()
        os.makedirs(OUTDIR, exist_ok=True)
        if not self.model or not self.dtm:
            missing = [n for n, p in (("model.pkl", self.model),
                                      ("dtm_phil_1km.tif", self.dtm)) if not p]
            self.after(300, lambda: messagebox.showerror(
                "Missing files",
                "Could not find %s.\n\nPut a copy next to this program:\n%s"
                % (" and ".join(missing), HOME)))

    # ---- chrome ----------------------------------------------------------
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=INK,
                    font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background="#ffffff", relief="solid",
                    borderwidth=1)
        s.configure("TLabel", background=BG)
        s.configure("Field.TLabel", background=BG, foreground=ACCENT,
                    font=("Segoe UI", 8, "bold"))
        s.configure("Warn.TLabel", background="#fff8e8", foreground="#5c4a24",
                    font=("Segoe UI", 9), padding=9)
        s.configure("Status.TLabel", background=BG, foreground="#7a6a58",
                    font=("Segoe UI", 9))
        s.configure("Run.TButton", background=BTN, foreground="#ffffff",
                    font=("Segoe UI", 10, "bold"), padding=(18, 6),
                    borderwidth=0)
        s.map("Run.TButton", background=[("active", BTN_HOT),
                                         ("disabled", "#cbbfae")])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(14, 6))

    def _build(self):
        head = tk.Frame(self, bg=HEAD)
        head.pack(fill="x")
        tk.Label(head, text="SynTC genesis tool", bg=HEAD, fg="#ffffff",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18,
                                                     pady=(11, 0))
        tk.Label(head, text="What storms forming here in this month have done, "
                            "over thousands of simulations. Not a forecast of "
                            "any actual storm.", bg=HEAD, fg="#e6dcd0",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(1, 2))
        # The track-model description is author metadata. A forecaster reading
        # a plume does not need it, so it is not shown in the normal case.
        # It IS shown when a model.pkl beside this program has replaced the
        # bundled one, because then the tool is not running what it shipped
        # with and the user has to be told.
        if _override_dir():
            tk.Label(head, text=model_note(), bg=HEAD, fg="#cdbba8",
                     font=("Segoe UI", 8)).pack(anchor="w", padx=18,
                                                pady=(0, 1))
        tk.Label(head, text="Blue line is the ridge, the crest of the plume. "
                            "The corridor half-width (150 km) is a set scale, "
                            "not calibrated to observed tracks; its % is a "
                            "property of that width.",
                 bg=HEAD, fg="#cdbba8", font=("Segoe UI", 8)).pack(
                     anchor="w", padx=18, pady=(0, 10))

        form = ttk.Frame(self, padding=(16, 12, 16, 6))
        form.pack(fill="x")
        self.v_lat = tk.StringVar(value="12")
        self.v_lon = tk.StringVar(value="135")
        self.v_month = tk.StringVar(value=MONTHS[7])
        self.v_n = tk.StringVar(value="2000")
        self.v_wind = tk.StringVar(value="")
        self.v_ka = tk.StringVar(value="5")
        self.v_kb = tk.StringVar(value="15")

        def field(col, text, var, width=9, values=None, row=0):
            box = ttk.Frame(form)
            box.grid(row=row, column=col, sticky="w", padx=(0, 14),
                     pady=(0, 6))
            ttk.Label(box, text=text.upper(), style="Field.TLabel").pack(
                anchor="w", pady=(0, 2))
            if values is None:
                w = ttk.Entry(box, textvariable=var, width=width)
            else:
                w = ttk.Combobox(box, textvariable=var, values=values,
                                 width=width, state="readonly")
            w.pack(anchor="w")
            return w

        field(0, "Latitude °N", self.v_lat)
        field(1, "Longitude °E", self.v_lon)
        field(2, "Month", self.v_month, 12, list(MONTHS))
        field(3, "Realisations", self.v_n)
        field(4, "Genesis wind kt", self.v_wind, 12)
        field(5, "Panel A tracks", self.v_ka, 6, [str(k) for k in KEEPS])
        field(6, "Panel B tracks", self.v_kb, 6, [str(k) for k in KEEPS])

        self.v_central = tk.StringVar(value="ridge (core of the plume)")
        field(7, "Central line", self.v_central, 20,
              [b for a, b in CENTRALS])

        self.v_cone = tk.BooleanVar(value=False)
        cbx = ttk.Frame(form)
        cbx.grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 14))
        ttk.Label(cbx, text=" ", style="Field.TLabel").pack(anchor="w",
                                                            pady=(0, 2))
        ttk.Checkbutton(cbx, text="draw 50% / 90% containment",
                        variable=self.v_cone).pack(anchor="w")

        # Ridge controls for the single genesis map. Corridor is the half-width
        # in km that defines "near the ridge"; top N draws only the N closest
        # tracks (blank draws the whole corridor); overlay adds all 10,000
        # faintly on top, the check that the ridge follows the real density.
        self.v_corridor = tk.StringVar(value="150")
        self.v_topn = tk.StringVar(value="")
        self.v_overlay = tk.BooleanVar(value=False)
        field(3, "Corridor km", self.v_corridor, 8, row=1)
        field(4, "Ridge top N", self.v_topn, 8, row=1)
        obx = ttk.Frame(form)
        obx.grid(row=1, column=5, columnspan=2, sticky="w", padx=(0, 14))
        ttk.Label(obx, text=" ", style="Field.TLabel").pack(anchor="w",
                                                            pady=(0, 2))
        ttk.Checkbutton(obx, text="overlay all 10k tracks",
                        variable=self.v_overlay).pack(anchor="w")

        # Track model. The standard single-step model is the default. The
        # extended-memory model is offered only if it was bundled, and is
        # labelled experimental: it predicts the next position better but
        # generates more intense landfalls than the record shows, so a run
        # made with it is not comparable with standard results.
        self.v_model = tk.StringVar(value=MODEL_LABELS[0])
        if len(MODEL_LABELS) > 1:
            field(2, "Track model", self.v_model, 22, MODEL_LABELS, row=1)

        run = ttk.Frame(form)
        run.grid(row=0, column=8, rowspan=2, sticky="w")
        ttk.Label(run, text=" ", style="Field.TLabel").pack(anchor="w",
                                                            pady=(0, 2))
        self.b_run = ttk.Button(run, text="Run", style="Run.TButton",
                                command=self.on_run)
        self.b_run.pack(anchor="w")

        ttk.Label(self, style="Warn.TLabel", wraplength=1180, justify="left",
                  text="Read this as conditional climatology, not a forecast. "
                       "Every number below is conditional on a storm forming "
                       "at the point and month you entered, and describes the "
                       "range of what such storms have done. It knows nothing "
                       "about today's atmosphere, the real vortex or the "
                       "steering flow, so a dynamical forecast will beat it on "
                       "a live storm at every lead time. Use it to judge how "
                       "usual or unusual a forecast track is, not to predict "
                       "one. For warnings and official guidance, use PAGASA "
                       "bulletins.").pack(fill="x", padx=16, pady=(2, 8))

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        self.tabs = ttk.Notebook(pane)
        self.fig_plume = ImagePanel(
            self.tabs, "Enter a genesis point and press Run.")
        self.fig_pair = ImagePanel(
            self.tabs, "The two ridge-following panels appear here after a run.")
        self.tabs.add(self.fig_plume, text="  Probability of passage  ")
        self.tabs.add(self.fig_pair, text="  Ridge-following tracks  ")
        pane.add(self.tabs, weight=3)

        right = ttk.Frame(pane)
        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="SUMMARY", style="Field.TLabel").pack(side="left")
        ttk.Button(bar, text="Save as .txt", command=self.on_save).pack(
            side="right")
        ttk.Button(bar, text="Open output folder", command=self.on_open).pack(
            side="right", padx=(0, 6))
        wrap = ttk.Frame(right, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        self.text = tk.Text(wrap, wrap="word", relief="flat", padx=10, pady=8,
                            font=("Consolas", 10), bg="#ffffff", fg=INK,
                            height=10)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        pane.add(right, weight=2)

        foot = ttk.Frame(self, padding=(16, 0, 16, 10))
        foot.pack(fill="x")
        self.bar = ttk.Progressbar(foot, mode="indeterminate", length=180)
        self.status = ttk.Label(foot, style="Status.TLabel",
                                text="model  %s" % (self.model or "NOT FOUND"))
        self.status.pack(side="left")
        self.bar.pack(side="right")

    # ---- actions ---------------------------------------------------------
    def say(self, msg):
        self.status.configure(text=msg)

    def write(self, msg):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", msg)

    def on_open(self):
        os.makedirs(OUTDIR, exist_ok=True)
        try:
            os.startfile(OUTDIR)                      # Windows
        except AttributeError:
            subprocess.Popen(["xdg-open", OUTDIR])

    def on_save(self):
        body = self.text.get("1.0", "end").strip()
        if not body:
            return
        p = filedialog.asksaveasfilename(
            initialdir=OUTDIR, defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile="syntc_summary.txt")
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body + "\n")
                # Record which model produced these numbers, so a saved summary
                # is still interpretable once it is away from this program.
                alt = getattr(self, "used_alt", False)
                if alt:
                    fh.write("\n-- EXPERIMENTAL RUN. These numbers were not "
                             "produced by the standard model and are not "
                             "comparable with standard results.\n")
                fh.write("\n-- %s\n" % model_provenance(alt))
            self.say("saved  %s" % p)

    def on_run(self):
        if self.busy:
            return
        if not self.model or not self.dtm:
            messagebox.showerror("Missing files",
                                 "model.pkl or dtm_phil_1km.tif is missing.")
            return
        try:
            lat = float(self.v_lat.get())
            lon = float(self.v_lon.get())
            n = int(self.v_n.get())
        except ValueError:
            messagebox.showerror(
                "Check the inputs",
                "Latitude, longitude and realisations must be numbers.")
            return
        if n < 1:
            messagebox.showerror("Check the inputs",
                                 "Realisations must be at least 1.")
            return
        wind = self.v_wind.get().strip()
        if wind:
            try:
                float(wind)
            except ValueError:
                messagebox.showerror("Check the inputs",
                                     "Genesis wind must be a number in knots, "
                                     "or blank for climatology.")
                return
        month = MONTHS.index(self.v_month.get()) + 1
        ka, kb = int(self.v_ka.get()), int(self.v_kb.get())
        pick = "ridge"
        cone = bool(self.v_cone.get())
        central = CENTRAL_LABEL.get(self.v_central.get(), "ridge")
        corridor = self.v_corridor.get().strip() or "150"
        topn = self.v_topn.get().strip()
        overlay = bool(self.v_overlay.get())
        use_alt = (self.v_model.get() == ALT_LABEL) and bool(self.model_alt)
        self.used_alt = use_alt
        model_path = self.model_alt if use_alt else self.model

        self.busy = True
        self.b_run.configure(state="disabled")
        self.bar.start(12)
        self.fig_pair.clear("drawing...")
        self.say("running {:,} realisations, this takes a few minutes"
                 .format(n))
        self.write("")
        threading.Thread(target=self.work,
                         args=(lat, lon, month, n, wind, ka, kb, pick, cone,
                               central, corridor, topn, overlay,
                               model_path, use_alt),
                         daemon=True).start()
        self.after(150, self.poll)

    def work(self, lat, lon, month, n, wind, ka, kb, pick, cone, central,
             corridor="150", topn="", overlay=False,
             model_path=None, use_alt=False):
        try:
            model_path = model_path or self.model
            argv = ["--model", model_path, "--dtm", self.dtm,
                    "--lat", lat, "--lon", lon, "--month", month,
                    "--n", n, "--out", OUTDIR]
            if wind:
                argv += ["--wind", wind]
            argv += ["--central", central, "--corridor", corridor]
            if topn:
                argv += ["--top", topn]
            if overlay:
                argv += ["--underlay"]
            ok, out = run_script("genesis_forecast.py", argv)
            if not ok:
                self.queue.put(("fail", out))
                return
            stem = "genesis_%gN_%gE_m%02d" % (lat, lon, month)
            summary = out.split("\nwritten:")[0].strip()
            self.queue.put(("plume", (os.path.join(OUTDIR, stem + ".png"),
                                      summary)))

            ext = fit_window(lat, lon)
            pair = stem + "_pair"
            argv = [
                "--gen", OUTDIR, "--dtm", self.dtm,
                "--left", stem, "--right", stem,
                "--left-pt", lat, lon, "--right-pt", lat, lon,
                "--left-label", "(a) %d closest to ridge" % ka,
                "--right-label", "(b) %d closest to ridge" % kb,
                "--keep-left", ka, "--keep-right", kb,
                "--central", central, "--corridor", corridor,
                "--out", os.path.join(OUTDIR, pair)]
            if cone:
                argv += ["--cone"]
            if ext != WINDOW:
                argv += ["--ext"] + list(ext)
                self.queue.put(("info",
                                "%g\u00b0N %g\u00b0E sits outside the standard "
                                "%g-%g\u00b0E, %g-%g\u00b0N frame, so the track "
                                "panels were widened to %g-%g\u00b0E, "
                                "%g-%g\u00b0N."
                                % ((lat, lon) + WINDOW + ext)))
            note = headline(summary, lat, lon, month, n)
            if note:
                argv += ["--note", note]
            ok, out = run_script("plume_pair.py", argv)
            if ok:
                self.queue.put(("pair", os.path.join(OUTDIR, pair + ".png")))
            else:
                self.queue.put(("note", "Track panels could not be drawn:\n"
                                + out.strip()[-600:]))
            self.queue.put(("done", None))
        except Exception:
            self.queue.put(("fail", traceback.format_exc()))

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "plume":
                    path, summary = payload
                    self.write(summary)
                    self.fig_plume.show(path)
                    self.tabs.select(0)
                    self.say("plume written to  %s" % OUTDIR)
                elif kind == "pair":
                    self.fig_pair.show(payload)
                    self.tabs.select(1)
                elif kind == "note":
                    self.fig_pair.clear(payload)
                    self.text.insert("end", "\n\n" + payload)
                elif kind == "info":
                    self.text.insert("end", "\n\n" + payload)
                elif kind == "fail":
                    self.finish()
                    self.write(payload)
                    self.say("failed")
                    return
                elif kind == "done":
                    self.finish()
                    self.say("done. outputs in  %s" % OUTDIR)
                    return
        except queue.Empty:
            pass
        if self.busy:
            self.after(150, self.poll)

    def finish(self):
        self.busy = False
        self.bar.stop()
        self.b_run.configure(state="normal")


if __name__ == "__main__":
    App().mainloop()
