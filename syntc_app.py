"""
SynTC genesis tool, local web front end.

    pip install flask
    python syntc_app.py
    then open http://127.0.0.1:5000

Runs the real genesis_forecast.py against the real fitted model. Nothing is
approximated or precomputed. This is a convenience wrapper, not a second
implementation, so the numbers it shows are the numbers the tool prints.

Bind address is 127.0.0.1 by default, which means this machine only. To let
colleagues on the office LAN reach it, change HOST to "0.0.0.0" and tell them
http://<your-ip>:5000. Do not expose it to the open internet.
"""
import os, re, subprocess, sys
from flask import Flask, request, render_template_string, send_from_directory

# ---- configuration -------------------------------------------------------
REPO   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.environ.get("SYNTC_MODEL", os.path.join(REPO, "model.pkl"))
DTM    = os.path.join(REPO, "dtm_phil_1km.tif")
OUTDIR = os.path.join(REPO, "forecast")
HOST, PORT = "127.0.0.1", 5000
# --------------------------------------------------------------------------

# The fitted model is not in git: it is a binary and .gitignore excludes *.pkl.
# Fail here with the reason rather than on the first request with a traceback.
if not os.path.exists(MODEL):
    sys.exit(
        f"fitted model not found: {MODEL}\n"
        "Set SYNTC_MODEL to its path, or put model.pkl beside this script.\n"
        "The model ships with the Zenodo deposit linked in README.md."
    )

app = Flask(__name__)
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>SynTC genesis tool</title>
<style>
 :root{--ink:#2b2b2b;--line:#d8cfbe;--bg:#faf7f1;--accent:#8a6f4e}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.55 system-ui,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
 header{background:#6b4f3a;color:#fff;padding:14px 22px}
 header h1{margin:0;font-size:17px;letter-spacing:.3px}
 header p{margin:3px 0 0;font-size:12.5px;opacity:.85}
 main{max-width:1180px;margin:22px auto;padding:0 18px}
 form{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 18px;
      display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end}
 label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:1px;
       color:var(--accent);margin-bottom:4px}
 input,select{font:15px system-ui;padding:7px 9px;border:1px solid var(--line);
              border-radius:5px;background:#fff;width:120px}
 select{width:150px}
 button{font:15px system-ui;padding:8px 20px;border:0;border-radius:5px;
        background:#b07d62;color:#fff;cursor:pointer}
 button:hover{background:#96654c}
 .warn{background:#fff8e8;border:1px solid #e0c98a;border-radius:8px;
       padding:11px 15px;margin:16px 0;font-size:13.5px}
 .grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);
       gap:18px;margin-top:18px}
 @media(max-width:900px){.grid{grid-template-columns:1fr}}
 .card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px}
 .card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;
          letter-spacing:1px;color:var(--accent)}
 img{width:100%;height:auto;border-radius:4px}
 pre{margin:0;font:12.5px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap}
 .err{background:#fdecec;border:1px solid #e3a9a9;border-radius:8px;
      padding:12px 15px;margin-top:16px;font:12.5px/1.5 ui-monospace,monospace;
      white-space:pre-wrap}
 footer{max-width:1180px;margin:26px auto 40px;padding:0 18px;
        font-size:12.5px;color:#7a6a58}
</style></head><body>
<header>
  <h1>SynTC genesis tool</h1>
  <p>Climatological conditionals from the fitted SynTC model. Not a forecast.</p>
</header>
<main>
<form method="get">
  <div><label>Latitude °N</label><input name="lat" value="{{lat}}" required></div>
  <div><label>Longitude °E</label><input name="lon" value="{{lon}}" required></div>
  <div><label>Month</label><select name="month">
    {% for i,m in months %}<option value="{{i}}" {{'selected' if i==month}}>{{m}}</option>{% endfor %}
  </select></div>
  <div><label>Realisations</label><input name="n" value="{{n}}"></div>
  <div><label>Genesis wind kt</label><input name="wind" value="{{wind}}" placeholder="climatology"></div>
  <div><label>Panel A tracks</label><select name="ka">
    {% for k in keeps %}<option value="{{k}}" {{'selected' if k==ka}}>top {{k}}</option>{% endfor %}
  </select></div>
  <div><label>Panel B tracks</label><select name="kb">
    {% for k in keeps %}<option value="{{k}}" {{'selected' if k==kb}}>top {{k}}</option>{% endfor %}
  </select></div>
  <div><button type="submit">Run</button></div>
</form>

<div class="warn">
  <b>Read this as climatology.</b> The numbers below are conditional on a storm
  <i>forming</i> at the point you entered, in that month. They carry no information
  about today's synoptic situation, and they are not a track forecast. For warnings
  and official guidance, use PAGASA bulletins.
</div>

{% if error %}<div class="err">{{error}}</div>{% endif %}

{% if summary %}
<div class="grid">
  <div class="card"><h2>Probability of passage</h2>
    <img src="/fig/{{stem}}.png?v={{cachebust}}" alt="plume">
  </div>
  <div class="card"><h2>Summary</h2><pre>{{summary}}</pre></div>
</div>
{% if pair %}
<div class="card" style="margin-top:18px"><h2>Core-following tracks, top {{ka}} and top {{kb}}</h2>
  <img src="/fig/{{pair}}.png?v={{cachebust}}" alt="pair">
  <p style="font-size:12.5px;color:#7a6a58;margin:10px 0 0">
   The drawn tracks are the realisations that stay inside the high-probability
   corridor throughout, ranked by the 10th percentile of passage probability
   along their own path. The statistics above cover all realisations, not these.</p>
</div>
{% elif pairnote %}
<div class="warn" style="margin-top:18px">{{pairnote}}</div>
{% endif %}
{% endif %}
</main>
<footer>
  Runs <code>genesis_forecast.py</code> against the fitted model in
  <code>{{model}}</code>. Outputs are written to <code>forecast\\</code>.
  Super typhoon shares should be read as a floor and over-land winds as an upper
  bound; both biases are quantified in the manuscript.
</footer>
</body></html>"""


@app.route("/fig/<path:name>")
def fig(name):
    return send_from_directory(OUTDIR, name)


@app.route("/")
def index():
    lat  = request.args.get("lat", "12")
    lon  = request.args.get("lon", "135")
    n    = request.args.get("n", "2000")
    wind = request.args.get("wind", "").strip()
    try:
        month = int(request.args.get("month", 8))
    except ValueError:
        month = 8

    try:    ka = int(request.args.get("ka", 5))
    except: ka = 5
    try:    kb = int(request.args.get("kb", 15))
    except: kb = 15

    ctx = dict(lat=lat, lon=lon, month=month, n=n, wind=wind, ka=ka, kb=kb,
               keeps=[5, 10, 15], months=list(enumerate(MONTHS, 1)), model=MODEL,
               summary=None, stem=None, pair=None, pairnote=None,
               error=None, cachebust=os.times()[4])

    if "lat" not in request.args:
        return render_template_string(PAGE, **ctx)

    try:
        latf, lonf, ni = float(lat), float(lon), int(n)
    except ValueError:
        ctx["error"] = "Latitude, longitude and realisations must be numbers."
        return render_template_string(PAGE, **ctx)

    cmd = [sys.executable, "genesis_forecast.py", "--model", MODEL, "--dtm", DTM,
           "--lat", str(latf), "--lon", str(lonf), "--month", str(month),
           "--n", str(ni), "--out", OUTDIR]
    if wind:
        cmd += ["--wind", wind]

    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        ctx["error"] = "Timed out after 15 minutes. Try fewer realisations."
        return render_template_string(PAGE, **ctx)

    if r.returncode != 0:
        ctx["error"] = (r.stderr or r.stdout or "unknown failure").strip()
        return render_template_string(PAGE, **ctx)

    out = r.stdout
    ctx["summary"] = re.split(r"\nwritten:", out)[0].strip()
    stem = "genesis_%gN_%gE_m%02d" % (latf, lonf, month)
    ctx["stem"] = stem

    # plume_pair covers 112-145E, 2-28N only
    if 112 <= lonf <= 145 and 2 <= latf <= 28:
        pair = stem + "_pair"
        p = subprocess.run(
            [sys.executable, "plume_pair.py", "--gen", OUTDIR, "--dtm", DTM,
             "--left", stem, "--right", stem,
             "--left-pt", str(latf), str(lonf), "--right-pt", str(latf), str(lonf),
             "--left-label", "(a) top %d core-following" % ka,
             "--right-label", "(b) top %d core-following" % kb,
             "--keep-left", str(ka), "--keep-right", str(kb),
             "--out", os.path.join(OUTDIR, pair)],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            ctx["pair"] = pair
        else:
            ctx["pairnote"] = "Track panels could not be drawn: " + (
                p.stderr or p.stdout or "unknown").strip()[:400]
    else:
        ctx["pairnote"] = ("%g\u00b0N %g\u00b0E is outside the track-panel window "
                           "(112-145\u00b0E, 2-28\u00b0N), so only the single-panel "
                           "figure above is available." % (latf, lonf))
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    print("SynTC genesis tool at http://%s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=False)
