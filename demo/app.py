"""Derby dispatch demo server.

    PYTHONPATH=. python demo/app.py            # -> http://127.0.0.1:5045
Serves the playback UI over pre-generated timelines in demo/data/
(see demo/generate_timeline.py). No torch / no simulator needed at serve time.
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, abort

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SVG = HERE.parent / "derby_experiment_region.svg"

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/svg")
def svg():
    return SVG.read_text(encoding="utf-8"), 200, {"Content-Type": "image/svg+xml"}


@app.get("/api/topology")
def topology():
    p = DATA / "topology.json"
    if not p.exists():
        abort(404, "run demo/generate_timeline.py first")
    return send_from_directory(str(DATA), "topology.json")


@app.get("/api/runs")
def runs():
    """{day: {policy: summary}} for every generated timeline."""
    out = {}
    for f in sorted(DATA.glob("*_*.json")):
        if f.name == "topology.json":
            continue
        day, _, policy = f.stem.partition("_")
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.setdefault(day, {})[policy] = {**d.get("summary", {}),
                                           "split": d.get("meta", {}).get("split")}
    return jsonify(out)


@app.get("/api/data/<day>/<policy>")
def data(day, policy):
    name = f"{day}_{policy}.json"
    if not (DATA / name).exists():
        abort(404)
    return send_from_directory(str(DATA), name)


if __name__ == "__main__":
    print("Derby dispatch demo -> http://127.0.0.1:5045")
    app.run(host="127.0.0.1", port=5045, debug=False)
