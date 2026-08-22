"""Demo timeline generator — run one (day, policy) episode and dump a playback JSON.

Usage (repo root):
    PYTHONPATH=. python demo/generate_timeline.py --days 2023-11-16..2023-11-22 --policies signaller,fcfs
    PYTHONPATH=. python demo/generate_timeline.py --days 2023-11-16..2023-11-22 --policies rl \
        --ckpt artifacts/reports/ppo_d73_AB_s0_best.pt          # needs torch (run on your machine)
    PYTHONPATH=. python demo/generate_timeline.py --days 2023-11-24 --policies rl,fcfs,signaller

Policies: rl (torch ckpt), bc (torch ckpt via --bc-ckpt), signaller, fcfs, lookup, always-set.
Output:   demo/data/<day>_<policy>.json  +  demo/data/topology.json (once)
Env config mirrors scripts/39 (paper eval): proactive, wide window, control_classes={1,2}.
No project code is modified: lock/release events are recorded by wrapping RouteReservation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from derbysim.learn.baselines import lookup_table, make_fcfs_policy, make_lookup_policy
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.sim import reservation as RES
from derbysim.sim.reward import (RewardConfig, RewardedDynamicEnv, can_complete,
                                 has_reached_goal, realized_weighted_delay)

RCFG = RewardConfig()
SVG = ROOT / "derby_experiment_region.svg"
OUT = ROOT / "demo" / "data"


# ---------------------------------------------------------------- SVG topology
def route_svg_id(route_id: str) -> str:
    """RDC5061A(M) -> DYRDC5061A_M_   (matches ids inside derby_experiment_region.svg)."""
    return "DY" + route_id.replace("(", "_").replace(")", "_")


def parse_topology(net):
    s = SVG.read_text(encoding="utf-8")
    polys = {m.group(2): m.group(1)
             for m in re.finditer(r'<path d="M ([\d,. ]+)" id="(DYR[^"]+)"', s)}
    signals = {m.group(3): (float(m.group(1)), float(m.group(2)))
               for m in re.finditer(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="\d+" id="(DYS[^"]+)"', s)}
    berths = {m.group(5): (float(m.group(1)), float(m.group(2)))
              for m in re.finditer(r'<rect x="([\d.]+)" y="([\d.]+)" width="(\d+)" height="(\d+)" id="(B_[^"]+)"', s)}

    def pts_of(d):
        return [tuple(float(v) for v in p.split(",")) for p in d.split()]

    topo = {"routes": {}, "signals": {k: list(v) for k, v in signals.items()},
            "berths": {k: list(v) for k, v in berths.items()}}
    for rid, r in net.routes.items():
        sid = route_svg_id(rid)
        if sid not in polys:                     # call-on (C) variants share M geometry
            for alt in (sid.replace("_C_", "_M_"), sid.replace("_M_", "_C_")):
                if alt in polys:
                    sid = alt
                    break
        if sid not in polys:
            continue
        pts = pts_of(polys[sid])
        # orient the polyline so index 0 is at the route's START signal
        sig_key = "DYS" + r.prefix + str(r.signal)
        sc = signals.get(sig_key)
        if sc is not None:
            d0 = (pts[0][0] - sc[0]) ** 2 + (pts[0][1] - sc[1]) ** 2
            d1 = (pts[-1][0] - sc[0]) ** 2 + (pts[-1][1] - sc[1]) ** 2
            if d1 < d0:
                pts = pts[::-1]
        topo["routes"][rid] = {"svg": sid, "pts": [list(p) for p in pts],
                               "signal": sig_key, "n_tc": len(r.tc_sequence),
                               "tcs": list(r.tc_sequence)}
    return topo


def _cumlen(pts):
    acc, out = 0.0, [0.0]
    for a, b in zip(pts, pts[1:]):
        acc += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        out.append(acc)
    return out


def point_at(pts, cum, frac):
    """Point at fraction [0,1] of total polyline length."""
    if not pts:
        return None
    target = max(0.0, min(1.0, frac)) * cum[-1]
    for i in range(1, len(pts)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1] or 1.0
            u = (target - cum[i - 1]) / seg
            a, b = pts[i - 1], pts[i]
            return [round(a[0] + u * (b[0] - a[0]), 1), round(a[1] + u * (b[1] - a[1]), 1)]
    return [round(pts[-1][0], 1), round(pts[-1][1], 1)]


# ---------------------------------------------------------------- recorder
class Recorder:
    """Wraps RouteReservation methods; records lock / full-release with sim time."""

    def __init__(self):
        self.events = []          # [t, 'L'|'R', route_id, train_id]
        self.engine = None
        self.mute = 0             # >0 while inside deadlock look-ahead (trial locks roll back)
        self._orig = {}
        self._orig_wcd = {}

    def now(self):
        try:
            return float(self.engine.now) if self.engine is not None else None
        except Exception:
            return None

    def install(self):
        R = RES.RouteReservation
        rec = self

        def wrap_lock(orig):
            def f(self, route, train):
                before = set(self.holder_fp)
                r = orig(self, route, train)
                if not rec.mute:
                    for key in before - set(self.holder_fp):   # TC transfer removed a key
                        rec.events.append([rec.now(), "R", key[0], key[1]])
                    rec.events.append([rec.now(), "L", route.route_id, train])
                return r
            return f

        def wrap_release(orig):
            def f(self, *a, **k):
                before = set(self.holder_fp)
                r = orig(self, *a, **k)
                if not rec.mute:
                    for key in before - set(self.holder_fp):
                        rec.events.append([rec.now(), "R", key[0], key[1]])
                return r
            return f

        for name, w in (("_lock", wrap_lock), ("release_tc", wrap_release),
                        ("release_overlap", wrap_release), ("release_route", wrap_release)):
            self._orig[name] = getattr(R, name)
            setattr(R, name, w(self._orig[name]))

        # mute the one-step deadlock look-ahead: it trial-locks on the real object
        # then rolls back by snapshot restore (deadlock.py::would_create_deadlock).
        from derbysim.sim import deadlock as DL
        from derbysim.sim import env_dynamic as ED
        real = DL.would_create_deadlock

        def muted(*a, **k):
            rec.mute += 1
            try:
                return real(*a, **k)
            finally:
                rec.mute -= 1

        for mod in (DL, ED):
            if getattr(mod, "would_create_deadlock", None) is not None:
                self._orig_wcd[mod] = mod.would_create_deadlock
                mod.would_create_deadlock = muted

    def uninstall(self):
        for name, fn in self._orig.items():
            setattr(RES.RouteReservation, name, fn)
        self._orig.clear()
        for mod, fn in self._orig_wcd.items():
            mod.would_create_deadlock = fn
        self._orig_wcd.clear()


# ---------------------------------------------------------------- policies
def load_torch_policy(ckpt_path):
    import torch
    from derbysim.learn.panel_models import PanelOptionPolicy
    from derbysim.learn.ppo import RunningMoments, TorchPanelDriver
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net, params, dwell, ptc = load_assets()
    if "state_dict" in st:                     # scripts/31 full-panel BC checkpoint
        fv = 1
        pol = PanelOptionPolicy(st["encoder"], *st["dims"])
        pol.load_state_dict(st["state_dict"])
    else:                                       # scripts/13 PPO checkpoint (scripts/39 convention)
        ck = st.get("args", {})
        fv = int(ck.get("feature_version", ck.get("fv", 2)))
        ad = PanelAdapter(net, Featurizer(net, ptc, fv=fv))
        pol = PanelOptionPolicy(ck.get("encoder", "gnn"), ad.dec_dim, ad.conf_dim,
                                ad.cand_dim, ad.cancel_dim, ad.train_dim, ad.global_dim,
                                hidden=int(ck.get("hidden", 64)))
        pol.load_state_dict(st["model"])
    pol.eval()
    rn = RunningMoments()
    if st.get("ret_norm"):
        rn.load_state_dict(st["ret_norm"])
    drv = TorchPanelDriver(pol, torch.device("cpu"), greedy=True, ret_norm=rn)
    return (lambda env: drv), fv


def make_policies(names, args):
    """-> {name: (factory(env)->act, feature_version)}"""
    out = {}
    top = lookup_table()
    for n in names:
        if n == "rl":
            fac, fv = load_torch_policy(args.ckpt)
            out[n] = (fac, fv)
        elif n == "bc":
            fac, fv = load_torch_policy(args.bc_ckpt)
            out[n] = (fac, fv)
        elif n == "fcfs":
            out[n] = (make_fcfs_policy(top), 1)
        elif n == "lookup":
            out[n] = (make_lookup_policy(top), 1)
        elif n == "signaller":
            out[n] = (make_signaller_replay_policy, 1)
        elif n == "always-set":
            from derbysim.learn.rollout import make_greedy_set_policy
            out[n] = ((lambda env: make_greedy_set_policy()), 1)
        else:
            raise SystemExit(f"unknown policy {n}")
    return out


# ---------------------------------------------------------------- episode
def run_one(day, policy_name, fac, fv, assets):
    net, params, dwell, ptc = assets
    adapter = PanelAdapter(net, Featurizer(net, ptc, fv=fv))
    rec = Recorder()
    rec.install()
    try:
        env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True,
                                 reserve_window="wide", control_classes={"1", "2"})
        trains = build_day_trains(day, inject=False)
        horizon = day_horizon(day)
        obs, _ = env.reset(trains, horizon=horizon)
        rec.engine = env.engine
        act = fac(env)                       # build once per env (scripts/39 convention)
        t_wall = time.time()
        decisions, kpi = [], []
        n = 0
        while not env.done and n < 20000:
            dec = env.decision()
            if dec is None or obs is None:
                break
            enc = adapter.encode(obs, dec)
            idx, _lp, _v = act(enc)
            a = dec.options[int(idx)]
            eng = env.engine
            decisions.append([float(eng.now), a.kind, a.train_id, a.route_id,
                              len(dec.options)])
            obs, _r, _d, _ = env.step(a)
            kpi.append([float(eng.now),
                        round(realized_weighted_delay(eng, RCFG), 1),
                        sum(1 for t in eng.trains.values()
                            if not t.background and has_reached_goal(t)),
                        sum(t.n_stops for t in eng.trains.values() if not t.background)])
            n += 1
        wall = time.time() - t_wall
    finally:
        rec.uninstall()
    return env.engine, decisions, kpi, rec.events, wall


def waypoints_for(t, net, topo, cums):
    """Ordered (t, x, y) from sim_occ using route geometry; robust to unknown TCs."""
    if not t.path or not t.sim_occ:
        return []
    wps = []
    for i, tc in enumerate(t.path):
        ts = t.sim_occ.get(tc)
        rid = t.path_route_ids[i] if i < len(t.path_route_ids) else None
        if ts is None or rid is None:
            continue
        R = topo["routes"].get(rid)
        r = net.routes.get(rid)
        if R is None or r is None:
            continue
        try:
            j = r.tc_sequence.index(tc)
            frac = (j + 0.5) / max(1, len(r.tc_sequence))
        except ValueError:
            continue
        p = point_at(R["pts"], cums[rid], frac)
        if p:
            wps.append([float(ts), p[0], p[1]])
    wps.sort(key=lambda x: x[0])
    return wps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True,
                    help="YYYY-MM-DD, comma list, or A..B inclusive range")
    ap.add_argument("--policies", default="signaller,fcfs")
    ap.add_argument("--ckpt", default="artifacts/reports/ppo_d73_AB_s0_best.pt")
    ap.add_argument("--bc-ckpt", default="artifacts/dataset/bc_fullpanel_gnn.pt")
    ap.add_argument("--kpi-points", type=int, default=1500)
    args = ap.parse_args()

    # resolve days
    all_days = {str(w["start"])[:10]: (w, sp)
                for sp in ("train", "val", "test") for w in operating_days(sp)}
    if ".." in args.days:
        a, b = args.days.split("..")
        dates = sorted(d for d in all_days if a <= d <= b)
    else:
        dates = [d for d in args.days.split(",") if d]
    for d in dates:
        if d not in all_days:
            raise SystemExit(f"day {d} is not an operating day")

    assets = load_assets()
    net = assets[0]
    OUT.mkdir(parents=True, exist_ok=True)

    topo = parse_topology(net)
    cums = {rid: _cumlen(R["pts"]) for rid, R in topo["routes"].items()}
    topo_out = {"routes": {rid: {"svg": R["svg"], "signal": R["signal"]}
                           for rid, R in topo["routes"].items()},
                "unmapped_routes": sorted(set(net.routes) - set(topo["routes"]))}
    (OUT / "topology.json").write_text(json.dumps(topo_out), encoding="utf-8")

    pols = make_policies([p.strip() for p in args.policies.split(",") if p.strip()], args)

    for date in dates:
        day, split = all_days[date]
        for name, (fac, fv) in pols.items():
            eng, decisions, kpi, resv, wall = run_one(day, name, fac, fv, assets)

            trains, wps = {}, {}
            for t in eng.trains.values():
                occ = list(t.sim_occ.values())
                trains[t.train_id] = {
                    "hc": t.train_id.split("#")[0], "cls": str(t.service_class),
                    "bg": bool(t.background), "entry": float(t.entry_time),
                    "entered": (min(occ) if occ else None),
                    "exit": (max(occ) if occ else None),
                    "done": bool(has_reached_goal(t)),
                    "completable": bool(can_complete(t)),
                    "wdelay": round(RCFG.lam(t.service_class) * float(t.held_time), 1),
                    "stops": int(t.n_stops),
                    "delay_min": (None if t.background else eng._delay_min(t)),
                }
                w_ = waypoints_for(t, net, topo, cums)
                if w_:
                    wps[t.train_id] = w_

            if len(kpi) > args.kpi_points:
                import numpy as np
                idx = np.linspace(0, len(kpi) - 1, args.kpi_points).astype(int)
                kpi = [kpi[i] for i in idx]

            ctl = [t for t in eng.trains.values() if not t.background]
            comp = [t for t in ctl if can_complete(t)]
            done = [t for t in comp if has_reached_goal(t)]
            dm = [eng._delay_min(t) for t in done]
            dm = [x for x in dm if x is not None]
            out = {
                "meta": {"day": date, "split": split, "policy": name,
                         "t0": float(day["start"].timestamp()),
                         "t1": float(day["end"].timestamp()),
                         "horizon": day_horizon(day), "wall_s": round(wall, 1),
                         "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
                "summary": {
                    "n_controlled": len(ctl), "n_background": len(eng.trains) - len(ctl),
                    "n_completable": len(comp), "n_done": len(done),
                    "completion": round(len(done) / max(1, len(comp)), 4),
                    "wdelay": round(sum(RCFG.lam(t.service_class) * t.held_time
                                        for t in ctl), 1),
                    "stops": sum(t.n_stops for t in ctl),
                    "ppm300": round(sum(1 for x in dm if x * 60.0 <= 300.0)
                                    / max(1, len(comp)), 4),
                    "collisions": 0 if not getattr(eng, "collisions", None) else len(eng.collisions),
                    "deadlocked": bool(getattr(eng, "deadlocked", False)),
                    "decisions": len(decisions)},
                "kpi": kpi,
                "decisions": decisions,
                "reservations": [e for e in resv if e[0] is not None],
                "trains": trains,
                "waypoints": wps,
            }
            f = OUT / f"{date}_{name}.json"
            f.write_text(json.dumps(out), encoding="utf-8")
            s = out["summary"]
            print(f"[{date} {name}] comp {s['completion']:.4f} ({s['n_done']}/{s['n_completable']}) "
                  f"wdelay {s['wdelay']:.0f}s stops {s['stops']} ppm {s['ppm300']:.3f} "
                  f"dec {s['decisions']} wall {wall:.1f}s -> {f.name}", flush=True)


if __name__ == "__main__":
    main()
