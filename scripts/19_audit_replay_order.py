"""D33 step 5: audit Panel_Request order against observed physical traversal.

This script is read-only.  It compares the chronological request sequence used
by M8/D31 with a deterministic position order inferred from observed Track
occupations plus route continuity.  It reports BC natural days and RL 03:00
operating days separately; no dataset is rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.scenarios import OPERATING_DAY_CUT_HOUR, operating_days
from derbysim.sim.dataset import (_chain_continuity, chain_is_continuous,
                                  position_order_indices)
from derbysim.sim.env import build_env_trains_for_window


def _d31_chains(route_chain, occ_first, net, succ, pair):
    """Apply the exact D31 boundary split to one supplied route order."""
    cuts = [0]
    for i in range(1, len(route_chain)):
        if not chain_is_continuous(route_chain[i - 1:i + 1], net, succ, pair):
            cuts.append(i)
    cuts.append(len(route_chain))
    chains = []
    for lo, hi in zip(cuts, cuts[1:]):
        chain = list(route_chain[lo:hi])
        path = []
        for rid in chain:
            route = net.routes.get(rid)
            for tc in (route.tc_sequence if route else ()):
                if not path or path[-1] != tc:
                    path.append(tc)
        if len(path) >= 2 and path[0] in occ_first:
            chains.append(chain)
    return chains


def _audit_windows(windows, td, net, label):
    succ, pair = _chain_continuity(net)
    counts = Counter()
    examples = []
    # One sort plus binary slices avoids rescanning 2.3M rows for every day.
    td = td.sort_values("ts").reset_index(drop=True)
    for wi, win in enumerate(windows):
        lo = int(td["ts"].searchsorted(win["start"], side="left"))
        hi = int(td["ts"].searchsorted(win["end"], side="left"))
        sub = td.iloc[lo:hi].copy()
        built = build_env_trains_for_window(sub, net, {}, {})
        for et, occ_first in built:
            counts["trains"] += 1
            n_dec = max(len(et.route_chain) - 1, 0)
            counts["raw_decisions"] += n_dec
            raw_cont = chain_is_continuous(et.route_chain, net, succ, pair)
            counts["raw_discontinuous"] += not raw_cont
            old_segments = _d31_chains(et.route_chain, occ_first, net, succ, pair)
            old_dec = sum(max(len(chain) - 1, 0) for chain in old_segments)
            counts["d31_segments"] += len(old_segments)
            counts["d31_decisions"] += old_dec

            order = position_order_indices(et, occ_first, net, succ, pair)
            position_chain = [et.route_chain[i] for i in order]
            new_segments = _d31_chains(position_chain, occ_first, net, succ, pair)
            new_dec = sum(max(len(chain) - 1, 0) for chain in new_segments)
            counts["position_components"] += len(new_segments)
            counts["position_decisions"] += new_dec
            reordered = order != tuple(range(len(order)))
            counts["reordered_trains"] += reordered
            counts["reordered_routes"] += sum(i != j for i, j in enumerate(order))
            counts["recovered_decisions"] += max(new_dec - old_dec, 0)
            if reordered and len(examples) < 20:
                examples.append({
                    "window": str(win["start"]), "train": et.train_id,
                    "request_order": list(et.route_chain),
                    "position_order": position_chain,
                    "component_sizes": [len(chain) for chain in new_segments],
                    "old_decisions": old_dec, "position_decisions": new_dec,
                })
        if (wi + 1) % 20 == 0 or wi + 1 == len(windows):
            print(f"[{label}] {wi + 1}/{len(windows)} windows", flush=True)
    return {"scope": label, **dict(counts), "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("bc", "rl", "both"), default="both")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    net = InfrastructureNetwork.load()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    reports = []
    if args.scope in ("bc", "both"):
        days = sorted({pd.Timestamp(x).normalize() for x in td["ts"].dt.date})
        wins = [{"start": d, "end": d + pd.Timedelta(days=1)} for d in days]
        reports.append(_audit_windows(wins, td, net, "BC natural days (98 raw days)"))
    if args.scope in ("rl", "both"):
        reports.append(_audit_windows(operating_days("all"), td, net,
                                      f"RL complete operating days ({OPERATING_DAY_CUT_HOUR:02d}:00 cut)"))
    payload = {
        "audit": "D33 step 5 Panel_Request order versus observed traversal",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": ("unique route ids; every route has an observed exclusive TC; "
                 "accept only when physical discontinuities strictly decrease"),
        "reports": reports,
    }
    out = args.output
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = config.ARTIFACTS / "reports" / f"replay_order_audit_{args.scope}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
