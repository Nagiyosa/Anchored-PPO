#!/usr/bin/env python3
"""D33 step 1: derive route-budget evidence from all train operating days.

This script is deliberately read-only with respect to datasets.  It writes a
timestamped report so every audit remains reproducible and no run is overwritten.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.route_audit import (audit_operating_day_coverage,
                                         audit_training_route_chains,
                                         bc_dataset_fingerprint)
from derbysim.learn.scenarios import (_td_events, build_day_trains, load_assets,
                                      operating_days)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    days = operating_days("train")
    raw_days = operating_days("all", include_partial=True)
    net, _params, _dwell, ptc = load_assets()
    trains = []
    train_counts = []
    for i, day in enumerate(days, 1):
        day_trains = build_day_trains(day, inject=False)
        trains.extend(day_trains)
        train_counts.append(len(day_trains))
        print(f"  [{i:02d}/{len(days)}] {str(day['start'])[:10]} trains={len(trains)}", flush=True)

    dataset = config.ARTIFACTS / "dataset" / "bc_dataset_all_d31.parquet"
    frame = pd.read_parquet(dataset)
    td = _td_events()
    raw_coverage = audit_operating_day_coverage(
        raw_days, [None] * len(raw_days), td["ts"].min(), td["ts"].max())
    report = {
        "audit": "D33 route budget and BC invariance baseline",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_git_head": _git_head(),
        "split": "train",
        "n_operating_days": len(days),
        "first_day": str(days[0]["start"]),
        "last_day": str(days[-1]["start"]),
        "operating_day_coverage": audit_operating_day_coverage(
            days, train_counts, td["ts"].min(), td["ts"].max()),
        "excluded_partial_operating_days": raw_coverage["partial_boundary_days"],
        "route_chains": audit_training_route_chains(trains, net, ptc),
        "bc_dataset": str(dataset),
        "bc_fingerprint": bc_dataset_fingerprint(frame),
    }
    out = config.ARTIFACTS_REPORTS / f"route_budget_audit_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["route_chains"], indent=2, ensure_ascii=False))
    print(json.dumps(report["bc_fingerprint"], indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
