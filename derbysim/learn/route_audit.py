"""D33 pre-repair audit helpers.

The counterfactual engine must not use a test trajectory to decide how long a
train is allowed to run.  These helpers therefore derive route-budget evidence
from the complete *training* split and pin the logged BC dataset independently.

They deliberately inspect observed route chains only.  No ad-hoc graph search
is used here: an unvalidated reachability proxy would repeat the project's
earlier mistake of turning a diagnostic heuristic into ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np

from derbysim.sim.goal import train_goal


PERCENTILES = (50, 90, 95, 99)


def _distribution(values) -> dict:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"n": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None}
    out = {"n": int(xs.size), "min": int(xs.min()), "max": int(xs.max())}
    for q in PERCENTILES:
        out[f"p{q}"] = float(np.percentile(xs, q))
    return out


def _normalised_tc_path(route_chain, net) -> tuple[list[str], list[str]]:
    """Expand route bodies, removing only adjacent boundary duplicates.

    Repeated TCs are *not* a loop test: consecutive UK routes may legitimately
    share track-circuit names at their boundary, and overlap footprints are a
    separate reservation concept.  The expanded path is retained only as a
    descriptive topology diagnostic.
    """
    path: list[str] = []
    missing: list[str] = []
    for rid in route_chain:
        route = net.routes.get(rid)
        if route is None:
            missing.append(rid)
            continue
        for tc in route.tc_sequence:
            if not path or path[-1] != tc:
                path.append(tc)
    return path, missing


def audit_training_route_chains(trains, net, plat_tc_to_pid: dict) -> dict:
    """Summarise all observed train-split chains for a data-derived route budget.

    Loop evidence is based on repeated route IDs and repeated route-start
    signals, not repeated TCs.  Track circuits can legitimately recur across
    adjacent route bodies and therefore cannot identify a cycle by themselves.
    """
    trains = list(trains)
    total_routes: list[int] = []
    controlled_routes: list[int] = []
    by_class: dict[str, list[int]] = defaultdict(list)
    by_goal: dict[str, list[int]] = defaultdict(list)
    repeated_route_ids: list[dict] = []
    repeated_start_signals: list[dict] = []
    shared_tc_chains = 0
    missing_routes = Counter()

    for train in trains:
        chain = list(train.route_chain)
        n_route = len(chain)
        total_routes.append(n_route)
        controlled_routes.append(max(n_route - 1, 0))  # entry route is not an action (D14.8)
        by_class[str(train.service_class or "other")].append(n_route)

        goal = train_goal(chain, net, train.stop_pids, plat_tc_to_pid)
        goal_kind = ("exit" if goal.exit_group else
                     "terminal_platform" if goal.terminates_at_platform else "unresolved")
        by_goal[goal_kind].append(n_route)

        tc_path, missing = _normalised_tc_path(chain, net)
        missing_routes.update(missing)
        seen, duplicates = set(), []
        for tc in tc_path:
            if tc in seen:
                duplicates.append(tc)
            seen.add(tc)
        if duplicates:
            shared_tc_chains += 1

        route_counts = Counter(chain)
        route_repeats = sorted(rid for rid, n in route_counts.items() if n > 1)
        if route_repeats:
            repeated_route_ids.append({"train_id": train.train_id,
                                       "route_count": n_route,
                                       "route_ids": route_repeats})

        start_signals = [net.routes[rid].signal for rid in chain if rid in net.routes]
        signal_counts = Counter(start_signals)
        signal_repeats = sorted(str(sig) for sig, n in signal_counts.items() if n > 1)
        if signal_repeats:
            repeated_start_signals.append({"train_id": train.train_id,
                                           "route_count": n_route,
                                           "signals": signal_repeats})

    total = _distribution(total_routes)
    p99_ceil = int(math.ceil(total["p99"])) if total["p99"] is not None else None
    return {
        "n_trains": len(trains),
        "route_count_including_entry": total,
        "agent_route_count_excluding_entry": _distribution(controlled_routes),
        "by_service_class": {k: _distribution(v) for k, v in sorted(by_class.items())},
        "by_goal_kind": {k: _distribution(v) for k, v in sorted(by_goal.items())},
        "chains_with_repeated_body_tc_descriptive_only": shared_tc_chains,
        "chains_with_repeated_route_id": len(repeated_route_ids),
        "repeated_route_id_examples": repeated_route_ids[:20],
        "chains_with_repeated_start_signal": len(repeated_start_signals),
        "repeated_start_signal_examples": repeated_start_signals[:20],
        "missing_route_ids": dict(sorted(missing_routes.items())),
        "budget_candidates_including_entry": {
            "train_p99_ceil": p99_ceil,
            "two_times_train_p99": 2 * p99_ceil if p99_ceil is not None else None,
            "three_times_train_p99": 3 * p99_ceil if p99_ceil is not None else None,
            "max_observed": total["max"],
            "recommended_total_actual_route_cap": 3 * p99_ceil if p99_ceil is not None else None,
            "recommendation_basis": (
                "global train-only 3*ceil(p99), including the non-agent entry route; "
                "a conservative safety bound, not a success criterion"
            ),
        },
    }


def audit_operating_day_coverage(days, train_counts, source_min, source_max) -> dict:
    """Make boundary truncation explicit instead of hiding it in a day count."""
    rows = []
    for day, n_train in zip(days, train_counts):
        rows.append({
            "start": str(day["start"]),
            "end": str(day["end"]),
            "n_trains": int(n_train) if n_train is not None else None,
            "source_covers_start": bool(source_min <= day["start"]),
            "source_covers_end": bool(source_max >= day["end"]),
        })
    partial = [row for row in rows
               if not row["source_covers_start"] or not row["source_covers_end"]]
    return {
        "source_min": str(source_min),
        "source_max": str(source_max),
        "n_days": len(rows),
        "n_partial_boundary_days": len(partial),
        "partial_boundary_days": partial,
        "daily_train_counts": rows,
    }


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def bc_dataset_fingerprint(frame) -> dict:
    """Stable fingerprint of labels, masks and source observations in the BC parquet."""
    keys = ["split", "date", "time", "train_id", "signal", "chosen_route"]
    rows = frame.sort_values(keys, kind="mergesort").to_dict("records")
    hashes = {name: hashlib.sha256() for name in ("labels", "legal", "obs", "combined")}
    for row in rows:
        base = [_plain(row.get(k)) for k in keys]
        parts = {
            "labels": base,
            "legal": base + [_plain(row.get("legal_route_ids"))],
            "obs": base + [_plain(row.get("obs"))],
        }
        combined = {k: parts[k] for k in ("labels", "legal", "obs")}
        for name, value in (*parts.items(), ("combined", combined)):
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), default=str).encode("utf-8")
            hashes[name].update(payload)
            hashes[name].update(b"\n")
    return {
        "n_rows": int(len(frame)),
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()},
        "sha256_labels": hashes["labels"].hexdigest(),
        "sha256_legal_masks": hashes["legal"].hexdigest(),
        "sha256_observations": hashes["obs"].hexdigest(),
        "sha256_combined": hashes["combined"].hexdigest(),
    }
