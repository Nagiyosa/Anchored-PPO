#!/usr/bin/env python3
"""D33 Step 8 audit: timing funnel, same-time panels, and WAIT contract.

Read-only with respect to simulation semantics.  It runs the signaller replay
and greedy-set reference on all validation operating days, records the route
funnel before every action, and measures whether WAIT really advances time.
Every run is preserved as a timestamped JSON report.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (make_greedy_set_policy,
                                    make_signaller_replay_policy, run_episode)
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.deadlock import would_create_deadlock
from derbysim.sim.reward import RewardedDynamicEnv


def _percentile(values, q):
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _geometry_routes(engine, train) -> list[str]:
    """The exact pre-goal geometry stage used by DynamicEngine._legal."""
    routes = []
    signal = engine._decision_signal(train)
    if signal is not None:
        signals = {signal}
        if signal in engine._pair:
            signals.add(engine._pair[signal])
        for sig in signals:
            routes.extend(engine.net.routes_from(sig))
    if train.path:
        for tc in engine._tc_succ.get(train.path[-1], ()):
            routes.extend(engine._routes_by_first.get(tc, ()))
    out, seen = [], set()
    for route in routes:
        if route.route_id not in seen and getattr(route, "in_scope", True):  # D41: 只算 in-scope
            seen.add(route.route_id)
            out.append(route.route_id)
    return out


def _snapshot(env, enc) -> dict:
    engine = env.engine
    awaiting = engine._actionable()
    per_train = []
    for train in awaiting:
        geometry = _geometry_routes(engine, train)
        goal = [rid for rid in geometry if engine.reach.route_keeps_goal(
            engine.net.routes[rid], train.goal, train.served_platform)]
        safe = [rid for rid in goal if not would_create_deadlock(
            engine.res, engine.net.routes[rid], train.train_id)]
        per_train.append({"train_id": train.train_id,
                          "n_geometry": len(geometry),
                          "n_goal_kept": len(goal),
                          "n_deadlock_safe": len(safe)})
    slots = enc["slots"]
    set_slots = [s for s in slots if s.kind == "set"]
    next_event_dt = None
    if len(engine.q):
        next_event_dt = float(engine.q._h[0].time - engine.now)  # read-only diagnostic
    return {
        "now": float(engine.now),
        "next_event_dt": next_event_dt,
        "n_awaiting": len(awaiting),
        "n_cancellable": sum(s.kind == "cancel" for s in slots),
        "n_set_options": len(set_slots),
        "n_set_trains": len({s.train_id for s in set_slots}),
        "n_geometry": sum(x["n_geometry"] for x in per_train),
        "n_goal_kept": sum(x["n_goal_kept"] for x in per_train),
        "n_deadlock_safe": sum(x["n_deadlock_safe"] for x in per_train),
        "n_awaiting_no_geometry": sum(x["n_geometry"] == 0 for x in per_train),
        "n_awaiting_goal_filtered": sum(
            x["n_geometry"] > 0 and x["n_goal_kept"] == 0 for x in per_train),
        "n_awaiting_deadlock_filtered": sum(
            x["n_goal_kept"] > 0 and x["n_deadlock_safe"] == 0 for x in per_train),
    }


class AuditedPolicy:
    def __init__(self, env, base_policy):
        self.env = env
        self.base_policy = base_policy
        self.snapshots = []

    def __call__(self, enc):
        snap = _snapshot(self.env, enc)
        idx, logp, value = self.base_policy(enc)
        slot = enc["slots"][idx]
        snap.update({"selected_kind": slot.kind,
                     "selected_train": slot.train_id,
                     "selected_route": slot.route_id})
        self.snapshots.append(snap)
        return idx, logp, value


def _run_policy(name, factory, days, net, params, dwell, ptc, adapter, max_decisions):
    all_rows = []
    day_metrics = []
    for day_idx, day in enumerate(days, 1):
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        audited = AuditedPolicy(env, factory(env))
        result = run_episode(env, adapter, audited, build_day_trains(day, inject=False),
                             max_decisions=max_decisions, horizon=day_horizon(day))
        if len(audited.snapshots) != len(result.transitions):
            raise RuntimeError("timing snapshots and transitions are misaligned")
        previous_now = None
        for snap, transition in zip(audited.snapshots, result.transitions):
            snap["dt"] = float(transition.dt)
            snap["same_time_as_previous_panel"] = previous_now == snap["now"]
            previous_now = snap["now"]
            all_rows.append(snap)
        day_metrics.append({"day": str(day["start"])[:10], **result.metrics})
        print(f"{name:<16} day {day_idx:02d}/{len(days)} {str(day['start'])[:10]} "
              f"steps={len(result.transitions)} completion={result.metrics['completion_rate']:.3f}",
              flush=True)
    return {"policy": name, "summary": _summarise(all_rows, day_metrics),
            "days": day_metrics}


def _summarise(rows, day_metrics):
    actions = Counter(r["selected_kind"] for r in rows)
    waits = [r for r in rows if r["selected_kind"] == "wait"]
    wait_dt = [r["dt"] for r in waits]
    total_awaiting = sum(r["n_awaiting"] for r in rows)
    return {
        "n_days": len(day_metrics),
        "n_panels": len(rows),
        "action_counts": dict(sorted(actions.items())),
        "action_mix": {k: v / len(rows) for k, v in sorted(actions.items())} if rows else {},
        "n_panels_same_time_as_previous": sum(r["same_time_as_previous_panel"] for r in rows),
        "frac_panels_same_time_as_previous": (
            sum(r["same_time_as_previous_panel"] for r in rows) / len(rows) if rows else 0.0),
        "n_panels_with_next_event_same_time": sum(r["next_event_dt"] == 0.0 for r in rows),
        "n_panels_with_multiple_awaiting": sum(r["n_awaiting"] >= 2 for r in rows),
        "n_panels_with_multiple_set_trains": sum(r["n_set_trains"] >= 2 for r in rows),
        "max_awaiting": max((r["n_awaiting"] for r in rows), default=0),
        "max_set_trains": max((r["n_set_trains"] for r in rows), default=0),
        "funnel_totals": {
            "awaiting_train_rows": total_awaiting,
            "geometry_routes": sum(r["n_geometry"] for r in rows),
            "goal_kept_routes": sum(r["n_goal_kept"] for r in rows),
            "deadlock_safe_routes": sum(r["n_deadlock_safe"] for r in rows),
            "set_options": sum(r["n_set_options"] for r in rows),
            "awaiting_no_geometry": sum(r["n_awaiting_no_geometry"] for r in rows),
            "awaiting_goal_filtered": sum(r["n_awaiting_goal_filtered"] for r in rows),
            "awaiting_deadlock_filtered": sum(r["n_awaiting_deadlock_filtered"] for r in rows),
        },
        "wait_contract": {
            "n_wait": len(waits),
            "n_zero_dt_wait": sum(dt == 0.0 for dt in wait_dt),
            "n_negative_dt_wait": sum(dt < 0.0 for dt in wait_dt),
            "min_dt": min(wait_dt) if wait_dt else None,
            "median_dt": statistics.median(wait_dt) if wait_dt else None,
            "p95_dt": _percentile(wait_dt, 0.95),
            "max_dt": max(wait_dt) if wait_dt else None,
            "n_wait_while_same_time_event_pending": sum(
                r["next_event_dt"] == 0.0 for r in waits),
        },
        "closed_loop": {
            "n_completable": sum(m["n_completable"] for m in day_metrics),
            "n_reached": sum(m["n_reached_exit"] for m in day_metrics),
            "n_collisions": sum(m["n_collisions"] for m in day_metrics),
            "n_deadlocks": sum(bool(m["deadlocked"]) for m in day_metrics),
            "n_truncated": sum(bool(m["truncated"]) for m in day_metrics),
            "n_state_invariant_violations": sum(
                m["n_state_invariant_violations"] for m in day_metrics),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=999, help="default: all 11 val days")
    parser.add_argument("--max-decisions", type=int, default=200000)
    args = parser.parse_args()
    days = operating_days("val")[:args.days]
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    policies = [
        ("signaller_replay", lambda env: make_signaller_replay_policy(env)),
        ("greedy_set", lambda _env: make_greedy_set_policy()),
    ]
    started = time.time()
    results = [_run_policy(name, factory, days, net, params, dwell, ptc, adapter,
                           args.max_decisions) for name, factory in policies]
    report = {
        "audit": "D33 step 8 timing funnel, same-time panels, and WAIT contract",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.time() - started,
        "days": [str(d["start"])[:10] for d in days],
        "results": {r["policy"]: r for r in results},
    }
    out = config.ARTIFACTS_REPORTS / f"d33_step8_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in report["results"].items()},
                     indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
