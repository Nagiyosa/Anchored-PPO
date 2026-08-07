#!/usr/bin/env python3
"""D33 Step 7: full 11-day signaller replay and determinism gate.

Runs every validation operating day twice on fresh train objects.  The gate
checks closed-loop completion, safety, D36 state invariants, exact replay of
the D35 reference chain, and fingerprints of actions/paths/timings/metrics.
Every invocation writes a timestamped JSON report; no result is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv

HISTORICAL_REPORT = (config.ARTIFACTS_REPORTS /
                     "ppo_deepsets_beta0.1_seed0_20260802-200725.json")


def _sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_day(day, net, params, dwell, ptc, adapter, max_decisions: int,
             proactive: bool = False, reserve_window: str = "narrow") -> dict:
    env = RewardedDynamicEnv(net, params, ptc, dwell,
                             proactive=proactive, reserve_window=reserve_window)
    result = run_episode(
        env, adapter, make_signaller_replay_policy(env), build_day_trains(day, inject=False),
        max_decisions=max_decisions, horizon=day_horizon(day))
    engine = env.engine

    actions = []
    action_counts = Counter()
    for tr in result.transitions:
        slot = tr.enc["slots"][tr.option_idx]
        row = [slot.kind, slot.train_id, slot.route_id, tr.dt, tr.reward]
        actions.append(row)
        action_counts[slot.kind] += 1

    paths = []
    mismatches = []
    for train_id, train in sorted(engine.trains.items()):
        reference = list(train.route_chain)
        actual = list(train.actual_chain)
        if actual != reference:
            mismatches.append({"train_id": train_id, "reference": reference, "actual": actual})
        paths.append({
            "train_id": train_id,
            "reference_chain": reference,
            "actual_chain": actual,
            "path": list(train.path),
            "path_route_ids": list(train.path_route_ids),
            "path_commit_indices": list(train.path_commit_indices),
            "sim_occ": sorted((str(tc), float(ts)) for tc, ts in train.sim_occ.items()),
            "pos": int(train.pos),
            "done": bool(train.done),
            "termination_reason": train.termination_reason,
            "reached_exit": bool(train.reached_exit),
            "reached_terminal": bool(train.reached_terminal),
        })

    metrics = dict(result.metrics)
    core_metrics = {k: metrics[k] for k in (
        "n_trains", "n_completable", "n_impossible", "n_reached_exit",
        "completion_rate", "deadlocked", "n_collisions",
        "n_state_invariant_violations", "weighted_secondary_delay_s",
        "n_decisions", "n_actions", "sim_end_s", "truncated", "n_steps",
        "episode_reward", "frac_wait", "frac_set", "frac_cancel")}
    return {
        "day": str(day["start"])[:10],
        "metrics": core_metrics,
        "n_done": sum(bool(t.done) for t in engine.trains.values()),
        "action_counts": dict(sorted(action_counts.items())),
        "n_reference_chain_mismatches": len(mismatches),
        "reference_chain_mismatch_examples": mismatches[:20],
        "action_sha256": _sha(actions),
        "path_sha256": _sha(paths),
        "metrics_sha256": _sha(core_metrics),
    }


def _summarise(rows: list[dict]) -> dict:
    metrics = [row["metrics"] for row in rows]
    able = sum(m["n_completable"] for m in metrics)
    reached = sum(m["n_reached_exit"] for m in metrics)
    actions = Counter()
    for row in rows:
        actions.update(row["action_counts"])
    n_action = sum(actions.values())
    return {
        "n_days": len(rows),
        "n_trains": sum(m["n_trains"] for m in metrics),
        "n_done": sum(row["n_done"] for row in rows),
        "n_completable": able,
        "n_reached": reached,
        "completion_rate": reached / able if able else 1.0,
        "weighted_secondary_delay_s": sum(m["weighted_secondary_delay_s"] for m in metrics),
        "episode_reward": sum(m["episode_reward"] for m in metrics),
        "n_collisions": sum(m["n_collisions"] for m in metrics),
        "n_deadlocks": sum(bool(m["deadlocked"]) for m in metrics),
        "n_truncated": sum(bool(m["truncated"]) for m in metrics),
        "n_state_invariant_violations": sum(
            m["n_state_invariant_violations"] for m in metrics),
        "n_reference_chain_mismatches": sum(
            row["n_reference_chain_mismatches"] for row in rows),
        "n_steps": sum(m["n_steps"] for m in metrics),
        "action_counts": dict(sorted(actions.items())),
        "action_mix": {k: v / n_action for k, v in sorted(actions.items())} if n_action else {},
        "every_day_completion_one": all(m["completion_rate"] == 1.0 for m in metrics),
    }


def _historical() -> dict | None:
    if not HISTORICAL_REPORT.exists():
        return None
    report = json.loads(HISTORICAL_REPORT.read_text(encoding="utf-8"))
    old = report.get("baselines", {}).get("signaller_replay")
    if old is None:
        return None
    return {"source": str(HISTORICAL_REPORT), **old}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=999,
                        help="validation days to run; default 999 means all 11")
    parser.add_argument("--passes", type=int, default=2,
                        help="fresh deterministic repetitions; Step 7 requires 2")
    parser.add_argument("--max-decisions", type=int, default=200000)
    parser.add_argument("--proactive", action="store_true",
                        help="D50/D57: 用提前触发的 proactive 引擎(默认 reactive)")
    parser.add_argument("--reserve-window", default="narrow", choices=["narrow", "wide"],
                        help="D57: 回放/BC 恒 narrow(复现真人时机); wide 仅 RL 消融, 不作人类锚")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    days = operating_days("val")[:args.days]
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    runs = []
    started = time.time()
    for pass_idx in range(args.passes):
        rows = []
        for day_idx, day in enumerate(days, 1):
            row = _run_day(day, net, params, dwell, ptc, adapter, args.max_decisions,
                           proactive=args.proactive, reserve_window=args.reserve_window)
            rows.append(row)
            m = row["metrics"]
            print(f"pass {pass_idx + 1}/{args.passes} day {day_idx:02d}/{len(days)} "
                  f"{row['day']} completion={m['completion_rate']:.3f} "
                  f"delay={m['weighted_secondary_delay_s']:.0f} "
                  f"inv={m['n_state_invariant_violations']} "
                  f"chain_miss={row['n_reference_chain_mismatches']}", flush=True)
        runs.append({"pass": pass_idx + 1, "days": rows, "summary": _summarise(rows)})

    deterministic = True
    differences = []
    if runs:
        base = runs[0]["days"]
        for other in runs[1:]:
            for a, b in zip(base, other["days"]):
                for field in ("action_sha256", "path_sha256", "metrics_sha256"):
                    if a[field] != b[field]:
                        deterministic = False
                        differences.append({"day": a["day"], "pass": other["pass"],
                                            "field": field, "first": a[field], "other": b[field]})

    summary = runs[0]["summary"] if runs else {}
    full_protocol = len(days) == 11 and args.passes >= 2
    gate_pass = bool(full_protocol and deterministic and summary
                     and summary["every_day_completion_one"]
                     and summary["n_collisions"] == 0
                     and summary["n_deadlocks"] == 0
                     and summary["n_truncated"] == 0
                     and summary["n_state_invariant_violations"] == 0
                     and summary["n_reference_chain_mismatches"] == 0
                     and summary["n_done"] == summary["n_trains"])

    # ⚠️ proactive 门**不**与旧对照比: HISTORICAL_REPORT 是旧 reactive 引擎数据(D48 作废),
    # 拿它做 delta 会"指向错误内容"。proactive 只自证不变式(确定/完成/D36/参考链/指纹)。
    historical = None if args.proactive else _historical()
    delta = None
    if historical and summary:
        delta = {
            "completion_rate": summary["completion_rate"] - historical["eval_completion_rate"],
            "weighted_secondary_delay_s": (summary["weighted_secondary_delay_s"]
                                             - historical["eval_weighted_secondary_delay_s"]),
            "episode_reward": summary["episode_reward"] - historical["eval_episode_reward"],
            "n_completable": summary["n_completable"] - historical["eval_n_completable"],
            "n_steps": summary["n_steps"] - historical["eval_n_steps"],
        }

    report = {
        "audit": "D33 step 7 full validation replay and determinism gate",
        "engine_mode": ("proactive" if args.proactive else "reactive"),
        "reserve_window": args.reserve_window,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_git_head": _git_head(),
        "elapsed_s": time.time() - started,
        "args": vars(args) | {"out": str(args.out) if args.out else None},
        "days": [str(d["start"])[:10] for d in days],
        "full_protocol": full_protocol,
        "deterministic": deterministic,
        "determinism_differences": differences,
        "gate_pass": gate_pass,
        "summary": summary,
        "historical_pre_d33": historical,
        "delta_vs_historical": delta,
        "runs": runs,
    }
    _tag = "proactive" if args.proactive else "reactive"
    out = args.out or (config.ARTIFACTS_REPORTS /
                       f"d33_step7_{_tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"gate_pass": gate_pass, "deterministic": deterministic,
                      "summary": summary, "delta_vs_historical": delta},
                     indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
