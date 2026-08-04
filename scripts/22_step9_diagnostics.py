#!/usr/bin/env python3
"""D33 Step 9 诊断: 分叉普查(定 ①② 扫描规模) + ③ 卡住车再提供审计。

覆盖口径(D40): **train+val = 76 天, test 不碰**。本脚本先只跑**普查**(每天一遍无扰动信号员
回放 + 一遍 greedy 漏斗), 给出:
  - ① 单点偏离可注入点数(信号员 set 且该车 ≥2 goal-safe 选项);
  - ② WAIT 注入候选点数(所有 set panel);
  - ③ greedy 无-set 决策点的成因分层(几何/目标/死锁前瞻)。
拿到规模后再决定 ①② 全量扫描是"每点一条确定性替代"还是全枚举。每次运行写时间戳 JSON, 不覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.learn.step9 import census_signaller, census_stuck
from derbysim.sim.reward import RewardedDynamicEnv


def _days(split: str, limit: int):
    if split == "trainval":
        days = list(operating_days("train")) + list(operating_days("val"))
    else:
        days = list(operating_days(split))
    return days[:limit] if limit else days


def _merge_hist(dst: dict, src: dict):
    for k, v in src.items():
        dst[int(k)] = dst.get(int(k), 0) + v


def run_census(days, net, params, dwell, ptc, adapter, max_decisions):
    sig_tot = {"n_panels": 0, "n_set": 0, "n_wait": 0, "n_branchable": 0, "n_set_panels": 0}
    setopt_hist: dict = {}
    stuck_tot = {"awaiting_rows": 0, "no_geometry": 0, "goal_filtered": 0,
                 "deadlock_filtered": 0, "has_set": 0, "n_panels": 0}
    per_day = []
    base_completion = 0.0
    n_completable = 0
    n_reached = 0
    for day_idx, day in enumerate(days, 1):
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        sig = census_signaller(env, adapter, build_day_trains(day, inject=False),
                               day_horizon(day), max_decisions=max_decisions)
        env2 = RewardedDynamicEnv(net, params, ptc, dwell)
        stuck = census_stuck(env2, adapter, build_day_trains(day, inject=False),
                             day_horizon(day), lambda _e: make_greedy_set_policy(),
                             max_decisions=max_decisions)
        for k in sig_tot:
            sig_tot[k] += sig[k]
        _merge_hist(setopt_hist, sig["setopt_hist"])
        for k in stuck_tot:
            stuck_tot[k] += stuck[k]
        bm = sig["baseline_metrics"]
        n_completable += bm["n_completable"]
        n_reached += bm["n_reached_exit"]
        per_day.append({
            "day": str(day["start"])[:10],
            "signaller": {k: sig[k] for k in
                          ("n_panels", "n_set", "n_wait", "n_branchable", "n_set_panels")},
            "baseline_completion": bm["completion_rate"],
            "baseline_delay_s": bm["weighted_secondary_delay_s"],
            "greedy_stuck": stuck,
        })
        print(f"day {day_idx:02d}/{len(days)} {str(day['start'])[:10]} "
              f"branchable={sig['n_branchable']:>4} set={sig['n_set']:>4} "
              f"greedy_no_set={stuck['no_geometry']+stuck['goal_filtered']+stuck['deadlock_filtered']:>4}"
              f"/{stuck['awaiting_rows']:>5}", flush=True)
    base_completion = (n_reached / n_completable) if n_completable else 1.0
    return {
        "signaller_totals": sig_tot,
        "setopt_hist": dict(sorted(setopt_hist.items())),
        "baseline_completion_pooled": base_completion,
        "n_completable": n_completable,
        "n_reached": n_reached,
        "greedy_stuck_totals": stuck_tot,
        "per_day": per_day,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["census"], default="census")
    parser.add_argument("--split", default="trainval",
                        help="trainval(默认, 76 天) | train | val")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 天(冒烟用); 0=全部")
    parser.add_argument("--max-decisions", type=int, default=200000)
    args = parser.parse_args()

    days = _days(args.split, args.limit)
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    started = time.time()
    result = run_census(days, net, params, dwell, ptc, adapter, args.max_decisions)

    st = result["greedy_stuck_totals"]
    no_set = st["no_geometry"] + st["goal_filtered"] + st["deadlock_filtered"]
    report = {
        "audit": "D33 step 9 census: branching (exp1/2 sizing) + greedy stuck funnel (exp3)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": args.split,
        "n_days": len(days),
        "days": [str(d["start"])[:10] for d in days],
        "elapsed_s": time.time() - started,
        **result,
        "headline": {
            "exp1_injection_points": result["signaller_totals"]["n_branchable"],
            "exp2_wait_candidates": result["signaller_totals"]["n_set_panels"],
            "exp3_greedy_no_set_rows": no_set,
            "exp3_greedy_awaiting_rows": st["awaiting_rows"],
            "exp3_no_set_frac": (no_set / st["awaiting_rows"]) if st["awaiting_rows"] else 0.0,
        },
    }
    out = config.ARTIFACTS_REPORTS / f"d33_step9_census_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"split": args.split, "n_days": len(days),
                      "headline": report["headline"],
                      "setopt_hist": report["setopt_hist"],
                      "baseline_completion_pooled": report["baseline_completion_pooled"]},
                     indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
