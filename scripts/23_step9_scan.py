#!/usr/bin/env python3
"""D33 Step 9 单点扰动扫描（① reroute 延误级联 / ② wait 时机轴），可断点重续 + 日分片并行。

重定向口径（D41 连带发现）：修正后 in-scope 环境完成率近乎饱和（greedy 0.99），质量差异在
**加权次生延误**。故 ① 主看"换一条 in-scope goal-safe 替代路后, 次生延误相对基线涨多少 / 是否
滚雪球"（完成率作次要检查）; ② 主看"某个 set 点强制 WAIT 后延误怎么摆动"。

覆盖 train+val（D40, test 不碰）。每个可注入点单独跑一整天; 逐点把结果 append 进分片 JSONL,
重启自动跳过已完成的 (day,k) → 断点重续。日按 `--shard i/n` 取模分片, 6 核可并发。
用 `--summarize` 读回全部分片 JSONL 汇总成一个带时间戳 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.learn.step9 import census_signaller, run_injection
from derbysim.sim.reward import RewardedDynamicEnv


def _days(split: str):
    if split == "trainval":
        return list(operating_days("train")) + list(operating_days("val"))
    return list(operating_days(split))


def _load_done(path: Path) -> set:
    """断点重续: 已写入的 (day,k) 集合。"""
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["day"], r["k"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _fmt_eta(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def run_scan(args):
    mode = args.mode                       # 'reroute' (①) | 'wait' (②)
    days = _days(args.split)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        days = [d for idx, d in enumerate(days) if idx % n == i]
    if args.limit_days:
        days = days[:args.limit_days]

    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))

    out = Path(args.out) if args.out else (
        config.ARTIFACTS_REPORTS / f"d33_step9_{mode}_scan"
        f"{('_shard' + args.shard.replace('/', 'of')) if args.shard else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    print(f"mode={mode} split={args.split} days={len(days)} out={out.name} "
          f"resume_skip={len(done)}", flush=True)

    started = time.time()
    n_points_total = 0
    n_done_run = 0
    with out.open("a", encoding="utf-8") as fh:
        for di, day in enumerate(days, 1):
            dstr = str(day["start"])[:10]
            env = RewardedDynamicEnv(net, params, ptc, dwell)
            census = census_signaller(env, adapter, build_day_trains(day, inject=False),
                                      day_horizon(day))
            base = census["baseline_metrics"]
            base_delay = base["weighted_secondary_delay_s"]
            base_reached, base_able = base["n_reached_exit"], base["n_completable"]
            points = census["branchable_idx"] if mode == "reroute" else census["set_idx"]
            if args.limit_points:
                points = points[:args.limit_points]
            n_points_total += len(points)
            todo = [k for k in points if (dstr, k) not in done]
            for k in todo:
                env2 = RewardedDynamicEnv(net, params, ptc, dwell)
                res = run_injection(env2, adapter, build_day_trains(day, inject=False),
                                    day_horizon(day), inject_at=k, mode=mode)
                m = res["metrics"]
                row = {
                    "day": dstr, "k": int(k), "mode": mode,
                    "injected": res["injected"],
                    "info": res["info"],
                    "deviated": res["deviated"],
                    "delay_s": m["weighted_secondary_delay_s"],
                    "base_delay_s": base_delay,
                    "delta_delay_s": m["weighted_secondary_delay_s"] - base_delay,
                    "completion": m["completion_rate"],
                    "n_reached": m["n_reached_exit"], "n_completable": m["n_completable"],
                    "base_n_reached": base_reached, "base_n_completable": base_able,
                    "n_collisions": m["n_collisions"], "n_deadlocks": int(m["deadlocked"]),
                }
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                n_done_run += 1
            elapsed = time.time() - started
            rate = n_done_run / elapsed if n_done_run else 0.0
            eta = (elapsed / di) * (len(days) - di)      # 按天数估算剩余时间
            print(f"day {di:02d}/{len(days)} {dstr} points={len(points)} "
                  f"ran={len(todo)} (skip {len(points)-len(todo)}) "
                  f"base_delay={base_delay:.0f} | done_this_run={n_done_run} "
                  f"rate={rate:.2f}/s elapsed={_fmt_eta(elapsed)} eta={_fmt_eta(eta)}",
                  flush=True)
    print(f"DONE mode={mode} shard={args.shard or 'all'} rows_written={n_done_run} "
          f"points_seen={n_points_total} file={out}", flush=True)


def summarize(args):
    files = sorted(Path(config.ARTIFACTS_REPORTS).glob(args.summarize))
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no rows found for pattern", args.summarize)
        return
    deltas = [r["delta_delay_s"] for r in rows]
    worsened = [d for d in deltas if d > 0]
    improved = [d for d in deltas if d < 0]
    comp_drop = [r for r in rows if r["n_reached"] < r["base_n_reached"]]
    deltas_sorted = sorted(deltas)

    def pct(q):
        if not deltas_sorted:
            return None
        pos = (len(deltas_sorted) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(deltas_sorted) - 1)
        return deltas_sorted[lo] * (1 - (pos - lo)) + deltas_sorted[hi] * (pos - lo)

    mode = rows[0]["mode"]
    summary = {
        "audit": f"D33 step 9 {mode} scan summary",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode, "n_files": len(files),
        "n_points": len(rows),
        "delta_delay_s": {
            "mean": sum(deltas) / len(deltas),
            "median": pct(0.5), "p05": pct(0.05), "p95": pct(0.95),
            "min": min(deltas), "max": max(deltas),
            "n_worsened": len(worsened), "n_improved": len(improved),
            "n_zero": len(deltas) - len(worsened) - len(improved),
            "frac_worsened": len(worsened) / len(deltas),
            "total_worsening_s": sum(worsened),
        },
        "completion": {
            "n_with_completion_drop": len(comp_drop),
            "frac_with_completion_drop": len(comp_drop) / len(rows),
        },
        "safety": {
            "n_collisions": sum(r["n_collisions"] for r in rows),
            "n_deadlocks": sum(r["n_deadlocks"] for r in rows),
        },
    }
    outp = config.ARTIFACTS_REPORTS / f"d33_step9_{mode}_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {outp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reroute", "wait"], default="reroute",
                   help="reroute=① 延误级联; wait=② 时机轴")
    p.add_argument("--split", default="trainval")
    p.add_argument("--shard", help="日分片 i/n, 如 0/6 (6 核并发跑 0/6..5/6)")
    p.add_argument("--limit-days", type=int, default=0, help="只跑前 N 天(冒烟)")
    p.add_argument("--limit-points", type=int, default=0, help="每天只跑前 N 个点(冒烟/抽样)")
    p.add_argument("--out", help="JSONL 输出路径(默认按 mode+shard 命名, 支持断点重续)")
    p.add_argument("--summarize", help="读回匹配的分片 JSONL 汇总, 如 'd33_step9_reroute_scan*.jsonl'")
    args = p.parse_args()
    if args.summarize:
        summarize(args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
