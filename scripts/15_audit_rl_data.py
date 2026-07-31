#!/usr/bin/env python3
"""M12 数据面全面核查：把此前只在 1–3 个窗口上量过的结论，放到**全部拥堵窗口**上重验。

动机(2026-07-31 用户要求): 之前多处结论取样过小(窗口数 1–8), 且 RL 训练只用了 6 个窗口 = 18 小时
= 全量的 0.77%, 而 BC 用的是全 98 天。**取样过小的结论不能当依据。**

用法: python scripts/15_audit_rl_data.py --start 0 --end 30 --out /tmp/audit_a.json
      python scripts/15_audit_rl_data.py --start 30 --end 60 --out /tmp/audit_b.json
      python scripts/15_audit_rl_data.py --merge /tmp/audit_a.json /tmp/audit_b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import (build_episode_trains, congested_windows, episode_horizon,
                                      load_assets)
from derbysim.sim.reward import RewardedDynamicEnv, can_complete


def audit_window(w, net, params, dwell, ptc, ad):
    """一个窗口: 信号员回放 + 有路就办, 各自的决策密度/选项池/完成率/动作构成。"""
    out = {"window": str(w["start"]), "n_late": w["n_late"]}
    H = episode_horizon(w)
    for tag, mk in (("signaller", make_signaller_replay_policy),
                    ("greedy", lambda e: make_greedy_set_policy())):
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        pool, chosen, no_set, n = Counter(), Counter(), 0, 0
        pol_holder = {}

        def wrapped(enc):
            nonlocal no_set, n
            n += 1
            ks = Counter(s.kind for s in enc["slots"])
            for k, v in ks.items():
                pool[k] += v
            if not ks.get("set"):
                no_set += 1
            i, lp, v = pol_holder["p"](enc)
            chosen[enc["slots"][i].kind] += 1
            return i, lp, v

        trains = build_episode_trains(w, inject=False)
        env_trains = trains
        pol_holder["p"] = mk(env)
        m = run_episode(env, ad, wrapped, env_trains, horizon=H).metrics
        tot_pool = max(sum(pool.values()), 1)
        eng = env.engine
        out[tag] = {
            "n_decisions": n,
            "frac_decisions_without_set": no_set / max(n, 1),
            "pool_wait": pool["wait"] / tot_pool, "pool_set": pool["set"] / tot_pool,
            "pool_cancel": pool["cancel"] / tot_pool,
            "chosen_wait": chosen["wait"] / max(n, 1), "chosen_set": chosen["set"] / max(n, 1),
            "chosen_cancel": chosen["cancel"] / max(n, 1),
            "completion_rate": m["completion_rate"], "n_trains": m["n_trains"],
            "n_completable": m["n_completable"], "n_impossible": m["n_impossible"],
            "collisions": m["n_collisions"], "deadlocked": bool(m["deadlocked"]),
            "delay": m["weighted_secondary_delay_s"], "reward": m["episode_reward"],
            "dwell_p99_min": float(np.percentile(
                [max(t.sim_occ.values()) - t.entry_time for t in eng.trains.values() if t.sim_occ]
                or [0.0], 99) / 60.0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    args = ap.parse_args()

    if args.merge:
        rows = []
        for f in args.merge:
            rows += json.loads(Path(f).read_text())
        report(rows)
        return

    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    wins = congested_windows("all", top_k=999, pool=80)[args.start:args.end]
    rows = []
    for i, w in enumerate(wins):
        rows.append(audit_window(w, net, params, dwell, ptc, ad))
        print(f"  [{args.start + i}] {w['start']} ok", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
        print(f"wrote {args.out} ({len(rows)} 窗口)")
    else:
        report(rows)


def report(rows) -> None:
    n = len(rows)
    print(f"\n{'='*78}\n全部拥堵窗口核查 (n={n})\n{'='*78}")
    for tag, label in (("signaller", "信号员回放"), ("greedy", "有路就办")):
        g = lambda k: np.array([r[tag][k] for r in rows], dtype=float)
        print(f"\n【{label}】")
        for key, name, pct in (("n_decisions", "决策点数", False),
                               ("frac_decisions_without_set", "其中无 set 可选", True),
                               ("pool_cancel", "选项池 cancel 占比", True),
                               ("chosen_wait", "实际选 WAIT 占比", True),
                               ("completion_rate", "完成率", True)):
            v = g(key)
            f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.0f}")
            print(f"  {name:<20} 中位 {f(np.median(v)):>8} | p10 {f(np.percentile(v,10)):>8}"
                  f" | p90 {f(np.percentile(v,90)):>8}")
        print(f"  碰撞总数 {int(g('collisions').sum())} | 死锁窗口数 {int(g('deadlocked').sum())}"
              f" | 结构不可能完成的车 中位 {np.median(g('n_impossible')):.0f}"
              f"/{np.median(g('n_trains')):.0f}")
        print(f"  在区停留 p99 中位 {np.median(g('dwell_p99_min')):.1f} 分钟"
              f" (最大 {g('dwell_p99_min').max():.1f}) —— drain=15min 是否够")
    sc = np.array([r["signaller"]["completion_rate"] for r in rows])
    print(f"\n★ 信号员完成率 == 1.000 的窗口: {(sc >= 0.9999).sum()}/{n}"
          f"; 最低 {sc.min():.3f}")


if __name__ == "__main__":
    main()
