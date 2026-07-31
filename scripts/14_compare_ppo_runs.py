#!/usr/bin/env python3
"""汇总 artifacts/reports 下**所有** PPO 训练轮次, 便于跨版本对照(每一版都留档)。

用法: python scripts/14_compare_ppo_runs.py [--full]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="逐轮打印每次 EVAL")
    args = ap.parse_args()
    runs = sorted(config.ARTIFACTS_REPORTS.glob("ppo_*.json"))
    if not runs:
        raise SystemExit("没有找到 ppo_*.json")
    print(f"{'run':<46}{'步数':>8}{'λ':>5}{'预热':>5}{'β':>6}"
          f"{'起点完成率':>11}{'最佳完成率':>11}{'最佳回报':>11}{'最佳@it':>8}")
    for f in runs:
        d = json.loads(f.read_text(encoding="utf-8"))
        a, ev = d.get("args", {}), d.get("evals", [])
        if not ev:
            continue
        start, best = ev[0], max(ev, key=lambda e: e["eval_episode_reward"])
        print(f"{f.stem:<46}{d.get('total_steps', 0):>8}{a.get('lam', '-'):>5}"
              f"{a.get('critic_warmup_iters', 0):>5}{a.get('kl_beta', '-'):>6}"
              f"{start['eval_completion_rate']:>11.3f}{best['eval_completion_rate']:>11.3f}"
              f"{best['eval_episode_reward']:>11.1f}{best['iter']:>8}")
        if args.full:
            for e in ev:
                print(f"      it {e['iter']:>3} completion {e['eval_completion_rate']:.3f} "
                      f"delay {e['eval_weighted_secondary_delay_s']:>9.0f} "
                      f"reward {e['eval_episode_reward']:>9.1f}")
        b = d.get("baselines", {})
        if b:
            print("      基线: " + " | ".join(
                f"{k} {v['eval_episode_reward']:.1f}" for k, v in b.items()))


if __name__ == "__main__":
    main()
