"""D64: 生成离线转移数据集(供 IQL / AWAC)。

跑**规则策略**(signaller 专家 + lookup/greedy/fcfs/fsfs 坏状态样本)过 **proactive-wide** 环境,
逐决策记 `(enc, option_idx, reward, done)`,按 episode 分组存 pickle。**有界奖励**(D64,RewardConfig 默认)。
**torch-free**(规则策略),故可本地跑。

诚实局限:信号员全程完美(无坏状态)、规则基线进坏状态就 gridlock(不恢复)→ 数据里**没有"从坏状态
恢复"的样本**;IQL 只能学"避免进坏状态"(≈好放行),恢复留给 AWAC 的在线相(D64)。

用法: PYTHONPATH=. python scripts/33_build_offline_dataset.py [split] [n_days]
产出: artifacts/dataset/offline_transitions_<split>.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.baselines import (lookup_table, make_fcfs_policy, make_fsfs_policy,
                                      make_lookup_policy)
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (make_greedy_set_policy, make_signaller_replay_policy,
                                    run_episode)
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv


def main(argv) -> int:
    split = argv[1] if len(argv) > 1 else "train"
    n_days = int(argv[2]) if len(argv) > 2 else 24
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    top = lookup_table("train")
    days = operating_days(split)[:n_days]
    policies = {                                          # 都是 factory(env)->act
        "signaller": make_signaller_replay_policy,        # 专家(正样本/避免样本)
        "lookup": make_lookup_policy(top),                # 坏状态样本(gridlock)
        "greedy": lambda e: make_greedy_set_policy(),
        "fcfs": make_fcfs_policy(top),
        "fsfs": make_fsfs_policy(top),
    }
    episodes, t0 = [], time.time()
    for name, fac in policies.items():
        for w in days:
            env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
            res = run_episode(env, ad, fac(env), build_day_trains(w, inject=False),
                              max_decisions=300000, horizon=day_horizon(w))
            ep = [{"enc": t.enc, "opt_idx": int(t.option_idx),
                   "reward": float(t.reward), "done": bool(t.done)} for t in res.transitions]
            if ep:
                episodes.append({"policy": name, "day": str(w["start"])[:10], "steps": ep})
    n_trans = sum(len(e["steps"]) for e in episodes)
    out = config.ARTIFACTS / "dataset" / f"offline_transitions_{split}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"policies {list(policies)} | days {len(days)} | episodes {len(episodes)} | "
          f"transitions {n_trans} | {time.time()-t0:.0f}s -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
