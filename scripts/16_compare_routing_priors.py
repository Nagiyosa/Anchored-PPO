#!/usr/bin/env python3
"""闭环对照：**选路先验的强弱**如何决定闭环质量（M11 决定 A 的重新裁决，2026-07-31）。

**动机**: 实测闭环质量几乎是**选路准确率**的函数 —— 同一固定评估集上
  信号员(选路 100%) 1.000 / 636s ｜ 每信号最频进路(macro≈0.80) 0.883 / 67k
  ｜ RL 峰值(热启动自 no-prior BC, macro≈0.66) 0.791 / 110k ｜ 有路就办(随便挑) 0.737 / 144k
即**一张零学习的查表打败了训了 336k 步的 PPO**。而我们之所以热启动自 no-prior BC(0.663)
而非含先验 BC(0.857), 是 M11 §12 决定 A —— 理由是"含先验版长尾低于随机, 会压制我们要研究的偏离"。
**那个决定是在 BC 指标上做的, 当时无法测其闭环后果; 现在能测了, 应当拿数据重新裁决。**

本脚本在**同一固定评估集**上闭环比较各"选路先验", 时机规则统一为"有得办就办"(与查表基线一致):
  signaller(上界) · lookup(每信号最频, 无学习) · bc_noprior · bc_prior · random(下界)
输出可直接支撑"用哪个先验热启动"的决定, 以及 M12d 基线阶梯(**查表必须进阶梯** —— 打不过它的方法没价值)。

⚠️ 已被 scripts/24 取代(2026-08-05, F2/F3): 本脚本跑在旧 3h 拥堵窗口(D27 前)上。修正引擎(D41)+
   运营日(D27)+ D35 的闭环对照请用 `scripts/24_routing_prior_closed_loop.py`。此处仅保留历史;
   如仍要跑, 数据集已改 D35, checkpoint 现为 `bc_<enc>[_noprior]_nw_d35.pt`(M11 §18: 必带 --no-route-weight)。
用法: python scripts/16_compare_routing_priors.py --bc-noprior artifacts/reports/bc_deepsets_noprior_nw_d35.pt \
          --bc-prior artifacts/reports/bc_deepsets_nw_d35.pt --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer, fit_route_prior
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (make_greedy_set_policy, make_random_policy,
                                    make_signaller_replay_policy, run_episode)
from derbysim.learn.scenarios import (build_episode_trains, congested_windows, episode_horizon,
                                      load_assets)
from derbysim.sim.reward import RewardedDynamicEnv


def lookup_table(split: str = "train") -> dict:
    """每信号最频进路 —— 只用 BC **训练**段拟合(无泄露)。M11 实测该基线 macro≈0.80。"""
    bc = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all_d35.parquet")
    tr = bc[bc["split"] == split]
    cnt: dict = {}
    for sig, rid in zip(tr["signal"], tr["chosen_route"]):
        cnt.setdefault(sig, Counter())[rid] += 1
    return {s: c.most_common(1)[0][0] for s, c in cnt.items()}


def make_lookup_factory(top: dict):
    def factory(env):
        def act(enc):
            first = None
            for i, s in enumerate(enc["slots"]):
                if s.kind != "set":
                    continue
                if first is None:
                    first = i
                t = env.engine.trains.get(s.train_id)
                sig = env.engine._decision_signal(t) if t is not None else None
                if sig is not None and top.get(sig) == s.route_id:
                    return i, 0.0, 0.0
            return (first, 0.0, 0.0) if first is not None else (0, 0.0, 0.0)
        return act
    return factory


def make_bc_factory(model, adapter, device):
    """用 BC 网络选路, 时机=有得办就办(与查表基线同规则, 保证公平)。"""
    import torch

    def factory(_env):
        def act(enc):
            rows = enc["train_rows"]
            if not rows:
                return 0, 0.0, 0.0
            with torch.no_grad():
                logits = model(torch.as_tensor(enc["x_dec"]).float().to(device),
                               torch.as_tensor(enc["x_conf"]).float().to(device),
                               torch.as_tensor(enc["conf_mask"]).float().to(device),
                               torch.as_tensor(enc["x_cand"]).float().to(device),
                               torch.as_tensor(enc["cand_mask"]).float().to(device))
                pick = logits.argmax(dim=-1).to("cpu").numpy()
            first = None
            for i, s in enumerate(enc["slots"]):
                if s.kind != "set":
                    continue
                if first is None:
                    first = i
                if s.col == int(pick[s.row]):
                    return i, 0.0, 0.0
            return (first, 0.0, 0.0) if first is not None else (0, 0.0, 0.0)
        return act
    return factory


def load_bc(ckpt, adapter, device):
    import torch
    from derbysim.learn.models import PointerBC
    d = torch.load(ckpt, map_location=device, weights_only=False)
    st, enc = (d["model"], d.get("encoder", "deepsets")) if "model" in d else (d, "deepsets")
    hid = d.get("hidden", 64) if isinstance(d, dict) else 64
    m = PointerBC(enc, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim, hidden=hid)
    m.load_state_dict(st)
    m.to(device).eval()
    return m, d.get("use_prior") if isinstance(d, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--bc-noprior", default=None)
    ap.add_argument("--bc-prior", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, params, dwell, ptc = load_assets()
    fz_noprior = Featurizer(net, ptc)
    ad = PanelAdapter(net, fz_noprior)
    wins = congested_windows(args.split, top_k=99)
    print(f"固定评估集: {len(wins)} 个 {args.split} 窗口 × {args.seeds} 种子")

    cands = [("signaller_replay(选路 100%)", make_signaller_replay_policy, None),
             ("lookup_most_frequent(≈0.80)", make_lookup_factory(lookup_table()), None),
             ("greedy_first_legal(随便挑)", lambda e: make_greedy_set_policy(), None),
             ("random_legal(下界)", lambda e: make_random_policy(np.random.default_rng(0)), None)]

    if args.bc_noprior or args.bc_prior:
        from derbysim.learn.panel_models import pick_device
        dev = pick_device()
        for tag, ck in (("bc_noprior", args.bc_noprior), ("bc_prior", args.bc_prior)):
            if not ck:
                continue
            if not Path(ck).exists():
                print(f"  [skip] {tag}: 找不到 {ck}"
                      + ("  → 先跑 `python scripts/12_train_bc.py deepsets --epochs 30`"
                         " 生成含先验版 bc_deepsets.pt" if tag == "bc_prior" else ""))
                continue
            # 含先验版的候选特征需要 live prior → 用**训练段**拟合的 prior 建专用特征器/适配层
            if tag == "bc_prior":
                bcdf = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all_d35.parquet")
                trdf = bcdf[bcdf["split"] == "train"]
                prior = fit_route_prior([{"signal": r["signal"],
                                          "legal_route_ids": r["legal_route_ids"],
                                          "chosen_route": r["chosen_route"],
                                          "n_legal": r["n_legal"]} for r in trdf.to_dict("records")])
                ad_p = PanelAdapter(net, Featurizer(net, ptc, route_prior=prior),
                                    require_no_prior=False)
            else:
                ad_p = ad
            m, used_prior = load_bc(ck, ad_p, dev)
            print(f"  [{tag}] {ck} (use_prior={used_prior})")
            cands.append((f"{tag}", make_bc_factory(m, ad_p, dev), ad_p))

    print(f"\n{'策略':<32}{'完成率':>9}{'加权次生延误':>14}{'回报':>11}{'死锁':>6}")
    rows = {}
    for tag, fac, ad_use in cands:
        adapter = ad_use or ad
        c = d = r = 0.0; dl = 0; n = 0
        for w in wins:
            for sd in range(args.seeds):
                env = RewardedDynamicEnv(net, params, ptc, dwell)
                m = run_episode(env, adapter, fac(env), build_episode_trains(w, seed=sd),
                                horizon=episode_horizon(w)).metrics
                c += m["completion_rate"]; d += m["weighted_secondary_delay_s"]
                r += m["episode_reward"]; dl += int(m["deadlocked"]); n += 1
        rows[tag] = {"completion": c / n, "delay": d / n, "reward": r / n, "deadlocks": dl}
        print(f"{tag:<32}{c/n:>9.3f}{d/n:>14.0f}{r/n:>11.1f}{dl:>6}")
    out = Path(args.out) if args.out else (config.ARTIFACTS_REPORTS /
                                           f"routing_prior_compare_{args.split}.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
