#!/usr/bin/env python3
"""闭环对照(修正引擎 + 运营日 + D35): 选路先验够不够 —— 典型路查表 vs 信号员。

动机(Step 9 讨论): ① 显示"偏离到非典型路"才会级联(常走路灾难率仅 3%), 而走典型路正是 BC /
最频进路查表的活。本脚本在**修正后的 in-scope 引擎**(D41)、**运营日**(D27)、**D35 train 段**上
闭环比较:
  signaller_replay(选路 100%, 上界) · lookup_most_frequent(每信号最频, BC 代理, 纯典型路)
  · greedy_first_legal(任意合法, 无先验) · random_legal(下界)
所有非信号员策略统一"有得办就办"(greedy 时机), 以**隔离选路质量**。

要回答两问:①纯典型路(lookup)完成率是不是也≈信号员? ②它与信号员的差距是否几乎全在**延误**?
若是, 坐实"选路由先验解决、剩下是时机", Step 10 的 PPO(KL-to-BC + 学时机)专注压延误。

用法: python scripts/24_routing_prior_closed_loop.py --split val   (或 trainval)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer, fit_route_prior
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (evaluate_policy, make_greedy_set_policy,
                                    make_random_policy, make_signaller_replay_policy)
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.sim.reward import RewardedDynamicEnv


def lookup_table(split_value: str = "train") -> dict:
    """每信号最频进路 —— 只用 D35 数据集的 train 段拟合(无泄露, holdout=test 不碰)。"""
    bc = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all_d35.parquet")
    tr = bc[bc["split"] == split_value]
    cnt: dict = {}
    for sig, rid in zip(tr["signal"], tr["chosen_route"]):
        cnt.setdefault(sig, Counter())[rid] += 1
    return {s: c.most_common(1)[0][0] for s, c in cnt.items()}


def make_lookup_factory(top: dict):
    """典型路: 若该信号的最频进路在当前选项里就办它, 否则退回第一条合法(greedy)。"""
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


def make_bc_factory(model, device):
    """用 BC 网络选路, 时机=有得办就办(与查表基线同规则, 保证只比选路)。(移植自 scripts/16)"""
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
    return m


def _prior_adapter(net, ptc):
    """含先验 BC 需要 live route_prior 特征 → 用 D35 train 段拟合的 prior 建专用适配层。"""
    bc = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all_d35.parquet")
    tr = bc[bc["split"] == "train"]
    prior = fit_route_prior([{"signal": r["signal"], "legal_route_ids": r["legal_route_ids"],
                              "chosen_route": r["chosen_route"], "n_legal": r["n_legal"]}
                             for r in tr.to_dict("records")])
    return PanelAdapter(net, Featurizer(net, ptc, route_prior=prior), require_no_prior=False)


def _days(split: str):
    if split == "trainval":
        return list(operating_days("train")) + list(operating_days("val"))
    return list(operating_days(split))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", help="val(默认) | trainval | train")
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--bc-noprior", default=None, help="no-prior BC 权重 .pt(如 artifacts/reports/bc_deepsets_noprior.pt)")
    ap.add_argument("--bc-prior", default=None, help="含先验 BC 权重 .pt(如 artifacts/reports/bc_deepsets.pt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    days = _days(args.split)
    if args.limit_days:
        days = days[:args.limit_days]
    episodes = [(str(d["start"])[:10],
                 (lambda d=d: build_day_trains(d, inject=False)),
                 day_horizon(d)) for d in days]
    top = lookup_table()
    print(f"评估集: {len(days)} 个 {args.split} 运营日 | lookup 覆盖 {len(top)} 个决策信号", flush=True)

    def env_factory():
        return RewardedDynamicEnv(net, params, ptc, dwell)

    # (标签, 策略工厂, 该策略专用适配层 or None=默认)
    policies = [
        ("signaller_replay(选路100%,上界)", make_signaller_replay_policy, None),
        ("lookup_most_frequent(BC代理,典型路)", make_lookup_factory(top), None),
        ("greedy_first_legal(无先验)", lambda _e: make_greedy_set_policy(), None),
        ("random_legal(下界)", lambda _e: make_random_policy(np.random.default_rng(0)), None),
    ]
    if args.bc_noprior or args.bc_prior:
        from derbysim.learn.panel_models import pick_device
        dev = pick_device()
        for tag, ck, prior in (("bc_noprior", args.bc_noprior, False),
                               ("bc_prior", args.bc_prior, True)):
            if not ck:
                continue
            if not Path(ck).exists():
                print(f"  [skip] {tag}: 找不到 {ck}(先跑 scripts/12 生成)", flush=True)
                continue
            ad_bc = _prior_adapter(net, ptc) if prior else adapter
            model = load_bc(ck, ad_bc, dev)
            print(f"  [{tag}] loaded {ck} (device={dev})", flush=True)
            policies.append((f"{tag}({'含先验' if prior else 'no-prior'})",
                             make_bc_factory(model, dev), ad_bc))

    print(f"\n{'策略':<40}{'完成率(微)':>11}{'加权次生延误':>15}{'回报':>12}{'死锁':>6}{'不变式':>7}", flush=True)
    rows = {}
    for tag, fac, ad_use in policies:
        r = evaluate_policy(env_factory, ad_use or adapter, fac, episodes, max_decisions=200000)
        rows[tag] = {"completion": r["eval_completion_rate"],
                     "delay_s": r["eval_weighted_secondary_delay_s"],
                     "reward": r["eval_episode_reward"],
                     "n_reached": r["eval_n_reached_exit"], "n_completable": r["eval_n_completable"],
                     "deadlocks": r["eval_deadlocks"],
                     "invariants": r["eval_n_state_invariant_violations"],
                     "frac_wait": r["eval_frac_wait"], "frac_set": r["eval_frac_set"]}
        print(f"{tag:<40}{r['eval_completion_rate']:>11.4f}"
              f"{r['eval_weighted_secondary_delay_s']:>15.0f}{r['eval_episode_reward']:>12.0f}"
              f"{r['eval_deadlocks']:>6}{int(r['eval_n_state_invariant_violations']):>7}", flush=True)

    out = Path(args.out) if args.out else (
        config.ARTIFACTS_REPORTS / f"routing_prior_closedloop_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    report = {"audit": "routing-prior closed-loop (corrected engine, operating days, D35)",
              "split": args.split, "n_days": len(days),
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "lookup_signals": len(top), "results": rows}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
