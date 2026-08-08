#!/usr/bin/env python3
"""D64: 训练 IQL(离线 RL)——从 scripts/33 的离线转移数据学策略。

**评估口径与 PPO 完全一致**(同一 `evaluate_policy` + 贪心 driver, 同一 val 集 + 参照线)→ 直接可比。
actor 复用 PanelOptionPolicy;可选 `--bc-ckpt` 热启动 actor(整网, 同 scripts/13)。
断点续训: `--run-tag` + `--resume`(存 model/opt/iter/best)。

用法(服务器):
    python scripts/34_train_iql.py --dataset artifacts/dataset/offline_transitions_train.pkl \
        --encoder gnn --bc-ckpt artifacts/dataset/bc_fullpanel_gnn.pt \
        --iters 2000 --batch 128 --eval-every 100 --run-tag iql_gnn_s0 --resume
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.iql import IQL, iql_update
from derbysim.learn.panel import PanelAdapter, collate_panels
from derbysim.learn.panel_models import pick_device, to_tensors
from derbysim.learn.rollout import (as_factory, evaluate_policy, make_greedy_set_policy,
                                    make_random_policy, signaller_replay_factory)
from derbysim.learn.baselines import lookup_table, make_lookup_policy
from derbysim.learn.ppo import TorchPanelDriver
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="scripts/33 的离线转移 pickle")
    ap.add_argument("--encoder", default="gnn")
    ap.add_argument("--bc-ckpt", default=None, help="(可选)热启动 actor: full-panel BC {state_dict,encoder,dims}")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-days", type=int, default=999)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99, help="IQL 折扣(离线 RL 常用 0.99, 稳)")
    ap.add_argument("--tau", type=float, default=0.7, help="expectile(τ→1 越像 max)")
    ap.add_argument("--beta", type=float, default=3.0, help="AWR 温度(越大越像纯模仿)")
    ap.add_argument("--w-max", type=float, default=100.0, help="AWR 权重上限")
    ap.add_argument("--reserve-window", default="wide", choices=["wide", "narrow"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = pick_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))

    # ---- 离线数据 → 扁平转移 (enc, next_enc, action, reward, done) ----
    # --dataset 可以是**分片目录**(scripts/33 产出的 offline_shards_*, 每分片一局)或单个 pkl(episodes 列表)。
    dp = Path(args.dataset)
    if dp.is_dir():
        episodes = []
        for shard in sorted(dp.glob("*.pkl")):
            with shard.open("rb") as f:
                episodes.append(pickle.load(f))
    else:
        with dp.open("rb") as f:
            episodes = pickle.load(f)
    trans = []
    for ep in episodes:
        steps = ep["steps"]
        for i, s in enumerate(steps):
            nxt = steps[i + 1]["enc"] if (i + 1 < len(steps) and not s["done"]) else s["enc"]  # done 时 s' 被掩掉
            trans.append((s["enc"], nxt, s["opt_idx"], s["reward"], float(s["done"])))
    print(f"[data] {args.dataset}: episodes {len(episodes)} | transitions {len(trans)}")

    model = IQL(args.encoder, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                adapter.cancel_dim, adapter.train_dim, adapter.global_dim, hidden=args.hidden).to(dev)
    if args.bc_ckpt:
        ck = torch.load(args.bc_ckpt, map_location=dev, weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            model.actor.load_state_dict(ck["state_dict"])
            print(f"[bc] actor 热启动自 {args.bc_ckpt}")
    opt_critic = torch.optim.Adam(model.critic_parameters(), lr=args.lr)
    opt_actor = torch.optim.Adam(model.actor.parameters(), lr=args.lr)

    # ---- 固定 val 评估集 + 参照线(与 PPO 同口径, D28/D43)----
    rcfg = RewardConfig()          # bounded_fail 默认 True(D64); 评估指标与 reward 无关
    ev_wins = operating_days("val")[:args.eval_days]
    eval_eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w)) for w in ev_wins]
    env_factory = lambda: RewardedDynamicEnv(net, params, ptc, dwell, reward_config=rcfg,
                                             proactive=True, reserve_window=args.reserve_window)
    top = lookup_table("train")
    baselines = {}
    for tag, fac in (("signaller_replay", signaller_replay_factory),
                     ("lookup_most_frequent", make_lookup_policy(top)),
                     ("greedy_set", as_factory(make_greedy_set_policy())),
                     ("random_legal", as_factory(make_random_policy(np.random.default_rng(0))))):
        b = evaluate_policy(env_factory, adapter, fac, eval_eps); b.pop("eval_episodes", None)
        baselines[tag] = b
    print(f"[baseline] 人类 {baselines['signaller_replay']['eval_episode_reward']:.1f} / "
          f"lookup {baselines['lookup_most_frequent']['eval_episode_reward']:.1f} / "
          f"有路就办 {baselines['greedy_set']['eval_episode_reward']:.1f}")

    run_id = args.run_tag or f"iql_{args.encoder}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    last_ck = config.ARTIFACTS_REPORTS / f"{run_id}_last.pt"
    history, evals = [], []
    best = {"reward": float("-inf"), "iter": -1}
    start_iter = 0
    if args.resume and last_ck.exists():
        st = torch.load(last_ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt_critic.load_state_dict(st["opt_critic"])
        opt_actor.load_state_dict(st["opt_actor"]); start_iter = int(st["next_iter"]); best = st["best"]
        history = st.get("history", []); evals = st.get("evals", [])
        print(f"[resume] 从 it{start_iter} 续训(best it{best['iter']} reward {best['reward']:.1f})")
    print(f"[run] {run_id} | dev {dev} | γ{args.gamma} τ{args.tau} β{args.beta}")

    def sample_batch(k):
        idx = rng.integers(0, len(trans), size=k)
        b = to_tensors(collate_panels([trans[i][0] for i in idx]), dev)
        nb = to_tensors(collate_panels([trans[i][1] for i in idx]), dev)
        a = torch.as_tensor([trans[i][2] for i in idx], device=dev, dtype=torch.long)
        r = torch.as_tensor([trans[i][3] for i in idx], device=dev, dtype=torch.float32)
        d = torch.as_tensor([trans[i][4] for i in idx], device=dev, dtype=torch.float32)
        return b, nb, a, r, d

    def save_last(next_iter):
        torch.save({"model": model.state_dict(), "opt_critic": opt_critic.state_dict(),
                    "opt_actor": opt_actor.state_dict(), "next_iter": next_iter, "best": best,
                    "history": history, "evals": evals, "args": vars(args), "run_id": run_id}, last_ck)

    t0 = time.time()
    for it in range(start_iter, args.iters):
        model.train()
        b, nb, a, r, d = sample_batch(args.batch)
        stats = iql_update(model, opt_critic, opt_actor, b, nb, a, r, d,
                           gamma=args.gamma, tau=args.tau, beta=args.beta, w_max=args.w_max)
        history.append({"iter": it, **stats})
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            model.eval()
            ev = evaluate_policy(env_factory, adapter,
                                 as_factory(TorchPanelDriver(model.actor, dev, greedy=True)), eval_eps)
            ev.pop("eval_episodes", None); ev["iter"] = it
            evals.append(ev)
            if ev["eval_episode_reward"] > best["reward"]:
                best = {"reward": ev["eval_episode_reward"], "iter": it,
                        "completion": ev["eval_completion_rate"], "delay": ev["eval_weighted_secondary_delay_s"]}
                torch.save({"model": {k: v.detach().to("cpu") for k, v in model.actor.state_dict().items()},
                            "encoder": args.encoder,
                            "dims": [adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                                     adapter.cancel_dim, adapter.train_dim, adapter.global_dim],
                            "iter": it, "eval": best, "run_id": run_id},
                           config.ARTIFACTS_REPORTS / f"{run_id}_best.pt")
            print(f"it {it:>5} | loss_q {stats['loss_q']:.3f} loss_v {stats['loss_v']:.3f} "
                  f"loss_pi {stats['loss_pi']:.3f} | adv {stats['adv_mean']:>7.2f} w {stats['w_mean']:.2f} "
                  f"|| val完成率 {ev['eval_completion_rate']:.3f} val_reward {ev['eval_episode_reward']:>10.1f} "
                  f"WAIT {ev['eval_frac_wait']:.0%}/set {ev['eval_frac_set']:.0%} "
                  f"{'✅压过lookup' if ev['eval_episode_reward'] > baselines['lookup_most_frequent']['eval_episode_reward'] else '❌'}")
            save_last(it + 1)
    print(f"\n完成 {args.iters} it / {time.time()-t0:.0f}s | 最佳 val: it{best['iter']} "
          f"完成率 {best.get('completion')} reward {best['reward']:.1f}")
    out = Path(args.out) if args.out else (config.ARTIFACTS_REPORTS / f"{run_id}.json")
    out.write_text(json.dumps({"args": vars(args), "best": best, "baselines": baselines,
                               "evals": evals, "history": history}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
