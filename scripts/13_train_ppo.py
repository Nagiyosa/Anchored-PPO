#!/usr/bin/env python3
"""M12b: 在真实拥堵窗口上训练 maskable PPO(KL-to-BC + no-prior BC 热启动, SMDP)。

用法(本地 GPU):
    python scripts/13_train_ppo.py --iters 20 --episodes-per-iter 2 --kl-beta 0.1 \
        --bc-ckpt artifacts/reports/bc_deepsets_noprior.pt --seed 0

关键旋钮(与 docs/M12_DESIGN §10 对应):
    --kl-beta      KL-to-BC 固定档位(§10-F 手动扫描: 0 / 0.01 / 0.1 / 1 / 10)
    --half-life    SMDP 折扣半衰期(秒)。**默认不给 = γ=1 不打折**(§10-G); M12e 扫 300/900/3600
    --encoder      mlp/deepsets/settransformer/gnn (D9 消融; bring-up 用 deepsets)
    --split        train/val (test 是完全留出集, 本脚本**拒绝**跑 test, §10-E)

诚实: 无 `--bc-ckpt` 时 KL 参照是随机初始化的网络 —— 那只是**跑通用**, 不是热启动, 结果无意义。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.panel_models import PanelOptionPolicy, pick_device
from derbysim.learn.ppo import PPOConfig, TorchPanelDriver, frozen_bc, ppo_update, prepare_batch
from derbysim.learn.rollout import run_episode
from derbysim.learn.scenarios import build_episode_trains, congested_windows, load_assets
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="deepsets")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes-per-iter", type=int, default=2)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--half-life", type=float, default=None,
                    help="SMDP 折扣半衰期(秒); 缺省 = γ=1 不打折(§10-G)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-ckpt", default=None, help="M11 **no-prior** BC 权重(PointerBC state_dict)")
    ap.add_argument("--no-warm-start", action="store_true", help="消融: 不做 BC 热启动")
    ap.add_argument("--allow-prior-ckpt", action="store_true",
                    help="放行含先验的 BC(仅对照实验; 默认拒绝, 见 M11 §12 决定 A)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = pick_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))          # route_prior={} 由适配层强制(§10-A③)
    windows = congested_windows(args.split, top_k=args.windows)
    if not windows:
        raise SystemExit(f"no windows for split={args.split}")

    policy = PanelOptionPolicy(args.encoder, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                               adapter.cancel_dim, adapter.train_dim, adapter.global_dim,
                               hidden=args.hidden).to(dev)
    bc_state = None
    if args.bc_ckpt:
        ck = torch.load(args.bc_ckpt, map_location=dev, weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            # **硬校验**: 静默的不匹配会毁掉热启动而不报错(§10-A 三条纪律)。
            if ck.get("use_prior", False) and not args.allow_prior_ckpt:
                raise SystemExit(
                    "该 checkpoint 是**含先验**的 BC。M11 §12 决定 A + M12_DESIGN §10-A③: "
                    "热启动必须用 no-prior 版(scripts/12 --no-prior) —— 含先验版长尾低于随机, "
                    "KL 会把 RL 拽向'永不选罕见路', 正好压制我们要研究的偏离。"
                    "确要对照实验请加 --allow-prior-ckpt。")
            if ck.get("encoder") and ck["encoder"] != args.encoder:
                raise SystemExit(f"checkpoint 编码器={ck['encoder']} 与 --encoder={args.encoder} 不符")
            if ck.get("hidden") and int(ck["hidden"]) != args.hidden:
                raise SystemExit(f"checkpoint hidden={ck['hidden']} 与 --hidden={args.hidden} 不符")
            want = [adapter.dec_dim, adapter.conf_dim, adapter.cand_dim]
            if ck.get("dims") and list(ck["dims"]) != want:
                raise SystemExit(f"checkpoint dims={ck['dims']} 与当前特征器 {want} 不符")
            print(f"[bc] {args.bc_ckpt}: encoder={ck.get('encoder')} hidden={ck.get('hidden')} "
                  f"use_prior={ck.get('use_prior')} holdout_macro={ck.get('holdout_macro')}")
            bc_state = ck["model"]
        else:
            bc_state = ck
        if not args.no_warm_start:
            policy.load_bc_state_dict(bc_state)                 # 选路分支整体热启动(§10-A)
    else:
        print("[warn] 未给 --bc-ckpt: KL 参照与热启动都是随机网络, 仅供跑通, 结果无意义")
    bc_ref = frozen_bc(args.encoder, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                       args.hidden, bc_state, dev)

    cfg = PPOConfig(lr=args.lr, lam=args.lam, half_life_s=args.half_life, kl_beta=args.kl_beta,
                    minibatch=args.minibatch, update_epochs=args.update_epochs)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    driver = TorchPanelDriver(policy, dev, generator=gen)
    rcfg = RewardConfig()

    history = []
    t0 = time.time()
    for it in range(args.iters):
        transitions, ep_metrics = [], []
        for _ in range(args.episodes_per_iter):
            w = windows[int(rng.integers(len(windows)))]
            trains = build_episode_trains(w, seed=int(rng.integers(1 << 30)))
            env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=rcfg)
            res = run_episode(env, adapter, driver, trains, max_decisions=args.max_decisions)
            transitions.extend(res.transitions)
            m = dict(res.metrics); m["window"] = str(w["start"])
            ep_metrics.append(m)
        stats = ppo_update(policy, bc_ref, opt, prepare_batch(transitions, cfg), cfg, dev)
        row = {"iter": it, "steps": len(transitions), **stats,
               "reward_mean": float(np.mean([m["episode_reward"] for m in ep_metrics])),
               "completion_mean": float(np.mean([m["completion_rate"] for m in ep_metrics])),
               "delay_mean": float(np.mean([m["weighted_secondary_delay_s"] for m in ep_metrics])),
               "deadlocks": int(sum(m["deadlocked"] for m in ep_metrics)),
               "collisions": int(sum(m["n_collisions"] for m in ep_metrics)),
               "episodes": ep_metrics}
        history.append(row)
        print(f"it {it:>3} | steps {row['steps']:>5} | reward {row['reward_mean']:>9.1f} "
              f"| completion {row['completion_mean']:.3f} | delay {row['delay_mean']:>10.0f} "
              f"| KL {row['kl_to_bc']:.4f} | ent {row['entropy']:.3f} | dl {row['deadlocks']}")

    out = Path(args.out) if args.out else (
        config.ARTIFACTS_REPORTS /
        f"ppo_{args.encoder}_beta{args.kl_beta}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "device": str(dev),
                               "elapsed_s": time.time() - t0, "history": history},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
