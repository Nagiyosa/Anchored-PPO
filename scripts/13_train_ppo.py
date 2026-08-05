#!/usr/bin/env python3
"""M12b: 在真实运营日上训练 maskable PPO(KL-to-BC + no-prior BC 热启动, SMDP)。

用法(本地 GPU):
    python scripts/13_train_ppo.py --iters 40 --steps-per-iter 2048 --kl-beta 0.1 \
        --bc-ckpt artifacts/reports/bc_deepsets_noprior_nw_d35.pt --seed 0
    # ↑ 当前正确的 no-prior 热启动权重(D35 + M11 §18 修好配置 --no-route-weight)。
    #   旧名 bc_deepsets_noprior.pt / _noprior_nw.pt / _noprior_d35.pt 均已过时, 勿用(F2, 2026-08-05)。

关键旋钮(与 docs/M12_DESIGN §10 对应):
    --kl-beta      KL-to-BC 固定档位(§10-F 手动扫描: 0 / 0.01 / 0.1 / 1 / 10)
    --half-life    SMDP 折扣半衰期(秒)。**默认不给 = γ=1 不打折**(§10-G); M12e 扫 300/900/3600
    --encoder      mlp/deepsets/settransformer/gnn (D9 消融; bring-up 用 deepsets)
    --split        train/val (test 是完全留出集, 本脚本**拒绝**跑 test, §10-E)
    场景 = **运营日**(03:00→次日03:00, 完整覆盖的 96 天按时间划分 train64/val11/test21),
    不筛拥堵窗口(D27+D34),
                   **真实历史、不注入合成扰动**(D28)
    --steps-per-iter 每轮更新的步数预算(不是局数 —— 局长在 180-4586 间晃, batch 会不稳)
    --eval-every   每 N 轮在**固定评估集**上评一次(val 运营日, **真实历史不注入** D28),
                   并与三条基线参照线(信号员回放 / 有路就办 / 随机)同图对照

诚实: 无 `--bc-ckpt` 时 KL 参照是随机初始化的网络 —— 那只是**跑通用**, 不是热启动, 结果无意义。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 使 derbysim 可导入(从任意 cwd 运行)

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.panel_models import PanelOptionPolicy, pick_device
from derbysim.learn.ppo import (PPOConfig, RunningMoments, TorchPanelDriver, frozen_bc,
                                ppo_update, prepare_batch)
from derbysim.learn.rollout import (as_factory, collect_rollout, evaluate_policy,
                                    make_greedy_set_policy, make_random_policy,
                                    signaller_replay_factory)
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="deepsets")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--steps-per-iter", type=int, default=2048,
                    help="每轮更新的**步数预算**(凑够才更新); 固定局数会让 batch 在 180-4586 间晃")
    ap.add_argument("--critic-warmup-iters", type=int, default=8,
                    help="开局只训 critic、策略不动的轮数。critic 没学会前优势只编码'步在 episode "
                         "中的位置'(实测相关 0.77-0.96), 照它更新会把策略推坏")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--early-stop-patience", type=int, default=6,
                    help="固定评估集上连续 N 次评估未创新高就停(0=不早停)。三轮实测均在解冻后约"
                         "20 轮见顶后塌, 继续跑纯属浪费且最终策略远差于峰值")
    ap.add_argument("--lr-final-frac", type=float, default=0.1,
                    help="学习率线性退火到初始值的这个比例(1.0=不退火)")
    ap.add_argument("--target-kl", type=float, default=0.05,
                    help="单次更新的策略 KL 上限, 超过就提前结束该轮 epoch 循环。**0.05 而不是 0.02**"
                         "(2026-08-02 重标): 0.02 是在**三轮已知有缺陷的训练**(dropout 污染 ratio / "
                         "扁平 argmax / 热启动被稀释)上定的; dropout 修好后 approx_kl 才第一次反映"
                         "**真实的策略位移**, 实测稳定在 0.019-0.024 —— 正好卡在旧阈值上, 25-50% 的 "
                         "epoch 被无谓截断。护栏是防爆不是限速")
    ap.add_argument("--eval-days", type=int, default=999,
                    help="固定评估集用几个 val 运营日; 默认取满(11 天)")
    ap.add_argument("--days", type=int, default=999,
                    help="训练用几个**运营日**(03:00→次日03:00); 默认取满(train 段 65 天)。"
                         "此前是'最堵的 6 个 3h 窗口'= 18 小时 = 全量的 0.77%%(见 D26 更正)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--kind-anchor-beta", type=float, default=0.05,
                    help="**时机锚**: p(动作类别) 到信号员实测构成的 KL 系数。0=关(消融)。"
                         "不加锚时实测 cancel 占比会从 0%% 漂到 58%%(真实撤销率仅 10.44%%), "
                         "completion 从 0.306 退回 0.241 —— kind_head 是唯一没有行为锚的分支")
    ap.add_argument("--half-life", type=float, default=None,
                    help="SMDP 折扣半衰期(秒); 缺省 = γ=1 不打折(§10-G)")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="GAE λ; 默认 1.0 = 蒙特卡洛(实测 λ=0.95 让第0步只看到 0.2%% 的回报)")
    ap.add_argument("--reward-scale", type=float, default=1.0,
                    help="固定奖励缩放; **只在 --no-return-norm 时有意义**(旧默认 0.01 是在 3h "
                         "窗口上标的, 运营日回报量级差 50 倍)")
    ap.add_argument("--no-return-norm", action="store_true",
                    help="关掉回报标准化, 回退到固定 --reward-scale(消融对照)")
    ap.add_argument("--rollout-device", default="cpu", choices=["cpu", "cuda", "same"],
                    help="rollout 期策略跑在哪(默认 cpu): batch=1 的 GPU 前向 73%% 是 kernel 启动"
                         "往返, 小网在 CPU 上更快; 副作用是训练变成纯 CPU 单核 → 矩阵可并发")
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
    windows = operating_days(args.split)[:args.days]
    if not windows:
        raise SystemExit(f"no operating days for split={args.split}")

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
                    minibatch=args.minibatch, update_epochs=args.update_epochs,
                    reward_scale=args.reward_scale,
                    normalise_returns=not args.no_return_norm,
                    kind_anchor_beta=args.kind_anchor_beta,
                    target_kl=(args.target_kl if args.target_kl > 0 else None))
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    # critic 预测归一化值 → driver 必须还原成原始单位再交给 GAE(见 smdp.RunningMoments)。
    ret_norm = RunningMoments() if cfg.normalise_returns else None
    # 方案 B(D32): rollout 期把策略放 CPU —— batch=1 的 GPU 前向 73% 是 kernel 启动往返。
    # `--rollout-device` 缺省 cpu; 传 same 则沿用训练设备(消融/对照用)。
    roll_dev = dev if args.rollout_device == "same" else torch.device(args.rollout_device)
    roll_gen = gen if roll_dev == dev else torch.Generator(device=roll_dev).manual_seed(args.seed)
    driver = TorchPanelDriver(policy, roll_dev, generator=roll_gen, ret_norm=ret_norm)
    rcfg = RewardConfig()

    # ---- 固定评估集(同窗口/同注入种子, 跨 iteration 与跨策略完全可比) + 基线参照线 ----
    # **D28: 一律真实历史, 不注入合成扰动**(旧代码用注入种子 0/1, 与 M10_DESIGN §4.4 的设计
    # "固定评估=真实历史窗口, 不注入"自相矛盾; 且注入把信号员的延误抬到 2.5 倍而人无法重新决策)。
    # 环境是确定的 → 同一天跑两次逐位相同, 不需要评估种子。
    ev_wins = operating_days("val")[:args.eval_days]
    eval_eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
                for w in ev_wins]
    env_factory = lambda: RewardedDynamicEnv(net, params, ptc, dwell, reward_config=rcfg)
    print(f"[eval] 固定集 = {len(eval_eps)} 个 val 运营日(**真实历史, 不注入**, D28)"
          f" | 训练 {len(windows)} 个 {args.split} 运营日")
    print(f"[warmup] 前 {args.critic_warmup_iters} 轮只训 critic(行首标 *), 策略冻结; "
          f"**EV 是关键诊断** —— EV≈0 表示优势只是'位置噪声', 此时放开策略必被推坏")
    # **热启动体检(§17)**: 训练前先在固定评估集上跑一次 argmax, 看初始动作构成。
    # 上一轮冒烟正是栽在这里: 所有单点指标正常, 但策略从第一步就在刷 cancel。
    if roll_dev != dev:
        policy.to(roll_dev)          # 体检也走 rollout 设备(否则 driver 的输入在 cpu、权重在 cuda)
    _chk = evaluate_policy(env_factory, adapter,
                           as_factory(TorchPanelDriver(policy, roll_dev, greedy=True,
                                                       ret_norm=ret_norm)), eval_eps)
    if roll_dev != dev:
        policy.to(dev)
    print(f"[热启动体检] (**完整评估集**, 与后续 EVAL 可比) 初始 argmax 动作构成: WAIT {_chk['eval_frac_wait']:.0%} / "
          f"set {_chk['eval_frac_set']:.0%} / cancel {_chk['eval_frac_cancel']:.0%} "
          f"| 初始 completion {_chk['eval_completion_rate']:.3f}")
    if _chk["eval_frac_cancel"] > 0.20 or not (0.02 <= _chk["eval_frac_wait"] <= 0.60):
        print("  ⚠️ **动作构成异常** —— 参考: 信号员实测 WAIT 10% / set 90% / cancel 0%, "
              "真实撤销率 10.44%。这一类'热启动静默失效'不会有任何其它指标报警, 先查再跑。")

    baselines = {}
    for tag, fac in (("signaller_replay", signaller_replay_factory),
                     ("greedy_set", as_factory(make_greedy_set_policy())),
                     ("random_legal", as_factory(make_random_policy(np.random.default_rng(0))))):
        b = evaluate_policy(env_factory, adapter, fac, eval_eps)
        b.pop("eval_episodes", None)
        baselines[tag] = b
        print(f"[baseline] {tag:<17} completion {b['eval_completion_rate']:.3f} "
              f"| delay {b['eval_weighted_secondary_delay_s']:>9.0f} "
              f"| reward {b['eval_episode_reward']:>9.1f} "
              f"| WAIT {b['eval_frac_wait']:.0%}/set {b['eval_frac_set']:.0%}")

    # **每一版都留下来**(用户 2026-07-31 要求): 文件名带时间戳, 不再覆盖上一轮。
    run_id = (f"ppo_{args.encoder}_beta{args.kl_beta}_seed{args.seed}"
              f"_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[run] {run_id}")
    history, evals = [], []
    best = {"reward": float("-inf"), "iter": -1}
    since_best = 0
    t0 = time.time()
    total_steps = 0
    for it in range(args.iters):
        def episode_stream():
            while True:
                w = windows[int(rng.integers(len(windows)))]
                yield (str(w["start"])[:16],
                       (lambda w=w: build_day_trains(w)),      # D28: 真实历史, 不注入
                       day_horizon(w))
        t_roll = time.time()
        if roll_dev != dev:
            policy.to(roll_dev)                       # 权重搬到 rollout 设备(小模型, 开销可忽略)
        transitions, ep_metrics = collect_rollout(
            env_factory, adapter, as_factory(driver), episode_stream(),
            target_steps=args.steps_per_iter, max_decisions=args.max_decisions)
        if roll_dev != dev:
            policy.to(dev)                            # 更新在训练设备上做
        roll_s = time.time() - t_roll
        total_steps += len(transitions)
        t_upd = time.time()
        warming = it < args.critic_warmup_iters
        if args.lr_final_frac < 1.0 and args.iters > 1:     # 线性退火, 减少后期外飘
            frac = it / max(args.iters - 1, 1)
            for gp in opt.param_groups:
                gp["lr"] = args.lr * (1.0 - frac * (1.0 - args.lr_final_frac))
        stats = ppo_update(policy, bc_ref, opt,
                           prepare_batch(transitions, cfg, ret_norm=ret_norm), cfg, dev,
                           value_only=warming)
        upd_s = time.time() - t_upd
        row = {"iter": it, "steps": len(transitions), "episodes": len(ep_metrics), **stats,
               "train_reward_mean": float(np.mean([m["episode_reward"] for m in ep_metrics])),
               "train_completion_mean": float(np.mean([m["completion_rate"] for m in ep_metrics])),
               "rollout_s": roll_s, "update_s": upd_s,
               "steps_per_s": len(transitions) / max(roll_s, 1e-9),
               "ret_mean": (ret_norm.mean if ret_norm else 0.0),
               "ret_std": (ret_norm.std if ret_norm else 1.0)}
        print(f"it {it:>3}{'*' if warming else ' '}| steps {row['steps']:>5} ({row['episodes']} eps) "
              f"| **完成率 {row['train_completion_mean']:.3f}** | EV {row['explained_variance']:>5.3f} | KL {row['kl_to_bc']:.4f} "
              f"| 锚 {row.get('kind_anchor_kl', 0):.3f} "
              f"| ent {row['entropy']:.3f} | aKL {row.get('approx_kl', 0):.3f} "
              f"| 早停 {row.get('kl_stop_rate', 0):.0%}"
              f"| rollout {roll_s:>5.1f}s ({row['steps_per_s']:>4.0f}步/s) upd {upd_s:>4.1f}s")
        if not warming and row.get("kl_stop_rate", 0) >= 0.5:
            print("        ⚠️ **target_kl 护栏在主导训练** —— 半数以上 epoch 在第一个 minibatch 就 "
                  "break, 实际只执行了很小一部分更新。查 approx_kl 为何在学习前就偏高。")
        if args.eval_every and ((it + 1) % args.eval_every == 0 or it == args.iters - 1):
            if roll_dev != dev:
                policy.to(roll_dev)
            ev = evaluate_policy(env_factory, adapter,
                                 as_factory(TorchPanelDriver(policy, roll_dev, greedy=True,
                                                             ret_norm=ret_norm)), eval_eps)
            if roll_dev != dev:
                policy.to(dev)
            ev.pop("eval_episodes", None)
            ev["iter"] = it
            evals.append(ev)
            if ev["eval_episode_reward"] > best["reward"]:
                best = {"reward": ev["eval_episode_reward"], "iter": it,
                        "completion": ev["eval_completion_rate"],
                        "delay": ev["eval_weighted_secondary_delay_s"]}
                ck = config.ARTIFACTS_REPORTS / f"{run_id}_best.pt"
                torch.save({"model": {k: v.detach().to("cpu") for k, v in policy.state_dict().items()},
                            "iter": it, "eval": {k: v for k, v in ev.items() if k != "eval_episodes"},
                            # critic 输出是归一化值 → 不存 (μ,σ) 就还原不回原始单位, 断点无法复用
                            "ret_norm": (ret_norm.state_dict() if ret_norm else None),
                            "args": vars(args), "run_id": run_id}, ck)
                print(f"        ↑ 新最佳, 已存 {ck.name}")
                since_best = 0
            else:
                since_best += 1
                if args.early_stop_patience and since_best >= args.early_stop_patience:
                    print(f"        早停: 连续 {since_best} 次评估未创新高 "
                          f"(最佳 it{best['iter']} reward {best['reward']:.1f})")
                    history.append(row)
                    break
            print(f"    >>> EVAL it {it}: completion {ev['eval_completion_rate']:.3f} "
                  f"| delay {ev['eval_weighted_secondary_delay_s']:>9.0f} "
                  f"| reward {ev['eval_episode_reward']:>9.1f} "
                  f"| WAIT {ev['eval_frac_wait']:.0%}/set {ev['eval_frac_set']:.0%}"
                  f"/cancel {ev['eval_frac_cancel']:.0%} | 死锁 {ev['eval_deadlocks']} "
                  f"| (人类 {baselines['signaller_replay']['eval_episode_reward']:.1f} / "
                  f"有路就办 {baselines['greedy_set']['eval_episode_reward']:.1f} / "
                  f"随机 {baselines['random_legal']['eval_episode_reward']:.1f})")
        history.append(row)
    print(f"\n总计 {total_steps} 步 / {time.time()-t0:.0f}s")
    if best["iter"] >= 0:
        print(f"最佳: it {best['iter']} | completion {best['completion']:.3f} "
              f"| delay {best['delay']:.0f} | reward {best['reward']:.1f}  "
              f"(权重已存 {run_id}_best.pt)")
        b = baselines
        print(f"对照: 人类 {b['signaller_replay']['eval_episode_reward']:.1f} / "
              f"有路就办 {b['greedy_set']['eval_episode_reward']:.1f} / "
              f"随机 {b['random_legal']['eval_episode_reward']:.1f}")

    out = Path(args.out) if args.out else (config.ARTIFACTS_REPORTS / f"{run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "device": str(dev),
                               "elapsed_s": time.time() - t0, "total_steps": total_steps,
                               "run_id": run_id, "best": best,
                               "baselines": baselines, "evals": evals, "history": history},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
