#!/usr/bin/env python3
"""M12b (D59 重写): 在真实**proactive**运营日上训练 maskable PPO(选路 KL-to-rule + full-panel BC 整网热启动, SMDP)。

用法(本地 GPU / 服务器):
    python scripts/13_train_ppo.py --iters 40 --steps-per-iter 2048 --encoder gnn \
        --bc-ckpt artifacts/dataset/bc_fullpanel_gnn.pt --kl-anchor schedule --kl-beta 0.1 --seed 0
    # ↑ 热启动 = full-panel BC(整个 PanelOptionPolicy, D9 最优 gnn), **整网**载入(不只选路头)。
    #   旧的 M11 只选路 PointerBC 热启动(bc_deepsets_noprior_*)与 reactive 环境已作废(D48/D59)。

D59 关键改动(全部去旧引擎):
    * **环境 proactive**(--reserve-window wide, 默认按 RR): 旧脚本建的是 reactive 环境(致命)。
    * **选路 KL 锚 = 规则参照**(--kl-anchor schedule|lookup|none): schedule=计划路(dec.logged)+lookup 兜底
      (RR faithful, 默认); lookup=只最频路; none=关 KL。取代旧的冻结神经 BC 锚。偏离靠 reward 挣。
    * **热启动整网**从 full-panel BC 载入(init); init 与 KL 锚**解耦**(init=BC 避免冷启死锁, 锚=schedule)。
    * **reward = per_train**(D58, RewardConfig 默认); **kind-anchor 默认关**(RL 自由学时机, 旧信号员混合作废)。

其它旋钮:
    --kl-beta      选路 KL 锚系数(先松: 0 / 0.01 / 0.1 / 1 看 EV 能否自恢复, D45 作废不预设强锚)
    --half-life    SMDP 折扣半衰期(秒)。**默认不给 = γ=1 不打折**(§10-G); 扫 300/900/3600
    --encoder      mlp/deepsets/settransformer/gnn (D9 消融; 热启动首选 gnn)
    --split        train/val (test 是完全留出集, 本脚本**拒绝**跑 test, §10-E)
    场景 = **运营日**(03:00→次日03:00, 按运营日划分 train/val/test, D43), 不筛拥堵窗口(D27+D34),
                   **真实历史、不注入合成扰动**(D28)
    --steps-per-iter 每轮更新的步数预算(不是局数 —— 局长会晃, batch 会不稳)
    --eval-every   每 N 轮在**固定评估集**(val 运营日, 真实历史不注入 D28)评一次 + 基线参照线对照

诚实: 无 `--bc-ckpt` 时是随机网络热启动 —— 只**跑通用**, 结果无意义。
"""
from __future__ import annotations

import argparse
import json
import os
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
from derbysim.learn.ppo import (PPOConfig, RunningMoments, TorchPanelDriver,
                                ppo_update, prepare_batch)
from derbysim.learn.rollout import (as_factory, collect_rollout, evaluate_policy,
                                    make_greedy_set_policy, make_random_policy,
                                    signaller_replay_factory)
from derbysim.learn.baselines import (lookup_table, make_ars_proxy_policy, make_fcfs_policy,
                                      make_fsfs_policy, make_lookup_policy, make_route_reference)
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
                    help="单次更新的策略 KL 上限, 超过就提前结束该轮 epoch 循环。护栏是防爆不是限速")
    ap.add_argument("--eval-days", type=int, default=999,
                    help="固定评估集用几个 val 运营日; 默认取满(11 天)")
    ap.add_argument("--days", type=int, default=999,
                    help="训练用几个**运营日**(03:00→次日03:00); 默认取满(train 段 65 天)。"
                         "此前是'最堵的 6 个 3h 窗口'= 18 小时 = 全量的 0.77%%(见 D26 更正)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--reserve-window", default="wide", choices=["wide", "narrow"],
                    help="proactive 预留窗口(D57): wide=按 RR 从已知前沿即可预留(默认); "
                         "narrow=校准的位置触发窗。RL 默认 wide, narrow 作消融")
    ap.add_argument("--kl-anchor", default="schedule", choices=["schedule", "lookup", "none"],
                    help="选路 KL 的**规则参照**(D59): schedule=计划路(dec.logged)+lookup 兜底(RR, 默认); "
                         "lookup=只最频路; none=关 KL。取代旧的冻结神经 BC 锚")
    ap.add_argument("--kl-beta", type=float, default=0.1,
                    help="选路 KL 锚系数(先松: 0/0.01/0.1/1 看 EV 自恢复); --kl-anchor none 时无效")
    ap.add_argument("--kind-anchor-beta", type=float, default=0.0,
                    help="**时机锚**(默认 0 关): p(动作类别) 到某参照的 KL。D59 主线要 RL 自由学时机, "
                         "旧信号员类别混合(reactive 引擎实测)已作废; >0 仅消融, 须先在新引擎重测构成")
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
    ap.add_argument("--max-decisions", type=int, default=20000,
                    help="单局决策上限。**3000 太低**: 宽窗每天要 ~3848 个决策, 3000 会截断 episode "
                         "→ 完成率被低估(旧'宽窗信号员 0.964'就是这个假象)。正常天 ≤3848, 2万足够。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-ckpt", default=None,
                    help="full-panel BC 权重(scripts/31 产出 {state_dict,encoder,dims}); 首选 gnn")
    ap.add_argument("--no-warm-start", action="store_true", help="消融: 不做 BC 热启动")
    ap.add_argument("--fail-penalty", type=float, default=50.0,
                    help="D64 有界搁浅罚(每辆固定, 取代 D46 到地平线无界项)。默认 50(大而有界)")
    ap.add_argument("--no-bounded-fail", action="store_true",
                    help="消融: 关掉有界搁浅罚, 回退 D46 到地平线(无界)")
    ap.add_argument("--run-tag", default=None,
                    help="稳定 run_id(无时间戳): 断点续训/矩阵幂等续跑用。给了它, checkpoint/输出用该名; "
                         "不给则沿用带时间戳的旧行为(每版留档)")
    ap.add_argument("--resume", action="store_true",
                    help="从 <run-tag>_last.pt 续训(需 --run-tag): 载 model/opt/ret_norm/iter/best/history/rng。"
                         "服务器单次 12h 上限时用 —— 到点被杀后重跑同一命令即从断点继续")
    ap.add_argument("--no-baseline-cache", action="store_true",
                    help="不缓存/不复用 baseline 参照线(默认缓存)。baseline 是**策略无关**的(signaller/lookup/"
                         "规则),(val, eval_days, reserve_window)相同即可跨 run 复用 → 省每 run ~3min 启动")
    ap.add_argument("--baselines-only", action="store_true",
                    help="只算并缓存 baseline 参照线然后退出(不热启动/不体检/不训练)。矩阵开跑前预热缓存用")
    # ---- curriculum(由易到难, 累加式; 只改'喂哪些天', 不动环境/奖励/网络)----
    ap.add_argument("--curriculum", action="store_true",
                    help="开启课程学习: 按每天车流量由易到难、累加式喂 train 日(档满/达标才升档)")
    ap.add_argument("--curriculum-tiers", default="185,286,300,999999",
                    help="各档**车数上限**(累加): 默认 ≤185/≤286/≤300/全部 = 9/33/48/64 天")
    ap.add_argument("--promote-at", type=float, default=0.9,
                    help="当前档 train 完成率≥此值就升档")
    ap.add_argument("--tier-max-iters", type=int, default=0,
                    help="**默认 0 = 不设上限, 只按达标(≥promote-at)升档**。>0 才启用'到点强制升'的兜底; "
                         "但强制升=把没学会的推去更难档=淤死(smoke 实证), 故默认关。练不到就留在本档接着练")
    ap.add_argument("--train-eval-days", type=int, default=6,
                    help="每次评估时在**当前档**抽几天算 train 完成率(升档判据 + train 曲线)")
    ap.add_argument("--final-test", action="store_true",
                    help="**训练结束后**用 best 权重在 test 段评**一次**(D43: test 全程封存, 只在最后碰一次)")
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
    # ---- 热启动: full-panel BC(整个 PanelOptionPolicy, D9 最优 gnn)**整网**载入(D59) ----
    # 与旧路线的区别: 旧的载 M11 只选路 PointerBC(load_bc_state_dict, 只迁选路头); 现在 BC 本身就是
    # 同一个 PanelOptionPolicy(scripts/31 训, 存 {state_dict,encoder,dims}), 整网 load_state_dict ——
    # 编码器 + 选路头 + kind/cancel 头 + critic 全给个能跑的起点(避免冷启动死锁)。init 与 KL 锚解耦。
    want_dims = [adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                 adapter.cancel_dim, adapter.train_dim, adapter.global_dim]
    if args.bc_ckpt and not args.no_warm_start and not args.baselines_only:
        ck = torch.load(args.bc_ckpt, map_location=dev, weights_only=False)
        if not (isinstance(ck, dict) and "state_dict" in ck):
            raise SystemExit(f"{args.bc_ckpt} 不是 full-panel BC(应含 state_dict/encoder/dims, "
                             f"scripts/31 产出)。旧的 M11 PointerBC 热启动已作废(D59)。")
        if ck.get("encoder") and ck["encoder"] != args.encoder:
            raise SystemExit(f"checkpoint 编码器={ck['encoder']} 与 --encoder={args.encoder} 不符")
        if ck.get("dims") and list(ck["dims"]) != want_dims:
            raise SystemExit(f"checkpoint dims={ck['dims']} 与当前特征器 {want_dims} 不符")
        print(f"[bc] {args.bc_ckpt}: encoder={ck.get('encoder')} dims={ck.get('dims')}")
        policy.load_state_dict(ck["state_dict"])               # **整网**热启动(strict, 维度不符即报错)
    else:
        print("[warn] 未做热启动(无 --bc-ckpt 或 --no-warm-start): 随机网络起步, 仅供跑通, 结果无意义")

    # ---- 选路 KL 的规则参照(D59, 取代冻结神经 BC) ----
    # schedule=计划路(dec.logged)+lookup 兜底(RR faithful); lookup=只最频路; none=关。lookup 表 held-out
    # 只用 train 段拟合(D43)。ref_builder 逐决策把参照目标挂到 enc, 只在 rollout 采集时用(评估不需要)。
    ref_builder = None
    if args.kl_anchor != "none" and args.kl_beta > 0:
        _top_ref = lookup_table("train")
        ref_builder = make_route_reference(_top_ref, mode=args.kl_anchor)
    print(f"[kl] anchor={args.kl_anchor} beta={args.kl_beta} kind_anchor_beta={args.kind_anchor_beta}")

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
    rcfg = RewardConfig(bounded_fail=not args.no_bounded_fail, fail_penalty=args.fail_penalty)
    # reward_mode 默认 per_train(D58); D64: 搁浅罚默认 RR 式有界(fail_penalty=50), 砍掉到地平线无界项的方差

    # ---- 固定评估集(同窗口/同注入种子, 跨 iteration 与跨策略完全可比) + 基线参照线 ----
    # **D28: 一律真实历史, 不注入合成扰动**(旧代码用注入种子 0/1, 与 M10_DESIGN §4.4 的设计
    # "固定评估=真实历史窗口, 不注入"自相矛盾; 且注入把信号员的延误抬到 2.5 倍而人无法重新决策)。
    # 环境是确定的 → 同一天跑两次逐位相同, 不需要评估种子。
    ev_wins = operating_days("val")[:args.eval_days]
    eval_eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
                for w in ev_wins]
    env_factory = lambda: RewardedDynamicEnv(net, params, ptc, dwell, reward_config=rcfg,
                                             proactive=True, reserve_window=args.reserve_window)
    print(f"[eval] 固定集 = {len(eval_eps)} 个 val 运营日(**真实历史, 不注入**, D28)"
          f" | 训练 {len(windows)} 个 {args.split} 运营日")
    print(f"[warmup] 前 {args.critic_warmup_iters} 轮只训 critic(行首标 *), 策略冻结; "
          f"**EV 是关键诊断** —— EV≈0 表示优势只是'位置噪声', 此时放开策略必被推坏")
    # **热启动体检(§17)**: 训练前先在固定评估集上跑一次 argmax, 看初始动作构成。(baselines-only 跳过)
    if not args.baselines_only:
        if roll_dev != dev:
            policy.to(roll_dev)      # 体检也走 rollout 设备(否则 driver 的输入在 cpu、权重在 cuda)
        _chk = evaluate_policy(env_factory, adapter,
                               as_factory(TorchPanelDriver(policy, roll_dev, greedy=True,
                                                           ret_norm=ret_norm)), eval_eps)
        if roll_dev != dev:
            policy.to(dev)
        print(f"[热启动体检] (**完整评估集**, 与后续 EVAL 可比) 初始 argmax 动作构成: WAIT {_chk['eval_frac_wait']:.0%} / "
              f"set {_chk['eval_frac_set']:.0%} / cancel {_chk['eval_frac_cancel']:.0%} "
              f"| 初始 completion {_chk['eval_completion_rate']:.3f}")
        # 真正的热启动失效: cancel 刷屏 / set 塌缩(全 WAIT) / WAIT 塌缩(全 set, 扁平 argmax bug §17.2)。
        # **宽窗里信号员本身 WAIT~68%**, 故高 WAIT 正常, 不再按旧 reactive 的 10% 判"异常"(那是误报)。
        if _chk["eval_frac_cancel"] > 0.20 or _chk["eval_frac_wait"] < 0.02 or _chk["eval_frac_set"] < 0.05:
            print("  ⚠️ **热启动动作构成异常**(cancel 刷屏 / 全 WAIT / 全 set 塌缩之一), 先查再跑。")

    # ---- baseline 参照线: 策略无关 → 按 (val, eval_days, reserve_window) 缓存, 跨 run 复用(省 ~3min/run) ----
    # 只影响**对照打印/写入的参照值**, 与 policy/optimizer/reward/训练**完全无关**(不碰主模型)。
    _blcache = config.ARTIFACTS_REPORTS / "rl_baseline_cache" / f"val_ed{len(ev_wins)}_{args.reserve_window}_v1.json"
    _bl_tags = ("signaller_replay", "lookup_most_frequent", "ars_proxy", "fcfs", "fsfs",
                "greedy_set", "random_legal")
    baselines = None
    if not args.no_baseline_cache and _blcache.exists():
        try:
            cached = json.loads(_blcache.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and all(t in cached for t in _bl_tags):
                baselines = cached
                print(f"[baseline] 命中缓存 {_blcache.name}(7 条参照线, 策略无关)")
        except Exception:
            baselines = None                                  # 损坏/半写 → 退回重算
    if baselines is None:
        baselines = {}
        _top = lookup_table()      # D42/D44: lookup 最频路是 held-out 最强安全基线, PPO 必须打败它
        for tag, fac in (("signaller_replay", signaller_replay_factory),
                         ("lookup_most_frequent", make_lookup_policy(_top)),
                         ("ars_proxy", make_ars_proxy_policy(_top)),
                         ("fcfs", make_fcfs_policy(_top)),
                         ("fsfs", make_fsfs_policy(_top)),
                         ("greedy_set", as_factory(make_greedy_set_policy())),
                         ("random_legal", as_factory(make_random_policy(np.random.default_rng(0))))):
            b = evaluate_policy(env_factory, adapter, fac, eval_eps)
            b.pop("eval_episodes", None)
            baselines[tag] = b
            print(f"[baseline] {tag:<17} completion {b['eval_completion_rate']:.3f} "
              f"| delay {b['eval_weighted_secondary_delay_s']:>9.0f} "
              f"| reward {b['eval_episode_reward']:>9.1f} "
              f"| WAIT {b['eval_frac_wait']:.0%}/set {b['eval_frac_set']:.0%}")
        if not args.no_baseline_cache:                        # 原子写缓存(临时文件 + 替换, 并发安全)
            _blcache.parent.mkdir(parents=True, exist_ok=True)
            _tmp = _blcache.with_suffix(f".tmp.{os.getpid()}")
            _tmp.write_text(json.dumps(baselines, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(_tmp, _blcache)
            print(f"[baseline] 已缓存 → {_blcache.name}")

    if args.baselines_only:      # 只预热 baseline 缓存, 不训练(矩阵开跑前调一次即可让所有 run 命中缓存)
        print("[baselines-only] 缓存就绪, 退出(未热启动/未训练)。")
        return

    # run_id: 给 --run-tag 用稳定名(续训/矩阵幂等), 否则带时间戳(每版留档, 用户 2026-07-31 要求)。
    run_id = args.run_tag or (f"ppo_{args.encoder}_beta{args.kl_beta}_seed{args.seed}"
                              f"_{time.strftime('%Y%m%d-%H%M%S')}")
    last_ck = config.ARTIFACTS_REPORTS / f"{run_id}_last.pt"
    history, evals = [], []
    best = {"reward": float("-inf"), "iter": -1}
    since_best = 0
    start_iter = 0
    # ---- 断点续训(D60): 从 <run-tag>_last.pt 恢复全状态 ----
    if args.resume:
        if not args.run_tag:
            raise SystemExit("--resume 需要 --run-tag(稳定名)才能定位 _last.pt")
        if last_ck.exists():
            st = torch.load(last_ck, map_location=dev, weights_only=False)
            policy.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            if ret_norm is not None and st.get("ret_norm"):
                ret_norm.load_state_dict(st["ret_norm"])
            start_iter = int(st["next_iter"]); best = st["best"]; since_best = st["since_best"]
            history = st.get("history", []); evals = st.get("evals", [])
            total_steps_resumed = int(st.get("total_steps", 0))
            cur_tier_resumed = int(st.get("cur_tier", 0)); tier_iter_resumed = int(st.get("tier_iter", 0))
            try:
                rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"])
            except Exception:
                pass
            print(f"[resume] {last_ck.name}: 从 it{start_iter} 续训(best it{best['iter']} "
                  f"reward {best['reward']:.1f}, since_best {since_best})")
        else:
            print(f"[resume] 无 {last_ck.name}, 从头开始")
    print(f"[run] {run_id}")

    def save_last(next_iter):                    # 周期全状态存档(续训用)
        torch.save({"model": {k: v.detach().to("cpu") for k, v in policy.state_dict().items()},
                    "opt": opt.state_dict(),
                    "ret_norm": (ret_norm.state_dict() if ret_norm else None),
                    "next_iter": next_iter, "best": best, "since_best": since_best,
                    "history": history, "evals": evals, "total_steps": total_steps,
                    "cur_tier": cur_tier, "tier_iter": tier_iter,        # curriculum 续训状态
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "args": vars(args), "run_id": run_id}, last_ck)

    # ---- curriculum(由易到难, 累加式): 只改'喂哪些天', 不动环境/奖励/网络 ----
    def _sample_eps(pool_days, k):
        """从 pool 里确定性均匀取最多 k 天, 做 train 完成率(升档判据 + train 曲线)。"""
        if not pool_days:
            return []
        step = max(1, len(pool_days) // max(k, 1))
        picks = pool_days[::step][:k]
        return [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w)) for w in picks]

    if args.curriculum:
        tier_thr = [int(x) for x in args.curriculum_tiers.split(",")]
        _dc = sorted(((w, len(build_day_trains(w))) for w in windows), key=lambda x: x[1])
        def _pool_for(k):
            return [w for (w, c) in _dc if c <= tier_thr[min(k, len(tier_thr) - 1)]]
        cur_tier = int(locals().get("cur_tier_resumed", 0))
        tier_iter = int(locals().get("tier_iter_resumed", 0))
        pool = _pool_for(cur_tier)
        print(f"[curriculum] 起始 档{cur_tier+1} ≤{tier_thr[cur_tier]}车 → {len(pool)} 天; "
              f"升档@train完成率≥{args.promote_at} 或满 {args.tier_max_iters} 次评估(累加式)")
    else:
        tier_thr = None; cur_tier = 0; tier_iter = 0
        pool = windows
    train_eval_eps = _sample_eps(pool, args.train_eval_days)

    t0 = time.time()
    total_steps = int(locals().get("total_steps_resumed", 0))
    for it in range(start_iter, args.iters):
        def episode_stream():
            while True:
                w = pool[int(rng.integers(len(pool)))]           # curriculum: 从当前档池子抽
                yield (str(w["start"])[:16],
                       (lambda w=w: build_day_trains(w)),      # D28: 真实历史, 不注入
                       day_horizon(w))
        t_roll = time.time()
        if roll_dev != dev:
            policy.to(roll_dev)                       # 权重搬到 rollout 设备(小模型, 开销可忽略)
        transitions, ep_metrics = collect_rollout(
            env_factory, adapter, as_factory(driver), episode_stream(),
            target_steps=args.steps_per_iter, max_decisions=args.max_decisions,
            ref_builder=ref_builder)          # D59: 采集时挂选路 KL 的规则参照(schedule/lookup)
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
        stats = ppo_update(policy, opt,
                           prepare_batch(transitions, cfg, ret_norm=ret_norm), cfg, dev,
                           value_only=warming)
        upd_s = time.time() - t_upd
        row = {"iter": it, "steps": len(transitions), "episodes": len(ep_metrics), **stats,
               "train_reward_mean": float(np.mean([m["episode_reward"] for m in ep_metrics])),
               "train_completion_mean": float(np.mean([m["completion_rate"] for m in ep_metrics])),
               "train_delay_mean": float(np.mean([m["weighted_secondary_delay_s"] for m in ep_metrics])),
               "rollout_s": roll_s, "update_s": upd_s,
               "steps_per_s": len(transitions) / max(roll_s, 1e-9),
               "ret_mean": (ret_norm.mean if ret_norm else 0.0),
               "ret_std": (ret_norm.std if ret_norm else 1.0)}
        row["tier"] = cur_tier
        print(f"it {it:>3}{'*' if warming else ' '}| 档{cur_tier+1} | steps {row['steps']:>5} ({row['episodes']} eps) "
              f"| **完成率 {row['train_completion_mean']:.3f} 延误 {row['train_delay_mean']:>9.0f} "
              f"reward {row['train_reward_mean']:>8.1f}** | EV {row['explained_variance']:>5.3f} | 选路KL {row['routing_kl']:.4f} "
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
            greedy_fac = as_factory(TorchPanelDriver(policy, roll_dev, greedy=True, ret_norm=ret_norm))
            ev = evaluate_policy(env_factory, adapter, greedy_fac, eval_eps)          # val(固定 11 日)
            tev = (evaluate_policy(env_factory, adapter, greedy_fac, train_eval_eps)  # train(当前档抽样)
                   if train_eval_eps else {})
            if roll_dev != dev:
                policy.to(dev)
            ev.pop("eval_episodes", None); tev.pop("eval_episodes", None)
            ev["iter"] = it; ev["tier"] = cur_tier
            # **point4: train(greedy, 当前档) 与 val 两套都记录**(之前只记 val)
            _tc = tev.get("eval_completion_rate")
            ev["train_completion"] = _tc
            ev["train_episode_reward"] = tev.get("eval_episode_reward")
            ev["train_weighted_secondary_delay_s"] = tev.get("eval_weighted_secondary_delay_s")
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
            print(f"    >>> EVAL it {it} 档{cur_tier+1}: "
                  f"train完成率 {(_tc if _tc is not None else float('nan')):.3f} | "
                  f"val完成率 {ev['eval_completion_rate']:.3f} | val_delay {ev['eval_weighted_secondary_delay_s']:>9.0f} "
                  f"| val_reward {ev['eval_episode_reward']:>9.1f} "
                  f"| WAIT {ev['eval_frac_wait']:.0%}/set {ev['eval_frac_set']:.0%}/cancel {ev['eval_frac_cancel']:.0%} "
                  f"| 死锁 {ev['eval_deadlocks']} "
                  f"|| 靶子 lookup reward {baselines['lookup_most_frequent']['eval_episode_reward']:.1f} "
                  f"{'✅压过' if ev['eval_weighted_secondary_delay_s'] < baselines['lookup_most_frequent']['eval_weighted_secondary_delay_s'] else '❌未过'} "
                  f"| 人类 {baselines['signaller_replay']['eval_episode_reward']:.1f}")
            # ---- 升档(累加式): 当前档 train 完成率达标 或 到 cap → 扩池(最终档不再升)----
            at_final_tier = (not args.curriculum) or (cur_tier >= len(tier_thr) - 1)
            if args.curriculum and not at_final_tier:
                tier_iter += 1
                cap_hit = args.tier_max_iters > 0 and tier_iter >= args.tier_max_iters   # 默认 0 = 不强制
                if (_tc is not None and _tc >= args.promote_at) or cap_hit:
                    cur_tier += 1; tier_iter = 0
                    pool = _pool_for(cur_tier); train_eval_eps = _sample_eps(pool, args.train_eval_days)
                    at_final_tier = cur_tier >= len(tier_thr) - 1
                    print(f"        ⏫ 升档 → 档{cur_tier+1} ≤{tier_thr[cur_tier]}车 → {len(pool)} 天"
                          + ("(最终档/全高峰)" if at_final_tier else ""))
            # early-stop **只在最终档生效**: 课程中途 val 会平台期(还没练到难日), 不能误判为收敛
            if at_final_tier and args.early_stop_patience and since_best >= args.early_stop_patience:
                print(f"        早停: 最终档连续 {since_best} 次评估未创新高 "
                      f"(最佳 it{best['iter']} reward {best['reward']:.1f})")
                history.append(row)
                save_last(args.iters)        # 标记已完成(next_iter=iters → 续跑不再重训)
                break
            save_last(it + 1)          # D60: 每 eval 周期存全状态 → 12h 被杀后可续训
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

    # ---- final-test(D43: test 全程封存, 只在最后用 best 权重碰一次)----
    final_test_result = None
    if args.final_test:
        best_ck = config.ARTIFACTS_REPORTS / f"{run_id}_best.pt"
        if best_ck.exists():
            st = torch.load(best_ck, map_location=dev, weights_only=False)
            policy.load_state_dict(st["model"])
            if ret_norm is not None and st.get("ret_norm"):
                ret_norm.load_state_dict(st["ret_norm"])
            test_eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
                        for w in operating_days("test")]
            if roll_dev != dev:
                policy.to(roll_dev)
            tst = evaluate_policy(env_factory, adapter,
                                  as_factory(TorchPanelDriver(policy, roll_dev, greedy=True,
                                                              ret_norm=ret_norm)), test_eps)
            if roll_dev != dev:
                policy.to(dev)
            tst.pop("eval_episodes", None)
            final_test_result = tst
            print(f"[final-test] (D43 一次性) test {len(test_eps)} 日: completion {tst['eval_completion_rate']:.3f} "
                  f"| delay {tst['eval_weighted_secondary_delay_s']:.0f} | reward {tst['eval_episode_reward']:.1f}")
        else:
            print("[final-test] 无 best 权重, 跳过")

    out = Path(args.out) if args.out else (config.ARTIFACTS_REPORTS / f"{run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "device": str(dev),
                               "elapsed_s": time.time() - t0, "total_steps": total_steps,
                               "run_id": run_id, "best": best, "final_test": final_test_result,
                               "baselines": baselines, "evals": evals, "history": history},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
