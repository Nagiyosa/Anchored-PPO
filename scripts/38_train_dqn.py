"""D76 基线: DQN(Double)—— 面板选项 Q 学习(与主方法同编码器/同评估口径, 供导师要求的 RL 基线对比)。

Q 网络 = PanelOptionPolicy(把逐选项 logits 解释为 Q(s,option); value/aux 头不用)。
ε-greedy 采样 → 回放池(顺序转移, done 掩蔽 bootstrap)→ Double DQN 目标:
    y = r + γ·(1−done)·Q_tgt(s', argmax_{a'} Q_online(s'))(a' 只在 s' 的合法选项上取)
Huber 损失; target 网络 Polyak 慢更新。评估 = evaluate_policy(greedy argmax)与主方法完全同口径。

用法(服务器): PYTHONPATH=. python scripts/38_train_dqn.py --iters 40 --run-tag base_dqn_s0 --resume
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter, collate_panels
from derbysim.learn.panel_models import PanelOptionPolicy, to_tensors
from derbysim.learn.ppo import TorchPanelDriver, masked_categorical
from derbysim.learn.rollout import as_factory, collect_rollout, evaluate_policy
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv


def make_eps_driver(policy, dev, eps_fn, rng):
    """ε-greedy 采样驱动: ε 随机合法选项, 否则 argmax Q。签名同 act_fn(enc)->(idx,logp,value)。"""
    greedy = TorchPanelDriver(policy, dev, greedy=True)

    def act(enc):
        if rng.random() < eps_fn():
            return int(rng.integers(enc["n_options"])), 0.0, 0.0
        i, lp, v = greedy(enc)
        return i, 0.0, 0.0
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--steps-per-iter", type=int, default=4096)
    ap.add_argument("--grad-steps", type=int, default=300, help="每 iter 的梯度步数")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=120000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-iters", type=int, default=20)
    ap.add_argument("--polyak", type=float, default=0.01)
    ap.add_argument("--encoder", default="gnn")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--control-classes", default="12")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-days", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-tag", default="base_dqn_s0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc, fv=args.feature_version))
    q = PanelOptionPolicy(args.encoder, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                          adapter.cancel_dim, adapter.train_dim, adapter.global_dim,
                          hidden=args.hidden)
    import copy
    q_tgt = copy.deepcopy(q)
    for p in q_tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    cc = None if args.control_classes in ("all", "") else set(args.control_classes)
    env_factory = lambda: RewardedDynamicEnv(net, params, ptc, dwell, proactive=True,
                                             reserve_window="wide", control_classes=cc)
    windows = operating_days("train")
    ev_wins = operating_days("val")[:args.eval_days]
    eval_eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
                for w in ev_wins]

    buf = deque(maxlen=args.buffer)          # (enc, a, r, next_enc, done)
    run_id = args.run_tag
    last_ck = config.ARTIFACTS_REPORTS / f"{run_id}_last.pt"
    best = {"reward": float("-inf"), "iter": -1}
    evals, history, start_iter = [], [], 0
    if args.resume and last_ck.exists():
        st = torch.load(last_ck, map_location=dev, weights_only=False)
        try:
            q.load_state_dict(st["model"]); q_tgt.load_state_dict(st["tgt"])
            opt.load_state_dict(st["opt"]); start_iter = st["next_iter"]
            best = st["best"]; evals = st.get("evals", []); history = st.get("history", [])
            print(f"[resume] it{start_iter} best it{best['iter']}")
        except Exception as e:
            print(f"[resume] 不兼容({str(e)[:80]}), 从头")
    print(f"[run] {run_id} dqn fv{args.feature_version} γ{args.gamma}")

    def eps_now(it):
        f = min(1.0, it / max(args.eps_decay_iters, 1))
        return args.eps_start + (args.eps_end - args.eps_start) * f

    def q_of(batch, acts, model):
        logits, _v, _aux = model(batch)
        return logits.gather(1, acts.view(-1, 1)).squeeze(1)

    t0 = time.time()
    for it in range(start_iter, args.iters):
        eps_val = eps_now(it)

        def episode_stream():
            while True:
                w = windows[int(rng.integers(len(windows)))]
                yield (str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
        drv = make_eps_driver(q, dev, lambda: eps_val, rng)
        trans, ep_m = collect_rollout(env_factory, adapter, as_factory(drv),
                                      episode_stream(), target_steps=args.steps_per_iter)
        for i, tr in enumerate(trans):
            nxt = trans[i + 1].enc if (i + 1 < len(trans) and not tr.done) else tr.enc
            buf.append((tr.enc, tr.option_idx, tr.reward, nxt, float(tr.done)))
        losses = []
        q.eval()
        for _g in range(args.grad_steps):
            if len(buf) < args.batch:
                break
            idx = rng.integers(0, len(buf), size=args.batch)
            enc_b = [buf[i][0] for i in idx]; a_b = torch.as_tensor([buf[i][1] for i in idx])
            r_b = torch.as_tensor([buf[i][2] for i in idx], dtype=torch.float32)
            nx_b = [buf[i][3] for i in idx]
            d_b = torch.as_tensor([buf[i][4] for i in idx], dtype=torch.float32)
            b = to_tensors(collate_panels(enc_b), dev)
            nb = to_tensors(collate_panels(nx_b), dev)
            with torch.no_grad():
                nl, _nv, _na = q(nb)                          # online 选动作(Double)
                nl = nl.masked_fill(nb["opt_mask"] < 0.5, float("-inf"))
                a_star = nl.argmax(dim=1)
                tq = q_of(nb, a_star, q_tgt)                  # target 评值
                y = r_b * 0.1 + args.gamma * (1.0 - d_b) * tq  # 奖励×0.1 与 IQL 同缩放
            qs = q_of(b, a_b, q)
            loss = F.smooth_l1_loss(qs, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for tp, lp in zip(q_tgt.parameters(), q.parameters()):
                    tp.data.mul_(1 - args.polyak).add_(args.polyak * lp.data)
            losses.append(float(loss.item()))
        comp = float(np.mean([m["completion_rate"] for m in ep_m])) if ep_m else 0.0
        row = {"iter": it, "steps": len(trans), "eps": eps_val,
               "loss": float(np.mean(losses)) if losses else 0.0, "train_completion": comp}
        history.append(row)
        print(f"it {it:>3} | steps {len(trans):>5} ε{eps_val:.2f} | loss {row['loss']:.4f} "
              f"| train完成 {comp:.3f} | buf {len(buf)}", flush=True)
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            ev = evaluate_policy(env_factory, adapter,
                                 as_factory(TorchPanelDriver(q, dev, greedy=True)), eval_eps)
            ev.pop("eval_episodes", None); ev["iter"] = it
            evals.append(ev)
            print(f"    >>> EVAL it{it}: val完成 {ev['eval_completion_rate']:.3f} "
                  f"| 延误 {ev['eval_weighted_secondary_delay_s']:.0f} "
                  f"| reward {ev['eval_episode_reward']:.1f}", flush=True)
            if ev["eval_episode_reward"] > best["reward"]:
                best = {"reward": ev["eval_episode_reward"], "iter": it,
                        "completion": ev["eval_completion_rate"],
                        "delay": ev["eval_weighted_secondary_delay_s"]}
                torch.save({"model": {k: v.cpu() for k, v in q.state_dict().items()},
                            "iter": it, "args": vars(args)},
                           config.ARTIFACTS_REPORTS / f"{run_id}_best.pt")
            torch.save({"model": {k: v.cpu() for k, v in q.state_dict().items()},
                        "tgt": {k: v.cpu() for k, v in q_tgt.state_dict().items()},
                        "opt": opt.state_dict(), "next_iter": it + 1, "best": best,
                        "evals": evals, "history": history, "args": vars(args)}, last_ck)
    (config.ARTIFACTS_REPORTS / f"{run_id}.json").write_text(
        json.dumps({"args": vars(args), "best": best, "evals": evals, "history": history},
                   ensure_ascii=False), encoding="utf-8")
    print(f"done {time.time()-t0:.0f}s | best it{best['iter']} reward {best['reward']:.1f}")


if __name__ == "__main__":
    main()
