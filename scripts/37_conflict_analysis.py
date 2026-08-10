"""D74 收尾①: 冲突归因分析 —— 把「RL 比 fcfs 延误低 32%」拆成「赢在哪里」。

对同一批日子确定性重放 RL(best ckpt, greedy) 和 fcfs, **逐车配对**比较, 输出:
  1) **按拥挤度分箱的延误差**(核心图): 每辆车的拥挤度 = 它在场期间与多少辆受控车共存(fcfs 口径,
     策略无关); RL−fcfs 的加权延误差按拥挤度分箱求和。预期: 低拥挤箱 ≈0, 差距集中在高拥挤箱
     ⇒ 实证「优势在冲突时刻」。
  2) **放行顺序偏离率**: 受控车按到达序 vs 实际入区序的逆序对占比(RL vs fcfs 对照) ——
     偏离先来先到 + 偏离与「车种权重×等待」相关 = 涌现加权优先调度。
  3) **Top-10 案例**: 单车延误差最大的车(日/车次/车种/拥挤度), 供逐帧剖析。
  4) 逐日汇总表 + json + png。

用法(服务器, 需 torch):
  PYTHONPATH=. python scripts/37_conflict_analysis.py --ckpt artifacts/reports/ppo_d73_AB_s0_best.pt \
      [--split val] [--days 999] [--with-signaller]
产出: artifacts/reports/conflict_analysis_<split>.{json,png} (不覆盖训练产物)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from derbysim import config
from derbysim.learn.baselines import lookup_table, make_fcfs_policy
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.panel_models import PanelOptionPolicy
from derbysim.learn.ppo import RunningMoments, TorchPanelDriver
from derbysim.learn.rollout import make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv, can_complete, has_reached_goal


def collect(env, adapter, act_factory, trains, horizon):
    """跑一局, 抽逐车记录(受控可计分车)。"""
    run_episode(env, adapter, act_factory(env), trains, max_decisions=20000, horizon=horizon)
    eng = env.engine
    cfg = RewardConfig()
    rows = {}
    for t in eng.trains.values():
        if t.background or not can_complete(t):
            continue
        occ = list(getattr(t, "sim_occ", {}).values())
        rows[t.train_id] = {
            "cls": str(t.service_class),
            "entry": float(t.entry_time),
            "entered": (min(occ) if occ else None),
            "exit": (max(occ) if occ else None),
            "wdelay": cfg.lam(t.service_class) * float(getattr(t, "held_time", 0.0)),
            "stops": int(getattr(t, "n_stops", 0)),
            "done": bool(has_reached_goal(t)),
        }
    return rows


def inversions(order_pairs):
    """(到达序, 入区序) 的逆序对占比。"""
    xs = [b for (_, b) in sorted(order_pairs)]
    n = len(xs)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if xs[i] > xs[j])
    return inv / (n * (n - 1) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--days", type=int, default=999)
    ap.add_argument("--with-signaller", action="store_true")
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = st.get("args", {})
    fv = int(ck_args.get("feature_version", 1))
    enc_name = ck_args.get("encoder", "gnn")
    cc = ck_args.get("control_classes", "12")
    cc_set = None if cc in ("all", "") else set(cc)
    print(f"[ckpt] {args.ckpt}: encoder={enc_name} fv={fv} control={cc} (it{st.get('iter','?')})")

    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc, fv=fv))
    policy = PanelOptionPolicy(enc_name, adapter.dec_dim, adapter.conf_dim, adapter.cand_dim,
                               adapter.cancel_dim, adapter.train_dim, adapter.global_dim,
                               hidden=int(ck_args.get("hidden", 64)))
    policy.load_state_dict(st["model"])
    policy.eval()
    ret_norm = RunningMoments() if st.get("ret_norm") else None
    if ret_norm is not None:
        ret_norm.load_state_dict(st["ret_norm"])
    dev = torch.device("cpu")
    rl_factory = lambda e: TorchPanelDriver(policy, dev, greedy=True, ret_norm=ret_norm)
    fcfs_factory = make_fcfs_policy(lookup_table())

    def env_factory():
        return RewardedDynamicEnv(net, params, ptc, dwell, proactive=True,
                                  reserve_window="wide", control_classes=cc_set)

    days = operating_days(args.split)[:args.days]
    per_train = []          # 配对记录
    day_rows = []
    inv_rl, inv_fc = [], []
    for w in days:
        date = str(w["start"])[:10]
        hz = day_horizon(w)
        rl = collect(env_factory(), adapter, rl_factory, build_day_trains(w, inject=False), hz)
        fc = collect(env_factory(), adapter, fcfs_factory, build_day_trains(w, inject=False), hz)
        if args.with_signaller:
            sg = collect(env_factory(), adapter, make_signaller_replay_policy,
                         build_day_trains(w, inject=False), hz)
        # 拥挤度(fcfs 口径, 策略无关): 该车在场区间与多少辆受控车区间重叠
        iv = {tid: (r["entry"], r["exit"] if r["exit"] is not None else r["entry"])
              for tid, r in fc.items()}
        for tid in rl:
            if tid not in fc:
                continue
            a0, a1 = iv.get(tid, (rl[tid]["entry"], rl[tid]["entry"]))
            cont = sum(1 for u, (b0, b1) in iv.items()
                       if u != tid and not (b1 < a0 or b0 > a1))
            per_train.append({
                "day": date, "train": tid, "cls": rl[tid]["cls"], "contention": cont,
                "rl_delay": rl[tid]["wdelay"], "fcfs_delay": fc[tid]["wdelay"],
                "d": rl[tid]["wdelay"] - fc[tid]["wdelay"],
                "rl_done": rl[tid]["done"], "fcfs_done": fc[tid]["done"],
                **({"sig_delay": sg[tid]["wdelay"]} if args.with_signaller and tid in sg else {}),
            })
        # 顺序偏离率(仅入区时间可得的车)
        def pairs(rows):
            ent = [(r["entry"], r["entered"]) for r in rows.values() if r["entered"] is not None]
            arrival_rank = {e: i for i, (e, _) in enumerate(sorted(ent))}
            entered_sorted = sorted(range(len(ent)), key=lambda k: ent[k][1])
            return [(arrival_rank[ent[k][0]], j) for j, k in enumerate(entered_sorted)]
        inv_rl.append(inversions(pairs(rl))); inv_fc.append(inversions(pairs(fc)))
        rl_sum = sum(r["wdelay"] for r in rl.values()); fc_sum = sum(r["wdelay"] for r in fc.values())
        day_rows.append({"day": date, "n": len(rl), "rl": rl_sum, "fcfs": fc_sum, "d": rl_sum - fc_sum})
        print(f"  {date}: n={len(rl):>3} RL {rl_sum:>9.0f} vs fcfs {fc_sum:>9.0f} (Δ{rl_sum-fc_sum:>+9.0f}) "
              f"| 逆序率 RL {inv_rl[-1]:.2f} fcfs {inv_fc[-1]:.2f}", flush=True)

    # ---- 按拥挤度分箱 ----
    bins = [(0, 4), (5, 9), (10, 14), (15, 24), (25, 10 ** 9)]
    labels = ["0-4", "5-9", "10-14", "15-24", "25+"]
    tab = []
    for (lo, hi), lab in zip(bins, labels):
        sel = [r for r in per_train if lo <= r["contention"] <= hi]
        tab.append({"bin": lab, "n": len(sel),
                    "rl": sum(r["rl_delay"] for r in sel),
                    "fcfs": sum(r["fcfs_delay"] for r in sel),
                    "d": sum(r["d"] for r in sel)})
    print("\n== 按拥挤度分箱(该车在场期间共存的受控车数) ==")
    print(f"{'bin':<8}{'n':>6}{'RL延误':>12}{'fcfs延误':>12}{'Δ(RL−fcfs)':>14}")
    for r in tab:
        print(f"{r['bin']:<8}{r['n']:>6}{r['rl']:>12.0f}{r['fcfs']:>12.0f}{r['d']:>14.0f}")
    print(f"\n顺序偏离率(逆序对占比): RL {np.mean(inv_rl):.3f} vs fcfs {np.mean(inv_fc):.3f}")
    top = sorted(per_train, key=lambda r: r["d"])[:10]
    print("\n== Top-10 省延误案例(Δ最负 = RL 比 fcfs 省最多) ==")
    for r in top:
        print(f"  {r['day']} {r['train']:<10} cls{r['cls']} 拥挤{r['contention']:>3} Δ {r['d']:>9.0f} "
              f"(RL {r['rl_delay']:.0f} vs fcfs {r['fcfs_delay']:.0f})")

    out = config.ARTIFACTS_REPORTS / f"conflict_analysis_{args.split}.json"
    out.write_text(json.dumps({"per_train": per_train, "bins": tab, "days": day_rows,
                               "inv_rl": float(np.mean(inv_rl)), "inv_fcfs": float(np.mean(inv_fc)),
                               "ckpt": args.ckpt}, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(tab)); w = 0.38
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.bar(x - w / 2, [r["rl"] for r in tab], w, label="RL", color="#c0392b")
        ax.bar(x + w / 2, [r["fcfs"] for r in tab], w, label="FCFS", color="#1a7a3a", alpha=.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{r['bin']}\n(n={r['n']})" for r in tab], fontsize=8)
        ax.set_xlabel("contention (co-present controlled trains)"); ax.set_ylabel("weighted delay (s)")
        ax.set_title(f"Where the advantage lives ({args.split})"); ax.legend()
        plt.tight_layout(); plt.savefig(config.ARTIFACTS_REPORTS / f"conflict_analysis_{args.split}.png", dpi=130)
        print(f"\n[out] {out} + .png")
    except Exception as e:
        print("plot skip", e)


if __name__ == "__main__":
    main()
