#!/usr/bin/env python3
"""M12 瓶颈诊断: 到底是"何时办(时机/排序)"卡住了, 还是环境容量被建模得过紧?

**问题背景(docs/M12_DESIGN §17.6)**: 全部自动策略都卡在 completion 0.14-0.31, 而信号员是 1.000。
分时段送达率显示典型形态是 **开局尚可、随后单调衰减、永不恢复**(有路就办: 0.83→0.24), 即
**区域被逐渐淤死**。而零学习查表以 90.6% 的准确率复制了人的选路, 闭环反而比"随便挑第一条"更差
⇒ **选路不是瓶颈**。

本脚本跑两组对照, 各自单独可跑:

  A) overlap 敏感性 —— 环境容量是否被建模得过紧(M10 就记过"M9 自由时序+overlap 让绝对延误膨胀~4×")。
     若 overlap_depth=0 让 completion 大幅上升, 那是**必须写进 limitations 的建模约束**, 且立刻
     改善所有结果。

  B) **时机/排序**对照 —— 这正是 D24/Path A 定义的"何时办、是否办"那一半, 也正是 **BC 按定义
     完全不覆盖**的那一半(M11 §0: "BC 学的是办哪条, 不是何时办")。把决策拆成两个正交的轴:
        · 选路轴: first(面板第一条) / lookup(零学习查表, 与人选路一致率 90.6%)
        · 时机轴: first(面板顺序) / fifo(先到先服务) / most_delayed(最晚点优先)
                  / priority(TOPS 优先级) / fewest_remaining(剩余路径最短优先)
     两轴交叉 ⇒ 直接读出"闭环质量的方差由哪个轴解释"。若时机轴的跨度远大于选路轴, 就坐实了
     M12 的价值全在时机上 —— 那也正是 RL 相对 BC 的唯一增量。

用法:
    python scripts/17_diagnose_bottleneck.py overlap [--days 4]
    python scripts/17_diagnose_bottleneck.py timing  [--days 4]
    python scripts/17_diagnose_bottleneck.py both    [--days 4]

产出: artifacts/reports/diag_<mode>_<时间戳>.json (每一版都留下来) + 屏幕表格。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from derbysim import config
from derbysim.learn.features import Featurizer, fit_route_prior
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (as_factory, make_greedy_set_policy, run_episode,
                                    signaller_replay_factory)
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv, can_complete, has_reached_goal

N_BINS = 6              # 分时段送达率的段数(按入区时刻等分)


# ---------------------------------------------------------------- 选路轴 / 时机轴
def _route_prior():
    """零学习查表: 训练段上的 P(被选 | 信号, 进路)。**只用 train 段拟合, 无泄露。**"""
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all_d31.parquet")
    tr = df[df["split"] == "train"]
    pr = fit_route_prior([{"signal": r["signal"], "legal_route_ids": r["legal_route_ids"],
                           "chosen_route": r["chosen_route"], "n_legal": r["n_legal"]}
                          for r in tr.to_dict("records")])
    sig_of = {rid: sg for (sg, rid) in pr}
    return pr, sig_of


def make_policy(env, route_axis: str, timing_axis: str, prior=None, sig_of=None):
    """决策 = 先按**时机轴**挑一辆车, 再按**选路轴**挑它的一条进路。

    两轴都只用**环境已提供的选项**(D6: 安全不由策略负责)。没有 set 选项时一律 WAIT ——
    所有策略在这一点上相同, 故差异纯粹来自"服务谁 / 走哪条"。
    """
    def score_train(t):
        if t is None:
            return 0.0
        if timing_axis == "fifo":
            return -float(t.entry_time)                       # 越早到越优先
        if timing_axis == "most_delayed":
            ref = getattr(t, "planned_ref", None)
            return float(env.engine.now - ref) if ref else 0.0
        if timing_axis == "priority":
            c = str(getattr(t, "service_class", "") or "9")
            return -float(c) if c.isdigit() else -9.0         # TOPS 1=最快 → 分最高
        if timing_axis == "fewest_remaining":
            return -float(max(len(t.route_chain) - len(t.actual_chain), 0))  # 临时参考深度；D33 后重审
        return 0.0                                            # "first" = 面板顺序

    def score_route(slot):
        if route_axis == "lookup":
            return float(prior.get((sig_of.get(slot.route_id), slot.route_id), 0.5))
        return 0.0                                            # "first" = 面板顺序

    def act(enc):
        best, best_key = None, None
        for i, s in enumerate(enc["slots"]):
            if s.kind != "set":
                continue
            t = env.engine.trains.get(s.train_id)
            key = (round(score_train(t), 6), round(score_route(s), 6), -i)   # -i: 稳定取先出现的
            if best_key is None or key > best_key:
                best, best_key = i, key
        return (best, 0.0, 0.0) if best is not None else (0, 0.0, 0.0)
    return act


# ---------------------------------------------------------------- 跑一组
def run_one(fac, days, overlap_depth=1, max_decisions=200000):
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    tot_r = tot_n = 0
    tot_delay = 0.0
    bins = defaultdict(lambda: [0, 0])
    for d in days:
        env = RewardedDynamicEnv(net, params, ptc, dwell, overlap_depth=overlap_depth)
        m = run_episode(env, ad, fac(env), build_day_trains(d),
                        horizon=day_horizon(d), max_decisions=max_decisions).metrics
        tot_r += m["n_reached_exit"]; tot_n += m["n_completable"]
        tot_delay += m["weighted_secondary_delay_s"]
        able = [t for t in env.engine.trains.values() if can_complete(t)]
        if able:
            t0 = min(t.entry_time for t in able); t1 = max(t.entry_time for t in able)
            for t in able:
                b = min(int(N_BINS * (t.entry_time - t0) / max(t1 - t0, 1.0)), N_BINS - 1)
                bins[b][1] += 1; bins[b][0] += bool(has_reached_goal(t))
    prof = [round(bins[b][0] / max(bins[b][1], 1), 2) for b in range(N_BINS)]
    return {"completion": tot_r / max(tot_n, 1), "n_reached": tot_r, "n_completable": tot_n,
            "delay": tot_delay, "profile": prof}


def _show(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'配置':<34}{'completion':>12}{'延误':>14}   分时段送达率(按入区时刻 6 等分)")
    for name, r in rows:
        print(f"{name:<34}{r['completion']:>12.3f}{r['delay']:>14.0f}   "
              + " ".join(f"{x:.2f}" for x in r["profile"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["overlap", "timing", "both"])
    ap.add_argument("--days", type=int, default=4, help="用几个 val 运营日(默认 4; 全量 11)")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    args = ap.parse_args()
    days = operating_days(args.split)[:args.days]
    print(f"[diag] {len(days)} 个 {args.split} 运营日, 真实历史不注入(D28)")
    out = {"mode": args.mode, "days": [str(d["start"])[:10] for d in days], "results": {}}
    t0 = time.time()

    if args.mode in ("overlap", "both"):
        rows = []
        for od in (0, 1, 2):
            for tag, fac in (("信号员", signaller_replay_factory),
                             ("有路就办", as_factory(make_greedy_set_policy()))):
                r = run_one(fac, days, overlap_depth=od)
                name = f"overlap={od}  {tag}"
                rows.append((name, r)); out["results"][name] = r
                print(f"  ...{name}: {r['completion']:.3f}")
        _show(rows, "A) overlap 敏感性 —— 环境容量是否被建模得过紧")
        print("读法: 若 overlap=0 让'有路就办'大幅上升, 说明区域容量被系统性低估, "
              "这是必须写进 limitations 的建模约束(默认 overlap=1 是 M9 的既定口径)。")

    if args.mode in ("timing", "both"):
        prior, sig_of = _route_prior()
        rows = [("信号员(参照)", run_one(signaller_replay_factory, days))]
        for timing in ("first", "fifo", "most_delayed", "priority", "fewest_remaining"):
            for route in ("first", "lookup"):
                fac = (lambda env, tm=timing, ro=route:
                       make_policy(env, ro, tm, prior, sig_of))
                r = run_one(fac, days)
                name = f"时机={timing:<16} 选路={route}"
                rows.append((name, r)); out["results"][name] = r
                print(f"  ...{name}: {r['completion']:.3f}")
        _show(rows, "B) 时机(何时办/服务谁) × 选路 —— 闭环质量的方差由哪个轴解释")
        cs = {n: r["completion"] for n, r in rows if n.startswith("时机")}
        by_t = defaultdict(list); by_r = defaultdict(list)
        for n, c in cs.items():
            by_t[n.split("时机=")[1].split()[0]].append(c)
            by_r[n.split("选路=")[1]].append(c)
        sp = lambda d: max(max(v) - min(v) for v in [[x for vs in d.values() for x in vs]])  # noqa
        t_sp = max(sum(v) / len(v) for v in by_t.values()) - min(sum(v) / len(v) for v in by_t.values())
        r_sp = max(sum(v) / len(v) for v in by_r.values()) - min(sum(v) / len(v) for v in by_r.values())
        print(f"\n**主效应**: 时机轴跨度 {t_sp:.3f} | 选路轴跨度 {r_sp:.3f}")
        print("读法: 时机轴跨度 >> 选路轴跨度 ⇒ 坐实'何时办'才是瓶颈 —— 而那正是 D24/Path A 的"
              "核心, 也正是 BC 按定义完全不覆盖、只有 RL 能学的那一半(M11 §0)。")

    out["elapsed_s"] = time.time() - t0
    p = config.ARTIFACTS_REPORTS / f"diag_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {p}  ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
