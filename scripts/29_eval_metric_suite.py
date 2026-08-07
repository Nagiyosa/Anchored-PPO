"""D58 评估指标套: 在 narrow proactive 环境上跑策略, 报铁路调度通用指标, 按 λ 优先级分层。

指标(D58):
  headline = 加权次生延误(λ,D16) · 完成率 · 准点率(PPM 式: 区内次生延误 ≤ T 的完成车占比) · 非计划停车数
  辅助     = 延误长尾(p50/p90/p95/max, 单车次生延误) · 站台遵从率 · 吞吐(车/小时) · 总等待
  (占用率 UIC406 需按 TC 记预留区间、要动引擎, 标记待补, 见 _meta.todo)

**评估纪律**: 信号员是**参照基线非靶子**(D58); 拟合型模型(BC/lookup/RL)看 held-out(D40/D43),
本脚本默认 val(参照/冒烟), test 全程不碰。reward 用 per_train(D58) 但指标本身与 reward 无关。

用法: PYTHONPATH=. python scripts/29_eval_metric_suite.py [val|train|test] [n_days]
产出: artifacts/reports/eval_metric_suite_<split>_<YYYYmmdd-HHMMSS>.json
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import (make_greedy_set_policy, make_signaller_replay_policy,
                                     run_episode)
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv, can_complete, has_reached_goal

LAM = RewardConfig()          # 复用 λ 权重(D16)
PPM_THRESH_S = (60.0, 180.0, 300.0)


def _pctl(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def _summarise(trains_by_day, horizons):
    """trains_by_day: list[list[DynTrain]]; horizons: list[float](各日地平线)。"""
    rows = []       # 每完成/可完成车一条
    for trains in trains_by_day:
        for t in trains:
            if not can_complete(t):
                continue
            done_ok = bool(t.done and has_reached_goal(t))
            pr = getattr(t, "planned_ref", None)
            # RR 式绝对到达晚点(D58): 实际到达/通过(max sim_occ) − 计划(planned_ref, 已 coalesce arr/dep/pass)。
            arr_delay = ((max(t.sim_occ.values()) - pr) if (done_ok and pr is not None and t.sim_occ)
                         else None)
            rows.append({
                "cls": (t.service_class or "?")[:1],
                "done": done_ok,
                "has_plan": pr is not None,             # PPM 分母 = 有计划的应完成车(搁浅计不准点)
                "arr_delay": arr_delay,                 # 绝对晚点(vs 计划); None=未到站/无计划
                "sec_delay": float(getattr(t, "held_time", 0.0)),   # 区内次生延误(秒, D23/D24 价值锚)
                "stops": int(getattr(t, "n_stops", 0)),
                "booked": getattr(t, "booked_platform", None),
                "served": set(getattr(t, "served_platforms", set()) or set()),
            })
    n = len(rows)
    done = [r for r in rows if r["done"]]
    total_h = sum((h - min(h - 3600 * 24, h)) for h in horizons)  # 名义: 每日窗口(占位, 见吞吐口径)

    def block(rs):
        if not rs:
            return {"n": 0}
        sd = np.array([r["sec_delay"] for r in rs], float)
        sd_done = np.array([r["sec_delay"] for r in rs if r["done"]], float)
        wsum = sum(LAM.lam(r["cls"]) * r["sec_delay"] for r in rs)
        n_done = sum(r["done"] for r in rs)
        plat = [r for r in rs if r["done"] and r["booked"] is not None]
        plat_ok = sum(1 for r in plat if r["booked"] in r["served"])
        out = {
            "n_completable": len(rs),
            "n_done": int(n_done),
            "completion_rate": round(n_done / len(rs), 4),
            "weighted_secondary_delay_s": round(wsum, 1),
            "unscheduled_stops_total": int(sum(r["stops"] for r in rs)),
            "stops_per_train_mean": round(float(np.mean([r["stops"] for r in rs])), 3),
            "sec_delay_p50_s": _pctl(sd_done, 50), "sec_delay_p90_s": _pctl(sd_done, 90),
            "sec_delay_p95_s": _pctl(sd_done, 95),
            "sec_delay_max_s": round(float(sd_done.max()), 1) if len(sd_done) else None,
            "total_held_time_s": round(float(sd.sum()), 1),
            "platform_adherence": (round(plat_ok / len(plat), 4) if plat else None),
        }
        # PPM(D58, RR 式绝对): 准点到达 / **所有有计划的应完成车**(搁浅=不准点); 非以信号员为标准。
        ppm_denom = sum(1 for r in rs if r["has_plan"])
        arr = [r["arr_delay"] for r in rs if r["arr_delay"] is not None]
        for T in PPM_THRESH_S:
            ok = sum(1 for r in rs if r["arr_delay"] is not None and r["arr_delay"] <= T)
            out[f"ppm_le_{int(T)}s"] = round(ok / ppm_denom, 4) if ppm_denom else None
        out["arr_delay_p50_s"] = _pctl(arr, 50); out["arr_delay_p90_s"] = _pctl(arr, 90)
        out["arr_delay_p95_s"] = _pctl(arr, 95)
        return out

    res = {"overall": block(rows)}
    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    res["by_class"] = {c: block(rs) for c, rs in sorted(by.items())}
    res["throughput_trains_per_h"] = round(len(done) / (total_h / 3600.0), 2) if total_h else None
    return res


def eval_policy(name, make_policy, days, net, params, dwell, ptc, adapter):
    trains_by_day, horizons = [], []
    for day in days:
        env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="narrow")
        run_episode(env, adapter, make_policy(env), build_day_trains(day, inject=False),
                    max_decisions=300000, horizon=day_horizon(day))
        trains_by_day.append(list(env.engine.trains.values()))
        horizons.append(day_horizon(day))
    return _summarise(trains_by_day, horizons)


def main(argv):
    split = argv[1] if len(argv) > 1 else "val"
    n_days = int(argv[2]) if len(argv) > 2 else 999
    days = operating_days(split)[:n_days]
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    policies = {"signaller": make_signaller_replay_policy,
                "greedy": lambda e: make_greedy_set_policy()}
    t0 = time.time()
    out = {}
    for name, mk in policies.items():
        out[name] = eval_policy(name, mk, days, net, params, dwell, ptc, adapter)
        o = out[name]["overall"]
        print(f"{name:10s} comp={o['completion_rate']} wdelay={o['weighted_secondary_delay_s']}s "
              f"ppm300={o.get('ppm_le_300s')} stops={o['unscheduled_stops_total']} "
              f"platadh={o['platform_adherence']}")
    rep = {"_meta": {"script": "scripts/29_eval_metric_suite.py",
                     "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": split,
                     "n_days": len(days), "env": "proactive narrow", "elapsed_s": round(time.time() - t0, 1)},
           "by_policy": out}
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    outp = config.ARTIFACTS_REPORTS / f"eval_metric_suite_{split}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
