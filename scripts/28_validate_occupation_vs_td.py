"""对原始 TD 验占用: 比 reactive / proactive-narrow / proactive-wide 三种时机模型下,
信号员回放的仿真占用时刻 `sim_occ` 与**真实 TD 观测占用** `occ_first` 的误差。谁贴 TD 谁 faithful。

背景(2026-08-07 关键分支点): D57.1 试"宽窗+按 req_time 延后"faithful 锚,发现车堵在前沿、延误
比 narrow 高 25×(held_time 34×)。悬而未决:那多出的等待是**真** secondary delay(信号员车本就在
信号前等, narrow 提前锁**低估**)还是**假**(自由推进跑太快)?本脚本用地面真值裁定。

度量: 每 (车, TC) 对(仿真与真实都占过该 TC)算 err = sim_occ[tc] - occ_first[tc] (秒);
报每模式的 median|err| / MAE / median(bias) / p90|err| / p95|err| / n。误差最小者最 faithful。

用法(本地; wide 慢, 默认 3 天):
    PYTHONPATH=. python scripts/28_validate_occupation_vs_td.py [n_days]
产出: artifacts/reports/occupation_vs_td_<YYYYmmdd-HHMMSS>.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import (_cif_index, _td_events, day_horizon, load_assets,
                                      operating_days)
from derbysim.sim.reward import RewardedDynamicEnv

MODES = [("reactive", False, "narrow"),
         ("proactive_narrow", True, "narrow"),
         ("proactive_wide", True, "wide")]


def _build_trains_with_occ(day, net, dwell, ptc):
    """复制 build_clean_window_dyn_trains, 但保留 {train_id: occ_first}(真实 TD 首占)。"""
    from derbysim.sim.env import build_env_trains_for_window
    from derbysim.sim.dataset import _chain_continuity, split_broken_chains
    from derbysim.sim.reward import to_dyn_trains
    td = _td_events()
    win = td[(td["ts"] >= day["start"]) & (day["end"] > td["ts"])].copy()
    built = build_env_trains_for_window(win, net, dwell, ptc, cif_map=_cif_index())
    succ, pair = _chain_continuity(net)
    clean = split_broken_chains(built, net, succ, pair, ptc)      # [(et, occ_first), ...]
    occ_map = {et.train_id: dict(occ_first) for et, occ_first in clean}
    return to_dyn_trains(clean), occ_map


def _errors_for_mode(proactive, window, days, net, params, dwell, ptc, adapter):
    errs = []
    comp_reach = comp_able = 0
    for day in days:
        trains, occ_map = _build_trains_with_occ(day, net, dwell, ptc)
        env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=proactive,
                                 reserve_window=window)
        res = run_episode(env, adapter, make_signaller_replay_policy(env),
                          trains, max_decisions=500000, horizon=day_horizon(day))
        comp_reach += res.metrics["n_reached_exit"]; comp_able += res.metrics["n_completable"]
        for t in env.engine.trains.values():
            occ_first = occ_map.get(t.train_id, {})
            for tc, st in t.sim_occ.items():
                real = occ_first.get(tc)
                if real is not None:
                    errs.append(float(st) - float(real))
    a = np.asarray(errs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median_abs_s": round(float(np.median(np.abs(a))), 2),
        "MAE_s": round(float(np.mean(np.abs(a))), 2),
        "median_bias_s": round(float(np.median(a)), 2),
        "p90_abs_s": round(float(np.percentile(np.abs(a), 90)), 1),
        "p95_abs_s": round(float(np.percentile(np.abs(a), 95)), 1),
        "frac_within_5s": round(float((np.abs(a) <= 5).mean()), 4),
        "frac_within_30s": round(float((np.abs(a) <= 30).mean()), 4),
        "completion_rate": round(comp_reach / comp_able, 6) if comp_able else 1.0,
    }


def main(argv):
    n_days = int(argv[1]) if len(argv) > 1 else 3
    days = operating_days("val")[:n_days]
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    t0 = time.time()
    out = {}
    for name, pro, win in MODES:
        out[name] = _errors_for_mode(pro, win, days, net, params, dwell, ptc, adapter)
        m = out[name]
        print(f"{name:18s} n={m.get('n',0):7d} median|err|={m.get('median_abs_s'):>7}s "
              f"MAE={m.get('MAE_s'):>8}s bias={m.get('median_bias_s'):>8}s "
              f"within5s={m.get('frac_within_5s')} comp={m.get('completion_rate')}")
    rep = {
        "_meta": {"script": "scripts/28_validate_occupation_vs_td.py",
                  "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "n_days": len(days), "elapsed_s": round(time.time() - t0, 1),
                  "metric": "err = sim_occ[tc] - real TD occ_first[tc] (s), over (train,TC) both occupied",
                  "note": "faithful anchor = lowest |err| vs raw TD occupation"},
        "by_mode": out,
    }
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    outp = config.ARTIFACTS_REPORTS / f"occupation_vs_td_{time.strftime('%Y%m%d-%H%M%S')}.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
