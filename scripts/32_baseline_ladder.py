"""⑧ 基线阶梯(D26/D30)在 **新 proactive narrow 环境**上重做 —— "BC/RL 必须打败的零学习参照"。

规则基线免训练、不做策略性 WAIT(有可办就办, baselines.py 口径), 只差选路/服务顺序; 安全通过由
mask∩可达∩in-scope(D41)结构性保证。lookup 表的"逐信号最频路"是**基础设施数据事实**(与环境无关,
表有效); 只是它的**闭环性能**须在新环境重测(旧 +8.7% 是 reactive/D41 假象下的, D47 作废)。

报 headline(D58): 完成率 · 加权次生延误 · PPM≤300s · 非计划停车数 · 0 碰撞/死锁(安全)。
**信号员是参照非靶子**; 拟合型(lookup)看 held-out(D40): lookup 表用 train 段拟合。

用法: PYTHONPATH=. python scripts/32_baseline_ladder.py [val|test] [n_days]
产出: artifacts/reports/baseline_ladder_<split>_<YYYYmmdd-HHMMSS>.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.baselines import (lookup_table, make_fcfs_policy, make_fsfs_policy,
                                      make_lookup_policy)
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.redteam import random_factory
from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days


def eval_policy(make_policy, days, net, params, dwell, ptc, adapter):
    from derbysim.sim.reward import RewardedDynamicEnv
    able = reach = wd = stops = col = dead = ppm_denom = 0
    ppm_ok = {60: 0, 180: 0, 300: 0}
    for day in days:
        env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="narrow")
        run_episode(env, adapter, make_policy(env), build_day_trains(day, inject=False),
                    max_decisions=300000, horizon=day_horizon(day))
        m = run_metrics(env)
        able += m["able"]; reach += m["reach"]; wd += m["wd"]; stops += m["stops"]
        col += m["col"]; dead += m["dead"]; ppm_denom += m["ppm_denom"]
        for T in ppm_ok:
            ppm_ok[T] += m["ppm"][T]
    return {
        "completion_rate": round(reach / able, 4) if able else 1.0,
        "weighted_secondary_delay_s": round(wd, 1),
        "unscheduled_stops": int(stops),
        # PPM = **对 CIF 计划的绝对准点**(D58): 完成车中"实际到达/通过 − planned_ref ≤ T"占比。
        # planned_ref 已 coalesce(arr→dep→pass); 实际取 max(sim_occ)=区内末占用≈到站/通过点。
        # **非以信号员为标准**; 分母=有 planned_ref 的完成车。信号员因外生 primary 晚点, PPM<1。
        "ppm_le_60s": round(ppm_ok[60] / ppm_denom, 4) if ppm_denom else None,
        "ppm_le_300s": round(ppm_ok[300] / ppm_denom, 4) if ppm_denom else None,
        "n_collisions": int(col), "n_deadlocks": int(dead),
    }


def run_metrics(env):
    from derbysim.sim.reward import realized_weighted_delay, has_reached_goal, can_complete
    eng = env.engine
    able = [t for t in eng.trains.values() if can_complete(t)]
    reach = [t for t in able if t.done and has_reached_goal(t)]
    # PPM 分母 = **所有应完成车**(有 planned_ref); 搁浅/没到站 = 不准点(RR: 没到站没分)。
    able_pr = [t for t in able if getattr(t, "planned_ref", None) is not None]
    def arrival_delay(t):                             # 实际到达/通过 − 计划(RR 式绝对晚点)
        return max(t.sim_occ.values()) - t.planned_ref
    ppm = {T: sum(1 for t in reach if getattr(t, "planned_ref", None) is not None
                  and t.sim_occ and arrival_delay(t) <= T) for T in (60, 180, 300)}
    return {"able": len(able), "reach": len(reach), "ppm_denom": len(able_pr),
            "wd": realized_weighted_delay(eng, env.cfg, horizon=env.horizon),
            "stops": sum(getattr(t, "n_stops", 0) for t in eng.trains.values()),
            "col": len(eng.collisions), "dead": int(bool(eng.deadlocked)), "ppm": ppm}


def main(argv):
    split = argv[1] if len(argv) > 1 else "val"
    n_days = int(argv[2]) if len(argv) > 2 else 999   # 默认全量该 split(之前误设 4, 只跑了 4 天)
    days = operating_days(split)[:n_days]
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    top = lookup_table("train")                       # held-out 纪律: 只用 train 段拟合
    ladder = {
        "signaller": make_signaller_replay_policy,    # 参照(非靶子)
        "lookup": make_lookup_policy(top),            # 最强安全基线
        "fcfs": make_fcfs_policy(top),
        "fsfs": make_fsfs_policy(top),
        "greedy": lambda e: make_greedy_set_policy(),
        "random_legal": random_factory(0),
    }
    t0 = time.time(); out = {}
    for name, mk in ladder.items():
        out[name] = eval_policy(mk, days, net, params, dwell, ptc, adapter)
        o = out[name]
        print(f"{name:13s} comp={o['completion_rate']:.4f} wdelay={o['weighted_secondary_delay_s']:>10} "
              f"ppm300={o['ppm_le_300s']} stops={o['unscheduled_stops']:>5} col={o['n_collisions']} dead={o['n_deadlocks']}")
    rep = {"_meta": {"script": "scripts/32_baseline_ladder.py", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "split": split, "n_days": len(days), "env": "proactive narrow",
                     "elapsed_s": round(time.time() - t0, 1),
                     "note": "signaller=参照非靶子; lookup 表 held-out(train)拟合; 闭环性能在新环境重测(旧+8.7%已作废 D47)"},
           "ladder": out}
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    outp = config.ARTIFACTS_REPORTS / f"baseline_ladder_{split}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
