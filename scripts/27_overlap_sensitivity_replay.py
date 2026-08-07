"""让数据现在就说话:信号员回放对 overlap_depth ∈ {0,1,2} 的敏感性(C4 的可现在跑部分)。

动机(2026-08-06):第④/⑤块争论"proactive 提前锁 overlap 会不会多挡真实车流"。带学习策略的
proactive 闭环 C4 扫描要等新环境;但**信号员回放是有效的、没被作废的**(逐位确定、completion 1.0,
D46/D37),且**当前引擎回放时本就按 req_time 提前 ~147s 锁路** —— 故"提前占 overlap 挡别人"的效应
现在就在发生。把全部车按真实日志路径/时机跑,只调 overlap 深度,看完成率/加权延误/排队动多少:
  · 0/1/2 几乎不动 → overlap 锁多锁少不改故事,(a) 安全、无需上摆动 overlap (c);
  · 明显分歧 → overlap 要紧,该认真处理 (c)。
depth-0 那档 = "完全不锁 overlap"(≈ RR 式 / 用户直觉),depth-1 = 保守默认,depth-2 = 更保守。

用法(本地可跑 val;全量在服务器):
    PYTHONPATH=. python scripts/27_overlap_sensitivity_replay.py val
    PYTHONPATH=. python scripts/27_overlap_sensitivity_replay.py all      # 全运营日
产出: artifacts/reports/overlap_sensitivity_replay_<split>_<YYYYmmdd-HHMMSS>.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv

DEPTHS = (0, 1, 2)


def run_depth(depth, days, net, params, dwell, ptc, adapter):
    able = reach = wdelay = steps = ncol = ndead = ntrunc = 0
    per_day = []
    frac_wait_num = 0
    for day in days:
        env = RewardedDynamicEnv(net, params, ptc, dwell, overlap_depth=depth)
        res = run_episode(env, adapter, make_signaller_replay_policy(env),
                          build_day_trains(day, inject=False),
                          max_decisions=200000, horizon=day_horizon(day))
        m = res.metrics
        able += m["n_completable"]; reach += m["n_reached_exit"]
        wdelay += m["weighted_secondary_delay_s"]; steps += m["n_steps"]
        ncol += m["n_collisions"]; ndead += int(bool(m["deadlocked"]))
        ntrunc += int(bool(m.get("truncated", False)))
        frac_wait_num += m.get("frac_wait", 0.0) * m["n_steps"]
        per_day.append({"day": str(day["start"]), "completion": m["completion_rate"],
                        "wdelay_s": round(m["weighted_secondary_delay_s"], 1)})
    return {
        "overlap_depth": depth,
        "n_completable": able, "n_reached": reach,
        "completion_rate": round(reach / able, 6) if able else 1.0,
        "every_day_completion_one": all(d["completion"] == 1.0 for d in per_day),
        "weighted_secondary_delay_s": round(wdelay, 1),
        "delay_per_train_s": round(wdelay / able, 2) if able else 0.0,
        "n_steps": steps, "frac_wait": round(frac_wait_num / steps, 4) if steps else 0.0,
        "n_collisions": ncol, "n_deadlocks": ndead, "n_truncated": ntrunc,
        "per_day": per_day,
    }


def main(argv):
    split = argv[1] if len(argv) > 1 else "val"
    days = operating_days(split if split != "all" else "all")
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    t0 = time.time()
    results = [run_depth(d, days, net, params, dwell, ptc, adapter) for d in DEPTHS]
    dt = time.time() - t0

    base = next(r for r in results if r["overlap_depth"] == 1)
    for r in results:
        r["delay_vs_depth1_pct"] = (round(100 * (r["weighted_secondary_delay_s"]
                                    / base["weighted_secondary_delay_s"] - 1), 2)
                                    if base["weighted_secondary_delay_s"] else 0.0)
    rep = {
        "_meta": {
            "script": "scripts/27_overlap_sensitivity_replay.py",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": split, "n_days": len(days), "elapsed_s": round(dt, 1),
            "policy": "signaller_replay (valid, bit-deterministic; engine locks ~147s ahead at req_time)",
            "note": "depth-0 = no overlap held (~RR/user intuition); depth-1 = conservative default; depth-2 = more conservative",
        },
        "by_depth": results,
    }
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    outp = config.ARTIFACTS_REPORTS / f"overlap_sensitivity_replay_{split}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[overlap-sens {split}] {len(days)} days x depths {DEPTHS} in {dt:.1f}s")
    for r in results:
        print(f"  depth={r['overlap_depth']}: completion={r['completion_rate']} "
              f"all1={r['every_day_completion_one']} wdelay={r['weighted_secondary_delay_s']}s "
              f"({r['delay_vs_depth1_pct']:+}% vs d1) col={r['n_collisions']} dead={r['n_deadlocks']} trunc={r['n_truncated']}")
    print(f"  -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
