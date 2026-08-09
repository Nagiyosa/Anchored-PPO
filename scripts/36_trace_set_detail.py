"""D65b 诊断(只读): 把 greedy(set-everything)在指定日的**前 N 次决策**逐条打出来——
每条选中的 option 是给**哪辆车、哪条进路、什么 kind、时钟走没走、该车入区没**,
以判定"冻结式反复预留"到底是 **1 辆车被无限 reserve-ahead** 还是 **多车入区冲突堆积**。

torch-free: PYTHONPATH=. python scripts/36_trace_set_detail.py [date] [N]
默认 date=2023-10-23(最忙、冻结态最典型), N=250
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv, has_reached_goal


def main(argv) -> int:
    date = argv[1] if len(argv) > 1 else "2023-10-23"
    N = int(argv[2]) if len(argv) > 2 else 250
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    w = next(w for w in operating_days("train") if str(w["start"])[:10] == date)
    trains = build_day_trains(w, inject=False)
    print(f"[detail] {date}  {len(trains)} 车  前 {N} 次决策\n")
    env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
    act = make_greedy_set_policy()
    obs, _ = env.reset(trains, horizon=day_horizon(w))
    t0 = None
    per_train = Counter()          # train_id -> 被 SET 次数
    per_route = Counter()          # route_id -> 次数
    entered_ids = set()
    rows = []
    n = 0
    while not env.done and n < N:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        idx, _, _ = act(enc)
        s = enc["slots"][idx]
        now = env.engine.now
        if t0 is None:
            t0 = now
        in_reg = sum(1 for t in env.engine.trains.values() if t.entered and not has_reached_goal(t))
        entered_now = {tid for tid, t in env.engine.trains.items() if t.entered}
        rows.append((round(now - t0, 1), s.kind, getattr(s, "train_id", "?"),
                     getattr(s, "route_id", "?"), in_reg, len(entered_now)))
        if s.kind == "set":
            per_train[getattr(s, "train_id", "?")] += 1
            per_route[getattr(s, "route_id", "?")] += 1
        obs, _r, _done, _ = env.step(dec.options[idx])
        n += 1

    for i, r in enumerate(rows):
        print(f"  #{i:>3} Δt={r[0]:>7.1f}s {r[1]:<6} train={str(r[2]):<10} route={str(r[3]):<14} "
              f"in_region={r[4]:>2} entered_total={r[5]:>2}")
    print(f"\n  === 汇总(前 {n} 次)===")
    print(f"  时钟跨度 Δt = {rows[-1][0]:.1f}s | SET 涉及 **{len(per_train)}** 辆不同车 / **{len(per_route)}** 条不同进路")
    print(f"  被 SET 最多的车 top5: {per_train.most_common(5)}")
    print(f"  被 SET 最多的进路 top5: {per_route.most_common(5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
