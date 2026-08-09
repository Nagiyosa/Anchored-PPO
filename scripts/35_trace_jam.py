"""D65 诊断(只读, 不改任何代码/env): trace 一局 **set-everything(greedy)** 的淤死时序,
和信号员对照,看清 "过量 SET → 咽喉塞满(区内)→ 边界堆积(入区)" 的**时间先后**,
以决定 game-over 触发点该落在**区内塞满**还是**入区堆积**(见新对话争议)。

逐决策快照(读 `engine.trains`):
  · in_region     = entered 且未到达目标         → **咽喉占用**
  · set_pending   = pending_request(入区路QUEUED被占)且未 entered → **入区堆积/边界受阻**
  · awaiting      = awaiting_entry(到区界等放行决策)               → **边界候车**
  · blocked_in    = 区内 entered 且 blocked/held                    → 区内动不了
  · advance       = Σ pos(总物理推进)—— 停涨=淤死
  · cum_set       = 累计 SET 次数

**torch-free**, 本地可跑: PYTHONPATH=. python scripts/35_trace_jam.py [n_busiest_days] [maxdec]
产出: 打印时间线 + 关键时刻; 每 (策略,日) 存 artifacts/reports/trace/<policy>_<date>.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv, has_reached_goal


def _snap(engine) -> tuple:
    tr = engine.trains.values()
    in_region = sum(1 for t in tr if t.entered and not has_reached_goal(t))
    done = sum(1 for t in tr if has_reached_goal(t))
    awaiting = sum(1 for t in tr if t.awaiting_entry)
    set_pending = sum(1 for t in tr if (t.pending_request is not None) and not t.entered)
    blocked_in = sum(1 for t in tr if t.entered and not has_reached_goal(t)
                     and (t.blocked_since is not None or t.hold_start is not None))
    advance = sum(int(t.pos) for t in tr)
    return in_region, done, awaiting, set_pending, blocked_in, advance


def _trace(env, adapter, act, trains, horizon, maxdec) -> list:
    obs, _ = env.reset(trains, horizon=horizon)
    rows, cum_set, n = [], 0, 0
    while not env.done and n < maxdec:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        idx, _, _ = act(enc)
        kind = enc["slots"][idx].kind
        if kind == "set":
            cum_set += 1
        obs, _r, _done, _ = env.step(dec.options[idx])
        rows.append((round(env.engine.now, 1), kind, cum_set, *_snap(env.engine)))
        n += 1
    return rows


def _first_time(rows, pred):
    for r in rows:
        if pred(r):
            return r[0]
    return None


def _report(tag, rows, n_trains):
    if not rows:
        print(f"  [{tag}] 空局"); return
    # 列: now,kind,cum_set,in_region,done,awaiting,set_pending,blocked_in,advance
    T = rows[-1][0]
    peak_in = max(r[3] for r in rows)
    t_peak_in = _first_time(rows, lambda r: r[3] == peak_in)
    peak_pileup = max(r[6] + r[5] for r in rows)                 # set_pending + awaiting
    t_pileup = _first_time(rows, lambda r: (r[6] + r[5]) >= max(2, peak_pileup // 2))
    final_done = rows[-1][4]
    cum_set = rows[-1][2]
    # 近瞬 SET 爆发: 前 60s 内办了多少 SET(量化"一瞬间把预留全占了")
    t0 = rows[0][0]
    burst = max((r[2] for r in rows if r[0] <= t0 + 60.0), default=0)
    # 最后一次"总推进"上涨的时刻 → 之后即淤死停滞
    last_adv_t, prev = rows[0][0], rows[0][8]
    for r in rows[1:]:
        if r[8] > prev:
            last_adv_t, prev = r[0], r[8]
    print(f"  [{tag}] 决策 {len(rows)} | 累计SET {cum_set}(前60s内 {burst}) | 完成 {final_done}/{n_trains} | 地平线跨度~{T:.0f}s")
    print(f"       咽喉占用峰值 {peak_in} @ {t_peak_in:.0f}s | 边界堆积(pending+await)峰值 {peak_pileup} "
          f"起于~{('%.0f'%t_pileup+'s') if t_pileup is not None else 'NA'}")
    print(f"       最后一次推进 @ {last_adv_t:.0f}s → 其后停滞 {T-last_adv_t:.0f}s(停滞占全程 {100*(T-last_adv_t)/max(T,1):.0f}%)")
    # 时间线: 均匀取 ~14 个快照
    idxs = [int(k*(len(rows)-1)/13) for k in range(14)]
    print("       t(s)   in_reg  await  set_pend  blk_in  advance  cumSET")
    for i in idxs:
        r = rows[i]
        print(f"       {r[0]:>6.0f}  {r[3]:>5}  {r[5]:>5}  {r[6]:>7}  {r[7]:>5}  {r[8]:>7}  {r[2]:>6}")


def main(argv) -> int:
    n_days = int(argv[1]) if len(argv) > 1 else 3
    maxdec = int(argv[2]) if len(argv) > 2 else 20000       # 与评估一致, 才能跑到自然停滞
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    days = operating_days("train")
    # 选最忙的 n_days(淤死最明显); 只造车不跑仿真, 便宜
    counts = sorted(((i, len(build_day_trains(w, inject=False))) for i, w in enumerate(days)),
                    key=lambda x: -x[1])[:n_days]
    outdir = config.ARTIFACTS / "reports" / "trace"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[trace] 最忙 {n_days} 天(train) maxdec={maxdec} | 咽喉≈63 berth\n")

    for i, ntr in counts:
        w = days[i]
        date = str(w["start"])[:10]
        hz = day_horizon(w)
        print(f"===== {date}  ({ntr} 车) =====")
        for pol_name, mk in (("greedy(set-everything)", lambda e: make_greedy_set_policy()),
                             ("signaller(专家)", make_signaller_replay_policy)):
            env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
            rows = _trace(env, adapter, mk(env), build_day_trains(w, inject=False), hz, maxdec)
            _report(pol_name, rows, ntr)
            with (outdir / f"{pol_name.split('(')[0]}_{date}.csv").open("w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["now", "kind", "cum_set", "in_region", "done", "awaiting",
                             "set_pending", "blocked_in", "advance"])
                cw.writerows(rows)
            print()
    print(f"[trace] CSV -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
