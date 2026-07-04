"""M4 入口: 在1小时窗口回放真实决策, 检验引擎+联锁(无崩溃/无双占/顺序一致).

运行: python scripts/07_replay_smoke.py [YYYY-MM-DD HH]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.engine import Engine
from derbysim.sim.occupation import load_params

WINDOW_START = "2023-09-04 07:00:00"
WINDOW_END = "2023-09-04 08:00:00"


def build_requests(td, net):
    """从 TD 构造回放请求: t_req=Panel_Request时刻, t_entry=首TC实际占用时刻."""
    track = td[(td["type"] == "Track") & (td["state"] == "1")][["ts", "trainid_filled", "id"]].copy()
    track["id"] = track["id"].str.strip()
    tr_by = defaultdict(list)
    for x in track.itertuples():
        tr_by[x.trainid_filled].append((x.ts, x.id))
    for hc in tr_by:
        tr_by[hc].sort()

    pr = td[(td["type"] == "Panel_Request") & (td["state"] == "1")].sort_values("ts")
    reqs = []
    for x in pr.itertuples():
        r = net.routes.get(x.id.strip())
        if not r or not r.tc_sequence:
            continue
        t_req = x.ts.timestamp()
        first = [ts for ts, tc in tr_by.get(x.trainid_filled, [])
                 if tc == r.tc_sequence[0] and x.ts - pd.Timedelta(seconds=60) <= ts <= x.ts + pd.Timedelta(seconds=400)]
        t_entry = min(first).timestamp() if first else t_req + 137.0
        reqs.append({"train": x.trainid_filled, "route_id": x.id.strip(),
                     "t_req": t_req, "t_entry": t_entry})
    return reqs, tr_by


def main() -> int:
    start = sys.argv[1] + " " + sys.argv[2] + ":00:00" if len(sys.argv) > 2 else WINDOW_START
    end = (pd.to_datetime(start) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    win = td[(td["ts"] >= start) & (td["ts"] < end)].copy()

    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "tc_params.json")

    reqs, tr_by = build_requests(win, net)
    res = Engine(net, params).run(reqs)

    # 顺序一致性: 每车 sim占用TC顺序 vs 日志占用TC顺序
    sim_by_train = defaultdict(list)
    for o in sorted(res.sim_occupations, key=lambda o: o.t_occupied):
        sim_by_train[o.train].append(o.tc)
    order_ok = order_tot = 0
    for hc, sim_seq in sim_by_train.items():
        log_seq = [tc for ts, tc in tr_by.get(hc, [])
                   if pd.to_datetime(start) <= ts < pd.to_datetime(end)]
        sset = set(sim_seq)
        log_f = [tc for tc in log_seq if tc in sset]
        seen = set(); log_dedup = [t for t in log_f if not (t in seen or seen.add(t))]
        seen = set(); sim_dedup = [t for t in sim_seq if not (t in seen or seen.add(t))]
        it = iter(sim_dedup)
        if all(any(o == s for o in it) for s in log_dedup):
            order_ok += 1
        order_tot += 1

    report = {
        "window": f"{start} ~ {end}",
        "n_requests": res.n_requests,
        "n_locked": res.n_locked,
        "n_deferred": res.n_deferred,
        "persistent_conflicts": len(res.persistent_conflicts),
        "interlocking_violations": len(res.interlocking_violations),
        "n_sim_occupations": len(res.sim_occupations),
        "trains_order_consistent": f"{order_ok}/{order_tot}",
    }
    print("=" * 60)
    print("回放冒烟 (M4)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:26} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "replay_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = report["interlocking_violations"] == 0
    print("\n验收:", "PASS ✅" if ok else "CHECK ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
