"""回放验证驱动(M6): 留出集喂真实决策前推, 逐事件对比 TD 观测.

方法(brief §8②描述性有效): 对每条进路, 以**观测的首TC占用时刻为进入条件**,
用标定占用模型前推该route各TC占用时刻, 与观测逐TC对比(留出集=未用于标定).
platform TC 上的占用误差 = 到达时刻再现误差(→到发延误再现精度).
"""

from __future__ import annotations

import csv
from collections import defaultdict

import pandas as pd

from derbysim import config
from derbysim.sim.occupation import occupancy_intervals
from derbysim.validation.calibration import pair_route_occupations


def _platform_tcs() -> set[str]:
    out = set()
    if config.DATA_PLATFORM_TC.exists():
        with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("platform_id") or "").strip().isdigit():
                    out.add(row["tc_id"].strip())
    return out


def forward_predict(td, net, params, split):
    """前推留出集占用. 返回 (records, occ_durations_sim, occ_durations_obs).

    records: list of dict(tc, pred, obs, signed, is_platform).
    """
    plat_tcs = _platform_tcs()
    records = []
    dur_sim, dur_obs = [], []
    for rid, train, occ, rel, seq in pair_route_occupations(td, net, split):
        present = [tc for tc in seq if tc in occ]
        if len(present) < 2:
            continue
        entry = occ[present[0]]
        ivs = occupancy_intervals(seq, entry, params)
        pred = {o.tc: o.t_occupied for o in ivs}
        pred_rel = {o.tc: o.t_released for o in ivs}
        for tc in present:
            records.append({
                "tc": tc, "pred": pred[tc], "obs": occ[tc],
                "signed": pred[tc] - occ[tc], "is_platform": tc in plat_tcs,
            })
            if tc in rel:
                dur_obs.append(rel[tc] - occ[tc])
                dur_sim.append(pred_rel[tc] - pred[tc])
    return records, dur_sim, dur_obs


# ── Cut1+Cut2: 端到端反应式前推(一趟一锚点, 等待涌现) ──────────────────────
def build_trains_for_window(td_win, net, dwell_by_platform, plat_tcs, plat_tc_to_pid):
    """从窗口TD为每车构造 反应式Train(path/entry/dwell). entry=观测首TC占用(唯一锚点)."""
    from derbysim.sim.reactive import Train
    td_win = td_win.sort_values("ts")
    trains = []
    # 每车: 办的进路(按时序) + 观测占用 + 是否停站(TRTS)
    for hc, g in td_win.groupby("trainid_filled"):
        if hc in ("0", "****", ""):
            continue
        prs = g[(g["type"] == "Panel_Request") & (g["state"] == "1")]
        path, ready_at = [], []
        for x in prs.itertuples():
            r = net.routes.get(x.id.strip())
            if not r:
                continue
            rt = x.ts.timestamp()
            for tc in r.tc_sequence:
                if not path or path[-1] != tc:
                    path.append(tc)
                    ready_at.append(rt)
        if len(path) < 2:
            continue
        occ = g[(g["type"] == "Track") & (g["state"] == "1")]
        occ_first = {}
        for x in occ.itertuples():
            tc = x.id.strip()
            if tc not in occ_first:
                occ_first[tc] = x.ts.timestamp()
        if path[0] not in occ_first:
            continue
        entry = occ_first[path[0]]
        # 停站: 有TRTS则在其站台TC停 dwell
        stops = g[g["type"] == "TRTS"]
        dwell_idx, dwell_s = None, 0.0
        if len(stops):
            import re as _re
            pids = set()
            for sid in stops["id"]:
                m = _re.search(r"LPLAT(\d+)", str(sid))
                if m:
                    pids.add(int(m.group(1)))
            for i, tc in enumerate(path):
                if tc in plat_tc_to_pid and plat_tc_to_pid[tc] in pids:
                    dwell_idx = i
                    dwell_s = dwell_by_platform.get(plat_tc_to_pid[tc], 0.0)
                    break
        trains.append((Train(hc, path, entry, ready_at, dwell_idx, dwell_s), occ_first))
    return trains


def forward_sim_window(td, net, params, dwell_by_platform, start, end):
    """端到端前推一个时间窗, 返回 (records, engine). records: (tc,pred,obs) per车per TC."""
    from derbysim.sim.reactive import ReactiveEngine
    import csv as _csv
    plat_tc_to_pid = {}
    if config.DATA_PLATFORM_TC.exists():
        with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if (row.get("platform_id") or "").strip().isdigit():
                    plat_tc_to_pid[row["tc_id"].strip()] = int(row["platform_id"])
    plat_tcs = set(plat_tc_to_pid)
    win = td[(td["ts"] >= start) & (td["ts"] < end)].copy()
    built = build_trains_for_window(win, net, dwell_by_platform, plat_tcs, plat_tc_to_pid)
    trains = [b[0] for b in built]
    obs_by_train = {b[0].train_id: b[1] for b in built}
    eng = ReactiveEngine(params)
    eng.run(trains)
    records = []
    for t in trains:
        obs = obs_by_train[t.train_id]
        for tc in t.path:
            if tc in t.sim_occ and tc in obs:
                records.append({"train": t.train_id, "tc": tc,
                                "pred": t.sim_occ[tc], "obs": obs[tc],
                                "is_platform": tc in plat_tcs})
    return records, eng, trains


# ── 主验证(Cut2/M6+): 次生延误 观测 vs 引擎涌现(秒级, TD内部) ──────────────
def knockon_records(td, net, params, dwell_by_platform, start, end):
    """一个窗口: 每车 观测次生延误(超额通过时长) vs 仿真次生延误(引擎等待). 秒级."""
    from derbysim.sim.reactive import ReactiveEngine, DEFAULT_ADVANCE, DEFAULT_OCC
    import csv as _csv
    plat_tc_to_pid = {}
    if config.DATA_PLATFORM_TC.exists():
        with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if (row.get("platform_id") or "").strip().isdigit():
                    plat_tc_to_pid[row["tc_id"].strip()] = int(row["platform_id"])
    win = td[(td["ts"] >= start) & (td["ts"] < end)].copy()
    built = build_trains_for_window(win, net, dwell_by_platform, set(plat_tc_to_pid), plat_tc_to_pid)
    trains = [b[0] for b in built]
    obs_by = {b[0].train_id: b[1] for b in built}
    eng = ReactiveEngine(params)
    eng.run(trains)

    def free_adv(tc):
        return params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    # 每车的路由列表(用于逐路由excess, 避免跨停站间隔污染)
    routes_by_train = {}
    for hc, g in win.sort_values("ts").groupby("trainid_filled"):
        seqs = []
        for x in g[(g["type"] == "Panel_Request") & (g["state"] == "1")].itertuples():
            r = net.routes.get(x.id.strip())
            if r and r.tc_sequence:
                seqs.append(r.tc_sequence)
        routes_by_train[hc] = seqs

    recs = []
    for t in trains:
        obs = obs_by[t.train_id]
        # 观测次生延误 = Σ_route max(0, 该route实际通过时长 − 自由流通过时长)
        obs_ko = 0.0
        for seq in routes_by_train.get(t.train_id, []):
            pres = [tc for tc in seq if tc in obs]
            if len(pres) < 2:
                continue
            observed = obs[pres[-1]] - obs[pres[0]]
            freeflow = sum(free_adv(tc) for tc in seq[:-1])
            obs_ko += max(0.0, observed - freeflow)
        recs.append({"train": t.train_id, "obs_ko": obs_ko, "sim_ko": t.wait_time,
                     "n_path": len(t.path)})
    return recs, eng
