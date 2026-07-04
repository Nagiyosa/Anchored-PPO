"""标定(M5): 事件驱动配对 → 精修per-TC占用参数 + 停站时分; 按标定/留出集切分.

事件驱动配对(替代M3临时窗口, brief §8): 每条进路的活跃期 = Panel_Request(锁路,s1) →
    Route(解锁,s0); 在此区间内收集该车对该route各TC的占用/出清, 沿route顺序算:
      advance[tc_i] = occ[tc_{i+1}] - occ[tc_i]   (route内相邻, 非跨车全局)
      occ_dur[tc]  = release[tc] - occ[tc]
标定集(9-10月)拟合; 留出集(11-12月)由 M6 验证. 用中位数(稳健), min_samples 过滤.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict

import pandas as pd

from derbysim import config


def _in_split(ts, split):
    if split is None:
        return True
    a, b = split
    return pd.Timestamp(a) <= ts <= pd.Timestamp(b) + pd.Timedelta(days=1)


def pair_route_occupations(td, net, split=None):
    """产出 [(route_id, train, {tc: (occ_epoch, rel_epoch)}, order)] —— 事件驱动配对."""
    td = td.copy()
    if "ts" not in td.columns:
        td["ts"] = pd.to_datetime(td["time"])

    # 每车的 Track 占用/出清 (tc -> 依时间的 (ts,state))
    tr = td[td["type"] == "Track"][["ts", "trainid_filled", "id", "state"]].copy()
    tr["id"] = tr["id"].str.strip()
    track_by_train = defaultdict(list)
    for x in tr.sort_values("ts").itertuples():
        track_by_train[x.trainid_filled].append((x.ts, x.id, x.state))

    locks = td[(td["type"] == "Panel_Request") & (td["state"] == "1")][["ts", "trainid_filled", "id"]]
    rels = td[(td["type"] == "Route") & (td["state"] == "0")][["ts", "trainid_filled", "id"]]
    rel_by = defaultdict(list)
    for x in rels.sort_values("ts").itertuples():
        rel_by[(x.trainid_filled, x.id.strip())].append(x.ts)

    for x in locks.sort_values("ts").itertuples():
        rid = x.id.strip()
        route = net.routes.get(rid)
        if not route or not route.tc_sequence:
            continue
        if not _in_split(x.ts, split):
            continue
        # 配对: 该(train,route)在 lock 之后最近的 release
        cand = [t for t in rel_by.get((x.trainid_filled, rid), []) if t >= x.ts]
        if not cand:
            continue
        t_rel = min(cand)
        lo = x.ts - pd.Timedelta(seconds=10)
        hi = t_rel + pd.Timedelta(seconds=30)
        seq = route.tc_sequence
        occ, rel = {}, {}
        open_t = {}
        for ts, tc, state in track_by_train.get(x.trainid_filled, []):
            if ts < lo:
                continue
            if ts > hi:
                break
            if tc not in seq:
                continue
            if state == "1" and tc not in occ:
                occ[tc] = ts.timestamp()
                open_t[tc] = ts
            elif state == "0" and tc in open_t and tc not in rel:
                rel[tc] = ts.timestamp()
        if len(occ) >= 2:
            yield rid, x.trainid_filled, occ, rel, seq


def calibrate_tc_params(td, net, split=None, min_samples: int = 5):
    """事件驱动配对 → per-TC (advance, occ_dur) 中位数."""
    advance = defaultdict(list)
    occ_dur = defaultdict(list)
    n_pairs = 0
    for rid, train, occ, rel, seq in pair_route_occupations(td, net, split):
        n_pairs += 1
        for tc in seq:
            if tc in occ and tc in rel:
                d = rel[tc] - occ[tc]
                if 0 < d < 600:
                    occ_dur[tc].append(d)
        occ_seq = [(tc, occ[tc]) for tc in seq if tc in occ]
        for i in range(len(occ_seq) - 1):
            dt = occ_seq[i + 1][1] - occ_seq[i][1]
            if 0 < dt < 120:
                advance[occ_seq[i][0]].append(dt)

    params = {}
    for tc in set(occ_dur) | set(advance):
        od = occ_dur.get(tc, [])
        ad = advance.get(tc, [])
        if len(od) < min_samples:
            continue
        params[tc] = (
            round(st.median(ad), 1) if len(ad) >= min_samples else 8.0,
            round(st.median(od), 1),
        )
    return params, n_pairs


def calibrate_dwell(td, net, split=None, min_samples: int = 5):
    """每站台停站时分中位数: 停车(有TRTS)车次的站台端信号TC占用时长近似."""
    td = td.copy()
    if "ts" not in td.columns:
        td["ts"] = pd.to_datetime(td["time"])
    # 站台端信号->站台号; 站台端信号TC(进路首TC)占用时长, 仅取有TRTS(停站)的车
    plat_of_signal = {}
    for pid, plat in net.platforms.items():
        for direction, sig in plat.end_signals.items():
            plat_of_signal[sig] = pid
    # 停站车集合: 出现过 TRTS(该站台) 的 (train)
    trts = td[td["type"] == "TRTS"].copy()
    trts["plat"] = trts["id"].str.extract(r"LPLAT(\d+)").astype("float")
    stopping = set(zip(trts["trainid_filled"], trts["plat"]))

    # 站台占用时长: 用站台 TC (platform_tc_map) 的占用时长
    plat_tc = {}
    if config.DATA_PLATFORM_TC.exists():
        import csv
        with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = (row.get("platform_id") or "").strip()
                if pid.isdigit():
                    plat_tc[row["tc_id"].strip()] = int(pid)

    tr = td[(td["type"] == "Track") & (td["id"].str.strip().isin(plat_tc))].copy()
    tr["id"] = tr["id"].str.strip()
    dwell = defaultdict(list)
    for hc, g in tr.sort_values("ts").groupby("trainid_filled"):
        open_t = {}
        for x in g.itertuples():
            if not _in_split(x.ts, split):
                continue
            if x.state == "1":
                open_t[x.id] = x.ts
            elif x.state == "0" and x.id in open_t:
                pid = plat_tc[x.id]
                dur = (x.ts - open_t.pop(x.id)).total_seconds()
                if (hc, float(pid)) in stopping and 20 < dur < 1200:
                    dwell[pid].append(dur)
    return {pid: round(st.median(v), 1) for pid, v in dwell.items() if len(v) >= min_samples}
