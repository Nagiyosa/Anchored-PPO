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
