"""M3 入口: 估计per-TC占用参数并缓存; 在真实 traversal 上验证占用模型.

运行: python scripts/06_calibrate_occupation.py
产出: artifacts/calib/tc_params.json + 验证报告(预测vs实际占用时刻MAE)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import (
    estimate_tc_params, occupancy_intervals, save_params,
)


def main() -> int:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()

    params = estimate_tc_params(td)
    save_params(params, config.ARTIFACTS_CALIB / "tc_params.json")

    # 验证: 抽样 route-setting, 对齐到route首TC, 比预测vs实际占用时刻
    track = td[(td["type"] == "Track") & (td["state"] == "1")][["ts", "trainid_filled", "id"]].copy()
    track["id"] = track["id"].str.strip()
    from collections import defaultdict
    tr_by = defaultdict(list)
    for x in track.itertuples():
        tr_by[x.trainid_filled].append((x.ts, x.id))
    for hc in tr_by:
        tr_by[hc].sort()

    pr = td[(td["type"] == "Panel_Request") & (td["state"] == "1")].sample(800, random_state=7)
    errs = []
    golden = None
    for x in pr.itertuples():
        r = net.routes.get(x.id.strip())
        if not r or not r.tc_sequence:
            continue
        seq = list(r.tc_sequence)
        lst = [(ts, tc) for ts, tc in tr_by.get(x.trainid_filled, [])
               if x.ts - pd.Timedelta(seconds=30) <= ts <= x.ts + pd.Timedelta(seconds=400)]
        first_occ = {}
        for ts, tc in lst:
            if tc not in first_occ:
                first_occ[tc] = ts
        if seq[0] not in first_occ:
            continue
        entry = first_occ[seq[0]].timestamp()
        pred = {o.tc: o.t_occupied for o in occupancy_intervals(seq, entry, params)}
        pair = [(pred[tc], first_occ[tc].timestamp()) for tc in seq if tc in first_occ]
        if len(pair) < 3:
            continue
        e = [abs(p - a) for p, a in pair]
        errs.extend(e)
        if golden is None and len(pair) >= 5 and max(e) < 20:
            golden = {
                "route": x.id.strip(), "headcode": x.trainid_filled,
                "tc_sequence": [tc for tc in seq if tc in first_occ],
                "entry_time": entry,
                "obs_occ": [first_occ[tc].timestamp() - entry for tc in seq if tc in first_occ],
                "params": {tc: list(params.get(tc, ())) for tc in seq if tc in first_occ},
            }
    errs = np.array(errs)
    report = {
        "n_tc_params": len(params),
        "validation_tc_occupations": int(len(errs)),
        "occ_time_MAE_s": round(float(errs.mean()), 2),
        "occ_time_median_err_s": round(float(np.median(errs)), 2),
        "within_10s_frac": round(float((errs <= 10).mean()), 3),
        "within_20s_frac": round(float((errs <= 20).mean()), 3),
        "params_json": str(config.ARTIFACTS_CALIB / "tc_params.json"),
    }
    print("=" * 60)
    print("占用模型标定+验证 (M3)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:26} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "occupation_calib.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if golden:
        (config.ARTIFACTS_CALIB / "golden_sample.json").write_text(
            json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
        print("golden 样例已存:", config.ARTIFACTS_CALIB / "golden_sample.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
