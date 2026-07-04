"""M5 入口: 事件驱动配对标定 per-TC占用参数 + 停站时分(9-10月标定集), 验证残差.

运行: python scripts/03_calibrate.py
产出: artifacts/calib/params.json (精修) + dwell.json + 标定报告
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
from derbysim.sim.occupation import occupancy_intervals, save_params
from derbysim.validation.calibration import (
    calibrate_tc_params, calibrate_dwell, pair_route_occupations,
)


def main() -> int:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()

    params, n_pairs = calibrate_tc_params(td, net, split=config.CALIB_SPLIT)
    dwell = calibrate_dwell(td, net, split=config.CALIB_SPLIT)

    save_params(params, config.ARTIFACTS_CALIB / "params.json")
    (config.ARTIFACTS_CALIB / "dwell.json").write_text(
        json.dumps({str(k): v for k, v in dwell.items()}, indent=2), encoding="utf-8")

    # 标定集残差验证: 事件驱动配对上, 预测占用 vs 实际(带符号=检偏)
    errs, signed = [], []
    for rid, train, occ, rel, seq in pair_route_occupations(td, net, split=config.CALIB_SPLIT):
        present = [tc for tc in seq if tc in occ]
        if len(present) < 3:
            continue
        entry = occ[present[0]]
        pred = {o.tc: o.t_occupied for o in occupancy_intervals(seq, entry, params)}
        for tc in present:
            e = pred[tc] - occ[tc]
            signed.append(e); errs.append(abs(e))
    errs = np.array(errs); signed = np.array(signed)

    report = {
        "calib_split": config.CALIB_SPLIT,
        "holdout_split": config.HOLDOUT_SPLIT,
        "n_route_pairs_calib": n_pairs,
        "n_tc_params": len(params),
        "n_platform_dwell": len(dwell),
        "dwell_medians_s": {str(k): v for k, v in dwell.items()},
        "occ_MAE_s_mean": round(float(errs.mean()), 2),
        "occ_median_abs_s": round(float(np.median(errs)), 2),
        "occ_median_bias_s": round(float(np.median(signed)), 2),
        "occ_p90_s": round(float(np.percentile(errs, 90)), 1),
        "occ_p95_s": round(float(np.percentile(errs, 95)), 1),
        "within_10s_frac": round(float((errs <= 10).mean()), 3),
        "within_20s_frac": round(float((errs <= 20).mean()), 3),
        "gt60s_frac_midroute_waits": round(float((errs > 60).mean()), 3),
    }
    print("=" * 60)
    print("标定 (M5) —— 事件驱动配对, 9-10月标定集")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:22} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = abs(report["occ_median_bias_s"]) < 2.0 and report["occ_median_abs_s"] < 5.0
    print("\n验收(中位残差无偏|中位abs<5s):", "PASS ✅" if ok else "CHECK ⚠️")
    print("注: 均值MAE被~3%中途停等车长尾抬高(这类由引擎deferral处理, 非自由流占用模型)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
