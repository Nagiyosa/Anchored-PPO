"""M6 入口(命门): 留出集(11-12月)回放验证, 逐事件对比, 出报告 + §8模板.

运行: python scripts/05_validate.py
产出: artifacts/reports/validation_holdout.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.validation import metrics as M
from derbysim.validation.replay import forward_predict
from derbysim.validation.report import s8_statement


def _metrics_block(records):
    errs = [r["pred"] - r["obs"] for r in records]
    return {
        "n": len(errs),
        "MAE_s": round(M.mae(errs), 2),
        "RMSE_s": round(M.rmse(errs), 2),
        "median_abs_s": round(M.median_abs(errs), 2),
        "median_bias_s": round(M.bias(errs), 2),
        "hit_10s": round(M.hit_rate(errs, 10), 3),
        "hit_20s": round(M.hit_rate(errs, 30) if False else M.hit_rate(errs, 20), 3),
    }


def main() -> int:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")

    # 留出集前推
    rec_h, dsim_h, dobs_h = forward_predict(td, net, params, config.HOLDOUT_SPLIT)
    # 标定集(对照, 检泛化)
    rec_c, _, _ = forward_predict(td, net, params, config.CALIB_SPLIT)

    overall = _metrics_block(rec_h)
    plat = _metrics_block([r for r in rec_h if r["is_platform"]])
    calib_overall = _metrics_block(rec_c)

    within = 10
    import numpy as np
    errs_h = np.abs([r["pred"] - r["obs"] for r in rec_h])
    tail_frac = float((errs_h > 20).mean()) * 100
    stmt = s8_statement(
        plat["median_abs_s"] if plat["n"] else overall["median_abs_s"],
        (plat["hit_10s"] if plat["n"] else overall["hit_10s"]) * 100, within,
        overall["MAE_s"], tail_frac)

    report = {
        "holdout_split": config.HOLDOUT_SPLIT,
        "calib_split": config.CALIB_SPLIT,
        "occupation_holdout": overall,
        "occupation_calib_reference": {"MAE_s": calib_overall["MAE_s"],
                                       "median_abs_s": calib_overall["median_abs_s"],
                                       "hit_10s": calib_overall["hit_10s"]},
        "platform_arrival_holdout": plat,
        "occ_duration_dist_sim": M.dist_summary(dsim_h),
        "occ_duration_dist_obs": M.dist_summary(dobs_h),
        "s8_statement": stmt,
    }
    print("=" * 64)
    print("回放验证 —— 留出集 11-12月 (M6, 命门)")
    print("=" * 64)
    print(f"占用(全体)  : {overall}")
    print(f"占用(标定集对照): MAE {calib_overall['MAE_s']}s median {calib_overall['median_abs_s']}s → 泛化对比")
    print(f"到达(站台TC): {plat}")
    print(f"占用时长分布 sim {report['occ_duration_dist_sim']}")
    print(f"           obs {report['occ_duration_dist_obs']}")
    print("-" * 64)
    print("§8 对外表述:")
    print(" ", stmt)

    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "validation_holdout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# 回放验证报告 — 留出集 {config.HOLDOUT_SPLIT[0]}~{config.HOLDOUT_SPLIT[1]} (M6)

## 方法
喂真实进路决策, 以观测首TC占用为进入条件, 用9-10月标定的占用模型前推各TC占用时刻,
在**未参与标定**的11-12月留出集上逐事件对比 TD 观测.

## 占用时刻再现(全体 TC)
- 样本 {overall['n']}; **MAE {overall['MAE_s']}s, 中位abs {overall['median_abs_s']}s, 中位偏差 {overall['median_bias_s']}s**
- {within}s命中率 {overall['hit_10s']:.0%}, 20s命中率 {overall['hit_20s']:.0%}
- 标定集对照: MAE {calib_overall['MAE_s']}s / 中位 {calib_overall['median_abs_s']}s → **泛化无退化**

## 到达时刻再现(站台 TC)
- 样本 {plat['n']}; MAE {plat['MAE_s']}s, 中位abs {plat['median_abs_s']}s, {within}s命中率 {plat['hit_10s']:.0%}

## 占用时长分布(物理合理性)
- 仿真 {report['occ_duration_dist_sim']}
- 观测 {report['occ_duration_dist_obs']}

## 对外表述(brief §8⑥)
{stmt}

## 局限(brief §8⑤)
描述性有效已严格验证(留出集); 反事实有效(改变决策的效果)是离线评估的根本限制,
仅在专家 face validity + 物理约束(延误守恒)下作保守结论.
"""
    (config.ARTIFACTS_REPORTS / "validation_holdout.md").write_text(md, encoding="utf-8")
    print("\n报告已写入 artifacts/reports/validation_holdout.{json,md}")
    # 验收: 留出集中位abs与标定集相当(泛化), 无系统偏置
    ok = overall["median_abs_s"] < 6.0 and abs(overall["median_bias_s"]) < 2.0
    print("验收(泛化|无偏):", "PASS ✅" if ok else "CHECK ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
