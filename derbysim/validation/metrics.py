"""验证度量: MAE / RMSE / N秒命中率 / 分布对比(纯函数)."""

from __future__ import annotations

import numpy as np


def mae(errors) -> float:
    e = np.abs(np.asarray(errors, dtype=float))
    return float(e.mean()) if len(e) else float("nan")


def rmse(errors) -> float:
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def median_abs(errors) -> float:
    e = np.abs(np.asarray(errors, dtype=float))
    return float(np.median(e)) if len(e) else float("nan")


def bias(signed_errors) -> float:
    e = np.asarray(signed_errors, dtype=float)
    return float(np.median(e)) if len(e) else float("nan")


def hit_rate(errors, within_s: float) -> float:
    e = np.abs(np.asarray(errors, dtype=float))
    return float((e <= within_s).mean()) if len(e) else float("nan")


def dist_summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {}
    return {
        "n": int(len(v)), "median": round(float(np.median(v)), 1),
        "p10": round(float(np.percentile(v, 10)), 1),
        "p90": round(float(np.percentile(v, 90)), 1),
        "mean": round(float(v.mean()), 1),
    }
