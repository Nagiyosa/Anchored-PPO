"""TRUST 到发/延误 → 规范化 parquet.

核心: 重构带符号延误 signed_variation(见 PROJECT_BRIEF §7 / IMPLEMENTATION_PLAN §4).
    timetable_variation 恒为非负量级; 方向由 variation_status 承载.
    signed = +var(LATE) / -var(EARLY) / 0(ON TIME); OFF ROUTE/缺值 → invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from derbysim import config

DERBY_STATION_STANOX = "57403"  # 已实测=德比主体站台

_KEEP = [
    "event_type", "planned_timestamp", "actual_timestamp", "gbtt_timestamp",
    "timetable_variation", "variation_status", "platform", "loc_stanox",
    "reporting_stanox", "next_report_stanox", "delay_monitoring_point",
    "offroute_ind", "train_id", "current_train_id", "toc_id", "direction_ind",
]


def signed_variation(var: pd.Series, status: pd.Series) -> pd.Series:
    """带符号延误(分钟): +var(LATE) / -var(EARLY) / 0(ON TIME) / NaN(其它)."""
    v = pd.to_numeric(var, errors="coerce")
    out = pd.Series(np.nan, index=var.index, dtype="float64")
    out = out.mask(status == "LATE", v)
    out = out.mask(status == "EARLY", -v)
    out = out.mask(status == "ON TIME", 0.0)
    return out


def build_trust_subset() -> dict:
    df = pd.read_csv(config.DATA_TRUST, dtype=str, keep_default_na=False)
    df = df[[c for c in _KEEP if c in df.columns]].copy()

    df["signed_variation"] = signed_variation(df["timetable_variation"], df["variation_status"])
    df["lateness"] = df["signed_variation"].clip(lower=0)
    df["earliness"] = (-df["signed_variation"]).clip(lower=0)
    # 有效性: 有带符号延误 且 非 OFF ROUTE 且 有实际时刻
    df["valid"] = (
        df["signed_variation"].notna()
        & (df["variation_status"] != "OFF ROUTE")
        & (df["actual_timestamp"].str.len() > 0)
    )
    df["headcode"] = df["train_id"].str[2:6]
    df["is_derby_station"] = df["loc_stanox"] == DERBY_STATION_STANOX

    config.ARTIFACTS_SUBSET.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_SUBSET / "trust_moves.parquet"
    df.to_parquet(path, index=False)

    valid = df[df["valid"]]
    derby = df[df["is_derby_station"]]
    report = {
        "trust_total_rows": int(len(df)),
        "event_types": {k: int(v) for k, v in df["event_type"].value_counts().items()},
        "variation_status": {k: int(v) for k, v in df["variation_status"].value_counts().items()},
        "n_valid": int(df["valid"].sum()),
        "n_invalid": int((~df["valid"]).sum()),
        "n_offroute": int((df["variation_status"] == "OFF ROUTE").sum()),
        "n_derby_station_rows": int(len(derby)),
        "signed_variation_min": float(valid["signed_variation"].min()),
        "signed_variation_max": float(valid["signed_variation"].max()),
        "median_lateness_of_late": float(
            valid.loc[valid["signed_variation"] > 0, "signed_variation"].median()
        ),
        "n_distinct_headcodes": int(df["headcode"].nunique()),
        "trust_parquet": str(path),
    }
    return report
