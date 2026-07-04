"""统一"事件-决策"时间线: 取单个车次实例的有序 TD 事件, 附其计划/实际.

供调试与 Phase 1 决策数据集抽取; 不物化全部(按需取用).
"""

from __future__ import annotations

import pandas as pd

from derbysim import config


def train_event_timeline(headcode: str, date=None) -> pd.DataFrame:
    """某 headcode(可选某日)的有序 TD 事件时间线(区域内)."""
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    sub = td[td["trainid_filled"] == headcode].copy()
    sub["ts"] = pd.to_datetime(sub["time"], errors="coerce")
    if date is not None:
        d = pd.to_datetime(date).date()
        sub = sub[sub["ts"].dt.date == d]
    return sub.sort_values("ts")[
        ["time", "type", "id", "state", "from_berth", "to_berth", "signal", "platform"]
    ].reset_index(drop=True)


def occurrence(headcode: str, date) -> dict:
    """某车次实例的对齐摘要(从 train_occurrences.parquet)."""
    occ = pd.read_parquet(config.ARTIFACTS_SUBSET / "train_occurrences.parquet")
    d = pd.to_datetime(date).date()
    m = occ[(occ["headcode"] == headcode) & (occ["date"] == d)]
    return m.iloc[0].to_dict() if len(m) else {}
