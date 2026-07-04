"""闭塞时间占用模型(M3): 给定 车+进路+进入时刻 → 各TC占用/出清区间.

模型(blocking-time 骨架, 参数化):
    列车沿 route 的 tc_sequence 依序占用 TC. 每 TC 两个参数:
      advance[tc]   —— 车头从占用本TC到占用下一TC的时距(前进时间)
      occ_dur[tc]   —— 本TC占用时长(车头进入→车尾出清)
    t_occ[0]=entry; t_occ[i]=t_occ[i-1]+advance[tc_{i-1}]; t_rel[i]=t_occ[i]+occ_dur[tc_i].
参数由 TD 聚合估计(estimate_tc_params, 中位数); M5 再做约束优化精修.
时间单位: 秒(float). 纯函数, 不依赖数据.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# 缺省参数(全体TC中位量级, 见M3勘探): advance~8s, occ~18s
DEFAULT_ADVANCE_S = 8.0
DEFAULT_OCC_S = 18.0


@dataclass(frozen=True)
class TCOccupation:
    tc: str
    t_occupied: float
    t_released: float

    @property
    def duration(self) -> float:
        return self.t_released - self.t_occupied


def occupancy_intervals(
    tc_sequence,
    entry_time: float,
    tc_params: dict[str, tuple[float, float]] | None = None,
    default_advance: float = DEFAULT_ADVANCE_S,
    default_occ: float = DEFAULT_OCC_S,
) -> list[TCOccupation]:
    """沿 tc_sequence 逐TC推占用/出清区间(闭塞时间骨架)."""
    tc_params = tc_params or {}
    out: list[TCOccupation] = []
    t_occ = float(entry_time)
    for i, tc in enumerate(tc_sequence):
        adv, occ = tc_params.get(tc, (default_advance, default_occ))
        out.append(TCOccupation(tc, t_occ, t_occ + occ))
        t_occ = t_occ + adv   # 下一TC在前进时间后被占用
    return out


def estimate_tc_params(td_events, min_samples: int = 5) -> dict[str, tuple[float, float]]:
    """从 TD 事件(DataFrame, 含 type/id/state/trainid_filled/ts) 估计每TC (advance, occ) 中位数."""
    import pandas as pd

    tr = td_events[td_events["type"] == "Track"].copy()
    tr["id"] = tr["id"].str.strip()
    if "ts" not in tr.columns:
        tr["ts"] = pd.to_datetime(tr["time"])
    tr = tr.sort_values("ts")

    occ_dur: dict[str, list[float]] = defaultdict(list)
    advance: dict[str, list[float]] = defaultdict(list)
    for hc, g in tr.groupby("trainid_filled"):
        if hc in ("0", "****", ""):
            continue
        open_t: dict[str, object] = {}
        seq_occ: list[tuple[str, object]] = []
        for x in g.itertuples():
            tc = x.id
            if x.state == "1":
                open_t[tc] = x.ts
                seq_occ.append((tc, x.ts))
            elif x.state == "0" and tc in open_t:
                occ_dur[tc].append((x.ts - open_t.pop(tc)).total_seconds())
        for i in range(len(seq_occ) - 1):
            dt = (seq_occ[i + 1][1] - seq_occ[i][1]).total_seconds()
            if 0 < dt < 120:
                advance[seq_occ[i][0]].append(dt)

    import statistics as st
    params: dict[str, tuple[float, float]] = {}
    tcs = set(occ_dur) | set(advance)
    for tc in tcs:
        od = occ_dur.get(tc, [])
        ad = advance.get(tc, [])
        if len(od) < min_samples:
            continue
        occ_m = st.median(od)
        adv_m = st.median(ad) if len(ad) >= min_samples else DEFAULT_ADVANCE_S
        params[tc] = (round(adv_m, 1), round(occ_m, 1))
    return params


def save_params(params: dict[str, tuple[float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: list(v) for k, v in params.items()}, indent=2), encoding="utf-8")


def load_params(path: Path) -> dict[str, tuple[float, float]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in d.items()}
