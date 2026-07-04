"""TD 秒级事件 → 区域子集 parquet + berth 邻接图.

- 分块流式读取 272MB TD, 向量化区域过滤(见 region.py 的判定规则)
- 输出: artifacts/subset/td_events.parquet
- 副产: artifacts/subset/berth_adjacency.parquet (CA 步进重建, 供仿真器)
"""

from __future__ import annotations

import pandas as pd

from derbysim import config
from derbysim.data import region as R
from derbysim.infra import InfrastructureNetwork

_KEEP_COLS = [
    "time", "type", "id", "state", "from_berth", "to_berth",
    "trainid_filled", "Panel_Request", "signal", "platform",
]
_USECOLS = [
    "time", "descr", "from_berth", "to_berth", "type", "id",
    "state", "trainid_filled", "Panel_Request",
]


def _region_mask(df: pd.DataFrame, region_tc: frozenset[str]) -> pd.Series:
    sig_str = {str(s) for s in config.IN_SCOPE_SIGNALS}
    berths = set(config.REGION_BERTHS) | {f"PLT{p}" for p in config.DERBY_PLATFORMS}
    plats = {str(p) for p in config.DERBY_PLATFORMS}

    id_s = df["id"].astype(str)
    frm = df["from_berth"].astype(str).str.strip()
    to = df["to_berth"].astype(str).str.strip()
    sig = id_s.str.extract(r"^[SR][A-Z]{2}(\d+)", expand=False)
    plat = id_s.str.extract(r"LPLAT(\d+)", expand=False)
    df["signal"] = sig
    df["platform"] = plat

    t = df["type"]
    m_sig = t.isin(["Signal", "Route", "Panel_Request"]) & sig.isin(sig_str)
    m_ca = (t == "CA") & (frm.isin(berths) | to.isin(berths))
    m_cb = (t == "CB") & frm.isin(berths)
    m_cc = (t == "CC") & to.isin(berths)
    m_track = (t == "Track") & id_s.str.strip().isin(region_tc)
    m_trts = (t == "TRTS") & plat.isin(plats)
    return m_sig | m_ca | m_cb | m_cc | m_track | m_trts


def build_td_subset(chunksize: int = 400_000) -> dict:
    net = InfrastructureNetwork.load()
    region_tc = R.region_tc_set_from_network(net)

    kept_frames: list[pd.DataFrame] = []
    adjacency: set[tuple[str, str]] = set()
    total_rows = 0
    kept_rows = 0
    type_counts_total: dict = {}

    reader = pd.read_csv(
        config.DATA_TD, usecols=_USECOLS, dtype=str,
        chunksize=chunksize, keep_default_na=False,
    )
    for chunk in reader:
        total_rows += len(chunk)
        for tp, c in chunk["type"].value_counts().items():
            type_counts_total[tp] = type_counts_total.get(tp, 0) + int(c)
        ca = chunk[chunk["type"] == "CA"]
        for a, b in zip(ca["from_berth"].str.strip(), ca["to_berth"].str.strip()):
            if a and b:
                adjacency.add((a, b))
        mask = _region_mask(chunk, region_tc)
        sub = chunk.loc[mask, _KEEP_COLS].copy()
        kept_rows += len(sub)
        kept_frames.append(sub)

    out = pd.concat(kept_frames, ignore_index=True)
    config.ARTIFACTS_SUBSET.mkdir(parents=True, exist_ok=True)
    td_path = config.ARTIFACTS_SUBSET / "td_events.parquet"
    out.to_parquet(td_path, index=False)

    adj_df = pd.DataFrame(sorted(adjacency), columns=["from_berth", "to_berth"])
    adj_path = config.ARTIFACTS_SUBSET / "berth_adjacency.parquet"
    adj_df.to_parquet(adj_path, index=False)

    in_region_trains = sorted(
        t for t in out["trainid_filled"].unique() if t and t not in ("0", "****")
    )
    pr = out[(out["type"] == "Panel_Request") & (out["state"] == "1")]
    report = {
        "td_total_rows": total_rows,
        "td_kept_rows": kept_rows,
        "kept_fraction": round(kept_rows / total_rows, 4) if total_rows else 0,
        "type_counts_total": type_counts_total,
        "kept_type_counts": {k: int(v) for k, v in out["type"].value_counts().items()},
        "n_region_trains": len(in_region_trains),
        "n_panel_requests_kept": int(len(pr)),
        "n_distinct_decision_routes": int(pr["id"].nunique()),
        "berth_adjacency_edges": len(adjacency),
        "td_parquet": str(td_path),
        "adjacency_parquet": str(adj_path),
    }
    return report
