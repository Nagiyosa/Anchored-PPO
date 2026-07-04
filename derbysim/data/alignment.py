"""三源对齐: 把 TD/TRUST/CIF 连成每车"占用记录"(按趟, 非按天).

连接键(实测): headcode 主键(TD↔TRUST 1127/1128, TD↔CIF 756).
    headcode 日内可复用/多趟(实测 3%) → 先按时间间隙(>RUN_GAP)把一天切成多趟,
    每趟用其锚点时刻各自就近匹配 CIF 层与 TRUST 到发(H6 修复).
产出: artifacts/subset/train_occurrences.parquet —— 一行=一趟车次实例.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from derbysim import config
from derbysim.data.cif_parser import ScheduleLayer, resolve

RUN_GAP_S = 90 * 60  # 相邻事件间隔超过此值判为新的一趟


def _sec_to_hms(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _derby_time(ly: ScheduleLayer):
    for v in (ly.derby_arr, ly.derby_dep, ly.derby_pass):
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return int(v)
    return None


def load_cif_index():
    df = pd.read_parquet(config.ARTIFACTS_SUBSET / "cif_schedules.parquet")
    by_uid: dict[str, list[ScheduleLayer]] = defaultdict(list)
    hc_uids: dict[str, set[str]] = defaultdict(set)
    for r in df.itertuples():
        ly = ScheduleLayer(
            uid=r.uid, stp=r.stp, date_from=r.date_from, date_to=r.date_to,
            days_run=r.days_run, headcode=r.headcode,
            train_service_code=r.train_service_code, derby_rec=r.derby_rec,
            derby_arr=r.derby_arr, derby_dep=r.derby_dep,
            derby_pass=r.derby_pass, derby_platform=r.derby_platform,
        )
        by_uid[r.uid].append(ly)
        if r.headcode:
            hc_uids[r.headcode].add(r.uid)
    return by_uid, hc_uids


def _span(ly: ScheduleLayer) -> int:
    if ly.date_from and ly.date_to:
        return (ly.date_to - ly.date_from).days
    return 10_000


def resolve_for_headcode(by_uid, hc_uids, hc, date, anchor_sec=None) -> ScheduleLayer | None:
    """headcode+日期 → 生效且经停德比的层; anchor_sec(该趟锚点时刻)就近消歧(H6)."""
    cands = []
    for uid in hc_uids.get(hc, ()):
        r = resolve(by_uid[uid], date)
        if r is not None and r.calls_derby:
            cands.append(r)
    if not cands:
        return None
    if anchor_sec is not None:
        def gap(l):
            t = _derby_time(l)
            return abs(t - anchor_sec) if t is not None else 86400
        cands.sort(key=lambda l: (gap(l), l.derby_platform in (None, ""), _span(l)))
    else:
        cands.sort(key=lambda l: (l.derby_platform in (None, ""), _span(l)))
    return cands[0]


def _split_runs(ts_sorted: np.ndarray) -> list[tuple[int, int]]:
    """按时间间隙切趟, 返回 [(起始idx, 结束idx+1)] 的切片区间."""
    if len(ts_sorted) == 0:
        return []
    secs = ts_sorted.astype("datetime64[s]").astype("int64")
    breaks = np.where(np.diff(secs) > RUN_GAP_S)[0]
    bounds = [0, *(int(b) + 1 for b in breaks), len(secs)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _nearest(events, anchor_ts):
    best, bestd = None, None
    for ts, sv, pl in events:
        if pd.isna(ts):
            continue
        d = abs((ts - anchor_ts).total_seconds())
        if bestd is None or d < bestd:
            bestd, best = d, (ts, sv, pl)
    return best


def build_train_occurrences() -> dict:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td = td[td["trainid_filled"].notna() & ~td["trainid_filled"].isin(["0", "****"])].copy()
    td["ts"] = pd.to_datetime(td["time"], errors="coerce")
    td["date"] = td["ts"].dt.date
    td["sig_i"] = pd.to_numeric(td["signal"], errors="coerce")

    trust = pd.read_parquet(config.ARTIFACTS_SUBSET / "trust_moves.parquet")
    trust["ats"] = pd.to_datetime(trust["actual_timestamp"], errors="coerce")
    trust["date"] = trust["ats"].dt.date
    td_d = trust[trust["is_derby_station"]]
    # TRUST: (headcode,date) -> {ARRIVAL:[(ts,sv,plat)], DEPARTURE:[...]}
    trust_by_key: dict = defaultdict(lambda: {"ARRIVAL": [], "DEPARTURE": []})
    for r in td_d.itertuples():
        if r.event_type in ("ARRIVAL", "DEPARTURE"):
            trust_by_key[(r.headcode, r.date)][r.event_type].append(
                (r.ats, r.signed_variation, str(r.platform).strip())
            )

    by_uid, hc_uids = load_cif_index()
    dp_set = set(config.DECISION_POINTS)

    rows = []
    for (hc, date), gg in td.groupby(["trainid_filled", "date"], sort=False):
        g = gg.sort_values("ts")
        ts_arr = g["ts"].values
        runs = _split_runs(ts_arr)
        tk = trust_by_key.get((hc, date), {"ARRIVAL": [], "DEPARTURE": []})
        for ri, (i0, i1) in enumerate(runs):
            r = g.iloc[i0:i1]
            is_pr = (r["type"] == "Panel_Request") & (r["state"] == "1")
            dps = sorted({int(s) for s in r.loc[is_pr, "sig_i"].dropna()} & dp_set)
            plats = sorted({p for p in r.loc[r["type"] == "CC", "to_berth"] if str(p).startswith("PL")})
            t0 = r["ts"].iloc[0]
            anchor = t0.hour * 3600 + t0.minute * 60 + t0.second

            cif = resolve_for_headcode(by_uid, hc_uids, hc, date, anchor)
            cif_t = _derby_time(cif) if cif else None
            gap_min = (abs(cif_t - anchor) / 60.0) if cif_t is not None else None

            a = _nearest(tk["ARRIVAL"], t0)
            d = _nearest(tk["DEPARTURE"], t0)
            rows.append({
                "headcode": hc, "date": date, "run": ri, "n_runs": len(runs),
                "n_events": int(len(r)), "n_decisions": int(is_pr.sum()),
                "decision_points": ",".join(map(str, dps)),
                "td_platforms": ",".join(plats),
                "td_first": t0, "td_last": r["ts"].iloc[-1],
                "matched_cif": cif is not None,
                "cif_uid": cif.uid if cif else None,
                "cif_stp": cif.stp if cif else None,
                "cif_rec": cif.derby_rec if cif else None,
                "cif_arr": _sec_to_hms(cif.derby_arr) if cif else None,
                "cif_dep": _sec_to_hms(cif.derby_dep) if cif else None,
                "cif_pass": _sec_to_hms(cif.derby_pass) if cif else None,
                "cif_platform": cif.derby_platform if cif else None,
                "cif_time_gap_min": round(gap_min, 1) if gap_min is not None else None,
                "matched_trust": (a is not None or d is not None),
                "trust_arr_actual": a[0] if a else None,
                "trust_arr_delay": float(a[1]) if a and pd.notna(a[1]) else None,
                "trust_dep_actual": d[0] if d else None,
                "trust_dep_delay": float(d[1]) if d and pd.notna(d[1]) else None,
                "trust_platform": (a[2] if a and a[2] else (d[2] if d and d[2] else None)),
            })

    occ = pd.DataFrame(rows)
    # CIF 可靠性: gap<=30min 视为可靠(正常先于站台数分钟); 大 gap=headcode复用伪匹配
    occ["cif_reliable"] = occ["cif_time_gap_min"].notna() & (occ["cif_time_gap_min"] <= 30)
    config.ARTIFACTS_SUBSET.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_SUBSET / "train_occurrences.parquet"
    occ.to_parquet(path, index=False)

    def first_plat(s):
        s = str(s)
        return s.replace("PLT", "").replace("PLAT", "").replace("PLA", "").strip() if s.startswith("PL") else ""
    occ_p = occ[(occ["td_platforms"] != "") & occ["cif_platform"].notna() & (occ["cif_platform"] != "")].copy()
    occ_p["td_p"] = occ_p["td_platforms"].str.split(",").str[0].map(first_plat)
    occ_p["cif_p"] = occ_p["cif_platform"].str.replace(r"[A-Za-z]", "", regex=True).str.strip()
    plat_agree = int((occ_p["td_p"] == occ_p["cif_p"]).sum())

    well = occ[occ["cif_time_gap_min"].notna()]
    return {
        "n_occurrences": int(len(occ)),
        "n_multi_run_days": int((occ["run"] > 0).groupby([occ["headcode"], occ["date"]]).any().sum()),
        "n_matched_cif": int(occ["matched_cif"].sum()),
        "n_matched_trust": int(occ["matched_trust"].sum()),
        "n_matched_all_three": int((occ["matched_cif"] & occ["matched_trust"]).sum()),
        "n_cif_reliable": int(occ["cif_reliable"].sum()),
        "n_with_decisions": int((occ["n_decisions"] > 0).sum()),
        "n_stp_overlaid": int(occ["cif_stp"].isin(["O", "N"]).sum()),
        "cif_time_gap_le2min_frac": round(float((well["cif_time_gap_min"] <= 2).mean()), 3),
        "platform_check_n": int(len(occ_p)),
        "platform_agree_td_vs_cif": plat_agree,
        "occurrences_parquet": str(path),
    }
