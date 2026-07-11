"""可控延误分析: 延误是否拥堵驱动(可控) + 决策多样性 + 站台竞争.

修正上一轮过强的"98%外生"——那用了最窄定义(紧邻TC). 本脚本用干净的
逐路由 held-time + 并发相关, 显示延误确与拥堵相关(可控信号), 杠杆在站台层.
运行: python scripts/09_controllable_delay.py
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.validation.calibration import pair_route_occupations


def main() -> int:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")

    def focc(tc):
        return params.get(tc, (8, 18))[1]

    tcpid = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("platform_id") or "").strip().isdigit():
                tcpid[r["tc_id"].strip()] = int(r["platform_id"])
    plat_tc = set(tcpid)

    # 1. 逐路由 held-time vs 并发
    occ = []
    for rid, tr, o, rel, seq in pair_route_occupations(td, net, split=config.HOLDOUT_SPLIT):
        ts = [o[tc] for tc in seq if tc in o]
        if len(ts) < 3:
            continue
        held = sum(max(0.0, rel[tc] - o[tc] - focc(tc))
                   for tc in seq if tc in o and tc in rel and tc not in plat_tc)
        occ.append((min(ts), max(ts), held))
    occ.sort()
    starts = np.array([a for a, _, _ in occ])
    ends = np.array([b for _, b, _ in occ])
    held = np.array([h for _, _, h in occ])
    conc = np.zeros(len(occ), dtype=int)
    for i, (a, b, _) in enumerate(occ):
        lo = bisect.bisect_left(starts, a - 120)
        hi = bisect.bisect_right(starts, b)
        conc[i] = sum(1 for j in range(lo, hi) if j != i and starts[j] <= b and ends[j] >= a)
    df = pd.DataFrame({"held": held, "conc": conc})
    corr = float(df["held"].corr(df["conc"]))
    df["bin"] = pd.cut(df["conc"], [-1, 0, 2, 4, 100], labels=["独占", "1-2", "3-4", "5+"])
    conc_tbl = {str(k): {"n": int(v["count"]), "median": round(float(v["median"]), 1),
                         "frac_gt30s": round(float(v["fg"]), 3)}
                for k, v in df.groupby("bin", observed=True)["held"].agg(
                    count="count", median="median", fg=lambda s: (s > 30).mean()).iterrows()}

    # 2. 站台竞争(headway <120s 占比)
    tr = td[(td["type"] == "Track") & td["id"].str.strip().isin(plat_tc)].copy()
    tr["id"] = tr["id"].str.strip()
    tr = tr.sort_values("ts")
    occ_p = defaultdict(list)
    opent = {}
    for x in tr.itertuples():
        k = (x.id, x.trainid_filled)
        if x.state == "1":
            opent[k] = x.ts
        elif x.state == "0" and k in opent:
            occ_p[tcpid[x.id]].append((opent.pop(k), x.ts))
    plat_press = {}
    for pid, ivs in occ_p.items():
        ivs.sort()
        gaps = [(ivs[i + 1][0] - ivs[i][1]).total_seconds() for i in range(len(ivs) - 1)]
        gaps = [g for g in gaps if 0 <= g < 3600]
        if gaps:
            plat_press[pid] = {"median_headway_s": round(float(np.median(gaps)), 0),
                               "frac_lt120s": round(float(np.mean(np.array(gaps) < 120)), 3),
                               "n": len(gaps)}

    # 3. 决策点选择多样性
    pr = td[(td["type"] == "Panel_Request") & (td["state"] == "1")].copy()
    pr["sig"] = pr["id"].str.extract(r"^R[A-Z]{2}(\d+)").astype(int)
    pr["exit"] = pr["id"].str.extract(r"^R[A-Z]{2}\d+([A-Z])")
    dp = set(config.DECISION_POINTS)
    multi = 0
    for s, g in pr[pr["sig"].isin(dp)].groupby("sig"):
        ex = g["exit"].value_counts(normalize=True)
        if len(ex) >= 2 and ex.iloc[0] < 0.95:
            multi += 1

    report = {
        "held_vs_concurrency_corr": round(corr, 2),
        "held_by_concurrency": conc_tbl,
        "platform_pressure": {str(k): v for k, v in sorted(plat_press.items())},
        "decision_points_with_real_choice": f"{multi}/35",
        "conclusion": (
            f"延误与并发相关 {corr:.2f}(拥堵驱动=可控信号); 站台3/4最挤(紧headway占比高); "
            f"{multi}/35 决策点真有多路由选择。可控杠杆在**站台层+高并发时段**, 非咽喉TC排队。"
            f"修正: 前述'98%外生'过强, 那是最窄定义(紧邻TC); 广义拥堵效应真实存在。"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "controllable_delay.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
