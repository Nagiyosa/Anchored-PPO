"""M11a: 全 98 天 BC 决策数据集生成 + 时间留出划分 + 分布报告.

用法:
    python scripts/11_build_full_dataset.py gen [i0] [i1]   # 生成 days[i0:i1] (幂等, 跳过已存在), 分块跑
    python scripts/11_build_full_dataset.py combine          # 合并全部 per-day → 划分 + 报告

产出:
    artifacts/dataset/full/decisions_<date>.parquet          # 逐日(链清洗后)
    artifacts/dataset/bc_dataset_all.parquet                 # 合并 + split 列(train=9-10 / holdout=11-12)
    artifacts/dataset/bc_dataset_report.json                 # 分布/长尾/优先级/delay 报告
数据来源全 98 天(2023-09-01→12-07); 决策=M8 回放物理触发点, 链连续性过滤(dataset.py, 同 BC 干净车).
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 使 derbysim 可导入(从任意 cwd 运行)

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.dataset import build_cif_map, generate_decisions

FULL = config.ARTIFACTS / "dataset" / "full"


def _load_common():
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    return net, params, dwell, ptc


def _all_days(td):
    return sorted({str(d) for d in pd.to_datetime(td["time"]).dt.date})


def _augment(recs, date):
    """加分析列: date/service_class/delay_min(从 obs), 便于报告与优先级分层。"""
    for r in recs:
        o = json.loads(r["obs"]); dec = o.get("deciding", {})
        r["date"] = date
        r["service_class"] = dec.get("service_class", "")
        r["delay_min"] = dec.get("delay_min")
    return recs


def gen(i0, i1):
    FULL.mkdir(parents=True, exist_ok=True)
    net, params, dwell, ptc = _load_common()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    occ = pd.read_parquet(config.ARTIFACTS_SUBSET / "train_occurrences.parquet")
    days = _all_days(td)
    sel = days[i0:i1] if i1 else days[i0:]
    t0 = time.time()
    for d in sel:
        out = FULL / f"decisions_{d}.parquet"
        if out.exists():
            print(f"skip {d} (exists)"); continue
        win = td[(td["ts"] >= f"{d} 00:00:00") & (td["ts"] < f"{d} 23:59:59")].copy()
        if win.empty:
            print(f"empty {d}"); continue
        recs, stats = generate_decisions(net, params, ptc, dwell, win, cif_map=build_cif_map(occ, d))
        pd.DataFrame(_augment(recs, d)).to_parquet(out, index=False)
        print(f"{d}: dec={stats['n_decisions']} multi={stats['n_real_choices']} "
              f"miss={stats['n_mask_misses']} drop={stats['n_dropped_broken_chain']}")
    print(f"[gen {i0}:{i1}] {len(sel)} days in {time.time()-t0:.1f}s")


def combine():
    files = sorted(FULL.glob("decisions_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    mon = pd.to_datetime(df["date"]).dt.month
    df["split"] = mon.map(lambda m: "train" if m in (9, 10) else "holdout")
    allp = config.ARTIFACTS / "dataset" / "bc_dataset_all.parquet"
    df.to_parquet(allp, index=False)

    multi = df[df["n_legal"] > 1]
    dmin = pd.to_numeric(df["delay_min"], errors="coerce").dropna()
    route_ct = multi["chosen_route"].value_counts()
    rep = {
        "n_days": int(df["date"].nunique()),
        "n_decisions": int(len(df)),
        "n_multi_choice": int(len(multi)),
        "multi_choice_rate": round(len(multi) / max(len(df), 1), 3),
        "n_mask_size_gt1_by_split": df[df["n_legal"] > 1].groupby("split").size().to_dict(),
        "split_decisions": df.groupby("split").size().to_dict(),
        "split_multi": multi.groupby("split").size().to_dict(),
        "distinct_signals": int(df["signal"].nunique()),
        "distinct_chosen_routes": int(df["chosen_route"].nunique()),
        "n_legal_dist": df["n_legal"].value_counts().sort_index().to_dict(),
        "route_longtail": {"once": int((route_ct == 1).sum()), "le5": int((route_ct <= 5).sum()),
                           "total_routes": int(len(route_ct)),
                           "top1_share": round(float(route_ct.iloc[0] / route_ct.sum()), 3) if len(route_ct) else 0.0},
        "priority_multi_counts": multi["service_class"].value_counts().to_dict(),
        "delay_min_pct": {q: round(float(dmin.quantile(q / 100)), 1) for q in (1, 5, 50, 95, 99)},
        "delay_abs_gt120_rate": round(float((dmin.abs() > 120).mean()), 4),
        "n_holdout_routes_unseen_in_train": None,
    }
    tr = set(multi[multi["split"] == "train"]["chosen_route"])
    ho = set(multi[multi["split"] == "holdout"]["chosen_route"])
    rep["n_holdout_routes_unseen_in_train"] = int(len(ho - tr))
    outp = config.ARTIFACTS / "dataset" / "bc_dataset_report.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("wrote", allp, "and", outp)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        i0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        i1 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        gen(i0, i1)
    elif mode == "combine":
        combine()
