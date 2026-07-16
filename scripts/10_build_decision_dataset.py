"""M8 副产物: Phase 1 BC 决策数据集 (§14.9). 回放模式逐决策点吐 (状态,掩码,被选动作,弱标签).

运行: python scripts/10_build_decision_dataset.py [YYYY-MM-DD]  (默认 2023-11-06)
产出: artifacts/dataset/decisions_<date>.parquet + _summary.json
状态契约与 RL 环境同一份(DerbyEnv._build_state) → 杜绝 train/deploy 态失配.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.dataset import build_cif_map, generate_decisions
from derbysim.sim.occupation import load_params


def _load_common():
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    plat_tc_to_pid = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                plat_tc_to_pid[row["tc_id"].strip()] = int(row["platform_id"])
    return net, params, dwell, plat_tc_to_pid


def main() -> int:
    date = sys.argv[1] if len(sys.argv) > 1 else "2023-11-06"
    net, params, dwell, plat_tc_to_pid = _load_common()

    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    win = td[(td["ts"] >= f"{date} 00:00:00") & (td["ts"] < f"{date} 23:59:59")].copy()

    occ = pd.read_parquet(config.ARTIFACTS_SUBSET / "train_occurrences.parquet")
    cif_map = build_cif_map(occ, date)

    records, stats = generate_decisions(net, params, plat_tc_to_pid, dwell, win, cif_map=cif_map)

    out_dir = config.ARTIFACTS / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    pq = out_dir / f"decisions_{date}.parquet"
    df.to_parquet(pq, index=False)

    # 触发/掩码/弱标签概览
    real = df[df["n_legal"] > 1]
    kinds = df["trigger_kind"].value_counts().to_dict()
    summary = {
        "date": date,
        **stats,
        "trigger_kinds": kinds,
        "mask_size": {"min": int(df["n_legal"].min()), "median": float(df["n_legal"].median()),
                      "max": int(df["n_legal"].max())},
        "real_choice_decisions": int(len(real)),
        "weak_platform_deviation_rate": (
            round(float(df["weak_platform_deviation"].dropna().mean()), 3)
            if df["weak_platform_deviation"].notna().any() else None),
        "cif_delay_available": int(df["obs"].apply(
            lambda s: json.loads(s)["deciding"]["delay_min"] is not None).sum()),
        "parquet": str(pq),
    }
    (out_dir / f"decisions_{date}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"Phase 1 BC 决策数据集 — {date} (M8 副产物)")
    print("=" * 64)
    print(f"决策 {stats['n_decisions']} (真实多选 {stats['n_real_choices']}); 列车 {stats['n_trains']}; "
          f"掩码误划 {stats['n_mask_misses']}")
    print(f"触发类型: {kinds}")
    print(f"掩码大小: {summary['mask_size']}")
    print(f"弱标签(站台偏离)率: {summary['weak_platform_deviation_rate']}; "
          f"CIF 延误可得: {summary['cif_delay_available']}/{stats['n_decisions']}")
    print(f"写入: {pq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
