"""M8 Phase 1 数据集副产物测试: 结构完整 + 与 RL 环境同一状态契约 + 弱标签可得."""

from __future__ import annotations

import csv
import json

import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.dataset import build_cif_map, generate_decisions

WIN_START = "2023-11-06 07:00:00"
WIN_END = "2023-11-06 10:00:00"


def _load():
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    win = td[(td["ts"] >= WIN_START) & (td["ts"] < WIN_END)].copy()
    occ = pd.read_parquet(config.ARTIFACTS_SUBSET / "train_occurrences.parquet")
    cif_map = build_cif_map(occ, "2023-11-06")
    return win, net, params, dwell, ptc, cif_map


def test_dataset_structure_and_contract():
    win, net, params, dwell, ptc, cif_map = _load()
    records, stats = generate_decisions(net, params, ptc, dwell, win, cif_map=cif_map)
    assert stats["n_decisions"] > 0 and stats["n_real_choices"] > 0
    for rec in records:
        # 每条 = (状态, 掩码, 被选动作, 弱标签)
        assert rec["chosen_route"] in rec["legal_route_ids"], "被选动作必须在掩码内"
        obs = json.loads(rec["obs"])
        # 同一状态契约(DerbyEnv._build_state 的键)
        assert set(obs) >= {"deciding", "candidate_tc_occupancy", "conflict_trains",
                            "decision_signal", "tod_hour"}
        # 状态不含答案
        assert rec["chosen_route"] not in rec["obs"]
    # 弱标签字段存在(值可为 None, 弱监督噪声允许)
    assert "weak_platform_deviation" in records[0]


def test_cif_targets_available():
    win, net, params, dwell, ptc, cif_map = _load()
    records, stats = generate_decisions(net, params, ptc, dwell, win, cif_map=cif_map)
    have = sum(1 for r in records if json.loads(r["obs"])["deciding"]["delay_min"] is not None)
    assert have / len(records) > 0.5, "多数决策应有 CIF 延误(状态目标字段)"
