"""M8 Phase 1 数据集副产物测试: 结构完整 + 与 RL 环境同一状态契约 + 弱标签可得."""

from __future__ import annotations

import csv
import json

import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.dataset import (build_cif_map, generate_decisions,
                                  position_order_indices, split_broken_chains)
from derbysim.sim.env import EnvTrain

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


def test_preset_downstream_request_is_reordered_by_observed_position():
    """D33 step 5: request order is not traversal order (the real 1M00 pattern)."""
    _win, net, _params, _dwell, _ptc, _cif = _load()
    request_order = ["RDW5306B(C)", "RTD5054B(M)",
                     "RTD5040A(M)", "RTD5044A(M)"]
    physical_order = ["RDW5306B(C)", "RTD5054B(M)",
                      "RTD5044A(M)", "RTD5040A(M)"]
    et = EnvTrain("1M00", request_order, [0.0, 1.0, 2.0, 3.0], 0.0)
    occ = {}
    for j, rid in enumerate(physical_order):
        for k, tc in enumerate(net.routes[rid].tc_sequence):
            occ.setdefault(tc, 100.0 * j + k)
    assert position_order_indices(et, occ, net) == (0, 1, 3, 2)
    clean = split_broken_chains([(et, occ)], net)
    assert len(clean) == 1 and clean[0][0].train_id == "1M00"
    assert clean[0][0].route_chain == physical_order
    assert clean[0][0].req_times == [0.0, 1.0, 3.0, 2.0]  # request times move with their routes


def test_position_order_refuses_to_guess_repeated_route_occurrences():
    _win, net, _params, _dwell, _ptc, _cif = _load()
    rid = "RTD5045E(M)"
    et = EnvTrain("SHUNT", [rid, rid], [0.0, 1.0], 0.0)
    occ = {tc: float(i) for i, tc in enumerate(net.routes[rid].tc_sequence)}
    assert position_order_indices(et, occ, net) == (0, 1)
