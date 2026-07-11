"""M7 等价回归: DerbyEnv 回放模式(逐段构造) 必须与 ReactiveEngine(整条 path) 逐位一致.

对比每车: path / ready_at / dwell_idx / dwell_s / wait_time / sim_occ.
用真实留出集窗口(含冲突), 确保等价在有"车等车"涌现时也成立.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.reactive import ReactiveEngine
from derbysim.sim.env import DerbyEnv, build_env_trains_for_window
from derbysim.validation.replay import build_trains_for_window

WIN_START = "2023-11-06 07:00:00"
WIN_END = "2023-11-06 10:00:00"      # 3h 早高峰, 含冲突


def _load():
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    plat_tc_to_pid = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                plat_tc_to_pid[row["tc_id"].strip()] = int(row["platform_id"])
    return td, net, params, dwell, plat_tc_to_pid


def _run_reactive(td, net, params, dwell, plat_tc_to_pid):
    win = td[(td["ts"] >= WIN_START) & (td["ts"] < WIN_END)].copy()
    built = build_trains_for_window(win, net, dwell, set(plat_tc_to_pid), plat_tc_to_pid)
    trains = [b[0] for b in built]
    ReactiveEngine(params).run(trains)
    return {t.train_id: t for t in trains}


def _run_env(td, net, params, dwell, plat_tc_to_pid):
    win = td[(td["ts"] >= WIN_START) & (td["ts"] < WIN_END)].copy()
    built = build_env_trains_for_window(win, net, dwell, plat_tc_to_pid)
    trains = [b[0] for b in built]
    env = DerbyEnv(net, params, plat_tc_to_pid, dwell)
    env.reset(trains)
    n_dec = 0
    while not env.done:
        ctx = env.decision()
        env.step(ctx.logged_route_id)     # 回放: 每步选日志真实进路
        n_dec += 1
    return {t.train_id: t for t in trains}, n_dec


def test_env_equivalence():
    td, net, params, dwell, plat_tc_to_pid = _load()
    react = _run_reactive(td, net, params, dwell, plat_tc_to_pid)
    envd, n_dec = _run_env(td, net, params, dwell, plat_tc_to_pid)

    assert set(react) == set(envd), "列车集不一致"
    assert len(react) > 0, "窗口无列车"

    mism = []
    for tid, rt in react.items():
        et = envd[tid]
        if rt.path != et.path:
            mism.append((tid, "path", rt.path, et.path)); continue
        if rt.ready_at != et.ready_at:
            mism.append((tid, "ready_at", rt.ready_at, et.ready_at)); continue
        if rt.dwell_idx != et.dwell_idx:
            mism.append((tid, "dwell_idx", rt.dwell_idx, et.dwell_idx)); continue
        if abs(rt.dwell_s - et.dwell_s) > 1e-9:
            mism.append((tid, "dwell_s", rt.dwell_s, et.dwell_s)); continue
        if abs(rt.wait_time - et.wait_time) > 1e-6:
            mism.append((tid, "wait_time", rt.wait_time, et.wait_time)); continue
        if rt.sim_occ != et.sim_occ:
            mism.append((tid, "sim_occ", None, None)); continue

    assert not mism, f"{len(mism)} 车不一致, 例: {mism[:3]}"
    # 至少发生过一些决策 + 一些等待(否则等价是平凡的)
    total_wait = sum(t.wait_time for t in react.values())
    print(f"\n[M7] trains={len(react)} decisions={n_dec} "
          f"total_wait={total_wait:.0f}s n_wait_events(react-not-tracked)")


if __name__ == "__main__":
    import sys
    sys.exit(0 if pytest.main([__file__, "-s", "-q"]) == 0 else 1)
