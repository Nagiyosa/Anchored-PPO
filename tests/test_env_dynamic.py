"""M9d-1 测试: DynamicEngine 预留驱动 —— 回放模式全车完成、0 碰撞、目标终止."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynamicEngine, DynTrain

WIN = ("2023-11-06 07:00:00", "2023-11-06 10:00:00")


def _load():
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    win = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(win, net, dwell, ptc)
    dyn = [DynTrain(b[0].train_id, b[0].route_chain, b[0].entry_time,
                    b[0].stop_pids, b[0].booked_platform) for b in built]
    return net, params, dwell, ptc, dyn


def _run_replay(net, params, ptc, dwell, dyn):
    eng = DynamicEngine(net, params, ptc, dwell)
    gen = eng.run(dyn)
    pending = next(gen, None)
    while pending is not None:
        try:
            pending = gen.send(pending.logged_route_id)   # 回放: 选日志真进路
        except StopIteration:
            pending = None
    return eng


def test_dynamic_replay_completes_no_collision():
    net, params, dwell, ptc, dyn = _load()
    eng = _run_replay(net, params, ptc, dwell, dyn)
    assert len(eng.collisions) == 0, f"预留应无碰撞, 实得 {len(eng.collisions)}"
    assert all(t.done for t in dyn), "全车应完成(无卡死)"
    assert eng.n_decisions > 10
    # 有出区口目标的车应到达该方向组出口(允许少量区内终止车)
    reached = sum(t.reached_exit for t in dyn)
    assert reached >= 0.8 * sum(1 for t in dyn if t.goal and t.goal.exit_group)


def test_dynamic_reservation_released_at_end():
    net, params, dwell, ptc, dyn = _load()
    eng = _run_replay(net, params, ptc, dwell, dyn)
    # 全部完成后预留表应清空(无泄漏预留)
    assert not eng.res.reserved and not eng.res.route_train, "结束后预留应全部释放"
    assert not eng.res.queue, "结束后无残留排队请求"
