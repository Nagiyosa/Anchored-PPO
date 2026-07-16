"""M9d 测试: DynamicEngine 预留驱动 + Path A 面板动作(办/撤销/等待) + 死锁判负."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynamicEngine, DynTrain, PanelAction

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
    p = next(gen, None)
    while p is not None:
        act = None
        for tid, rid in p.logged.items():            # 回放: 给某待决车按日志办路
            act = PanelAction("set", tid, rid)
            break
        if act is None:
            act = PanelAction("wait")
        try:
            p = gen.send(act)
        except StopIteration:
            p = None
    return eng


def _disjoint_pair(net, R):
    P = net.routes["RDC5061A(M)"]
    fpP = R.footprint(P)
    for r in net.routes.values():
        if r.route_id != P.route_id and not (R.footprint(r) & fpP):
            return P, r
    raise RuntimeError("no disjoint route")


def test_dynamic_replay_completes_no_collision():
    net, params, dwell, ptc, dyn = _load()
    eng = _run_replay(net, params, ptc, dwell, dyn)
    assert len(eng.collisions) == 0
    assert all(t.done for t in dyn), "全车应完成"
    assert not eng.deadlocked
    assert eng.n_decisions > 10
    reached = sum(t.reached_exit for t in dyn)
    assert reached >= 0.8 * sum(1 for t in dyn if t.goal and t.goal.exit_group)


def test_reservation_released_at_end():
    net, params, dwell, ptc, dyn = _load()
    eng = _run_replay(net, params, ptc, dwell, dyn)
    assert not eng.res.reserved and not eng.res.route_train and not eng.res.queue


def test_cancel_returns_train_to_awaiting():
    net, params, _, _, _ = _load()
    eng = DynamicEngine(net, params)
    P, _ = _disjoint_pair(net, eng.res)
    eng.res.request(P, "A")                          # A 占 P
    tB = DynTrain("B", [P.route_id], 0.0)
    eng.trains = {"B": tB}
    eng.res.request(P, "B")                           # B 请求 P → QUEUED
    tB.pending_route = P.route_id
    tB.route_ptr = 1
    eng._apply(PanelAction("cancel", "B"))
    assert tB.pending_route is None and tB.awaiting_decision and tB.route_ptr == 0
    assert not any(rid == P.route_id and tr == "B" for rid, tr in eng.res.queue)


def test_deadlock_flags_terminal():
    net, params, _, _, _ = _load()
    eng = DynamicEngine(net, params)
    P, Q = _disjoint_pair(net, eng.res)
    eng.res.request(P, "A")
    eng.res.request(Q, "B")
    eng.res.request(Q, "A")                           # A 等 B (A→B)
    eng.trains = {"A": DynTrain("A", [], 0.0), "B": DynTrain("B", [], 0.0)}
    eng._apply(PanelAction("set", "B", P.route_id))   # B 请求 P → 排队 → B→A → 成环
    assert eng.deadlocked and "A" in eng.deadlock_cycle and "B" in eng.deadlock_cycle
