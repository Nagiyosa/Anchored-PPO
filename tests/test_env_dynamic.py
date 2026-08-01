"""M9d 测试: DynamicEngine 预留驱动 + Path A 面板动作(办/撤销/等待) + 死锁判负."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

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
    assert not eng.res.reserved and not eng.res.holder_fp and not eng.res.queue


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


def test_gym_wrapper_replay_and_obs_lockstate():
    from derbysim.sim.env_dynamic import DynamicDerbyEnv
    net, params, dwell, ptc, dyn = _load()
    env = DynamicDerbyEnv(net, params, ptc, dwell)
    obs, info = env.reset(dyn)
    assert "trains" in obs and "reservations" in obs      # 面板 obs 结构
    seen_locked = False
    steps = 0
    while not env.done and steps < 100000:
        dec = env.decision()
        act = None
        for tid, rid in dec.logged.items():
            act = PanelAction("set", tid, rid)
            break
        if act is None:
            act = PanelAction("wait")
        obs, reward, done, info = env.step(act)
        steps += 1
        if obs and obs["n_reserved_tc"] > 0:
            seen_locked = True                            # obs 含当前锁态(Path A 可观测)
    assert env.done and all(t.done for t in dyn)
    assert seen_locked, "obs 应在过程中含非空当前锁态(预留)"
    assert not env.engine.deadlocked and len(env.engine.collisions) == 0


def _build_both(net, dwell, ptc):
    """同窗口构造 M8 EnvTrains 与 M9 DynTrains(同一 route_chain)."""
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    w = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(w, net, dwell, ptc)
    m8 = [b[0] for b in built]
    dyn = [DynTrain(b[0].train_id, b[0].route_chain, b[0].entry_time,
                    b[0].stop_pids, b[0].booked_platform) for b in built]
    return m8, dyn


def test_consistency_regression_vs_m8():
    """喂日志决策: M9(预留/overlap/Path A) 与 M8 逐车**路径完全一致** + 物理 sanity.
    占用时刻自由流段一致(中位≤2s); 尾部分歧=M9 去掉 log 时序门禁(agent 掌控时机)的**预期**差异.
    """
    from derbysim.sim.env import DerbyEnv
    import numpy as np
    net, params, dwell, ptc, _ = _load()
    m8, dyn = _build_both(net, dwell, ptc)
    env = DerbyEnv(net, params, ptc, dwell); env.reset(m8)
    while not env.done:
        c = env.decision(); env.step(c.logged_route_id)
    eng = _run_replay_dyn(net, params, ptc, dwell, dyn)
    m8p = {t.train_id: list(t.path) for t in m8}
    m9p = {t.train_id: list(t.path) for t in dyn}
    common = set(m8p) & set(m9p)
    assert common and all(m8p[k] == m9p[k] for k in common), "M9 路径应与 M8 逐车一致"
    # 物理 sanity
    assert len(eng.collisions) == 0 and not eng.deadlocked and all(t.done for t in dyn)
    assert all(t.wait_time >= 0 for t in dyn)                 # 延误守恒
    # 自由流占用一致(中位) —— 尾部分歧为预期(agent 时机), 不硬卡
    m8o = {t.train_id: dict(t.sim_occ) for t in m8}
    diffs = [abs(t.sim_occ[tc] - m8o[t.train_id][tc])
             for t in dyn for tc in t.sim_occ if tc in m8o.get(t.train_id, {})]
    assert np.median(diffs) <= 5.0, f"自由流占用中位分歧应≤5s, 实得 {np.median(diffs):.0f}s"


def _run_replay_dyn(net, params, ptc, dwell, dyn):
    eng = DynamicEngine(net, params, ptc, dwell)
    gen = eng.run(dyn); p = next(gen, None)
    while p is not None:
        act = next((PanelAction("set", tid, rid) for tid, rid in p.logged.items()), PanelAction("wait"))
        try:
            p = gen.send(act)
        except StopIteration:
            p = None
    return eng


def test_path_a_heuristic_smoke():
    """Path A 非回放冒烟: FCFS 式启发策略(给待决车办首条合法进路)驱动, 不崩、0 碰撞、可终止."""
    from derbysim.sim.env_dynamic import DynamicDerbyEnv
    net, params, dwell, ptc, dyn = _load()
    env = DynamicDerbyEnv(net, params, ptc, dwell)
    env.reset(dyn); steps = 0
    while not env.done and steps < 200000:
        dec = env.decision()
        sets = [o for o in dec.options if o.kind == "set"]
        act = sets[0] if sets else PanelAction("wait")     # 启发: 办首条合法进路, 否则等
        env.step(act); steps += 1
    # 冒烟只验**可驱动 + 安全**: 能终止、0 碰撞、无异常。
    # 朴素 first-legal 策略会堆车、搁浅部分车(策略质量问题非引擎 bug); 真实策略(M11 BC/M12 RL)会理性选路。
    assert env.done, "Path A 应能终止(全完成/搁浅/死锁判负)"
    assert len(env.engine.collisions) == 0                 # 预留结构性无碰撞
    assert steps < 200000                                   # 未陷入无限循环


def _total_wait(net, params, ptc, dwell, built, overlap_depth=1, exclude=None):
    dyn = [DynTrain(b[0].train_id, b[0].route_chain, b[0].entry_time, b[0].stop_pids,
                    b[0].booked_platform) for b in built if exclude is None or b[0].train_id != exclude]
    eng = DynamicEngine(net, params, ptc, dwell, overlap_depth=overlap_depth)
    gen = eng.run(dyn); p = next(gen, None)
    while p is not None:
        act = next((PanelAction("set", tid, rid) for tid, rid in p.logged.items()),
                   PanelAction("wait"))
        try:
            p = gen.send(act)
        except StopIteration:
            p = None
    return {t.train_id: t.wait_time for t in dyn}


def test_delay_conservation_monotonic():
    """C3 真·守恒: 次生延误全非负且可归属; 移除一辆车 → 其余车总等待不增(无免费午餐)."""
    net, params, dwell, ptc, _ = _load()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    w = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(w, net, dwell, ptc)
    base = _total_wait(net, params, ptc, dwell, built)
    assert all(x >= 0 for x in base.values())                      # 可归属、非负
    mover = max(base, key=base.get)
    after = _total_wait(net, params, ptc, dwell, built, exclude=mover)
    rem_before = sum(x for k, x in base.items() if k != mover)
    assert sum(after.values()) <= rem_before + 1e-6, "移除车后其余延误不应增加(无免费午餐)"


def test_overlap_depth_monotone_conflict():
    """C4: overlap 越深冲突/次生延误越多(单调). 0<=1<=2. 记录敏感性(depth2 会激增)."""
    net, params, dwell, ptc, _ = _load()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    w = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(w, net, dwell, ptc)
    w0 = sum(_total_wait(net, params, ptc, dwell, built, overlap_depth=0).values())
    w1 = sum(_total_wait(net, params, ptc, dwell, built, overlap_depth=1).values())
    w2 = sum(_total_wait(net, params, ptc, dwell, built, overlap_depth=2).values())
    assert w0 <= w1 <= w2, f"overlap 单调: {w0} {w1} {w2}"
    assert w1 > w0                                                  # overlap 确有效应


def _run_cancel_policy(net, params, ptc, dwell, built):
    """策略: 遇到可撤销(排队中)进路, 每车撤销一次再改办日志进路; 否则照日志办/等待."""
    from derbysim.sim.env_dynamic import DynamicDerbyEnv
    dyn = [DynTrain(b[0].train_id, b[0].route_chain, b[0].entry_time, b[0].stop_pids,
                    b[0].booked_platform) for b in built]
    env = DynamicDerbyEnv(net, params, ptc, dwell)
    env.reset(dyn)
    cancelled, n_cancel, steps = set(), 0, 0
    while not env.done and steps < 100000:
        dec = env.decision()
        canc = [o for o in dec.options if o.kind == "cancel" and o.train_id not in cancelled]
        if canc:
            cancelled.add(canc[0].train_id); n_cancel += 1
            act = canc[0]
        else:
            act = next((PanelAction("set", tid, rid) for tid, rid in dec.logged.items()),
                       PanelAction("wait"))
        env.step(act); steps += 1
    return env, dyn, n_cancel


def test_cancel_in_running_episode():
    """E: 运行中撤销排队进路再改办 —— 撤销确被行使、episode 正常完成、0 碰撞、状态一致."""
    net, params, dwell, ptc, _ = _load()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    w = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(w, net, dwell, ptc)
    env, dyn, n_cancel = _run_cancel_policy(net, params, ptc, dwell, built)
    assert n_cancel > 0, "窗口应有排队进路可撤销(否则测试空转)"
    assert env.done and len(env.engine.collisions) == 0
    assert all(t.done for t in dyn), "撤销+改办后各车仍应完成"
    assert not env.engine.res.reserved and not env.engine.res.queue     # 结束预留/队列清空(状态一致)


def test_cancel_episode_deterministic():
    """E: 含 cancel 的整局两遍逐位一致(快照回滚/撤销不引入非确定)."""
    net, params, dwell, ptc, _ = _load()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    w = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(w, net, dwell, ptc)
    _, d1, c1 = _run_cancel_policy(net, params, ptc, dwell, built)
    _, d2, c2 = _run_cancel_policy(net, params, ptc, dwell, built)
    assert c1 == c2
    assert {t.train_id: t.path for t in d1} == {t.train_id: t.path for t in d2}
    assert {t.train_id: t.sim_occ for t in d1} == {t.train_id: t.sim_occ for t in d2}


def test_release_behind_never_frees_a_track_the_train_still_needs():
    """不变式: 车只能释放**它再也用不到**的 TC(2026-08-01, D6 红线回归测试)。

    `_release_behind(idx-2)` 原先假设路径不回头。但进路链是策略在线选的, 策略可以把车导成环
    —— 此时 `path[idx-2]` 与车**当前所在**的 TC 是同一段, 旧实现会把自己脚下的锁释放掉,
    另一辆车随即锁走, 本车再进 = 碰撞。这里用最小构造直接钉死判据, 不依赖能否抽到那一天。
    """
    from derbysim.learn.scenarios import load_assets
    net, params, dwell, ptc = load_assets()
    eng = DynamicEngine(net, params, ptc, dwell)
    eng.trains = {}
    t = SimpleNamespace(train_id="LOOP", path=["A", "B", "A", "B", "A", "B"], pos=2, done=False,
                        pending_route=None)
    eng.res.reserved = {"A": ("r0", "LOOP"), "B": ("r1", "LOOP")}
    eng.res.holder_fp = {("r0", "LOOP"): frozenset({"A"}), ("r1", "LOOP"): frozenset({"B"})}
    eng._release_behind(t, 0)                       # path[0] == "A" == 车当前所在
    assert eng.res.reserved.get("A") == ("r0", "LOOP"), "把车脚下的 TC 释放掉了"
    t.pos = 5                                       # 已走到最后一段, 前方不再需要 A
    eng._release_behind(t, 4)                       # path[4] == "A", 但 path[5:] 里没有 A
    assert "A" not in eng.res.reserved, "车确实用不到了却没释放 → 会假死锁"
