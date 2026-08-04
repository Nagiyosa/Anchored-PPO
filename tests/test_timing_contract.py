"""D33 Step 8 时机契约守门测试(D38): 同刻排空 · 非终局 WAIT Δt>0 · 两车同刻联合可见。

修引擎前应**全红**, 修好后应**全绿** —— 与 `scripts/21_audit_timing_contract.py`
在全部 11 个 val 运营日上报的三组指标口径一致(same-time panels / zero-dt WAIT / max_set_trains),
故直接在全部 val 日上度量, 保证与守门审计逐项对齐, 而不是挑一天可能测不出缺陷。
"""
from __future__ import annotations

import pytest

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.sim.reward import RewardedDynamicEnv, can_complete, has_reached_goal


@pytest.fixture(scope="module")
def assets():
    return load_assets()


def _drive_days(assets, policy_factory, days):
    """跑一个策略走完给定运营日; 每个面板记 next_event_dt / n_set_trains / kind / dt。

    完全复刻 rollout 的驱动方式(reset → decision → encode → step), 额外在每次决策**开面板时**
    读队列最前事件与当前时刻之差, 以及该面板暴露的可 set 车数 —— 即 Step 8 审计的度量点。
    """
    net, params, dwell, ptc = assets
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    panels = []
    agg = {"collisions": 0, "deadlocks": 0, "invariants": 0,
           "able": 0, "reached": 0, "undone": 0}
    for day in days:
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        policy = policy_factory(env)
        obs, _ = env.reset(build_day_trains(day, inject=False), horizon=day_horizon(day))
        while not env.done:
            dec = env.decision()
            if dec is None or obs is None:
                break
            enc = adapter.encode(obs, dec)
            eng = env.engine
            next_dt = (eng.q._h[0].time - eng.now) if len(eng.q) else None
            set_trains = {s.train_id for s in enc["slots"] if s.kind == "set"}
            idx, _, _ = policy(enc)
            kind = enc["slots"][idx].kind
            t0 = eng.now
            obs, _r, done, _ = env.step(dec.options[idx])
            panels.append({"next_dt": next_dt, "n_set_trains": len(set_trains),
                           "kind": kind, "dt": eng.now - t0})
            if done:
                break
        eng = env.engine
        agg["collisions"] += len(eng.collisions)
        agg["deadlocks"] += int(eng.deadlocked)
        agg["invariants"] += len(eng.state_invariant_violations())
        agg["able"] += sum(1 for t in eng.trains.values() if can_complete(t))
        agg["reached"] += sum(1 for t in eng.trains.values() if has_reached_goal(t))
        agg["undone"] += sum(1 for t in eng.trains.values() if not t.done)
    return panels, agg


@pytest.fixture(scope="module")
def signaller(assets):
    return _drive_days(assets, make_signaller_replay_policy, operating_days("val"))


@pytest.fixture(scope="module")
def greedy(assets):
    return _drive_days(assets, lambda _env: make_greedy_set_policy(), operating_days("val"))


def test_no_panel_opens_with_same_time_event_pending(signaller):
    """契约①: 面板只在时间戳 t 的事件全部排空后才打开(next_event_dt 恒 > 0 或队列空)。"""
    panels, _ = signaller
    offenders = [p for p in panels if p["next_dt"] == 0.0]
    assert offenders == [], f"{len(offenders)} 个面板在同刻事件未排空时就打开了"


def test_non_terminal_wait_advances_time(signaller):
    """契约③④: WAIT 是时间延展宏动作, Δt 必须 > 0(不得有零时长 WAIT)。"""
    panels, _ = signaller
    zero = [p for p in panels if p["kind"] == "wait" and p["dt"] == 0.0]
    assert zero == [], f"{len(zero)} 个零时长 WAIT"


def test_multiple_trains_are_jointly_settable(greedy):
    """契约②: 同刻多辆 awaiting 必须在同一面板联合暴露, 而非被串行化(max_set_trains ≥ 2)。"""
    panels, _ = greedy
    assert max(p["n_set_trains"] for p in panels) >= 2, "从未有面板同时给 ≥2 辆车 set 选项"


def test_signaller_replay_closed_loop_regression(signaller):
    """回归: 重组时间切面不得破坏它刚刚重排的闭环(完成率 1.000 · 0 碰撞 · 0 死锁 · 0 不变式违背)。"""
    _, agg = signaller
    assert agg["collisions"] == 0
    assert agg["deadlocks"] == 0
    assert agg["invariants"] == 0
    assert agg["undone"] == 0
    assert agg["able"] > 0 and agg["reached"] == agg["able"], "signaller 完成率必须为 1.000"
