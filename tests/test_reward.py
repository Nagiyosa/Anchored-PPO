"""M10a 测试: 稠密次生延误项(每步差分, TOPS λ) + realized(含 in-progress) + 死锁惩罚 + determinism."""

from __future__ import annotations

import csv
import json

import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynTrain, PanelAction
from derbysim.sim.reward import (RewardConfig, RewardedDynamicEnv,
                                 realized_weighted_delay, to_dyn_trains)

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
    return net, params, dwell, ptc, built


def _replay_rewards(env, dyn):
    """回放(按日志办路)驱动 RewardedDynamicEnv, 收集每步 reward。"""
    obs, info = env.reset(dyn)
    rewards, total_done = [], False
    p = info["decision"]
    while p is not None:
        act = None
        for tid, rid in p.logged.items():
            act = PanelAction("set", tid, rid)
            break
        if act is None:
            act = PanelAction("wait")
        obs, r, done, info = env.step(act)
        rewards.append(r)
        p = info["decision"]
        if done:
            break
    return rewards


# ---- F2: to_dyn_trains 携带 service_class/planned_ref ----
def test_to_dyn_trains_carries_priority_fields():
    net, params, dwell, ptc, built = _load()
    dyn = to_dyn_trains(built)
    assert dyn, "no trains built"
    # service_class = headcode 首位, 应非空且属 TOPS 集
    classes = {t.service_class for t in dyn}
    assert any(c in {"1", "2", "3", "5", "6"} for c in classes), classes
    # 与 EnvTrain 源逐字段一致
    src = {b[0].train_id: b[0] for b in built}
    for t in dyn:
        assert t.service_class == src[t.train_id].service_class
        assert t.booked_platform == src[t.train_id].booked_platform


# ---- realized 加权延误: 权重 + in-progress ----
def _stub(now, trains):
    class _E:
        pass
    e = _E(); e.now = now; e.trains = {t.train_id: t for t in trains}
    return e


def test_realized_weighted_delay_priority_and_inprogress():
    cfg = RewardConfig(tops_weights={"1": 4.0, "6": 1.0}, tops_fallback=2.0)
    a = DynTrain("A", [], 0.0, service_class="1"); a.wait_time = 100.0     # 特快
    b = DynTrain("B", [], 0.0, service_class="6"); b.wait_time = 100.0     # 货
    e = _stub(0.0, [a, b])
    # 同 100s 延误: 特快 ×4 vs 货 ×1 → 高优先级权重更大
    assert realized_weighted_delay(e, cfg) == 4.0 * 100 + 1.0 * 100
    # in-progress: B 现被阻塞, blocked_since=0, now=50 → +50s 计入(即便还没解阻塞累加)
    b.blocked_since = 0.0
    e2 = _stub(50.0, [a, b])
    assert realized_weighted_delay(e2, cfg) == 4.0 * 100 + 1.0 * (100 + 50)


# ---- 延误项 telescoping: Σ 每步延误 reward == -c_delay * 整局 realized ----
def test_delay_reward_telescopes_to_total():
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    dyn = to_dyn_trains(built)
    rewards = _replay_rewards(env, dyn)
    assert rewards
    final = realized_weighted_delay(env.engine, cfg)
    # 无死锁回放: Σ reward == -c_delay * 整局加权次生延误(差分累加=首尾)
    assert not env.engine.deadlocked
    assert abs(sum(rewards) - (-cfg.c_delay * final)) < 1e-6
    # 每步延误项 ≤ 0(realized 单调不减)
    assert all(r <= 1e-9 for r in rewards)


# ---- determinism: 两遍逐步 reward 逐位一致 ----
def test_reward_deterministic():
    net, params, dwell, ptc, built = _load()
    r1 = _replay_rewards(RewardedDynamicEnv(net, params, ptc, dwell), to_dyn_trains(built))
    r2 = _replay_rewards(RewardedDynamicEnv(net, params, ptc, dwell), to_dyn_trains(built))
    assert r1 == r2 and len(r1) > 0


# ---- 死锁 terminal 惩罚只发一次 ----
def test_deadlock_penalty_wired_once():
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig(p_deadlock=50.0)
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    env.reset(to_dyn_trains(built))
    # 白盒: 引擎判负 → _compute_reward 扣 p_deadlock 一次, 再算不再重复
    env.engine.deadlocked = True
    env._was_deadlocked = False
    r_first = env._compute_reward()
    r_again = env._compute_reward()
    assert r_first <= -cfg.p_deadlock + 1e-9
    assert r_again > -cfg.p_deadlock + 1e-9      # 第二次不再重复扣死锁
