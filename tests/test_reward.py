"""M10 reward 测试:
   M10a 稠密延误(差分, TOPS λ, in-progress) + 死锁; M10b 站台三档 + on-time + 搁浅判负 + determinism."""

from __future__ import annotations

import csv
import json

import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynTrain, PanelAction
from derbysim.sim.reward import (RewardConfig, RewardedDynamicEnv, realized_weighted_delay,
                                 to_dyn_trains, platform_tier_reward, on_time_bonus)

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
    obs, info = env.reset(dyn)
    rewards = []
    p = info["decision"]
    while p is not None:
        act = None
        for tid, rid in p.logged.items():
            act = PanelAction("set", tid, rid); break
        if act is None:
            act = PanelAction("wait")
        obs, r, done, info = env.step(act)
        rewards.append(r)
        p = info["decision"]
        if done:
            break
    return rewards


def _stub(now, trains):
    class _E: pass
    e = _E(); e.now = now; e.trains = {t.train_id: t for t in trains}
    return e


# ============ M10a ============
def test_to_dyn_trains_carries_priority_fields():
    net, params, dwell, ptc, built = _load()
    dyn = to_dyn_trains(built)
    assert dyn
    assert any(t.service_class in {"1", "2", "3", "5", "6"} for t in dyn)
    src = {b[0].train_id: b[0] for b in built}
    for t in dyn:
        assert t.service_class == src[t.train_id].service_class
        assert t.booked_platform == src[t.train_id].booked_platform


def test_realized_weighted_delay_priority_and_inprogress():
    cfg = RewardConfig(tops_weights={"1": 4.0, "6": 1.0}, tops_fallback=2.0)
    a = DynTrain("A", [], 0.0, service_class="1"); a.wait_time = 100.0
    b = DynTrain("B", [], 0.0, service_class="6"); b.wait_time = 100.0
    assert realized_weighted_delay(_stub(0.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * 100
    b.blocked_since = 0.0
    assert realized_weighted_delay(_stub(50.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * (100 + 50)


def test_delay_reward_telescopes_to_total():
    # 延误项 telescoping: 关掉所有事件项 → Σ reward == -c_delay*整局 realized(每步 ≤0)
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig(eps_replatform=0.0, p_nostop=0.0, b_ontime=0.0,
                       p_incomplete=0.0, p_deadlock=0.0)
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    rewards = _replay_rewards(env, to_dyn_trains(built))
    assert rewards and not env.engine.deadlocked
    final = realized_weighted_delay(env.engine, cfg)
    assert final >= 0.0
    assert abs(sum(rewards) - (-cfg.c_delay * final)) < 1e-6
    assert all(r <= 1e-9 for r in rewards)


def test_reward_deterministic():
    net, params, dwell, ptc, built = _load()
    r1 = _replay_rewards(RewardedDynamicEnv(net, params, ptc, dwell), to_dyn_trains(built))
    r2 = _replay_rewards(RewardedDynamicEnv(net, params, ptc, dwell), to_dyn_trains(built))
    assert r1 == r2 and len(r1) > 0


def test_deadlock_penalty_wired_once():
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig(p_deadlock=50.0)
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    env.reset(to_dyn_trains(built))
    env.engine.deadlocked = True; env._was_deadlocked = False
    r_first = env._compute_reward(done=True)
    r_again = env._compute_reward(done=True)
    assert r_first <= -cfg.p_deadlock + 1e-9
    assert r_again > -cfg.p_deadlock + 1e-9


# ============ M10b ============
def test_platform_tier_levels():
    # enable_nostop=True 时三档全测; 默认(R4 暂缓)时 nostop→0(不误罚边界碎片车)
    cfg = RewardConfig(eps_replatform=2.0, p_nostop=10.0, enable_nostop=True)
    ptc = {"P4": 4, "P3": 3}
    t0 = DynTrain("T0", [], 0.0, stop_pids={4}); t0.path = ["a", "P4", "b"]
    assert platform_tier_reward(t0, ptc, cfg) == 0.0            # booked=0
    t1 = DynTrain("T1", [], 0.0, stop_pids={4}); t1.path = ["a", "P3", "b"]
    assert platform_tier_reward(t1, ptc, cfg) == -2.0          # 改台=−ε
    t2 = DynTrain("T2", [], 0.0, stop_pids={4}); t2.path = ["a", "b"]
    assert platform_tier_reward(t2, ptc, cfg) == -10.0         # 不停(enable)=−P_nostop
    t3 = DynTrain("T3", [], 0.0); t3.path = ["a", "P3", "b"]
    assert platform_tier_reward(t3, ptc, cfg) == 0.0           # 直通=0
    # R4: 默认 enable_nostop=False → 不停台**不罚**(暂缓到 M13 边界机制)
    cfg_off = RewardConfig(eps_replatform=2.0, p_nostop=10.0)
    assert cfg_off.enable_nostop is False
    assert platform_tier_reward(t2, ptc, cfg_off) == 0.0
    assert platform_tier_reward(t1, ptc, cfg_off) == -2.0      # 改台档不受影响


def test_on_time_bonus_window():
    cfg = RewardConfig(b_ontime=1.0, on_time_tol_s=300.0)
    ptc = {"P4": 4}
    t = DynTrain("T", [], 0.0, stop_pids={4}); t.path = ["a", "P4", "b"]
    t.reached_exit = True; t.planned_ref = 900.0; t.sim_occ = {"P4": 1000.0}
    assert on_time_bonus(t, ptc, cfg) == 1.0           # 1000 <= 900+300
    t.planned_ref = 600.0
    assert on_time_bonus(t, ptc, cfg) == 0.0           # 1000 > 600+300
    t.reached_exit = False; t.planned_ref = 900.0
    assert on_time_bonus(t, ptc, cfg) == 0.0           # 未完成不给


def test_stranded_incomplete_penalty():
    # 白盒: 构造两辆 done 车(一到出区口/一搁浅)→ 终局判负只罚搁浅车
    cfg = RewardConfig(p_incomplete=20.0)
    net, params, dwell, ptc, built = _load()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    env.reset(to_dyn_trains(built))
    done_ok = DynTrain("OK", [], 0.0); done_ok.done = True; done_ok.reached_exit = True
    stuck = DynTrain("STUCK", [], 0.0); stuck.done = False; stuck.reached_exit = False
    env.engine.trains = {"OK": done_ok, "STUCK": stuck}
    env._last_realized = realized_weighted_delay(env.engine, cfg)
    r = env._compute_reward(done=True)
    assert abs(r - (-cfg.p_incomplete)) < 1e-6          # 仅 STUCK 罚一次
    assert "STUCK" in env._finalized and "OK" in env._finalized


def test_replay_platform_penalty_small_and_deterministic():
    # 回放真实决策 → 大多停 booked 台 → 站台惩罚应很小(sanity); 且确定性
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    r1 = _replay_rewards(env, to_dyn_trains(built))
    assert not env.engine.deadlocked
    reached = sum(t.reached_exit for t in env.engine.trains.values())
    assert reached >= 30                                # replay 应几乎全完成
    assert len(env._finalized) == len(env.engine.trains)
    env2 = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    r2 = _replay_rewards(env2, to_dyn_trains(built))
    assert r1 == r2
