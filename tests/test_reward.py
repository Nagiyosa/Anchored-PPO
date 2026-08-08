"""M10 reward 测试:
   M10a 稠密延误(差分, TOPS λ, in-progress) + 死锁; M10b 站台三档 + on-time + 搁浅判负 + determinism."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynTrain, PanelAction
from derbysim.sim.reward import (RewardConfig, RewardedDynamicEnv, can_complete,
                                 has_reached_goal, realized_weighted_delay,
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


def test_realized_weighted_delay_priority_and_inprogress_new_measure():
    """默认口径(2026-07-30 起) = **单一 hold 计时器**: 车"走不动"的总时长, 到真正驶入下一 TC 才结清。

    它已覆盖"排队等锁"那段, 故不再叠加 `wait_time`(否则重复计); 且 set/cancel 都不能重置它 ——
    实测旧口径下反复 办路→撤销 能把延误刷成 0(空转策略拿 +3.0 而正常派路 −1660)。
    """
    cfg = RewardConfig(tops_weights={"1": 4.0, "6": 1.0}, tops_fallback=2.0)
    a = DynTrain("A", [], 0.0, service_class="1"); a.held_time = 100.0
    b = DynTrain("B", [], 0.0, service_class="6"); b.held_time = 100.0
    assert realized_weighted_delay(_stub(0.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * 100
    b.hold_start = 0.0                                   # 仍在"走不动"中 → 计在途
    assert realized_weighted_delay(_stub(50.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * (100 + 50)
    b.done = True                                        # R1: done 车不计在途(防幻延误)
    assert realized_weighted_delay(_stub(50.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * 100


def test_realized_weighted_delay_old_measure_still_available():
    """旧口径(只算排队等锁)由 `count_awaiting_hold=False` 保留, 供 A/B 对照。"""
    cfg = RewardConfig(tops_weights={"1": 4.0, "6": 1.0}, tops_fallback=2.0,
                       count_awaiting_hold=False)
    a = DynTrain("A", [], 0.0, service_class="1"); a.wait_time = 100.0
    b = DynTrain("B", [], 0.0, service_class="6"); b.wait_time = 100.0
    assert realized_weighted_delay(_stub(0.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * 100
    b.blocked_since = 0.0
    assert realized_weighted_delay(_stub(50.0, [a, b]), cfg) == 4.0 * 100 + 1.0 * (100 + 50)


def test_delay_reward_telescopes_to_total():
    # 延误项 telescoping: 关掉所有事件项 → Σ reward == -c_delay*整局 realized(每步 ≤0)
    # **per_step 模式专属**(逐步 Δ 延误); per_train(D58 默认)是逐车终局, 不 telescoping。
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig(reward_mode="per_step", eps_replatform=0.0, p_nostop=0.0, b_ontime=0.0,
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
    t0 = DynTrain("T0", [], 0.0, stop_pids={4}); t0.served_platforms = {4}
    assert platform_tier_reward(t0, ptc, cfg) == 0.0            # booked=0
    t1 = DynTrain("T1", [], 0.0, stop_pids={4}); t1.served_platforms = {3}
    assert platform_tier_reward(t1, ptc, cfg) == -2.0          # 改台=−ε
    t2 = DynTrain("T2", [], 0.0, stop_pids={4})
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


def test_terminal_goal_requires_completed_dwell_not_platform_entry_only():
    t = DynTrain("TERM", [], 0.0)
    t.goal = SimpleNamespace(exit_group=set(), terminates_at_platform=True,
                             terminal_platforms=frozenset({6}))
    t.served_platform = True
    t.served_platforms = {6}
    assert not has_reached_goal(t)
    t.reached_terminal = True
    assert has_reached_goal(t)


def _goal(exit_group):
    return SimpleNamespace(exit_group=set(exit_group) if exit_group else set())


def test_stranded_incomplete_penalty():
    """终局判负只罚**结构上本可完成**却被搁浅的车。

    2026-07-30 修正(M12b 复审): 原测试的 STUCK 没有 goal, 其实属于"不可能完成"那一类;
    现在分成三辆——到达 / 可完成但被搁浅 / 结构不可能(空出区口组)——只有中间那辆该罚。
    真值依据: 信号员日志决策回放 8 个真实拥堵窗口, 在"有出区口目标"的车里 0 遗漏。
    """
    cfg = RewardConfig(p_incomplete=20.0)
    net, params, dwell, ptc, built = _load()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    env.reset(to_dyn_trains(built))
    done_ok = DynTrain("OK", [], 0.0); done_ok.done = True; done_ok.reached_exit = True
    done_ok.goal = _goal(["5093"])
    stuck = DynTrain("STUCK", [], 0.0); stuck.done = False; stuck.reached_exit = False
    stuck.goal = _goal(["5093"])                      # 有出区口目标 = 本可完成 → 该罚
    impossible = DynTrain("NOEXIT", [], 0.0); impossible.done = True
    impossible.reached_exit = False; impossible.goal = _goal([])   # 空出区口组 → 不该罚
    assert can_complete(stuck) and not can_complete(impossible)
    env.engine.trains = {"OK": done_ok, "STUCK": stuck, "NOEXIT": impossible}
    env._last_realized = realized_weighted_delay(env.engine, cfg)
    r = env._compute_reward(done=True)
    # per_train(D58 默认): OK 到站 +base_completion; STUCK 搁浅 −(w_d·延误到地平线 + p_incomplete)
    # (此处 STUCK 无 hold_start → 延误 0 → 只 −p_incomplete); NOEXIT 结构不可能豁免。
    assert abs(r - (cfg.base_completion - cfg.p_incomplete)) < 1e-6
    assert "STUCK" in env._finalized and "OK" in env._finalized
    assert "NOEXIT" not in env._finalized             # 结构不可能的车不进结算


def test_replay_platform_penalty_small_and_deterministic():
    # 回放真实决策 → 大多停 booked 台 → 站台惩罚应很小(sanity); 且确定性
    net, params, dwell, ptc, built = _load()
    cfg = RewardConfig()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    r1 = _replay_rewards(env, to_dyn_trains(built))
    assert not env.engine.deadlocked
    reached = sum(t.reached_exit for t in env.engine.trains.values())
    assert reached >= 30                                # replay 应几乎全完成
    # 结算集 = 到达的车 + 本可完成却搁浅的车; **结构不可能完成的车不结算**(2026-07-30 修正)
    able = [t for t in env.engine.trains.values() if can_complete(t)]
    assert len(env._finalized) == len(able)
    assert len(able) <= len(env.engine.trains)
    env2 = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    r2 = _replay_rewards(env2, to_dyn_trains(built))
    assert r1 == r2
