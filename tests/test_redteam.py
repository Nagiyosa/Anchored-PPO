"""M12b 红队审计：把"发现 reward hacking"变成常规检查（沙盒可跑，无 torch）。

**背景**: 首次真训时 agent 的评估回报逼近人类而完成率只有 0.217 —— 学到的是漏洞不是调度，
而当时**所有单元测试全绿**。这类目标错配只在闭环长程交互下显形，逐个症状去修是打地鼠。
故立**不变式 + 常备红队**：每改一次奖励/环境，这个文件都要重跑。

**两条结构性规则**（比逐个补丁根本）：
  - **终局结算与终止原因无关**：自然跑完 / 死锁判负 / 截断，同一套账都结；`p_deadlock` 额外加而非替代。
  - **零仿真时间内不得无限动作**：`cancel` 要求请求已排队 >0 仿真时间。
  加上**固定时间地平线**（窗末+drain）与**入区排队即起计**，让"少干活"无法把欠账截短。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.redteam import RED_TEAM, REFERENCES
from derbysim.learn.rollout import make_greedy_set_policy, run_episode
from derbysim.learn.scenarios import (build_episode_trains, congested_windows, episode_horizon,
                                      load_assets)
from derbysim.sim.env_dynamic import DynTrain
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv, can_complete

SPEC_SEEDS = (0,)


def _audit():
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    specs = [(w, sd) for w in congested_windows("val", top_k=2) for sd in SPEC_SEEDS]
    out = {}
    for name, fac in list(REFERENCES.items()) + list(RED_TEAM.items()):
        c = d = r = col = 0.0
        for w, sd in specs:
            env = RewardedDynamicEnv(net, params, ptc, dwell)
            m = run_episode(env, ad, fac(env), build_episode_trains(w, seed=sd),
                            horizon=episode_horizon(w)).metrics
            c += m["completion_rate"]; d += m["weighted_secondary_delay_s"]; r += m["episode_reward"]
            col += m["n_collisions"]
        n = len(specs)
        out[name] = {"completion": c / n, "delay": d / n, "reward": r / n, "collisions": col}
    return out


@pytest.fixture(scope="module")
def audit():
    return _audit()


def test_signaller_beats_every_red_team_policy(audit):
    """人类是对照锚(D23): 任何钻空子的脚本策略都不该赢过信号员。"""
    human = audit["signaller_replay"]["reward"]
    beaten = {k: v["reward"] for k, v in audit.items()
              if k in RED_TEAM and v["reward"] > human}
    assert not beaten, f"红队策略赢过了信号员 → 奖励有洞: {beaten}"


def test_no_policy_scores_higher_while_completing_less(audit):
    """核心不变式: **完成率更差却得分更高 = 奖励错了**。

    注意反向不成立 —— 完成率高但拖延巨大的策略(late_dispatch)得分低是**正确**的,
    所以不能要求"完成率与回报完全同序"。
    """
    ref = audit["greedy_set"]
    bad = {k: v for k, v in audit.items()
           if k in RED_TEAM and v["completion"] < ref["completion"] - 1e-9
           and v["reward"] > ref["reward"] + 1e-6}
    assert not bad, f"这些策略完成得更少却得分更高: {bad}"


def test_no_policy_freezes_the_clock(audit):
    """时钟被冻住的征兆: 延误接近 0 却完成率很低(空转 set/cancel 曾如此)。"""
    for k, v in audit.items():
        if v["completion"] < 0.6:
            assert v["delay"] > 1000.0, f"{k}: 完成率 {v['completion']:.3f} 却几乎没有延误 = 时钟没走"


def test_stalling_is_the_worst_strategy(audit):
    """什么都不做必须是最差的 —— 否则"不作为"就是可行解。"""
    worst = min(audit.items(), key=lambda kv: kv[1]["reward"])[0]
    assert worst == "always_wait", f"最差的应是 always_wait, 实际是 {worst}"


def _terminal_env(n, deadlocked):
    net, params, dwell, ptc = load_assets()
    cfg = RewardConfig()
    env = RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg)
    trains = {}
    for i in range(n):
        t = DynTrain(f"T{i}", [], 0.0, service_class="1")
        t.done = False; t.reached_exit = False
        t.goal = SimpleNamespace(exit_group={"5093"}); t.held_time = 3000.0
        trains[t.train_id] = t

    class _E:
        now = 0.0
        def __init__(s):
            s.trains = trains; s.deadlocked = deadlocked
    env.engine = _E(); env._last_realized = 0.0
    env._was_deadlocked = False; env._finalized = set()
    return env, cfg


def test_deadlock_cannot_be_used_to_escape_the_bill():
    """自杀死锁曾是最优解: 未完成罚整段被跳过, 只付 p_deadlock。

    实测(修前): 25 辆全搁浅时正常收尾 −3500、死锁收尾 −3050(**划算 +450**),
    而"有路就办"跑完一个真实拥堵窗口约 −760 ⇒ 故意死锁更优。
    规则改为"终局结算与终止原因无关"后, 死锁必须**更差**(多付 p_deadlock)。
    """
    n = 25
    a, cfg = _terminal_env(n, False)
    b, _ = _terminal_env(n, True)
    r_normal = a._compute_reward(done=True)
    r_deadlock = b._compute_reward(done=True)
    assert r_deadlock == pytest.approx(r_normal - cfg.p_deadlock)
    assert r_deadlock < r_normal
    assert len(b._finalized) == n            # 死锁下未完成罚照收


def test_horizon_makes_stranding_cost_the_same_span_for_everyone():
    """固定地平线: 被晾着的车按"站到共同地平线"计, 少干活不能把欠账截短。"""
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("val", top_k=1)[0]
    H = episode_horizon(w)
    env_h = RewardedDynamicEnv(net, params, ptc, dwell)
    m_h = run_episode(env_h, ad, make_greedy_set_policy(),
                      build_episode_trains(w, seed=0), horizon=H).metrics
    env_n = RewardedDynamicEnv(net, params, ptc, dwell)
    m_n = run_episode(env_n, ad, make_greedy_set_policy(),
                      build_episode_trains(w, seed=0)).metrics
    assert m_h["weighted_secondary_delay_s"] >= m_n["weighted_secondary_delay_s"]
    assert env_h.horizon == H and env_n.horizon is None


def test_no_policy_can_cause_a_collision(audit):
    """D6 红线: **任何**策略下碰撞恒为 0 —— 由预留结构性保证, 不靠奖励劝阻。

    2026-08-01 破过一次: 进路链是策略**在线**选的, 策略可以把车导成环
    (`RDC5061A→RDC5075A→RDC5083A` = `TDAB→TDAC→TDAB→TDAC→TDAB→TDAC`), 而
    `_release_behind(idx-2)` 假设"路径不回头" → 车释放掉**自己脚下**的锁 → 别的车锁走 →
    本车再进 = 碰撞。**不是数据问题**(98 天 18650 辆车的原始链 0 条有重复 TC)。
    旧的评估汇总按局**平均**碰撞数(3 局 1 次 → 0.33 → 打印成 0)把它藏住了, 改报总量后现形。
    `loop_the_train` 已进红队常备。
    """
    bad = {k: v["collisions"] for k, v in audit.items() if v["collisions"]}
    assert not bad, f"这些策略造成了碰撞(D6 破了): {bad}"
