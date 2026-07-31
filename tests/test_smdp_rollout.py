"""M12b 沙盒测试(无 torch): SMDP 折扣/GAE 数学 · 场景划分(留出集不可污染) · rollout 记账.

torch 侧(PPO 更新 / KL-to-BC)在 `tests/test_ppo_smoke.py`, 沙盒 skip、本地 GPU 跑。
"""
from __future__ import annotations

import numpy as np
import pytest

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_greedy_set_policy, make_random_policy, run_episode
from derbysim.learn.scenarios import (TEST_START, TRAIN_END, build_episode_trains,
                                      congested_windows, load_assets)
from derbysim.learn.smdp import (beta_from_half_life, compute_gae, discounted_return, smdp_gamma)
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv


# ---------- SMDP 折扣 / GAE ----------
def test_gamma_is_one_by_default_and_time_based_otherwise():
    assert beta_from_half_life(None) == 0.0 and beta_from_half_life(0) == 0.0
    assert np.allclose(smdp_gamma([0.0, 3.0, 600.0], 0.0), 1.0)          # β=0 → 恒 1
    g = smdp_gamma([0.0, 300.0, 600.0], beta_from_half_life(300.0))
    assert np.allclose(g, [1.0, 0.5, 0.25])                              # 按**时间**折, 非按步数
    # 关键: 同样两步, WAIT 推进 600s 与 set 瞬时 0s 的折扣必须不同(D24 精化点②)
    assert smdp_gamma([0.0], beta_from_half_life(300))[0] > smdp_gamma([600.0], beta_from_half_life(300))[0]


def test_undiscounted_gae_returns_equal_reward_to_go():
    """γ=1、λ=1 时 returns 必须等于**剩余奖励之和** —— 训练目标 ≡ 汇报指标(§10-G 理由①)。"""
    r = [1.0, -2.0, 3.0, 0.5]
    v = [7.0, -1.0, 2.0, 9.0]                       # 任意 critic, 不应影响 λ=1 的 returns
    dt = [0.0, 120.0, 0.0, 900.0]
    dones = [0, 0, 0, 1]
    _adv, ret = compute_gae(r, v, dt, dones, last_value=0.0, half_life_s=None, lam=1.0)
    assert np.allclose(ret, np.cumsum(r[::-1])[::-1])
    assert discounted_return(r, dt, None) == pytest.approx(sum(r))


def test_discounting_makes_deferred_cost_cheaper():
    """§10-G 理由②的可执行版: 打折会让"把麻烦往后推"变便宜; γ=1 则不会。"""
    # dt[t] = **该步推进**的仿真秒数 → 要表达"代价在 20 分钟之后才付", 1200s 必须在**前一步**,
    # 这样第二步的奖励才被 γ(1200s) 折过一次。
    now_cost = [-100.0, 0.0]                        # 现在自己扛 100s
    later_cost = [0.0, -150.0]                      # 20 分钟后赔 150s
    dt = [1200.0, 0.0]
    undisc_now = discounted_return(now_cost, dt, None)
    undisc_later = discounted_return(later_cost, dt, None)
    assert undisc_now > undisc_later                # 不打折: 现在扛 100 优于以后赔 150 ✔
    d_now = discounted_return(now_cost, dt, 300.0)
    d_later = discounted_return(later_cost, dt, 300.0)
    assert d_later > d_now                          # 5min 半衰期: 反过来了 ✘(故默认 γ=1)


def test_gae_rejects_ragged_input():
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.0], [0.0, 0.0], [0, 1])


# ---------- 场景划分: 留出集不可污染 ----------
def test_splits_are_disjoint_and_respect_the_holdout():
    tr = congested_windows("train", top_k=99)
    va = congested_windows("val", top_k=99)
    te = congested_windows("test", top_k=99)
    S = lambda ws: {w["start"] for w in ws}
    assert tr and va and te
    assert not (S(tr) & S(te)), "留出集窗口漏进训练集"          # 曾真的漏过(2026-07-30 修)
    assert not (S(tr) & S(va)) and not (S(va) & S(te))
    assert all(w["start"] >= TEST_START for w in te)
    assert all(w["start"] < TRAIN_END for w in tr + va)


def test_ranking_by_n_late_beats_total_lateness_on_train_density():
    """排序统计量之争: 按总晚点分钟排会挑到"少数极端晚点"的清淡窗口(实测 10 车/8 决策)。"""
    by_n = congested_windows("all", top_k=5, rank_by="n_late")
    by_tot = congested_windows("all", top_k=5, rank_by="total_lateness")
    n_trains = lambda ws: [len(build_episode_trains(w, inject=False)) for w in ws]
    assert min(n_trains(by_n)) > min(n_trains(by_tot))


def test_delay_injection_is_deterministic_and_keeps_trains():
    w = congested_windows("train", top_k=1)[0]
    base = build_episode_trains(w, inject=False)
    a = build_episode_trains(w, seed=7)
    b = build_episode_trains(w, seed=7)
    assert len(a) == len(b) == len(base)
    assert [t.entry_time for t in a] == [t.entry_time for t in b]
    assert [t.route_chain for t in a] == [t.route_chain for t in b]      # 路由不变 → 可达性保持


# ---------- rollout 记账 ----------
def _env_and_adapter(cfg=None):
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    return RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg), ad


def test_rollout_records_consistent_transitions():
    env, ad = _env_and_adapter()
    w = congested_windows("train", top_k=1)[0]
    res = run_episode(env, ad, make_greedy_set_policy(), build_episode_trains(w, seed=0))
    assert res.transitions and not res.truncated
    assert res.transitions[-1].done and not any(t.done for t in res.transitions[:-1])
    assert all(t.dt >= 0 for t in res.transitions)
    assert all(0 <= t.option_idx < t.enc["n_options"] for t in res.transitions)
    assert res.metrics["n_collisions"] == 0                         # 结构性无碰撞
    assert res.metrics["n_steps"] == len(res.transitions)
    assert res.metrics["episode_reward"] == pytest.approx(sum(t.reward for t in res.transitions))


def test_rollout_is_deterministic():
    w = congested_windows("train", top_k=1)[0]

    def once():
        env, ad = _env_and_adapter()
        r = run_episode(env, ad, make_random_policy(np.random.default_rng(3)),
                        build_episode_trains(w, seed=1))
        return [(t.option_idx, t.reward, t.dt) for t in r.transitions], r.metrics
    a, ma = once()
    b, mb = once()
    assert a == b and ma == mb


def test_holding_a_train_now_costs_delay():
    """M12b 复审修的洞: 旧度量下"不给车办路"零延误 → 随机策略回报**高于**正常派路。

    真值口径: 同窗口同种子, 三条策略的完成数是引擎终态直接读的, 不用自制判据。
    """
    w = congested_windows("train", top_k=1)[0]

    def score(policy, cfg):
        env, ad = _env_and_adapter(cfg)
        r = run_episode(env, ad, policy, build_episode_trains(w, seed=0))
        return r.metrics["episode_reward"], r.metrics["n_reached_exit"]

    greedy, rnd = make_greedy_set_policy(), make_random_policy(np.random.default_rng(0))
    old_g, ng = score(greedy, RewardConfig(count_awaiting_hold=False))
    old_r, nr = score(rnd, RewardConfig(count_awaiting_hold=False))
    new_g, _ = score(greedy, RewardConfig())
    new_r, _ = score(rnd, RewardConfig())
    assert ng > nr                                   # 有路就办完成得多
    assert old_r > old_g                             # 旧度量: 少派路反而得分高(病)
    assert new_g > new_r                             # 新度量: 排序纠正 ✔
    # 不误伤正常派路: 新旧口径下"有路就办"的分数应基本一致(几个百分点内)。
    # (2026-07-30 二次修订后不再逐位相等 —— 新口径是**单一 hold 计时器**"从停下到真正驶入下一 TC",
    #  已覆盖排队段、不再叠加 wait_time; 对派路及时的策略两者只差 ~1.5%。)
    assert abs(new_g - old_g) / abs(old_g) < 0.05


# ---------- 人类参照线 + 固定评估协议 ----------
def test_signaller_replay_completes_every_completable_train():
    """**真值对照**: 信号员按自己的日志决策跑, 应把所有"结构上可完成"的车都送到出区口。

    这既是 D23 的核心对照锚, 也验证了"空出区口组 = 结构不可能完成"这个判据 ——
    实测 8 个真实拥堵窗口 0 遗漏; 这里取 3 个窗口锁死。
    """
    from derbysim.learn.rollout import evaluate_policy, signaller_replay_factory
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    ef = lambda: RewardedDynamicEnv(net, params, ptc, dwell)
    eps = [(str(w["start"]), (lambda w=w: build_episode_trains(w, inject=False)))
           for w in congested_windows("train", top_k=3)]
    r = evaluate_policy(ef, ad, signaller_replay_factory, eps)
    assert r["eval_completion_rate"] == pytest.approx(1.0)
    assert r["eval_deadlocks"] == 0 and r["eval_n_collisions"] == 0


def test_structurally_impossible_trains_are_not_penalised():
    """空出区口组的车既不进完成率分母, 也不吃 p_incomplete(连信号员都完成不了它们)。"""
    from derbysim.sim.reward import can_complete
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("train", top_k=1)[0]
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    res = run_episode(env, ad, make_greedy_set_policy(), build_episode_trains(w, seed=0))
    m = res.metrics
    imp = [t for t in env.engine.trains.values() if not can_complete(t)]
    assert m["n_impossible"] == len(imp) > 0                 # 该窗口确实有这类车
    assert m["n_completable"] + m["n_impossible"] == m["n_trains"]
    assert all(t.train_id not in env._finalized for t in imp)


def test_fixed_eval_does_not_let_policies_contaminate_each_other():
    """DynTrain 有状态: 评估集必须传**造车函数**, 否则第二个策略拿到"已跑完"的车。

    2026-07-30 实测撞到过: 三个策略评出完全相同的数字(完成率 1.0/延误 620/回报 0)。
    """
    from derbysim.learn.rollout import as_factory, evaluate_policy
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    ef = lambda: RewardedDynamicEnv(net, params, ptc, dwell)
    eps = [(str(w["start"]), (lambda w=w: build_episode_trains(w, seed=0)))
           for w in congested_windows("val", top_k=2)]
    g1 = evaluate_policy(ef, ad, as_factory(make_greedy_set_policy()), eps)
    rnd = evaluate_policy(ef, ad, as_factory(make_random_policy(np.random.default_rng(0))), eps)
    g2 = evaluate_policy(ef, ad, as_factory(make_greedy_set_policy()), eps)
    assert g1["eval_episode_reward"] == g2["eval_episode_reward"]      # 可重复
    assert g1["eval_episode_reward"] != rnd["eval_episode_reward"]     # 未被污染
    assert g1["eval_n_steps"] > 0 and rnd["eval_n_steps"] > 0


def test_set_cancel_oscillation_cannot_beat_dispatching():
    """回归: 反复 办路→撤销 的空转策略**不得**优于正常派路(2026-07-30 实测的 reward hack)。

    当时两个洞叠加: ①`cancel` 可在**同一瞬间**撤掉刚排的进路 → 时钟不走、"站着等"计时器不走 →
    延误恒为 0; ②`max_decisions` 截断时 `done=False`, 终局未完成罚**整段不执行**。
    结果: 空转策略 3000 步全卡在一个时刻、延误 0、得分 **+3.0**, 而正常派路 −1660。
    修: cancel 要求请求已排队 >0 仿真时间; 截断也补做终局结算(`finalize_truncated`)。
    """
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("val", top_k=1)[0]

    def oscillate(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "cancel":
                return i, 0.0, 0.0
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                return i, 0.0, 0.0
        return 0, 0.0, 0.0

    def run(pol):
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        res = run_episode(env, ad, pol, build_episode_trains(w, seed=0), max_decisions=3000)
        span = env.engine.now - min(t.entry_time for t in env.engine.trains.values())
        return res, span

    osc, osc_span = run(oscillate)
    gre, gre_span = run(make_greedy_set_policy())
    assert not osc.truncated, "空转仍能耗尽决策预算 = 时钟被冻住"
    assert osc_span > 0.5 * gre_span, "空转让仿真时间几乎不前进"
    assert osc.metrics["episode_reward"] <= gre.metrics["episode_reward"] + 1e-6
    assert osc.metrics["weighted_secondary_delay_s"] > 0, "空转不该把延误刷成 0"


def test_truncated_episode_still_pays_the_incomplete_penalty():
    """截断不能让策略白嫖完成率罚分: 终局结算必须补做, 且结算集 = 所有可完成的车。"""
    from derbysim.sim.reward import can_complete
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("val", top_k=1)[0]
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    res = run_episode(env, ad, make_greedy_set_policy(),
                      build_episode_trains(w, seed=0), max_decisions=40)
    assert res.truncated and res.transitions[-1].done and res.transitions[-1].dt == 0.0
    able = [t for t in env.engine.trains.values() if can_complete(t)]
    assert len(env._finalized) == len(able)                 # 终局账已结
    assert res.transitions[-1].reward < 0                   # 且确实是罚分


def test_lambda_one_delivers_full_credit_to_early_decisions():
    """为什么 λ 默认 1.0: 实测 λ=0.95 时第 0 步只"看到"总回报的 ~0.2%。

    episode 长(143–642 步)且奖励集中在末尾(最后一步占 15–16%), 而 λ=0.95 的有效回溯长度
    只有 1/(1−λ)=20 步 → 早期决策几乎拿不到信号(表现为训练曲线全平、KL 不动)。
    """
    rng = np.random.default_rng(0)
    n = 400
    rewards = list(rng.normal(-0.5, 0.5, n - 1)) + [-300.0]      # 末尾一大笔
    dts = [0.0] * n
    dones = [0] * (n - 1) + [1]
    zero_v = [0.0] * n
    total = sum(rewards)
    frac = {}
    for lam in (0.95, 0.99, 1.0):
        _a, ret = compute_gae(rewards, zero_v, dts, dones, 0.0, None, lam)
        frac[lam] = ret[0] / total
    assert frac[1.0] == pytest.approx(1.0)          # λ=1: 完整回溯
    assert frac[0.95] < 0.10                        # λ=0.95: 早期几乎看不到
    assert frac[0.95] < frac[0.99] < frac[1.0]


def test_reward_scale_does_not_change_the_advantage_ordering():
    """奖励缩放是**纯数值**手段: 优势归一化后排序不变, 不改变最优策略。"""
    from types import SimpleNamespace
    from derbysim.learn.rollout import Transition
    from derbysim.learn.smdp import prepare_batch
    cfgs = lambda sc: SimpleNamespace(reward_scale=sc, half_life_s=None, lam=1.0)
    rng = np.random.default_rng(1)
    trs = [Transition({"n_options": 2}, 0, 0.0, 0.0, float(r), 0.0, i == 39)
           for i, r in enumerate(rng.normal(-5, 3, 40))]
    a = prepare_batch(trs, cfgs(1.0))["adv"]
    b = prepare_batch(trs, cfgs(0.01))["adv"]
    assert np.array_equal(np.argsort(a), np.argsort(b))
    assert np.allclose(a * 0.01, b, atol=1e-6)


def test_advantage_encodes_position_when_critic_is_useless():
    """钉住"为什么必须先预热 critic": critic≈0 时优势几乎只编码步序, 不编码动作质量。

    实测(真实窗口, 4 个场景): 归一化优势与步序的相关系数 **0.77–0.96**;
    前 1/4 步均值 ≈ −0.9、后 1/4 ≈ +1.5 —— 早期动作一律被压、晚期一律被抬。
    γ=1 下回报全负、G_t 随 t 单调趋近 0 使然。EV(explained variance) 是它的直接探测器。
    """
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("val", top_k=1)[0]
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    from derbysim.learn.scenarios import episode_horizon
    res = run_episode(env, ad, make_greedy_set_policy(),
                      build_episode_trains(w, seed=0), horizon=episode_horizon(w))
    rw = np.array([t.reward for t in res.transitions]) * 0.01
    n = len(rw)
    adv, _ = compute_gae(rw, np.zeros(n), [t.dt for t in res.transitions],
                         [t.done for t in res.transitions], 0.0, None, 1.0)
    a = (adv - adv.mean()) / (adv.std() + 1e-8)
    rho = np.corrcoef(np.arange(n), a)[0, 1]
    assert rho > 0.6, f"零 critic 下优势应强烈随步序单调(实测 0.77-0.96), 实得 {rho:.3f}"
    assert a[:n // 4].mean() < 0 < a[-n // 4:].mean()
