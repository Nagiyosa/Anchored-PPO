"""M12b rollout 采集器：用策略驱动 M10 `RewardedDynamicEnv`，逐决策记 SMDP 转移（无 torch）。

策略以**可调用对象注入**(`act_fn(enc) -> (option_idx, logp, value)`) → 本模块与 torch 解耦,
沙盒可用随机/启发式策略把记账逻辑完整单测; torch 策略在 `ppo.py` 里包一层即可。

**Δt 的含义(D24/§10-G)**: 一个 step 推进的**仿真秒数** = `engine.now` 前后之差。
`set`/`cancel` 通常瞬时(Δt≈0), `WAIT` 是时间延展宏动作(Δt 可达几分钟) —— SMDP 折扣按它算。

**安全**: 动作只能从 `PanelDecision.options` 里挑(env 已过 联锁∩可达∩死锁前瞻), 策略给的是**下标**,
不构造动作 → 结构性安全(D6)不依赖网络。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from derbysim.sim.reward import RewardConfig, can_complete, realized_weighted_delay


@dataclass
class Transition:
    enc: dict                 # panel.encode() 的输出(该决策的面板张量)
    option_idx: int           # 选中的选项在 PanelDecision.options 里的下标
    logp: float
    value: float
    reward: float
    dt: float                 # 本 step 推进的仿真秒数(SMDP)
    done: bool


@dataclass
class EpisodeResult:
    transitions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    truncated: bool = False


def episode_metrics(engine, cfg: RewardConfig, plat_tc_to_pid: dict,
                    horizon: float | None = None) -> dict:
    """闭环质量指标(D23 的价值锚点) —— 从引擎终态直接读, 不用自制判据。

    次生延误**只报相对量时才有意义**(M9 自由时序 + overlap 使绝对量膨胀 ~4×, M10 §5),
    这里记绝对值仅供**同环境内**跨策略相减。
    """
    trains = list(engine.trains.values())
    able = [t for t in trains if can_complete(t)]        # 结构上可能到达出区口的车(分母)
    reached = [t for t in trains if t.reached_exit]
    stops = [t for t in trains if (t.stop_pids or t.booked_platform is not None)]
    booked_ok = 0
    for t in stops:
        served = {plat_tc_to_pid[tc] for tc in t.path if tc in plat_tc_to_pid}
        want = set(t.stop_pids) | ({t.booked_platform} if t.booked_platform is not None else set())
        booked_ok += bool(served & want)
    return {
        "n_trains": len(trains),
        "n_completable": len(able),                       # 分母: 排除空出区口组的结构不可能车
        "n_impossible": len(trains) - len(able),
        "n_reached_exit": len(reached),
        "completion_rate": (len(reached) / len(able)) if able else 1.0,
        "deadlocked": bool(engine.deadlocked),
        "n_collisions": len(engine.collisions),          # 结构性应恒为 0
        "weighted_secondary_delay_s": float(realized_weighted_delay(engine, cfg, horizon)),
        "raw_wait_s": float(sum(t.wait_time for t in trains)),
        "n_stop_trains": len(stops),
        "n_served_booked": booked_ok,
        "n_decisions": int(engine.n_decisions),
        "n_actions": int(engine.n_actions),
        "sim_end_s": float(engine.now),
    }


def run_episode(env, adapter, act_fn, trains, max_decisions: int = 5000,
                horizon: float | None = None) -> EpisodeResult:
    """跑完一局: reset → 逐决策(编码→策略选下标→env.step) → 收集转移 + 闭环指标。

    `done=True` 只在**自然终局**(全车终局 / 死锁判负)时置位 → GAE 的 bootstrap 值为 0 是正确的。
    因 `max_decisions` 提前收尾时置 `truncated=True`(**如实标注**, 不当成终局)。
    """
    try:
        obs, _info = env.reset(trains, horizon=horizon)
    except TypeError:
        obs, _info = env.reset(trains)                 # 裸 DynamicDerbyEnv 无 horizon
    res = EpisodeResult()
    while not env.done and len(res.transitions) < max_decisions:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        idx, logp, value = act_fn(enc)
        if not (0 <= idx < len(dec.options)):
            raise IndexError(f"policy returned option index {idx} outside 0..{len(dec.options)-1}")
        t0 = env.engine.now
        obs, reward, done, _info = env.step(dec.options[idx])
        res.transitions.append(Transition(enc, int(idx), float(logp), float(value),
                                          float(reward), float(env.engine.now - t0), bool(done)))
        if done:
            break
    res.truncated = (not env.done) and len(res.transitions) >= max_decisions
    if res.truncated and res.transitions and hasattr(env, "finalize_truncated"):
        # 截断也要结终局账(未完成判负), 否则"拖到截断"= 白嫖掉完成率罚分。
        # 记成一条 dt=0 的终止转移 → GAE 以 0 bootstrap, 与自然终局一致。
        tail = env.finalize_truncated()
        last = res.transitions[-1]
        res.transitions.append(Transition(last.enc, last.option_idx, last.logp, last.value,
                                          float(tail), 0.0, True))
    res.metrics = episode_metrics(env.engine, env.cfg, env.plat_tc_to_pid,
                                  getattr(env, "horizon", None))
    res.metrics["truncated"] = res.truncated
    res.metrics["n_steps"] = len(res.transitions)
    res.metrics["episode_reward"] = float(sum(t.reward for t in res.transitions))
    # 动作构成: 这个任务的难点在**何时办**(实测信号员主动 WAIT 36–44% 且 100% 完成,
    # 而"一有机会就派路"只完成 71%)。BC 热启动只迁移选路头, WAIT/cancel 从零学 →
    # 必须能看到策略在时机上偏到哪去了。
    n = max(len(res.transitions), 1)
    for kind in ("wait", "set", "cancel"):
        res.metrics[f"frac_{kind}"] = sum(
            1 for t in res.transitions if t.enc["slots"][t.option_idx].kind == kind) / n
    return res


def make_random_policy(rng):
    """均匀随机合法策略(基线下界 + 沙盒测记账用)。value/logp 取真实均匀分布的值。"""
    import math

    def act(enc):
        n = enc["n_options"]
        i = int(rng.integers(n))
        return i, -math.log(n), 0.0
    return act


def make_signaller_replay_policy(env):
    """人类参照(M12d 基线之一): 按日志真实决策办路; 无可对应的日志动作则 WAIT。

    `PanelDecision.logged` 给出每辆待决车在日志里的下一条进路 —— 这是**信号员本人的选择**,
    在同一环境/同一选项集上跑, 即 D23 的核心对照锚。
    """
    def act(enc):
        dec = env.decision()
        if dec is not None:
            for i, (slot, opt) in enumerate(zip(enc["slots"], dec.options)):
                if (slot.kind == "set"
                        and dec.logged.get(slot.train_id) == slot.route_id):
                    return i, 0.0, 0.0
        return 0, 0.0, 0.0                                # options[0] 恒为 WAIT
    return act


def make_greedy_set_policy():
    """确定性启发式(冒烟用): 有 set 就办第一条, 否则 WAIT。**不是** M12d 的正式基线。"""
    def act(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act


def as_factory(act_fn):
    """无状态策略 → 统一的 `policy_factory(env) -> act_fn` 形态。"""
    return lambda _env: act_fn


def signaller_replay_factory(env):
    """人类参照的工厂形态(需要 env 才能读 `PanelDecision.logged`)。"""
    return make_signaller_replay_policy(env)


def collect_rollout(env_factory, adapter, policy_factory, episode_iter, target_steps: int,
                    max_decisions: int = 3000):
    """按**步数预算**收集 rollout(而非固定局数)。

    冒烟轮实测每 iter 的步数在 180–4586 之间晃(25×) —— batch 大小随机会让 PPO 本身不稳,
    且"这一轮回报"被"抽到哪个窗口"主导而非策略质量。改成凑够 `target_steps` 再更新。
    `episode_iter` 产出 **(标签, 造车函数)** —— 必须是**函数**不是车列表: `DynTrain` 有状态
    (引擎推进 route_ptr/path), 复用同一批对象跑第二次会拿到"已跑完"的车 → 完成率/延误全是假的
    (2026-07-30 实测撞到: 三个策略评出完全相同的数字)。`policy_factory(env) -> act_fn`(见 `as_factory`)。
    """
    transitions, metrics = [], []
    for item in episode_iter:
        tag, make_trains = item[0], item[1]
        horizon = item[2] if len(item) > 2 else None     # (标签, 造车函数[, 地平线])
        env = env_factory()
        res = run_episode(env, adapter, policy_factory(env), make_trains(),
                          max_decisions=max_decisions, horizon=horizon)
        transitions.extend(res.transitions)
        m = dict(res.metrics); m["episode"] = tag
        metrics.append(m)
        if len(transitions) >= target_steps:
            break
    return transitions, metrics


def evaluate_policy(env_factory, adapter, policy_factory, episodes,
                    max_decisions: int = 3000) -> dict:
    """在**固定** (窗口, 注入种子) 上评估 → 跨 iteration 可比的曲线。

    训练时每 iter 随机抽窗口, 窗口间难度差很大(实测步数 180–4586) → 训练回报不可比。
    评估必须固定场景, 这也是 M12d/M13 "同一环境/同一评估窗口/同一指标"的要求。
    `episodes` 产出 **(标签, 造车函数)** —— 同上, 必须每局新造车, 否则跨策略互相污染。
    """
    rows = []
    for item in episodes:
        tag, make_trains = item[0], item[1]
        horizon = item[2] if len(item) > 2 else None
        env = env_factory()
        m = dict(run_episode(env, adapter, policy_factory(env), make_trains(),
                             max_decisions=max_decisions, horizon=horizon).metrics)
        m["episode"] = tag
        rows.append(m)
    keys = ("completion_rate", "weighted_secondary_delay_s", "episode_reward",
            "n_reached_exit", "n_collisions", "n_steps",
            "frac_wait", "frac_set", "frac_cancel")
    out = {f"eval_{k}": float(sum(r[k] for r in rows) / len(rows)) for k in keys} if rows else {}
    out["eval_deadlocks"] = int(sum(bool(r["deadlocked"]) for r in rows))
    out["eval_episodes"] = rows
    return out
