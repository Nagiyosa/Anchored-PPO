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

from derbysim.sim.reward import RewardConfig, realized_weighted_delay


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


def episode_metrics(engine, cfg: RewardConfig, plat_tc_to_pid: dict) -> dict:
    """闭环质量指标(D23 的价值锚点) —— 从引擎终态直接读, 不用自制判据。

    次生延误**只报相对量时才有意义**(M9 自由时序 + overlap 使绝对量膨胀 ~4×, M10 §5),
    这里记绝对值仅供**同环境内**跨策略相减。
    """
    trains = list(engine.trains.values())
    reached = [t for t in trains if t.reached_exit]
    stops = [t for t in trains if (t.stop_pids or t.booked_platform is not None)]
    booked_ok = 0
    for t in stops:
        served = {plat_tc_to_pid[tc] for tc in t.path if tc in plat_tc_to_pid}
        want = set(t.stop_pids) | ({t.booked_platform} if t.booked_platform is not None else set())
        booked_ok += bool(served & want)
    return {
        "n_trains": len(trains),
        "n_reached_exit": len(reached),
        "completion_rate": (len(reached) / len(trains)) if trains else 0.0,
        "deadlocked": bool(engine.deadlocked),
        "n_collisions": len(engine.collisions),          # 结构性应恒为 0
        "weighted_secondary_delay_s": float(realized_weighted_delay(engine, cfg)),
        "raw_wait_s": float(sum(t.wait_time for t in trains)),
        "n_stop_trains": len(stops),
        "n_served_booked": booked_ok,
        "n_decisions": int(engine.n_decisions),
        "n_actions": int(engine.n_actions),
        "sim_end_s": float(engine.now),
    }


def run_episode(env, adapter, act_fn, trains, max_decisions: int = 5000) -> EpisodeResult:
    """跑完一局: reset → 逐决策(编码→策略选下标→env.step) → 收集转移 + 闭环指标。

    `done=True` 只在**自然终局**(全车终局 / 死锁判负)时置位 → GAE 的 bootstrap 值为 0 是正确的。
    因 `max_decisions` 提前收尾时置 `truncated=True`(**如实标注**, 不当成终局)。
    """
    obs, _info = env.reset(trains)
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
    res.metrics = episode_metrics(env.engine, env.cfg, env.plat_tc_to_pid)
    res.metrics["truncated"] = res.truncated
    res.metrics["n_steps"] = len(res.transitions)
    res.metrics["episode_reward"] = float(sum(t.reward for t in res.transitions))
    return res


def make_random_policy(rng):
    """均匀随机合法策略(基线下界 + 沙盒测记账用)。value/logp 取真实均匀分布的值。"""
    import math

    def act(enc):
        n = enc["n_options"]
        i = int(rng.integers(n))
        return i, -math.log(n), 0.0
    return act


def make_greedy_set_policy():
    """确定性启发式(冒烟用): 有 set 就办第一条, 否则 WAIT。**不是** M12d 的正式基线。"""
    def act(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act
