"""M10 奖励模块: 稠密次生延误(每步差分) + 站台分档 + on-time + 死锁/搁浅 terminal.

诚实口径(docs/M10_DESIGN.md §2/§5, PROJECT_BRIEF §7 更正):
    次生延误=**小的真实目标项**(非"不改变最优解的纯 shaping")——想让模型少造次生延误(D21),
    故必须真进目标, 只是权重小; 发放=每步差分。**不主张势函数策略不变性**(自相矛盾)。
    对 M9 绝对量膨胀~4× 的鲁棒靠: 只报相对量(共模抵消) + M13 overlap 稳健性。

复审修订(§12, 2026-07-23):
    - **R1**: realized 延误的 in-progress 项只对**未 done** 车计(引擎 M9 quirk: 车带未落锁排队 route
      完成时残留 blocked_since, 不影响 wait_time; 若对 done 车加 now-blocked_since → 随时钟膨胀的幻延误)。
    - **R4**: 站台 −P_nostop **暂缓**(enable_nostop 默认 False): 实测罚到窗口边界碎片车(如 1O06 07:00
      到达=窗口起点, 站台服务在窗外), 干净判别需边界机制 → 留 M13。现只保 booked=0 / 改台=−ε 两档。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from derbysim.sim.env_dynamic import DynamicDerbyEnv, DynTrain

# TOPS 优先级权重(D16: 1特快>2普客>3包裹>5ECS>6货). 温和默认, 主结论靠灵敏度扫描, 不押数值。
DEFAULT_TOPS_WEIGHTS = {"1": 4.0, "2": 3.0, "3": 2.0, "5": 1.5, "6": 1.0}
DEFAULT_TOPS_FALLBACK = 2.0


@dataclass
class RewardConfig:
    # 稠密延误(小)
    c_delay: float = 0.01                      # 100s 加权延误 ≈ 1
    tops_weights: dict = field(default_factory=lambda: dict(DEFAULT_TOPS_WEIGHTS))
    tops_fallback: float = DEFAULT_TOPS_FALLBACK
    # 站台分档(软)
    eps_replatform: float = 2.0                # 改台小扣
    p_nostop: float = 10.0                     # 客运不停任何台(暂缓, enable_nostop 控)
    enable_nostop: bool = False                # R4: 暂缓 −P_nostop 到 M13 边界机制(默认关)
    # on-time(小正) —— 粗信号, 容差可调
    b_ontime: float = 1.0
    on_time_tol_s: float = 300.0               # 5 min 容差(灵敏度旋钮)
    # 安全/完成(大, 结构性)
    p_deadlock: float = 50.0                   # 死锁判负(D6 唯一进 reward 的安全项)
    p_incomplete: float = 20.0                 # 未完成/搁浅每车判负

    def lam(self, service_class) -> float:
        return self.tops_weights.get(str(service_class), self.tops_fallback)


def realized_weighted_delay(engine, cfg: RewardConfig) -> float:
    """整局累计加权次生延误(含**未 done 车**的在途阻塞, R1), 单调不减 → 每步差分 ≤ 0 = 成本。"""
    total = 0.0
    for t in engine.trains.values():
        w = t.wait_time
        if t.blocked_since is not None and not t.done:      # R1: done 车 in-progress 不计(防幻延误)
            w += engine.now - t.blocked_since
        total += cfg.lam(t.service_class) * w
    return total


def _expects_stop(t) -> bool:
    return bool(t.stop_pids) or (t.booked_platform is not None)


def _served_platforms(t, plat_tc_to_pid: dict) -> set:
    return {plat_tc_to_pid[tc] for tc in t.path if tc in plat_tc_to_pid}


def platform_tier_reward(t, plat_tc_to_pid: dict, cfg: RewardConfig) -> float:
    """站台分档(客运): booked=0 / 其它台=−ε / 不停=−P_nostop(仅 enable_nostop, R4 默认关)。直通=0。"""
    if not _expects_stop(t):
        return 0.0
    booked = set(t.stop_pids)
    if t.booked_platform is not None:
        booked.add(t.booked_platform)
    served = _served_platforms(t, plat_tc_to_pid)
    if served & booked:
        return 0.0
    if served:
        return -cfg.eps_replatform
    return -cfg.p_nostop if cfg.enable_nostop else 0.0       # R4: 暂缓, 不误罚边界碎片车


def on_time_bonus(t, plat_tc_to_pid: dict, cfg: RewardConfig) -> float:
    """完成车在 planned_ref+容差内到达测量点(客运=首个服务站台占用时刻/直通=出区末TC)→ +b_ontime。"""
    if getattr(t, "planned_ref", None) is None or not t.reached_exit:
        return 0.0
    t_measure = None
    if _expects_stop(t):
        for tc in t.path:
            if tc in plat_tc_to_pid and tc in t.sim_occ:
                t_measure = t.sim_occ[tc]
                break
    if t_measure is None and t.path and t.path[-1] in t.sim_occ:
        t_measure = t.sim_occ[t.path[-1]]
    if t_measure is None:
        return 0.0
    return cfg.b_ontime if t_measure <= t.planned_ref + cfg.on_time_tol_s else 0.0


def to_dyn_trains(built) -> list[DynTrain]:
    """[(EnvTrain, occ), ...] → DynTrain, 携带 service_class/planned_ref/booked_platform(F2)。"""
    out = []
    for b in built:
        et = b[0] if isinstance(b, (tuple, list)) else b
        out.append(DynTrain(
            et.train_id, list(et.route_chain), et.entry_time,
            set(et.stop_pids), et.booked_platform,
            planned_ref=et.planned_ref, service_class=et.service_class,
        ))
    return out


class RewardedDynamicEnv(DynamicDerbyEnv):
    """M10: 在 M9 反事实环境上叠加奖励(非侵入子类, 单一接线点 = step 前后快照差)。

    每步 = -c_delay·Δ(realized 加权次生延误)
           + Σ[本步新完成车] (站台分档 + on-time)
           - [新判负死锁] p_deadlock
           - [终局未完成车] p_incomplete。
    确定性: reward 是引擎快照的纯函数; 完成/站台/on-time 逐完成归因, 搁浅在终局。
    """

    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 reward_config: RewardConfig | None = None) -> None:
        super().__init__(net, params, plat_tc_to_pid, dwell_by_platform)
        self.cfg = reward_config or RewardConfig()
        self._last_realized = 0.0
        self._was_deadlocked = False
        self._finalized: set = set()

    def reset(self, trains):
        obs, info = super().reset(trains)
        self._last_realized = realized_weighted_delay(self.engine, self.cfg)
        self._was_deadlocked = self.engine.deadlocked
        self._finalized = set()
        return obs, info

    def step(self, action):
        obs, _r0, done, info = super().step(action)
        r = self._compute_reward(done)
        return obs, r, done, info

    def _compute_reward(self, done: bool) -> float:
        cur = realized_weighted_delay(self.engine, self.cfg)
        r = -self.cfg.c_delay * (cur - self._last_realized)
        self._last_realized = cur
        if self.engine.deadlocked and not self._was_deadlocked:
            self._was_deadlocked = True
            return r - self.cfg.p_deadlock
        for t in self.engine.trains.values():
            if t.train_id in self._finalized:
                continue
            if t.done and t.reached_exit:
                r += platform_tier_reward(t, self.plat_tc_to_pid, self.cfg)
                r += on_time_bonus(t, self.plat_tc_to_pid, self.cfg)
                self._finalized.add(t.train_id)
        if done and not self.engine.deadlocked:
            for t in self.engine.trains.values():
                if t.train_id in self._finalized:
                    continue
                if not t.reached_exit:
                    r -= self.cfg.p_incomplete
                    self._finalized.add(t.train_id)
        return r
