"""M10 奖励模块: 稠密次生延误(每步差分) + 站台三档 + on-time + 死锁/搁浅 terminal.

诚实口径(docs/M10_DESIGN.md §2/§5, PROJECT_BRIEF §7 更正):
    次生延误是**小的真实目标项**(非"不改变最优解的纯 shaping")——想让模型少造次生延误(D21),
    故必须真进目标, 只是权重小; 发放=每步差分。**不主张势函数策略不变性**(自相矛盾)。
    对 M9 绝对量膨胀~4× 的鲁棒靠: 只报相对量(共模抵消) + M13 overlap 稳健性。

信号取引擎现成量: DynTrain.wait_time(=累计阻塞=次生延误), realized 另计 in-progress(now-blocked_since, 自查F1)。

M10b:
    - **站台三档软惩罚**(客运): 停 booked 台=0 / 停其它可达台=−ε / 不停任何台=−P_nostop。
      依据: 英国实操 booked 为默认(WTT/docker), 改台=经理拍板的刻意例外; M9 把站台从硬筛转软目标。
      客运判定用 stop_pids(TRTS 计划停站)∪booked_platform(CIF); 实际服务台从 path 事后算(无需改引擎, F3)。
    - **on-time**(小正): 完成车在 CIF 计划时刻(planned_ref)容差内到达测量点(客运=服务站台/直通=出区)。
      **局限**: planned_ref 是单一 arr/dep/pass 粗参考, on-time 是粗信号、容差可调(灵敏度), 不押。
    - **未完成/搁浅判负**: 自然终止(队列排空, 无 wall-clock 截断, F4)时未到出区口的车=策略饿死→每辆 −P_incomplete。
      时间界定窗口下的边界豁免(τ_feasible)留 M13(自然终止无此假象)。

reward = 引擎 step 前后快照的纯函数(确定性); 完成/站台/on-time 逐车完成时归因(信用分配), 搁浅在终局。
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
    # 站台三档(软)
    eps_replatform: float = 2.0                # 改台小扣
    p_nostop: float = 10.0                     # 客运不停任何台出区大扣
    # on-time(小正) —— 粗信号, 容差可调
    b_ontime: float = 1.0
    on_time_tol_s: float = 300.0               # 5 min 容差(灵敏度旋钮)
    # 安全/完成(大, 结构性)
    p_deadlock: float = 50.0                   # 死锁判负(D6 唯一进 reward 的安全项)
    p_incomplete: float = 20.0                 # 未完成/搁浅每车判负

    def lam(self, service_class) -> float:
        return self.tops_weights.get(str(service_class), self.tops_fallback)


def realized_weighted_delay(engine, cfg: RewardConfig) -> float:
    """整局累计加权次生延误(含当前在途阻塞), 单调不减 → 每步差分 ≤ 0 = 成本。"""
    total = 0.0
    for t in engine.trains.values():
        w = t.wait_time
        if t.blocked_since is not None:
            w += engine.now - t.blocked_since
        total += cfg.lam(t.service_class) * w
    return total


def _expects_stop(t) -> bool:
    return bool(t.stop_pids) or (t.booked_platform is not None)


def _served_platforms(t, plat_tc_to_pid: dict) -> set:
    return {plat_tc_to_pid[tc] for tc in t.path if tc in plat_tc_to_pid}


def platform_tier_reward(t, plat_tc_to_pid: dict, cfg: RewardConfig) -> float:
    """站台三档(客运): booked=0 / 其它台=−ε / 不停=−P_nostop。直通(不该停)=0。"""
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
    return -cfg.p_nostop


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
           + Σ[本步新完成车] (站台三档 + on-time)
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
        # 死锁失败: 只发 p_deadlock, 不叠站台/搁浅(失败信号=p_deadlock, 避免双计)
        if self.engine.deadlocked and not self._was_deadlocked:
            self._was_deadlocked = True
            return r - self.cfg.p_deadlock
        # 逐完成归因: 本步新到出区口的车 → 站台三档 + on-time
        for t in self.engine.trains.values():
            if t.train_id in self._finalized:
                continue
            if t.done and t.reached_exit:
                r += platform_tier_reward(t, self.plat_tc_to_pid, self.cfg)
                r += on_time_bonus(t, self.plat_tc_to_pid, self.cfg)
                self._finalized.add(t.train_id)
        # 终局(自然终止, 非死锁): 未完成/搁浅判负
        if done and not self.engine.deadlocked:
            for t in self.engine.trains.values():
                if t.train_id in self._finalized:
                    continue
                if not t.reached_exit:
                    r -= self.cfg.p_incomplete
                    self._finalized.add(t.train_id)
        return r
