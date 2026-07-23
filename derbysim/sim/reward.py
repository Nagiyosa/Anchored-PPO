"""M10a 奖励模块: 稠密次生延误项(势函数=每步差分) + 死锁 terminal.

诚实口径(见 docs/M10_DESIGN.md §2/§5, PROJECT_BRIEF §7 更正):
    次生延误是**小的真实目标项**(不是"不影响最优解的纯 shaping")——我们想让模型少造次生
    延误(D21 次要探索目标), 故它必须真进目标, 只是权重小。发放=每步差分(把这步新增的加权
    次生延误当成本扣给 agent), 好做信用分配。**不主张"势函数保证策略不变"**(自相矛盾)。
    对 M9 绝对量膨胀~4× 的鲁棒不靠不变性, 靠: 只报相对量(共模抵消) + M13 overlap 稳健性。

信号取引擎现成量 DynTrain.wait_time(=累计阻塞=次生延误); realized 延误另计 in-progress 阻塞
    (now - blocked_since), 免得 reward 成块滞后(自查 F1)。

站台三档/on-time/搁浅判负 → M10b(此处仅延误 + 死锁)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from derbysim.sim.env_dynamic import DynamicDerbyEnv, DynTrain

# TOPS 优先级权重(D16: 1特快>2普客>3包裹>5ECS>6货). 温和默认, 主结论靠灵敏度扫描, 不押数值。
DEFAULT_TOPS_WEIGHTS = {"1": 4.0, "2": 3.0, "3": 2.0, "5": 1.5, "6": 1.0}
DEFAULT_TOPS_FALLBACK = 2.0


@dataclass
class RewardConfig:
    c_delay: float = 0.01                      # 延误缩放(小); 100s 加权延误 ≈ 1
    tops_weights: dict = field(default_factory=lambda: dict(DEFAULT_TOPS_WEIGHTS))
    tops_fallback: float = DEFAULT_TOPS_FALLBACK
    p_deadlock: float = 50.0                    # 死锁判负大惩罚(安全兜底, D6 唯一进 reward 的安全项)

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


def to_dyn_trains(built) -> list[DynTrain]:
    """把 build_env_trains_for_window 的 [(EnvTrain, occ), ...] 转 DynTrain,
    **携带 service_class/planned_ref/booked_platform**(自查 F2: 否则 λ 退化全等/on-time 无法算)。"""
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
    """M10a: 在 M9 反事实环境上叠加奖励(非侵入子类, 单一接线点 = step 前后快照差)。

    reward(每步) = -c_delay * Δ(realized 加权次生延误) - [新判负死锁] p_deadlock。
    延误 realized 单调 → 每步延误项 ≤ 0。确定性: reward 是引擎快照的纯函数。
    """

    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 reward_config: RewardConfig | None = None) -> None:
        super().__init__(net, params, plat_tc_to_pid, dwell_by_platform)
        self.cfg = reward_config or RewardConfig()
        self._last_realized = 0.0
        self._was_deadlocked = False

    def reset(self, trains):
        obs, info = super().reset(trains)
        self._last_realized = realized_weighted_delay(self.engine, self.cfg)
        self._was_deadlocked = self.engine.deadlocked
        return obs, info

    def step(self, action):
        obs, _r0, done, info = super().step(action)
        r = self._compute_reward()
        return obs, r, done, info

    def _compute_reward(self) -> float:
        cur = realized_weighted_delay(self.engine, self.cfg)
        r = -self.cfg.c_delay * (cur - self._last_realized)
        self._last_realized = cur
        if self.engine.deadlocked and not self._was_deadlocked:
            r -= self.cfg.p_deadlock
            self._was_deadlocked = True
        return r
