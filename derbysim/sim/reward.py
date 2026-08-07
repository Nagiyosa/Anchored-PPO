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
    # D58: reward 结构 —— "per_train"=照 RR 逐车终局(默认); "per_step"=旧逐步全局 Δ(留作消融对照)。
    reward_mode: str = "per_train"
    # D58 逐车终局项(照 RR:到站+10 / 每晚 −2·min / 刹车扣分); 数值是灵敏度扫描旋钮, 不押。
    base_completion: float = 10.0              # RR: 车到站 +10
    w_delay: float = 0.01                      # 每单位加权次生延误(秒)扣; 1000 加权s ≈ 一次完成
    w_stops: float = 1.0                       # 每次区内非计划停车扣(RR 刹车罚代理, 小)
    # 稠密延误(小) —— 仅 per_step 旧模式用
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
    # M12b 复审(2026-07-30, 用户定): 把"站着等派路"的时长也计入次生延误。
    # 旧度量只算"请求了进路但被排队", 于是**不给车办路 = 零延误** → 实测随机策略(完成 18/30)
    # 回报反而高于"有路就办"(完成 26/30)。留开关供对照, **默认开**。
    count_awaiting_hold: bool = True
    # 固定时间地平线的排空期(M10 §3 的 drain)。**依据换成全量实测(80 个拥堵窗口, 2026-07-31)**:
    # 信号员回放下车在区内停留 p99 中位 7.4min、**最大 14.9min**; 原定 15min 只剩 0.1min 余量 →
    # 提到 20min。(之前的 6 窗口/191 辆样本给出 max 12.8min, 取样过小。)
    drain_s: float = 1200.0

    def lam(self, service_class) -> float:
        return self.tops_weights.get(str(service_class), self.tops_fallback)


def realized_weighted_delay(engine, cfg: RewardConfig, horizon: float | None = None) -> float:
    """整局累计加权次生延误(含**未 done 车**的在途阻塞, R1), 单调不减 → 每步差分 ≤ 0 = 成本。"""
    total = 0.0
    hold = bool(getattr(cfg, "count_awaiting_hold", True))
    for t in engine.trains.values():
        if hold:
            # **单一计时器**: 车"走不动"的总时长(从停下到真正驶入下一个 TC)。它已经覆盖了
            # "排队等锁"那段, 故**不再叠加** wait_time —— 否则重复计; 也不受 set/cancel 影响。
            w = getattr(t, "held_time", 0.0)
            if not t.done and getattr(t, "hold_start", None) is not None:
                # **固定地平线**: 被晾着的车按"一直站到共同地平线"计, 而不是按引擎停摆时刻计。
                # 否则**少干活的策略会把自己欠的账截短** —— 车不派 → 无事件 → 队列早排空 → 时钟停 →
                # "站着等"的计时跟着停。实测: 随机策略完成率 0.512 却比"有路就办"(0.735) 得分高。
                # 这同时也让跨策略比较公平(否则各策略的时间跨度都不一样, 对比本身无意义)。
                end = engine.now if horizon is None else max(engine.now, horizon)
                w += end - t.hold_start                  # R1: done 车不计在途(防幻延误)
        else:
            w = t.wait_time                              # 旧口径(仅排队等锁), 留作对照
            if t.blocked_since is not None and not t.done:
                w += engine.now - t.blocked_since
        total += cfg.lam(t.service_class) * w
    return total


def can_complete(t) -> bool:
    """该车是否有**可观测的目标**(出区口组非空, 或是终到站台的车)。

    M12b 复审(2026-07-30): 部分车的 `goal.exit_group` 为**空**, `_at_goal_exit` 对它们恒为 False。
    真值对照: 信号员回放 8 个真实拥堵窗口, "有出区口目标"的车**一辆未漏**, 漏掉的恰好全是这批空
    目标车 → 对它们罚 `p_incomplete` 是必然误罚。

    **D29 修正(2026-08-01)**: 当时的处理是把它们整体**排除**, 且把成因写成"窗口边界碎片" ——
    **两点都不对**。全 98 天实测 **1651/18650 = 8.85%** 的车 `exit_group` 为空; 3 天样本里 45 辆
    中只有 **2 辆**贴日界(其余在日中), 而 20 天样本里 **82% 末段落在站台 TC 上**, 车次号以
    1C/2A/1K/1F/1V/1D 为主 = **在德比终到的客运列车**(车次号在此消亡, 之后以 ECS 新号入库)。
    它们**有**目标, 就是"在观测目标站台完成必要 dwell"; 排除的后果是 agent 把终到车晾在咽喉区不受任何惩罚,
    而现实里它正堵着站台。故: **终到车(`goal.terminates_at_platform`)回到分母**, 完成判据 =
    `reached_terminal`。仍排除的只剩"无出区口目标且末段不在站台"的车(20 天样本 18%, 多为进入
    未建模的 Etches Park 库内区段), 那类我们确实没有可观测目标。
    """
    g = getattr(t, "goal", None)
    if not g:
        return False
    return bool(getattr(g, "exit_group", None) or getattr(g, "terminates_at_platform", False))


def has_reached_goal(t) -> bool:
    """达成目标: 出区车到方向组；终到车在目标站台完成必要 dwell。"""
    if getattr(t, "reached_exit", False):
        return True
    g = getattr(t, "goal", None)
    return bool(g and getattr(g, "terminates_at_platform", False)
                and getattr(t, "reached_terminal", False))


def _expects_stop(t) -> bool:
    return bool(t.stop_pids) or (t.booked_platform is not None)


def _served_platforms(t, plat_tc_to_pid: dict) -> set:
    if hasattr(t, "served_platforms"):
        return set(t.served_platforms)
    return {plat_tc_to_pid[tc] for tc in getattr(t, "sim_occ", {}) if tc in plat_tc_to_pid}


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
    if getattr(t, "planned_ref", None) is None or not has_reached_goal(t):
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
            req_times=tuple(getattr(et, "req_times", ()) or ()),
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
                 reward_config: RewardConfig | None = None, overlap_depth: int = 1,
                 proactive: bool = False, reserve_window: str = "narrow") -> None:
        super().__init__(net, params, plat_tc_to_pid, dwell_by_platform,
                         overlap_depth=overlap_depth, proactive=proactive,
                         reserve_window=reserve_window)
        self.cfg = reward_config or RewardConfig()
        self.horizon: float | None = None          # 由 reset(trains, horizon=...) 设定
        self._last_realized = 0.0
        self._was_deadlocked = False
        self._finalized: set = set()

    def reset(self, trains, horizon: float | None = None):
        self.horizon = horizon
        obs, info = super().reset(trains)
        self._last_realized = realized_weighted_delay(self.engine, self.cfg)
        self._was_deadlocked = self.engine.deadlocked
        self._finalized = set()
        return obs, info

    def step(self, action):
        obs, _r0, done, info = super().step(action)
        r = self._compute_reward(done)
        return obs, r, done, info

    def finalize_truncated(self) -> float:
        """episode 被**截断**(如 max_decisions 用尽)时补做终局结算。

        否则 `_compute_reward` 里 `if done:` 那段不执行 → **拖到截断就能白嫖掉所有未完成罚分**。
        实测(2026-07-30): 一个只会反复 办路→撤销 的空转策略靠此拿 **+3.0**, 而正常派路 −1660。
        """
        return self._compute_reward(done=True)

    def _train_weighted_delay(self, t, horizon) -> float:
        """单车加权次生延误 = λ · held_time(未 done 且在停 → 计到地平线, 同 realized_weighted_delay)。"""
        w = getattr(t, "held_time", 0.0)
        if not t.done and getattr(t, "hold_start", None) is not None:
            end = self.engine.now if horizon is None else max(self.engine.now, horizon)
            w += end - t.hold_start
        return self.cfg.lam(t.service_class) * w

    def _reward_per_train(self, done: bool) -> float:
        """D58: 照 RR 逐车终局。车到站结账 base − w_d·加权延误 − w_s·停车; 死锁大罚; 终局对未完成
        车按到地平线的加权延误 + p_incomplete 结账(D46 不丢车)。站台/on-time 不入 reward(只评估)。"""
        r = 0.0
        newly_deadlocked = self.engine.deadlocked and not self._was_deadlocked
        if newly_deadlocked:
            self._was_deadlocked = True
            r -= self.cfg.p_deadlock
        for t in self.engine.trains.values():                # 新完成的车逐车结账
            if t.train_id in self._finalized:
                continue
            if t.done and has_reached_goal(t):
                r += (self.cfg.base_completion
                      - self.cfg.w_delay * self._train_weighted_delay(t, horizon=None)
                      - self.cfg.w_stops * float(getattr(t, "n_stops", 0)))
                self._finalized.add(t.train_id)
        if done or newly_deadlocked:                         # 终局: 未完成车结账(不丢车, D46)
            for t in self.engine.trains.values():
                if t.train_id in self._finalized:
                    continue
                if can_complete(t) and not has_reached_goal(t):
                    r += (- self.cfg.w_delay * self._train_weighted_delay(t, horizon=self.horizon)
                          - self.cfg.p_incomplete)
                    self._finalized.add(t.train_id)
        return r

    def _compute_reward(self, done: bool) -> float:
        if getattr(self.cfg, "reward_mode", "per_train") == "per_train":
            return self._reward_per_train(done)
        cur = realized_weighted_delay(self.engine, self.cfg)
        r = -self.cfg.c_delay * (cur - self._last_realized)
        self._last_realized = cur
        newly_deadlocked = self.engine.deadlocked and not self._was_deadlocked
        if newly_deadlocked:
            self._was_deadlocked = True
            r -= self.cfg.p_deadlock          # **叠加**, 不再 return —— 见下面的终局结算规则
        for t in self.engine.trains.values():
            if t.train_id in self._finalized:
                continue
            if t.done and has_reached_goal(t):
                r += platform_tier_reward(t, self.plat_tc_to_pid, self.cfg)
                r += on_time_bonus(t, self.plat_tc_to_pid, self.cfg)
                self._finalized.add(t.train_id)
        # **结构性规则(2026-07-30)**: 终局结算与**终止原因无关** —— 自然跑完 / 死锁判负 / 截断,
        # 同一套账都要结。原先 `and not deadlocked` 让**故意死锁**成为逃费手段: 实测 25 辆车全搁浅时
        # 正常收尾 −3500、死锁收尾只 −3050(划算 +450), 而"有路就办"跑完真实拥堵窗口是 −760 ⇒
        # 自杀死锁是当时奖励下的更优解。`p_deadlock` 现在是**额外**罚, 不是**替代**罚。
        # 同理截断也要补结(`finalize_truncated`)。这一条同时关掉整类"提前结束逃费"。
        if done or newly_deadlocked:
            if self.horizon is not None:                 # 终局: 把欠账补到共同地平线
                cur_h = realized_weighted_delay(self.engine, self.cfg, horizon=self.horizon)
                r -= self.cfg.c_delay * (cur_h - self._last_realized)
                self._last_realized = cur_h
            for t in self.engine.trains.values():
                if t.train_id in self._finalized:
                    continue
                # 终到车须在观测目标台完成 dwell 才结算成功(D29/D33 step 4)；
                # 仅进入任意站台或计划路径经过站台都不算。无可观测目标的车仍不误罚。
                if not has_reached_goal(t) and can_complete(t):
                    r -= self.cfg.p_incomplete
                    self._finalized.add(t.train_id)
        return r
