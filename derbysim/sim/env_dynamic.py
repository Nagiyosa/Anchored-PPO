"""M9d DynamicEngine: 预留驱动的反事实引擎 + 完整 Path A 面板动作集 (stage 1-2).

与 M8 env.py 的根本区别:
    - 全物理锁: 办路即请求预留(reservation.py); 仅落锁(LOCKED)才把 TC 追加进 path 并放行;
      冲突则请求排队(QUEUED)、车停等; 车尾过后分段释放 + 完成时整车释放。车只进已锁属己 TC → 无碰撞。
    - **Path A 面板动作集(stage 2)**: 每个决策机会, 策略从 {WAIT} ∪ {set(t,r)} ∪ {cancel(t)} 选一个动作。
      逐动作(可同一时刻连办多个); WAIT=推进到下一事件。撤销=退掉某车未落锁(排队中)的进路请求, 该车退回待决。
    - **死锁**: 每次 set/释放后查等待图(deadlock.py); 有环 → episode **判负终止**(deadlocked=True)。
      前瞻屏蔽: 会当即成环的 set 从动作集剔除。M8 的 MAX_WAIT 超时兜底退役。
    - 目标=到达出区口方向组(goal.py)。入口进路(route_chain[0]) 非决策(D14.8), 到达时办理。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from derbysim.sim.events import EventQueue
from derbysim.sim.reactive import DEFAULT_ADVANCE, DEFAULT_OCC
from derbysim.sim.reservation import RouteReservation, LOCKED
from derbysim.sim.goal import Reachability, train_goal
from derbysim.sim.deadlock import has_deadlock, would_create_deadlock


# D34: complete train split p99=4 routes including entry; 3*ceil(p99)=12.
# This is only an infinite-cycle failure guard, never a success condition.
TOTAL_ACTUAL_ROUTE_CAP = 12


@dataclass
class PanelAction:
    kind: str                       # 'set' | 'cancel' | 'wait'
    train_id: str | None = None
    route_id: str | None = None


@dataclass
class PanelDecision:
    now: float
    options: list[PanelAction]      # 可选动作(含 WAIT); set 已过 几何∩可达∩死锁前瞻
    logged: dict = field(default_factory=dict)   # train_id -> 与当前实际位置兼容的未消费历史请求


@dataclass
class DynTrain:
    train_id: str
    route_chain: tuple[str, ...]                   # 只读历史参考链；不得表示反事实运行进度(D33)
    entry_time: float
    stop_pids: set = field(default_factory=set)
    booked_platform: int | None = None
    planned_ref: float | None = None       # CIF 计划参考时刻(供 signed 延误)
    service_class: str = ""                 # headcode 首位 = TOPS 优先级
    matched_reference_indices: set[int] = field(default_factory=set)  # 仅回放标签匹配；不表示实际进度
    # 四轨状态(D33 step 2): 历史参考 / 待锁请求 / 已锁实际链 / 物理位置必须互不代用。
    actual_chain: list[str] = field(default_factory=list)  # 仅 LOCKED 后追加，含入口进路
    pending_request: str | None = None                   # QUEUED 请求；尚不属于 actual_chain
    path: list[str] = field(default_factory=list)        # 已锁实际链展开出的 TC
    path_route_ids: list[str] = field(default_factory=list)  # 与 path 逐位对齐的来源进路
    path_commit_indices: list[int] = field(default_factory=list)  # 与 path 对齐的 actual_chain 实例序号
    entered_commit_idx: int = -1                        # 物理上已进入的最后一个实际进路实例
    awaiting_decision: bool = False
    # M8 提前触发(仅 proactive): 下一决策的物理触发点(沿已锁 path 的 idx)与类型。
    next_trigger_idx: int | None = None
    next_trigger_kind: str = ""
    # D50 去重(仅 proactive): 上次为该车提供 SET 决策时的上下文签名; 未变则不重复端上面板。
    set_offer_sig: tuple | None = None
    entered: bool = False
    pos: int = 0
    dwell_idx: int | None = None
    dwell_s: float = 0.0
    served_platform: bool = False
    served_platforms: set[int] = field(default_factory=set)
    completed_dwell_platforms: set[int] = field(default_factory=set)
    done: bool = False
    wait_time: float = 0.0
    blocked_since: float | None = None
    # M12b: "站在信号机前等面板给路"的时长。引擎原先只在**已请求进路但被排队**(blocked_since)时
    # 计延误 → 干脆不给车办路反而零延误(实测: 随机策略 12 辆车实际链 0/N、wait_time 全 0)。
    # 车停着就是在耗时间, 与"排队等锁"对旅客无区别 → 单列累计, 与 blocked_since **互斥**(见 _apply)。
    held_time: float = 0.0
    hold_start: float | None = None
    cancel_sig: tuple | None = None      # 上次为该车提供 cancel 时, 挡住其待办进路的持有者签名
    sim_occ: dict = field(default_factory=dict)
    reached_exit: bool = False
    reached_terminal: bool = False
    termination_reason: str | None = None
    goal: object = None

    def __post_init__(self):
        # 输入通常来自 EnvTrain.list；转 tuple 让“只读参考”成为结构约束，而非注释约定。
        self.route_chain = tuple(self.route_chain)


class DynamicEngine:
    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 overlap_depth: int = 1, proactive: bool = False, advance_cfg=None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        # D50/D48: proactive=True 恢复 M8 的**提前触发**(决策候选在前沿时间窗内产生, 非只在跑光锁定
        # 轨道时). 默认 False = 原 reactive 行为逐位不变(现有信号员回放 1.0/所有下游不受影响)。
        # advance_cfg 复用 M8 的 artifacts/calib/decision_advance.json(逐信号标定, 141s/N 格)。
        self.proactive = bool(proactive)
        if self.proactive and advance_cfg is None:
            from derbysim.sim.env import load_advance_cfg
            advance_cfg = load_advance_cfg()
        self.advance_cfg = advance_cfg or {}
        self.res = RouteReservation(net, overlap_depth=overlap_depth)
        self.reach = Reachability(net)
        self.q = EventQueue()
        self.trains: dict[str, DynTrain] = {}
        self.now = 0.0
        self.n_decisions = 0
        self.n_actions = 0
        self.collisions: list = []
        self.deadlocked = False
        self.deadlock_cycle: list = []
        self._pair: dict = {}                    # 站台两端信号互为配对(反向出站)
        for _p in net.platforms.values():
            _ends = list(_p.end_signals.values())
            if len(_ends) == 2:
                self._pair[_ends[0]] = _ends[1]
                self._pair[_ends[1]] = _ends[0]
        # 物理续接: TC 后继(进路内相邻) + 以某 TC 起始的进路 —— 供按物理位置推决策(D 修)
        self._tc_succ: dict = defaultdict(set)
        self._routes_by_first: dict = defaultdict(list)
        for _r in net.routes.values():
            if _r.tc_sequence:
                self._routes_by_first[_r.tc_sequence[0]].append(_r)
                for _i in range(1, len(_r.tc_sequence)):
                    self._tc_succ[_r.tc_sequence[_i - 1]].add(_r.tc_sequence[_i])

    def _adv(self, tc):
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    def _legal(self, t, signal):
        """合法进路 = (信号/配对端出口 ∪ **物理续接**: 首TC∈succ(车当前末TC)) ∩ 目标可达.

        物理续接项(D 修): 决策信号按信号名匹配会漏 21.8%(TC 连续但信号名不同, 如 5487->5484);
        改按车物理位置(末TC 的后继)提供进路, 是站台配对 fix 的推广。
        """
        routes = []
        if signal is not None:
            sigs = {signal}
            if signal in self._pair:
                sigs.add(self._pair[signal])
            for s in sigs:
                routes.extend(self.net.routes_from(s))
        if t.path:                                        # 物理续接: 首TC 是车末TC 的后继
            for y in self._tc_succ.get(t.path[-1], ()):
                routes.extend(self._routes_by_first.get(y, []))
        out, seen = [], set()
        for r in routes:
            if r.route_id in seen:
                continue
            seen.add(r.route_id)
            # D41 决策空间限于建模区域: 只提供 in-scope 进路(起始信号 ∈ 48 白名单)。可达性(_reach)
            # 本就只在区内扩展、把区边界当出区口, 故此过滤与可达性一致; 真实列车 0/2809 用过区外进路,
            # 对信号员回放零影响。旧行为会在区边界端出区外 route, 让策略被引出区、卡在 TDBN 等区外死点。
            if not getattr(r, "in_scope", True):
                continue
            if self.reach.route_keeps_goal(r, t.goal, t.served_platform):
                out.append(r.route_id)
        return out

    def _decision_signal(self, t):
        prev = self.net.routes.get(t.actual_chain[-1]) if t.actual_chain else None
        return prev.end_signal if prev else None

    def _schedule_next_decision(self, t):
        """proactive(D50/D48): 已提交一条进路后, 为**下一决策**算 M8 式提前触发点(沿已锁 path 的 idx)。

        决策信号 S = 前沿信号(上一已锁进路的 end_signal); 触发 = 站台出站 dwell 起点 / 上游 N 格
        (N 来自 decision_advance.json[S]); 前沿已到目标出口或无 in-scope 合法续路 → 无下一触发。
        当 pos 到达该 idx 时, `_actionable`(proactive) 把该车纳入候选(提前, 非等跑光轨道)。
        """
        t.next_trigger_idx = None
        t.next_trigger_kind = ""
        if not t.actual_chain or t.done or self._at_goal_exit(t):
            return
        S = self._decision_signal(t)
        if S is None or not self._legal(t, S):
            return
        seg_end = len(t.path) - 1
        cfg = self.advance_cfg.get(str(S), {})
        is_pex = bool(cfg.get("is_platform_exit")) or cfg.get("trigger") == "dwell"
        if is_pex and t.dwell_idx is not None and t.dwell_idx <= seg_end:
            t.next_trigger_idx = t.dwell_idx
            t.next_trigger_kind = "dwell"
        else:
            n = int(cfg.get("n_berths", 0) or 0)
            t.next_trigger_idx = max(0, seg_end - n)
            t.next_trigger_kind = "upstream_n"

    @staticmethod
    def _match_reference_index(t, route_ids) -> int | None:
        """Earliest unused historical request compatible with current position.

        ``route_ids`` must come from the current physical option set.  Historical
        array depth is deliberately irrelevant: downstream requests may have
        been preset before their upstream feeder (D33 step 5).
        """
        allowed = set(route_ids)
        for i, rid in enumerate(t.route_chain):
            if i not in t.matched_reference_indices and rid in allowed:
                return i
        return None

    def _append_locked(self, t, route_id):
        r = self.net.routes.get(route_id)
        if not r:
            return
        if not self.res.is_locked(route_id, t.train_id):
            raise RuntimeError(f"cannot commit unlocked route {(route_id, t.train_id)!r}")
        ref_idx = self._match_reference_index(t, {route_id})
        if ref_idx is not None:
            # A reference is consumed by physical commitment, never by a
            # QUEUED request.  Cancelling a queued entry must leave it matchable.
            t.matched_reference_indices.add(ref_idx)
        commit_idx = len(t.actual_chain)
        t.actual_chain.append(route_id)
        for tc in r.tc_sequence:
            if not t.path or t.path[-1] != tc:
                t.path.append(tc)
                t.path_route_ids.append(route_id)
                t.path_commit_indices.append(commit_idx)
                pid = self.plat_tc_to_pid.get(tc)
                dwell_targets = set(t.stop_pids)
                if t.booked_platform is not None:
                    dwell_targets.add(t.booked_platform)
                if t.goal and getattr(t.goal, "terminates_at_platform", False):
                    dwell_targets.update(getattr(t.goal, "terminal_platforms", ()))
                if t.dwell_idx is None and pid is not None and pid in dwell_targets:
                    t.dwell_idx = len(t.path) - 1
                    t.dwell_s = self.dwell_by_platform.get(pid, 0.0)
        if self.proactive:
            self._schedule_next_decision(t)          # 提交后重算下一决策的提前触发点

    def _request(self, t, route_id):
        r = self.net.routes.get(route_id)
        st = self.res.request(r, t.train_id) if r else LOCKED
        if st == LOCKED:
            self._append_locked(t, route_id)
            t.pending_request = None
            return True
        t.pending_request = route_id
        t.blocked_since = self.now
        if t.hold_start is None:
            # 请求被排队 = 车走不动 → 起计。**入区进路**也走这条路径: 卡在入区口的车原先
            # `hold_start` 一直是 None、一秒不计(实测随机策略 18 辆车堵在入区、延误 0,
            # 于是完成率 0.39 却比"有路就办"0.71 得分高)。
            t.hold_start = self.now
        return False

    # ---- Path A 面板动作 ----
    def _actionable(self):
        if not self.proactive:
            return [t for t in self.trains.values() if t.awaiting_decision and not t.done]
        # proactive(D50): 候选 = ①跑光锁定轨道的 deadline 车(awaiting) ∪ ②到达提前触发点、
        # 前沿有待决进路、且当前无排队请求在途的车。前沿窗口 = [提前触发点 … deadline]。
        out = []
        for t in self.trains.values():
            if t.done:
                continue
            if t.awaiting_decision:
                out.append(t)
                continue
            if t.pending_request is not None:
                continue
            if t.next_trigger_idx is not None and t.pos >= t.next_trigger_idx:
                # D50 去重: 只在决策上下文(触发点 + 合法续路的占用签名)自上次提供后变化才重新问,
                # 否则同一未变问题被每个 cut 重复端上(Stage 1 步数 6159→10327 的成因)。
                if self._set_offer_sig(t) != t.set_offer_sig:
                    out.append(t)
        return out

    def _set_offer_sig(self, t):
        """proactive SET 候选的上下文签名 = (触发点 idx, 合法续路 footprint 当前的他车持有者集)。

        前沿一提交(触发点变)或阻塞冲突变化 → 签名变 → 重新提供; 否则压住不重复问(D50)。
        """
        S = self._decision_signal(t)
        legal = self._legal(t, S) if S is not None else []
        holders = set()
        for rid in legal:
            r = self.net.routes.get(rid)
            if r is None:
                continue
            for tc in self.res.footprint(r):
                h = self.res.reserved.get(tc)
                if h is not None and h[1] != t.train_id:
                    holders.add(h)
        return (t.next_trigger_idx, frozenset(holders))

    def _blocker_sig(self, t):
        """挡住车 t 待办进路的持有者集合 (route_id, 车) —— 引擎原生量, 非自造指标。"""
        r = self.net.routes.get(t.pending_request) if t.pending_request else None
        if r is None:
            return ()
        hs = {self.res.reserved.get(tc) for tc in self.res.footprint(r)}
        return tuple(sorted(h for h in hs if h is not None and h[1] != t.train_id))

    def _cancellable(self):
        """可撤销 = 有待办(排队中)进路 · 已排了 >0 仿真时间 · **且阻塞来源自上次提供后发生了变化**。

        ①`blocked_since == now` 不给撤: 现实里信号员不可能同一瞬间把进路排上再撤掉; 允许的话策略能在
          单个时间点无限 办路→撤销 空转(时钟不走 ⇒ 延误恒为 0, 实测该空转策略得分反超正常派路)。
        ②**阻塞来源没变就不重复提供**(2026-07-31): 原实现每个引擎事件都把同一条待办进路重新端上来问
          "要不要撤"。全量实测 —— 一辆车在一个 3h 窗口里被问了 **614 次**同一个问题; "有路就办"的
          决策点中位 **660**、其中 **89.3%** 没有任何 set 可选, 选项池 **62.8%** 是 cancel;
          而**信号员**同期决策中位 **102**、选项池 cancel 仅 **3.4%**, 真实撤销率(全 98 天)
          **10.44%**(92534 请求 / 9664 撤销) ≈ 每 3 小时 14 次。
          即: 这不是"agent 权力大", 是**我们把每次'没动手'都记成一次决策**, 稀释了 9 倍学习信号,
          还让 episode 比信号员长 6 倍(打垮了 GAE 的信用回溯)。撤销能力在**情况真的变了**时完整保留。
        """
        out = []
        for t in self.trains.values():
            if t.pending_request is None or t.done:
                continue
            if not t.entered:
                # The entry route is an automatic infrastructure request
                # (D14.8), not an agent decision, so it cannot appear as cancel.
                continue
            if (t.pending_request, t.train_id) not in self.res.queue:
                continue
            if self.res.is_locked(t.pending_request, t.train_id):
                continue
            if t.blocked_since is None or t.blocked_since >= self.now:
                continue
            if self._blocker_sig(t) != t.cancel_sig:
                out.append(t)
        return out

    def _build_options(self, awaiting, cancellable):
        opts = [PanelAction("wait")]
        logged = {}
        for t in awaiting:
            dsig = self._decision_signal(t)
            offered = []
            for rid in self._legal(t, dsig):
                r = self.net.routes.get(rid)
                if r and not would_create_deadlock(self.res, r, t.train_id):   # 前瞻屏蔽
                    opts.append(PanelAction("set", t.train_id, rid))
                    offered.append(rid)
            ref_idx = self._match_reference_index(t, offered)
            if ref_idx is not None:
                logged[t.train_id] = t.route_chain[ref_idx]
        for t in cancellable:
            opts.append(PanelAction("cancel", t.train_id))
        return opts, logged

    def _apply(self, a: PanelAction):
        self.n_actions += 1
        t = self.trains.get(a.train_id) if a.train_id else None
        if a.kind == "set" and t is not None:
            t.awaiting_decision = False                  # hold_start **不清**: 车还没动
            if self._request(t, a.route_id):
                self.q.push(self.now, "try_advance", {"tr": t.train_id, "idx": t.pos})
            if has_deadlock(self.res):
                self._flag_deadlock()
        elif a.kind == "cancel" and t is not None:
            rid = t.pending_request
            if rid is not None and self.res.cancel_queued(rid, t.train_id):
                # queued request 从未进入 actual_chain；撤销只清 pending，不回退任何实际进度。
                t.pending_request = None
                if t.blocked_since is not None:          # 排队等锁的那段**必须结清**, 不能丢
                    t.wait_time += self.now - t.blocked_since
                    t.blocked_since = None
                t.awaiting_decision = True               # hold_start **不清**: 车自始至终没动
                t.cancel_sig = None
            elif rid is not None and self.res.is_locked(rid, t.train_id):
                # Queue pumping may have promoted it since the panel snapshot.
                # Queued cancel cannot undo a physical lock; observe/commit it.
                self._wake()

    def _flag_deadlock(self):
        from derbysim.sim.deadlock import deadlock_cycle
        self.deadlocked = True
        self.deadlock_cycle = deadlock_cycle(self.res) or []

    def _dispatch(self, ev):
        if ev.kind == "arrive":
            self._on_arrive(ev.data)
        elif ev.kind == "enter":
            self._on_enter(ev.data)
        elif ev.kind == "try_advance":
            self._on_try_advance(ev.data)

    def _drain_slice(self, t_cut):
        """D38 契约①: 弹尽并处理时间戳 == t_cut 的**全部**队列事件后才开面板。

        同刻事件(arrive / enter / try_advance, 以及本 cut 内 SET 派生的同刻事件)必须全部落地,
        否则面板会把此刻**物理上共同可见**的车串行化(Step 8 抓到的 307 个 next_event_same_time)。
        """
        while (len(self.q) and self.q._h[0].time <= t_cut
               and not self.deadlocked):
            ev = self.q.pop()
            self.now = ev.time
            self._dispatch(ev)

    def _time_slice(self, t_cut):
        """一个共同时间切面: 先排空 t_cut 的事件, 再开一个联合面板(D24 / D38 契约②③④)。

        agent 可在同一 cut 内**连续 SET**(每次 SET 会把它排在 t_cut 的派生事件再次排空);
        一旦 WAIT 就结束本 cut, 由 `run` 推进到严格 > t_cut 的下一批事件 → WAIT 恒为时间延展宏动作。
        若无未来事件, 卡住的面板只在无 set/cancel 时静默返回并进入终局结算, 不再吐零时长 WAIT。
        """
        guard = 0
        while not self.deadlocked:
            guard += 1
            if guard > 20000:
                break
            self._drain_slice(t_cut)
            if self.deadlocked:
                return
            awaiting = self._actionable()
            cancellable = self._cancellable()
            if not awaiting and not cancellable:
                return
            opts, logged = self._build_options(awaiting, cancellable)
            for t in cancellable:                    # 已经问过了; 阻塞来源再变才会重新问
                t.cancel_sig = self._blocker_sig(t)
            if self.proactive:                       # D50: 记下本次 SET 决策的上下文签名; 未变不重问
                for t in awaiting:
                    t.set_offer_sig = self._set_offer_sig(t)
            if all(o.kind == "wait" for o in opts):   # 无可办可撤 → 只能等
                return
            action = yield PanelDecision(self.now, opts, logged)
            if action is None or action.kind == "wait":
                return
            self._apply(action)

    def run(self, trains):
        self.trains = {t.train_id: t for t in trains}
        for t in trains:
            t.goal = train_goal(t.route_chain, self.net, t.stop_pids, self.plat_tc_to_pid)
            self.q.push(t.entry_time, "arrive", {"tr": t.train_id})
        guard = 0
        limit = 500000 + 5000 * len(trains)
        while len(self.q) and not self.deadlocked:
            guard += 1
            if guard > limit:
                break
            # D38 契约①: 每个共同时间切面开**一个**面板。先按队列最前时间戳排空该刻全部环境事件,
            # 使在 t_cut 变为 awaiting 的所有车联合可见(②③④); WAIT 后由本循环推进到严格 > t_cut。
            t_cut = self.q._h[0].time
            yield from self._time_slice(t_cut)

    def _on_arrive(self, d):
        t = self.trains[d["tr"]]
        if not t.route_chain:
            t.done = True
            t.termination_reason = "no_route"
            return
        if self._request(t, t.route_chain[0]):
            t.entered = True
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": 0})

    def _on_enter(self, d):
        t = self.trains[d["tr"]]
        idx = d["idx"]
        tc = t.path[idx]
        hr = self.res.reserved.get(tc)               # (route_id, 车) —— 身份含车(2026-07-31 修)
        if hr is not None and hr[1] != t.train_id:
            # 记下**当时**的持有者: 事后查 reserved 已被释放, 查不出所以然(2026-07-31 教训)
            self.collisions.append({"t": self.now, "tc": tc, "train": t.train_id,
                                    "holder_route": hr[0],
                                    "holder_train": hr[1],
                                    "intruder_route": t.actual_chain[-1]
                                    if t.actual_chain else None,
                                    "intruder_pos": idx})
        t.pos = idx
        commit_idx = t.path_commit_indices[idx]
        if commit_idx > t.entered_commit_idx:
            # GB-mainline-aligned approximation: once the train passes the old
            # route's end signal into the next locked route, release only the
            # old route's remaining overlap.  Route body release stays sectional.
            for old_idx in range(t.entered_commit_idx + 1, commit_idx):
                if old_idx >= 0:
                    self.res.release_overlap(
                        t.actual_chain[old_idx], t.train_id,
                        retain_tcs=t.path[t.pos:],
                    )
            t.entered_commit_idx = commit_idx
            self._wake()
        if t.hold_start is not None:                 # **只有真的动了**才结清"走不动"计时
            t.held_time += self.now - t.hold_start
            t.hold_start = None
        t.sim_occ[tc] = self.now
        pid = self.plat_tc_to_pid.get(tc)
        if pid is not None:
            t.served_platform = True
            t.served_platforms.add(pid)
        if idx >= 2:
            self._release_behind(t, idx - 2)
        hold = t.dwell_s if (t.dwell_idx == idx and t.dwell_s > 0) else self._adv(tc)
        self.q.push(self.now + hold, "try_advance", {"tr": t.train_id, "idx": idx})

    def _release_behind(self, t, idx):
        """车尾清出 `path[idx]` → 释放该段。

        **不得释放车前方仍要用的段(2026-08-01 修, D6 红线)**: `path[idx-2]` 只在"路径不回头"时
        才真的在车后面。但**进路链是策略在线选的**, 策略完全可以把车导成环 —— 实测随机策略选出
        `RDC5061A→RDC5075A→RDC5083A` = `TDAB→TDAC→TDAB→TDAC→TDAB→TDAC`, 于是 `path[idx-2]`
        与车**当前所在**的 TC 是同一段: 车把自己脚下的锁释放掉 → 另一辆车(5E38)把它锁走 →
        本车再进 → **碰撞**。全量核查确认这不是数据问题(98 天 18650 辆车的原始链 **0** 条有重复
        TC), 而是**策略可达的环境漏洞** —— 属于"RL 找到漏洞就系统性堵掉"那一类, 不是调参。
        判据: 该 TC 只要还出现在 `path[t.pos:]`(当前位置及前方)里, 就不许释放。
        """
        if not (0 <= idx < len(t.path)):
            return
        tc = t.path[idx]
        if tc in t.path[t.pos:]:                       # 当前所在或前方仍需 → 留着
            return
        key = self.res.reserved.get(tc)
        if key is not None and key[1] == t.train_id:   # 只释放**自己**的预留
            self.res.release_tc(tc, key)
            self._wake()

    def _wake(self):
        for t in self.trains.values():
            if t.done or t.pending_request is None:
                continue
            if self.res.is_locked(t.pending_request, t.train_id):
                self._append_locked(t, t.pending_request)
                t.pending_request = None
                if t.blocked_since is not None:
                    t.wait_time += self.now - t.blocked_since
                    t.blocked_since = None
                if not t.entered:
                    t.entered = True
                    self.q.push(self.now, "enter", {"tr": t.train_id, "idx": 0})
                else:
                    self.q.push(self.now, "try_advance", {"tr": t.train_id, "idx": t.pos})

    def _on_try_advance(self, d):
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.pos != idx or t.done or t.pending_request is not None or t.awaiting_decision:
            return
        if t.dwell_idx == idx:
            pid = self.plat_tc_to_pid.get(t.path[idx])
            if pid is not None:
                # This event fires after the dwell interval scheduled on entry.
                t.completed_dwell_platforms.add(pid)
        reason = self._goal_termination_reason(t)
        if reason is not None:
            self._finish_train(t, reason)
            return
        if idx + 1 < len(t.path):
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
            return
        if len(t.actual_chain) >= TOTAL_ACTUAL_ROUTE_CAP:
            self._finish_train(t, "route_cap")        # failure guard, never success
            return
        # D46(撤销 D41 的 left_region 终止): 车到路径末端无 in-scope 延续且非目标出口 = **搁浅**。
        # **不再终止+释放锁** —— 旧做法让"把难车丢进死点"零后果、且物理不真实(现实里搁浅车停在
        # 轨道上、一直占着锁、堵住后车 = gridlock)。改为**保持 awaiting、继续占锁、延误计到地平线**
        # (realized_weighted_delay 对 not-done 车本就计到地平线): 坏选路自食其果, 才是对的自然惩罚。
        # 信号员从不搁浅(完成 1.0), 故回放逐位不变。无 in-scope set 选项时面板对它只有 WAIT →
        # _time_slice 静默返回, 不空转; 阻塞若成环则由死锁检测判负(亦是真实后果)。
        t.awaiting_decision = True
        if t.hold_start is None:
            t.hold_start = self.now                  # 开始"走不动": 站着等面板给路(占着锁)
        self.n_decisions += 1

    def _goal_termination_reason(self, t):
        goal = t.goal
        if goal is None:
            return None
        if getattr(goal, "terminates_at_platform", False):
            targets = set(getattr(goal, "terminal_platforms", ()))
            if targets & t.completed_dwell_platforms:
                return "terminal_platform"
            return None
        at_path_end = bool(t.path) and t.pos == len(t.path) - 1
        if at_path_end and getattr(goal, "exit_group", None) and self._at_goal_exit(t):
            return "exit"
        unresolved = getattr(goal, "unresolved_end_signal", None)
        last = self.net.routes.get(t.actual_chain[-1]) if t.actual_chain else None
        if at_path_end and not getattr(goal, "exit_group", None) and unresolved is not None:
            if last is not None and last.end_signal == unresolved:
                return "unresolved_boundary"
        return None

    def _finish_train(self, t, reason: str):
        """Settle one train and release only its actual physical state."""
        t.done = True
        t.termination_reason = reason
        t.reached_exit = reason == "exit"
        t.reached_terminal = reason == "terminal_platform"
        if t.hold_start is not None:
            t.held_time += self.now - t.hold_start
            t.hold_start = None
        if t.pending_request is not None:
            rid = t.pending_request
            if self.res.is_locked(rid, t.train_id):
                self.res.release_route(rid, t.train_id)
            else:
                self.res.cancel_queued(rid, t.train_id)
            t.pending_request = None
            t.blocked_since = None
        for route_id in t.actual_chain:
            self.res.release_route(route_id, t.train_id)   # 释放实际锁过的键，不碰历史参考链
        self._wake()

    def state_invariant_violations(self) -> list[str]:
        """Return structural counterfactual-state violations (D33 step 6).

        This is an audit, not a repair: callers must see the original bad state.
        It is safe at natural termination, deadlock, or a fixed-horizon snapshot;
        unfinished trains may legitimately retain locks, done trains may not.
        """
        bad: list[str] = []
        trains = self.trains

        # Reservation ownership must point to one live train, and the reverse
        # index must describe exactly the same TC set.
        by_holder: dict[tuple[str, str], set[str]] = defaultdict(set)
        for tc, key in self.res.reserved.items():
            by_holder[key].add(tc)
            owner = trains.get(key[1])
            if owner is None:
                bad.append(f"reserved[{tc}] has unknown train {key[1]}")
            elif owner.done:
                bad.append(f"done train {key[1]} still reserves {tc} via {key[0]}")
        actual_fp = {key: frozenset(tcs) for key, tcs in by_holder.items()}
        if self.res.holder_fp != actual_fp:
            bad.append("reserved and holder_fp are not exact reverse indices")
        for rid, train_id in self.res.holder_fp:
            owner = trains.get(train_id)
            if owner is None:
                bad.append(f"holder_fp {(rid, train_id)} has unknown train")
            elif owner.done:
                bad.append(f"done train {train_id} still has holder_fp entry {rid}")

        for rid, train_id in self.res.queue:
            owner = trains.get(train_id)
            if owner is None:
                bad.append(f"queue {(rid, train_id)} has unknown train")
            elif owner.done:
                bad.append(f"done train {train_id} still has queued route {rid}")

        # Reconstruct the only path that actual_chain is allowed to imply.
        # This mirrors _append_locked, including adjacent shared-boundary TCs.
        for t in trains.values():
            exp_path: list[str] = []
            exp_routes: list[str] = []
            exp_commits: list[int] = []
            for commit_idx, rid in enumerate(t.actual_chain):
                route = self.net.routes.get(rid)
                if route is None:
                    bad.append(f"train {t.train_id} actual_chain has unknown route {rid}")
                    continue
                for tc in route.tc_sequence:
                    if not exp_path or exp_path[-1] != tc:
                        exp_path.append(tc)
                        exp_routes.append(rid)
                        exp_commits.append(commit_idx)
            if t.path != exp_path:
                bad.append(f"train {t.train_id} path is inconsistent with actual_chain")
            if t.path_route_ids != exp_routes:
                bad.append(f"train {t.train_id} path_route_ids are inconsistent with actual_chain")
            if t.path_commit_indices != exp_commits:
                bad.append(f"train {t.train_id} path_commit_indices are inconsistent with actual_chain")
            if t.path and not (0 <= t.pos < len(t.path)):
                bad.append(f"train {t.train_id} pos {t.pos} is outside path length {len(t.path)}")
            if t.done and t.pending_request is not None:
                bad.append(f"done train {t.train_id} still has pending route {t.pending_request}")
        return bad

    def _delay_min(self, t):
        pr = getattr(t, "planned_ref", None)
        if pr is None:
            return None
        # 无泄露(D54/D22): `entry_time` 是**实际**首占时刻。车尚未入区时它落在未来 → 暴露"接近车
        # 将来几点到/晚多少"= 泄露"已实现的未来"。只在车已入区(entry_time 已实现)后才给 signed 延误;
        # 接近车的时刻只能来自 CIF 计划(planned_ref), 不能来自 entry_time。回放不用特征, 门不受影响。
        if not t.entered:
            return None
        return round((t.entry_time - pr) / 60.0, 1)

    def _build_obs(self):
        """面板 obs(Path A 在线控制): 全区在读列车 + **当前预留锁态**(agent 自控, 非泄露, D22/M9) + 时段.

        与 M8 决策车快照相反 —— M9 是在线开面板, 当前哪些进路已锁是本该看见的现场态.
        """
        trains = []
        for t in self.trains.values():
            if t.done:
                continue
            trains.append({
                "train_id": t.train_id,
                "cur_tc": t.path[t.pos] if t.pos < len(t.path) else None,
                "pos_idx": t.pos,
                "awaiting": t.awaiting_decision,
                "has_pending": t.pending_request is not None,
                # 保持面板观测契约名称；值来自独立的 pending-request 状态。
                "pending_route": t.pending_request,
                "served_platform": t.served_platform,
                "booked_platform": t.booked_platform,
                "delay_min": self._delay_min(t),
                "service_class": t.service_class,
                "exit_group": sorted(t.goal.exit_group) if (t.goal and t.goal.exit_group) else [],
                # M12a: 面板 obs 补两项(供状态适配层重建 M8 式 context, docs/M12_DESIGN §10-A)
                #   rem_path = 该车**已锁**的剩余前方路径(自身既往动作的后果, 在线可观测, 非泄露)
                #   dec_signal = 该车当前所处的决策信号(物理位置, 非日志)
                "rem_path": list(t.path[t.pos:]) if t.pos < len(t.path) else [],
                "dec_signal": self._decision_signal(t),
            })
        return {
            "now": self.now,
            "tod_hour": int((self.now // 3600) % 24),
            "trains": trains,
            # 当前锁态: tc -> (route_id, 车)。特征侧只用**物理占用**(§10-A②), 这里供诊断/面板特征
            "reservations": {tc: k[0] for tc, k in self.res.reserved.items()},
            "n_reserved_tc": len(self.res.reserved),
            "queue_len": len(self.res.queue),
        }

    def _at_goal_exit(self, t):
        last = self.net.routes.get(t.actual_chain[-1]) if t.actual_chain else None
        return bool(last and t.goal and last.end_signal in t.goal.exit_group)


class DynamicDerbyEnv:
    """M9d gym 式包装: reset(trains) → 跑到首个面板决策; step(PanelAction) → 到下一个决策/终局.

    obs = 引擎面板 obs(含当前锁态); reward=0.0(留 M10). decision() 给 PanelDecision(可选动作集).
    done = 生成器耗尽(全车终局) 或 死锁判负.
    """

    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 overlap_depth: int = 1, proactive: bool = False) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.overlap_depth = int(overlap_depth)    # M10 F6: M13 overlap 鲁棒性扫描用; 默认 1 不改原行为
        self.proactive = bool(proactive)           # D50: 提前触发(默认 False = 原 reactive)
        self.engine: DynamicEngine | None = None
        self._gen = None
        self._pending: PanelDecision | None = None

    def reset(self, trains):
        self.engine = DynamicEngine(self.net, self.params, self.plat_tc_to_pid,
                                    self.dwell_by_platform, overlap_depth=self.overlap_depth,
                                    proactive=self.proactive)
        self._gen = self.engine.run(trains)
        self._pending = next(self._gen, None)
        return self._obs(), self._info()

    def step(self, action: PanelAction):
        if self._gen is None or self._pending is None:
            raise RuntimeError("step() called on finished/uninitialized env")
        try:
            self._pending = self._gen.send(action)
            done = self.engine.deadlocked
        except StopIteration:
            self._pending = None
            done = True
        return self._obs(), 0.0, done, self._info()

    @property
    def done(self) -> bool:
        return self._pending is None or (self.engine is not None and self.engine.deadlocked)

    def decision(self) -> PanelDecision | None:
        return self._pending

    def _obs(self):
        return self.engine._build_obs() if (self.engine and self._pending is not None) else None

    def _info(self) -> dict:
        return {"decision": self._pending,
                "deadlocked": self.engine.deadlocked if self.engine else False}
