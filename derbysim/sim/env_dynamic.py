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


@dataclass
class PanelAction:
    kind: str                       # 'set' | 'cancel' | 'wait'
    train_id: str | None = None
    route_id: str | None = None


@dataclass
class PanelDecision:
    now: float
    options: list[PanelAction]      # 可选动作(含 WAIT); set 已过 几何∩可达∩死锁前瞻
    logged: dict = field(default_factory=dict)   # train_id -> 日志下一进路(回放便利)


@dataclass
class DynTrain:
    train_id: str
    route_chain: list[str]
    entry_time: float
    stop_pids: set = field(default_factory=set)
    booked_platform: int | None = None
    planned_ref: float | None = None       # CIF 计划参考时刻(供 signed 延误)
    service_class: str = ""                 # headcode 首位 = TOPS 优先级
    path: list[str] = field(default_factory=list)      # 仅已锁进路的 TC
    route_ptr: int = 0
    pending_route: str | None = None
    awaiting_decision: bool = False
    entered: bool = False
    pos: int = 0
    dwell_idx: int | None = None
    dwell_s: float = 0.0
    served_platform: bool = False
    done: bool = False
    wait_time: float = 0.0
    blocked_since: float | None = None
    # M12b: "站在信号机前等面板给路"的时长。引擎原先只在**已请求进路但被排队**(blocked_since)时
    # 计延误 → 干脆不给车办路反而零延误(实测: 随机策略 12 辆车 route_ptr 0/N、wait_time 全 0)。
    # 车停着就是在耗时间, 与"排队等锁"对旅客无区别 → 单列累计, 与 blocked_since **互斥**(见 _apply)。
    held_time: float = 0.0
    hold_start: float | None = None
    cancel_sig: tuple | None = None      # 上次为该车提供 cancel 时, 挡住其待办进路的持有者签名
    sim_occ: dict = field(default_factory=dict)
    reached_exit: bool = False
    goal: object = None


class DynamicEngine:
    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 overlap_depth: int = 1) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
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
            if self.reach.route_keeps_goal(r, t.goal, t.served_platform):
                out.append(r.route_id)
        return out

    def _decision_signal(self, t):
        prev = self.net.routes.get(t.route_chain[t.route_ptr - 1]) if t.route_ptr >= 1 else None
        return prev.end_signal if prev else None

    def _append_locked(self, t, route_id):
        r = self.net.routes.get(route_id)
        if not r:
            return
        for tc in r.tc_sequence:
            if not t.path or t.path[-1] != tc:
                t.path.append(tc)
                if (t.dwell_idx is None and tc in self.plat_tc_to_pid
                        and self.plat_tc_to_pid[tc] in t.stop_pids):
                    t.dwell_idx = len(t.path) - 1
                    t.dwell_s = self.dwell_by_platform.get(self.plat_tc_to_pid[tc], 0.0)

    def _request(self, t, route_id):
        r = self.net.routes.get(route_id)
        t.route_ptr += 1
        st = self.res.request(r, t.train_id) if r else LOCKED
        if st == LOCKED:
            self._append_locked(t, route_id)
            t.pending_route = None
            return True
        t.pending_route = route_id
        t.blocked_since = self.now
        if t.hold_start is None:
            # 请求被排队 = 车走不动 → 起计。**入区进路**也走这条路径: 卡在入区口的车原先
            # `hold_start` 一直是 None、一秒不计(实测随机策略 18 辆车堵在入区、延误 0,
            # 于是完成率 0.39 却比"有路就办"0.71 得分高)。
            t.hold_start = self.now
        return False

    # ---- Path A 面板动作 ----
    def _actionable(self):
        return [t for t in self.trains.values() if t.awaiting_decision and not t.done]

    def _blocker_sig(self, t):
        """挡住车 t 待办进路的持有者集合 (route_id, 车) —— 引擎原生量, 非自造指标。"""
        r = self.net.routes.get(t.pending_route) if t.pending_route else None
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
            if t.pending_route is None or t.done:
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
            for rid in self._legal(t, dsig):
                r = self.net.routes.get(rid)
                if r and not would_create_deadlock(self.res, r, t.train_id):   # 前瞻屏蔽
                    opts.append(PanelAction("set", t.train_id, rid))
            if t.route_ptr < len(t.route_chain):
                logged[t.train_id] = t.route_chain[t.route_ptr]
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
            rid = t.pending_route
            if rid is not None:
                self.res.queue = [(q, tr) for q, tr in self.res.queue
                                  if not (q == rid and tr == t.train_id)]
                t.route_ptr -= 1                 # 撤销该决策 → 重回待决同一槽
                t.pending_route = None
                if t.blocked_since is not None:          # 排队等锁的那段**必须结清**, 不能丢
                    t.wait_time += self.now - t.blocked_since
                    t.blocked_since = None
                t.awaiting_decision = True               # hold_start **不清**: 车自始至终没动
                t.cancel_sig = None

    def _flag_deadlock(self):
        from derbysim.sim.deadlock import deadlock_cycle
        self.deadlocked = True
        self.deadlock_cycle = deadlock_cycle(self.res) or []

    def _panel_phase(self):
        guard = 0
        while not self.deadlocked:
            guard += 1
            if guard > 10000:
                break
            awaiting = self._actionable()
            cancellable = self._cancellable()
            if not awaiting and not cancellable:
                return
            opts, logged = self._build_options(awaiting, cancellable)
            for t in cancellable:                    # 已经问过了; 阻塞来源再变才会重新问
                t.cancel_sig = self._blocker_sig(t)
            if all(o.kind == "wait" for o in opts):   # 无可办可撤 → 只能等
                return
            action = yield PanelDecision(self.now, opts, logged)
            if action is None or action.kind == "wait":
                return
            self._apply(action)

    def run(self, trains):
        self.trains = {t.train_id: t for t in trains}
        for t in trains:
            t.goal = train_goal(t.route_chain, self.net, t.stop_pids)
            self.q.push(t.entry_time, "arrive", {"tr": t.train_id})
        guard = 0
        limit = 500000 + 5000 * len(trains)
        while len(self.q) and not self.deadlocked:
            guard += 1
            if guard > limit:
                break
            ev = self.q.pop()
            self.now = ev.time
            if ev.kind == "arrive":
                self._on_arrive(ev.data)
            elif ev.kind == "enter":
                self._on_enter(ev.data)
            elif ev.kind == "try_advance":
                self._on_try_advance(ev.data)
            yield from self._panel_phase()

    def _on_arrive(self, d):
        t = self.trains[d["tr"]]
        if not t.route_chain:
            t.done = True
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
                                    "intruder_route": t.route_chain[t.route_ptr - 1]
                                    if t.route_ptr >= 1 else None,
                                    "intruder_pos": idx})
        t.pos = idx
        if t.hold_start is not None:                 # **只有真的动了**才结清"走不动"计时
            t.held_time += self.now - t.hold_start
            t.hold_start = None
        t.sim_occ[tc] = self.now
        if self.plat_tc_to_pid.get(tc) is not None:
            t.served_platform = True
        if idx >= 2:
            self._release_behind(t, idx - 2)
        hold = t.dwell_s if (t.dwell_idx == idx and t.dwell_s > 0) else self._adv(tc)
        self.q.push(self.now + hold, "try_advance", {"tr": t.train_id, "idx": idx})

    def _release_behind(self, t, idx):
        if 0 <= idx < len(t.path):
            tc = t.path[idx]
            key = self.res.reserved.get(tc)
            if key is not None and key[1] == t.train_id:   # 只释放**自己**的预留
                self.res.release_tc(tc, key)
                self._wake()

    def _wake(self):
        for t in self.trains.values():
            if t.done or t.pending_route is None:
                continue
            if self.res.is_locked(t.pending_route, t.train_id):
                self._append_locked(t, t.pending_route)
                t.pending_route = None
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
        if t.pos != idx or t.done or t.pending_route is not None or t.awaiting_decision:
            return
        if idx + 1 < len(t.path):
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
            return
        if t.route_ptr < len(t.route_chain):        # 段末 → 待决(面板相位处理)
            t.awaiting_decision = True
            if t.hold_start is None:
                t.hold_start = self.now              # 开始"走不动": 站着等面板给路
            self.n_decisions += 1
            return
        t.done = True                                # 无更多进路 → 终局 + 释放
        if t.hold_start is not None:
            t.held_time += self.now - t.hold_start
            t.hold_start = None
        t.reached_exit = self._at_goal_exit(t)
        for k in range(t.route_ptr):
            self.res.release_route(t.route_chain[k], t.train_id)   # 带车: 不能释放别人的锁
        self._wake()

    def _delay_min(self, t):
        pr = getattr(t, "planned_ref", None)
        if pr is None:
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
                "has_pending": t.pending_route is not None,
                "pending_route": t.pending_route,      # M12a: cancel(t) 选项特征需要它(自身动作, 非泄露)
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
        last = self.net.routes.get(t.route_chain[t.route_ptr - 1]) if t.route_ptr >= 1 else None
        return bool(last and t.goal and last.end_signal in t.goal.exit_group)


class DynamicDerbyEnv:
    """M9d gym 式包装: reset(trains) → 跑到首个面板决策; step(PanelAction) → 到下一个决策/终局.

    obs = 引擎面板 obs(含当前锁态); reward=0.0(留 M10). decision() 给 PanelDecision(可选动作集).
    done = 生成器耗尽(全车终局) 或 死锁判负.
    """

    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None,
                 overlap_depth: int = 1) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.overlap_depth = int(overlap_depth)    # M10 F6: M13 overlap 鲁棒性扫描用; 默认 1 不改原行为
        self.engine: DynamicEngine | None = None
        self._gen = None
        self._pending: PanelDecision | None = None

    def reset(self, trains):
        self.engine = DynamicEngine(self.net, self.params, self.plat_tc_to_pid,
                                    self.dwell_by_platform, overlap_depth=self.overlap_depth)
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
