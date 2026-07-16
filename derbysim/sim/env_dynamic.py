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
    sim_occ: dict = field(default_factory=dict)
    reached_exit: bool = False
    goal: object = None


class DynamicEngine:
    def __init__(self, net, params=None, plat_tc_to_pid=None, dwell_by_platform=None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.res = RouteReservation(net)
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

    def _adv(self, tc):
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    def _legal(self, t, signal):
        """几何出口 ∩ 目标可达. 站台端信号并入其配对端(反向出站, 如 5064<->5054)."""
        if signal is None:
            return []
        sigs = {signal}
        if signal in self._pair:
            sigs.add(self._pair[signal])
        out = []
        for s in sigs:
            for r in self.net.routes_from(s):
                if self.reach.route_keeps_goal(r, t.goal, t.served_platform):
                    out.append(r.route_id)
        return list(dict.fromkeys(out))

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
        return False

    # ---- Path A 面板动作 ----
    def _actionable(self):
        return [t for t in self.trains.values() if t.awaiting_decision and not t.done]

    def _cancellable(self):
        return [t for t in self.trains.values() if t.pending_route is not None and not t.done]

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
            t.awaiting_decision = False
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
                t.blocked_since = None
                t.awaiting_decision = True

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
        hr = self.res.reserved.get(tc)
        if hr is not None and self.res.route_train.get(hr) not in (None, t.train_id):
            self.collisions.append({"t": self.now, "tc": tc, "train": t.train_id})
        t.pos = idx
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
            rid = self.res.reserved.get(tc)
            if rid is not None and self.res.route_train.get(rid) == t.train_id:
                self.res.release_tc(tc, rid)
                self._wake()

    def _wake(self):
        for t in self.trains.values():
            if t.done or t.pending_route is None:
                continue
            if (self.res.is_locked(t.pending_route)
                    and self.res.route_train.get(t.pending_route) == t.train_id):
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
            self.n_decisions += 1
            return
        t.done = True                                # 无更多进路 → 终局 + 释放
        t.reached_exit = self._at_goal_exit(t)
        for k in range(t.route_ptr):
            self.res.release_route(t.route_chain[k])
        self._wake()

    def _at_goal_exit(self, t):
        last = self.net.routes.get(t.route_chain[t.route_ptr - 1]) if t.route_ptr >= 1 else None
        return bool(last and t.goal and last.end_signal in t.goal.exit_group)
