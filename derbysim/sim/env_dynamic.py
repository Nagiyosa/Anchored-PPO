"""M9d DynamicEngine (stage 1): 预留驱动的反事实引擎 (Path A).

与 M8 env.py 的根本区别:
    - M8 解耦(保 M7 等价): 办路只把 TC 追加进 path, 推进仍照 log req_time 门禁。
    - M9 **全物理锁**: 办一条路即**请求预留**(reservation.py); **仅当落锁(LOCKED)** 才把其 TC 追加进
      path 并放行; 冲突则请求**排队(QUEUED)**、车停在当前位置等; 车尾过后**分段释放**、pump 队列放行后车。
    - 车只会进入**已锁属己**的 TC → 结构性无碰撞(collisions 应恒为 0, 仅作自检)。
    - 决策时机由**策略**掌控(Path A); 目标=到达**出区口方向组**(goal.py)。

stage 1: 预留驱动推进 + 逐路由决策(生成器) + 目标终止 + 排队/分段释放/放行。
撤销 / 死锁判负+前瞻 / obs锁态 / Path A 完整动作集(wait) 留 stage 2-3。
入口进路(route_chain[0]) 视为非决策(D14.8), 车到达(entry)时办理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from derbysim.sim.events import EventQueue
from derbysim.sim.reactive import DEFAULT_ADVANCE, DEFAULT_OCC
from derbysim.sim.reservation import RouteReservation, LOCKED
from derbysim.sim.goal import Reachability, train_goal


@dataclass
class DynDecision:
    train_id: str
    signal: int | None
    route_ptr: int
    legal_route_ids: list[str]
    logged_route_id: str | None
    now: float


@dataclass
class DynTrain:
    train_id: str
    route_chain: list[str]
    entry_time: float
    stop_pids: set = field(default_factory=set)
    booked_platform: int | None = None
    # 运行时:
    path: list[str] = field(default_factory=list)      # **仅已锁**进路的 TC 序列
    route_ptr: int = 0                                   # 已决策(请求)进路数
    pending_route: str | None = None                     # 已请求但未落锁的进路
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
        self.collisions: list = []

    def _adv(self, tc):
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    # ---- 合法动作 = 几何出口 ∩ 目标可达(D22: 不按预留/占用过滤) ----
    def _legal(self, t, signal):
        if signal is None:
            return []
        return [r.route_id for r in self.net.routes_from(signal)
                if self.reach.route_keeps_goal(r, t.goal, t.served_platform)]

    def _decision_signal(self, t):
        prev = self.net.routes.get(t.route_chain[t.route_ptr - 1]) if t.route_ptr >= 1 else None
        return prev.end_signal if prev else None

    def _append_locked(self, t, route_id):
        """把**已锁**进路的 TC 追加进 path(跨段去重), 标记停站."""
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
        """请求办理一条进路. LOCKED→追加 path 并返回 True; QUEUED→挂 pending 返回 False."""
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

    def run(self, trains):
        self.trains = {t.train_id: t for t in trains}
        for t in trains:
            t.goal = train_goal(t.route_chain, self.net, t.stop_pids)
            self.q.push(t.entry_time, "arrive", {"tr": t.train_id})
        guard = 0
        limit = 500000 + 5000 * len(trains)
        while len(self.q):
            guard += 1
            if guard > limit:
                break
            ev = self.q.pop()
            self.now = ev.time
            if ev.kind == "arrive":
                yield from self._on_arrive(ev.data)
            elif ev.kind == "enter":
                self._on_enter(ev.data)
            elif ev.kind == "try_advance":
                yield from self._on_try_advance(ev.data)
            elif ev.kind == "release":
                pass

    def _on_arrive(self, d):
        t = self.trains[d["tr"]]
        if not t.route_chain:
            t.done = True
            return
        if self._request(t, t.route_chain[0]):        # 入口进路(非决策): 落锁即入区
            t.entered = True
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": 0})
        # 若入口排队(罕见): 等释放 pump 落锁后由 _wake 放行
        if False:
            yield  # keep generator semantics

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
                self.res.release_tc(tc, rid)          # 内部 pump 队列
                self._wake()

    def _wake(self):
        """释放后, 唤醒 pending 已落锁的车(入区 or 续行)."""
        for t in self.trains.values():
            if t.done or t.pending_route is None:
                continue
            if self.res.is_locked(t.pending_route) and self.res.route_train.get(t.pending_route) == t.train_id:
                self._append_locked(t, t.pending_route)
                t.pending_route = None
                if t.blocked_since is not None:
                    t.wait_time += self.now - t.blocked_since
                    t.blocked_since = None
                if not t.entered:                     # 入口排队后落锁 → 入区
                    t.entered = True
                    self.q.push(self.now, "enter", {"tr": t.train_id, "idx": 0})
                else:
                    self.q.push(self.now, "try_advance", {"tr": t.train_id, "idx": t.pos})

    def _on_try_advance(self, d):
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.pos != idx or t.done or t.pending_route is not None:
            return
        if idx + 1 < len(t.path):                     # 已锁路内: 下一 TC 属己 → 进
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
            return
        # 到 path 末端
        if t.route_ptr < len(t.route_chain):          # 还有下一段 → 决策
            dsig = self._decision_signal(t)
            legal = self._legal(t, dsig)
            logged = t.route_chain[t.route_ptr]
            self.n_decisions += 1
            chosen = yield DynDecision(t.train_id, dsig, t.route_ptr, legal, logged, self.now)
            if self._request(t, chosen):              # 落锁 → 继续推进
                self.q.push(self.now, "try_advance", {"tr": t.train_id, "idx": t.pos})
            # 排队 → 等释放放行(_wake)
            return
        # 无更多进路: 终局 → 释放该车全部预留(含 overlap), 唤醒等待者
        t.done = True
        t.reached_exit = self._at_goal_exit(t)
        for k in range(t.route_ptr):
            self.res.release_route(t.route_chain[k])
        self._wake()

    def _at_goal_exit(self, t):
        last = self.net.routes.get(t.route_chain[t.route_ptr - 1]) if t.route_ptr >= 1 else None
        return bool(last and t.goal and last.end_signal in t.goal.exit_group)
