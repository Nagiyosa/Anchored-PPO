"""离散事件引擎(M4): heapq 主循环, 接联锁 + 占用模型, 回放真实决策.

流程(每条进路请求):
  route_request(t_req) → 联锁 can_lock?
      是: lock + 按占用模型排 tc_occupy/tc_release(占用起始 = max(t_entry, 锁路时刻));
      否: **延迟重试**(等被占TC空出再办, 模拟信号员等待); 超 MAX_WAIT 记为持续冲突(疑似死锁).
  tc_occupy → 标占用; 若已被他车占=联锁违例(应为0)。
  tc_release → 分段出清, 释放该TC。
锁路时刻与占用起始分离: TC 从锁路即预留, 至列车实际占用才进入.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from derbysim.sim.events import EventQueue, ROUTE_REQUEST, TC_OCCUPY, TC_RELEASE
from derbysim.sim.interlocking import Interlocking
from derbysim.sim.occupation import occupancy_intervals

RETRY_S = 10.0        # 被阻塞时重试间隔
MAX_WAIT_S = 900.0    # 最长等待(超过=持续冲突/疑似死锁)


@dataclass
class SimOccupation:
    tc: str
    t_occupied: float
    t_released: float
    train: str
    route_id: str


@dataclass
class EngineResult:
    sim_occupations: list = field(default_factory=list)
    n_requests: int = 0
    n_locked: int = 0
    n_deferred: int = 0                                     # 因等待而延迟过的请求
    persistent_conflicts: list = field(default_factory=list)  # 超MAX_WAIT仍锁不上
    interlocking_violations: list = field(default_factory=list)


class Engine:
    def __init__(self, net, tc_params: dict | None = None) -> None:
        self.net = net
        self.params = tc_params or {}
        self.q = EventQueue()
        self.interlock = Interlocking(net)
        self.now = 0.0
        self._occupied_by: dict[str, str] = {}
        self.result = EngineResult()

    def run(self, requests: list[dict]) -> EngineResult:
        for r in requests:
            self.q.push(r["t_req"], ROUTE_REQUEST, dict(r))
        self.result.n_requests = len(requests)
        guard = 0
        max_events = 50 * max(1, len(requests)) + 10000
        while len(self.q):
            guard += 1
            if guard > max_events:
                break
            ev = self.q.pop()
            self.now = ev.time
            if ev.kind == ROUTE_REQUEST:
                self._on_request(ev.data)
            elif ev.kind == TC_OCCUPY:
                self._on_occupy(ev.data)
            elif ev.kind == TC_RELEASE:
                self._on_release(ev.data)
        return self.result

    def _on_request(self, d: dict) -> None:
        route = self.net.routes.get(d["route_id"])
        if route is None or not route.tc_sequence:
            return
        if not self.interlock.can_lock(route):
            wait = d.get("_wait", 0.0)
            if wait >= MAX_WAIT_S:
                self.result.persistent_conflicts.append({
                    "t": self.now, "train": d["train"], "route": d["route_id"],
                    "blocked_tcs": [tc for tc in route.tc_sequence if not self.interlock.tc_free(tc)],
                })
                return
            if wait == 0.0:
                self.result.n_deferred += 1
            d["_wait"] = wait + RETRY_S
            self.q.push(self.now + RETRY_S, ROUTE_REQUEST, d)
            return
        self.interlock.lock(route, d["train"])
        self.result.n_locked += 1
        entry = d["t_entry"]   # 回放: 占用用真实时刻(deferral仅管锁路)
        for iv in occupancy_intervals(route.tc_sequence, entry, self.params):
            base = {"tc": iv.tc, "train": d["train"], "route_id": d["route_id"]}
            self.q.push(iv.t_occupied, TC_OCCUPY, dict(base))
            self.q.push(iv.t_released, TC_RELEASE, dict(base))
            self.result.sim_occupations.append(
                SimOccupation(iv.tc, iv.t_occupied, iv.t_released, d["train"], d["route_id"]))

    def _on_occupy(self, d: dict) -> None:
        tc = d["tc"]
        holder = self._occupied_by.get(tc)
        if holder is not None and holder != d["train"]:
            self.result.interlocking_violations.append(
                {"t": self.now, "tc": tc, "held_by": holder, "wanted_by": d["train"]})
        self._occupied_by[tc] = d["train"]

    def _on_release(self, d: dict) -> None:
        tc = d["tc"]
        if self._occupied_by.get(tc) == d["train"]:
            del self._occupied_by[tc]
        self.interlock.release_tc(tc, d["route_id"])
