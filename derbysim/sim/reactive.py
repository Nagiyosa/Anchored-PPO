"""反应式闭塞时间引擎(Cut2, keystone): 列车逐TC推进, 下一段被占则停在当前段等.

与旧 engine.py 的区别: 旧的锁路时一次性算完整条route占用(自由流, 车永不等);
本引擎让列车反应式前进——想进的下一TC被别车占着就停在当前TC等, 其出清再进; 且进入
某段还需该段进路已办理(route-ready 门禁)。于是冲突按信号员真实办理时序自动串行化,
"车等车"的次生延误因果涌现(engine累计每车 wait_time = 阻塞总时长 = 次生延误)。

只在区域入口给一次真实进入时刻(entry), 之后由 模型+真实决策时序 接力(端到端)。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from derbysim.sim.events import EventQueue

DEFAULT_ADVANCE = 8.0
DEFAULT_OCC = 18.0
MAX_WAIT_S = 600.0


@dataclass
class Train:
    train_id: str
    path: list[str]
    entry_time: float
    ready_at: list[float] = field(default_factory=list)
    dwell_idx: int | None = None
    dwell_s: float = 0.0
    pos: int = 0
    done: bool = False
    blocked_since: float | None = None
    wait_time: float = 0.0                       # 累计阻塞时长 = 次生延误
    sim_occ: dict = field(default_factory=dict)

    def ready(self, idx: int) -> float:
        return self.ready_at[idx] if self.ready_at and idx < len(self.ready_at) else -1e18


class ReactiveEngine:
    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.q = EventQueue()
        self.occupied: dict[str, str] = {}
        self.waiting: dict[str, deque] = defaultdict(deque)
        self.trains: dict[str, Train] = {}
        self.now = 0.0
        self.deadlocks: list = []
        self.n_wait_events = 0

    def _adv(self, tc: str) -> float:
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    def _tail(self, tc: str) -> float:
        a, o = self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))
        return max(1.0, o - a)

    def run(self, trains: list[Train]) -> dict[str, Train]:
        self.trains = {t.train_id: t for t in trains}
        for t in trains:
            if t.path:
                self.q.push(t.entry_time, "enter", {"tr": t.train_id, "idx": 0})
        guard = 0
        limit = 300 * sum(len(t.path) for t in trains) + 20000
        while len(self.q):
            guard += 1
            if guard > limit:
                break
            ev = self.q.pop()
            self.now = ev.time
            getattr(self, "_on_" + ev.kind)(ev.data)
        return self.trains

    def _on_enter(self, d: dict) -> None:
        t = self.trains[d["tr"]]
        idx = d["idx"]
        tc = t.path[idx]
        self.occupied[tc] = t.train_id
        t.pos = idx
        t.sim_occ[tc] = self.now
        t.blocked_since = None
        hold = t.dwell_s if (t.dwell_idx == idx and t.dwell_s > 0) else self._adv(tc)
        self.q.push(self.now + hold, "try_advance", {"tr": t.train_id, "idx": idx})

    def _on_try_advance(self, d: dict) -> None:
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.pos != idx or t.done:
            return
        tc = t.path[idx]
        if idx + 1 >= len(t.path):
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            t.done = True
            return
        nxt = t.path[idx + 1]
        ready = t.ready(idx + 1)
        if self.now < ready:                       # 门禁1: 进路未办理
            self.q.push(ready, "try_advance", {"tr": t.train_id, "idx": idx})
            return
        holder = self.occupied.get(nxt)
        if holder is None or holder == t.train_id:  # 门禁2: 下一段空 → 进
            if t.blocked_since is not None:         # 结束阻塞, 累计等待
                t.wait_time += self.now - t.blocked_since
                t.blocked_since = None
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
        else:                                       # 被占 → 停等
            if t.blocked_since is None:
                t.blocked_since = self.now
                self.n_wait_events += 1
                self.q.push(self.now + MAX_WAIT_S, "deadlock_check", {"tr": t.train_id, "idx": idx})
            if t.train_id not in self.waiting[nxt]:
                self.waiting[nxt].append(t.train_id)

    def _on_release(self, d: dict) -> None:
        tc, tr = d["tc"], d["tr"]
        if self.occupied.get(tc) == tr:
            del self.occupied[tc]
        while self.waiting[tc]:
            w = self.waiting[tc].popleft()
            wt = self.trains.get(w)
            if wt and not wt.done and wt.pos < len(wt.path) - 1 and wt.path[wt.pos + 1] == tc:
                self.q.push(self.now, "try_advance", {"tr": w, "idx": wt.pos})
                break

    def _on_deadlock_check(self, d: dict) -> None:
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.done or t.pos != idx or t.blocked_since is None:
            return
        if self.now - t.blocked_since >= MAX_WAIT_S:
            self.deadlocks.append({"t": self.now, "train": t.train_id, "tc": t.path[idx],
                                   "blocked_next": t.path[idx + 1]})
            t.wait_time += self.now - t.blocked_since
            t.blocked_since = None
            self.q.push(self.now + self._tail(t.path[idx]), "release",
                        {"tc": t.path[idx], "tr": t.train_id})
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
