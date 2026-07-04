"""离散事件类型 + 事件堆封装(heapq)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

# 事件种类
ROUTE_REQUEST = "route_request"   # 信号员办进路(锁路时刻)
TC_OCCUPY = "tc_occupy"           # 某TC被列车占用
TC_RELEASE = "tc_release"         # 某TC出清


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


class EventQueue:
    """最小堆事件队列; seq 保证同时刻稳定顺序."""

    def __init__(self) -> None:
        self._h: list[Event] = []
        self._seq = 0

    def push(self, time: float, kind: str, data: dict) -> None:
        heapq.heappush(self._h, Event(float(time), self._seq, kind, data))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._h)

    def __len__(self) -> int:
        return len(self._h)
