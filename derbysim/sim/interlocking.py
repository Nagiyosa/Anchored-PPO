"""联锁: TC 互斥占用/进路锁闭 + 合法动作掩码.

模型: 进路锁定即预留其全部 TC(从锁路到列车逐段出清); 任一 TC 被占则该进路不可锁.
动作掩码 = 某信号处 全部 TC 当前空闲的出口进路(安全结构性保证, brief §4).
"""

from __future__ import annotations


class Interlocking:
    def __init__(self, net) -> None:
        self.net = net
        self.tc_holder: dict[str, str] = {}     # tc -> 持有它的 route_id
        self.route_train: dict[str, str] = {}    # route_id -> train

    def tc_free(self, tc: str) -> bool:
        return tc not in self.tc_holder

    def can_lock(self, route) -> bool:
        return all(self.tc_free(tc) for tc in route.tc_sequence)

    def lock(self, route, train: str) -> None:
        for tc in route.tc_sequence:
            self.tc_holder[tc] = route.route_id
        self.route_train[route.route_id] = train

    def release_tc(self, tc: str, route_id: str) -> None:
        # 仅当该TC确由此route持有才释放(分段出清)
        if self.tc_holder.get(tc) == route_id:
            del self.tc_holder[tc]

    def legal_routes(self, signal: int) -> list:
        """动作掩码: 该信号处当前可安全锁定的出口进路."""
        return [r for r in self.net.routes_from(signal) if self.can_lock(r)]
