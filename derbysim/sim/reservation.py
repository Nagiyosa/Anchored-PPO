"""M9a 全物理锁: 进路预留 (含 overlap) + 请求排队 + 分段释放.

落地 D22 的 request/lock 分层:
    - **请求层**(策略的动作): 几何合法出口, 不看占用/预留 —— 由环境的动作掩码负责, 不在本模块.
    - **锁闭层**(本模块): 办一条进路要**预留**其 TC + overlap; 冲突则**排队**(非拒绝), 待冲突预留
      分段释放后自动落锁. 这对应真实"进路请求可越占用、联锁待轨道清空才锁"(实测 2.2% 请求越占用).

overlap(信号后紧邻一段安全余量)取自数据: 进路 r 之后(从 r.end_signal 出发)各进路的首 TC 集
—— 754/912 进路边界止于信号, 故"信号后紧邻 TC"明确可得. 作已声明近似(真实 overlap 长度不一).
安全(无碰撞)由"只有预留者能进其 TC"结构性保证(D6, 不进 reward).
"""

from __future__ import annotations

LOCKED = "LOCKED"
QUEUED = "QUEUED"


class RouteReservation:
    def __init__(self, net) -> None:
        self.net = net
        self.reserved: dict[str, str] = {}          # tc -> 持有它的 route_id
        self.route_train: dict[str, str] = {}        # route_id -> train
        self.route_fp: dict[str, frozenset] = {}     # route_id -> 预留足迹(route TC + overlap)
        self.queue: list[tuple[str, str]] = []        # FIFO 排队请求 (route_id, train)
        self._overlap_cache: dict[str, frozenset] = {}

    # ---- overlap / footprint ----
    def overlap(self, route) -> frozenset:
        rid = route.route_id
        if rid in self._overlap_cache:
            return self._overlap_cache[rid]
        tcs: set[str] = set()
        if route.end_signal is not None:
            for r in self.net.routes_from(route.end_signal):
                if r.tc_sequence:
                    tcs.add(r.tc_sequence[0])         # 信号后紧邻一格
        tcs -= set(route.tc_sequence)                # overlap 不含进路自身 TC
        fs = frozenset(tcs)
        self._overlap_cache[rid] = fs
        return fs

    def footprint(self, route) -> frozenset:
        return frozenset(route.tc_sequence) | self.overlap(route)

    # ---- 冲突判定 / 落锁 ----
    def _conflict_tc(self, tc: str, train: str) -> bool:
        """tc 是否被**别的车**的进路预留(同车自路不算冲突)."""
        holder_route = self.reserved.get(tc)
        if holder_route is None:
            return False
        return self.route_train.get(holder_route) != train

    def can_lock(self, route, train: str) -> bool:
        return not any(self._conflict_tc(tc, train) for tc in self.footprint(route))

    def _lock(self, route, train: str) -> None:
        fp = self.footprint(route)
        for tc in fp:
            self.reserved[tc] = route.route_id
        self.route_train[route.route_id] = train
        self.route_fp[route.route_id] = fp

    def request(self, route, train: str) -> str:
        """办理一条进路: 可锁→立即预留(LOCKED); 冲突→入队(QUEUED)."""
        if self.can_lock(route, train):
            self._lock(route, train)
            return LOCKED
        self.queue.append((route.route_id, train))
        return QUEUED

    # ---- 分段释放 ----
    def release_tc(self, tc: str, route_id: str) -> None:
        """车尾清出 tc → 释放该 TC 的预留(仅当由此 route 持有), 再尝试放行排队."""
        if self.reserved.get(tc) == route_id:
            del self.reserved[tc]
            self.route_fp[route_id] = self.route_fp.get(route_id, frozenset()) - {tc}
        self._pump_queue()

    def release_route(self, route_id: str) -> None:
        """整条进路一次性释放(车已出区/该进路作废)."""
        for tc in list(self.reserved):
            if self.reserved[tc] == route_id:
                del self.reserved[tc]
        self.route_train.pop(route_id, None)
        self.route_fp.pop(route_id, None)
        self._pump_queue()

    def _pump_queue(self) -> None:
        """FIFO 逐个尝试把已可锁的排队请求落锁; 保持相对顺序. 反复直到无变化."""
        changed = True
        while changed and self.queue:
            changed = False
            still: list[tuple[str, str]] = []
            for rid, tr in self.queue:
                r = self.net.routes.get(rid)
                if r is not None and self.can_lock(r, tr):
                    self._lock(r, tr)
                    changed = True
                else:
                    still.append((rid, tr))
            self.queue = still

    # ---- 查询 ----
    def is_locked(self, route_id: str) -> bool:
        return route_id in self.route_train

    def pending(self) -> list[tuple[str, str]]:
        return list(self.queue)
