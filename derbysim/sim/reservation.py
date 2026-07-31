"""M9a 全物理锁: 进路预留 (含 overlap) + 请求排队 + 分段释放.

落地 D22 的 request/lock 分层:
    - **请求层**(策略的动作): 几何合法出口, 不看占用/预留 —— 由环境的动作掩码负责, 不在本模块.
    - **锁闭层**(本模块): 办一条进路要**预留**其 TC + overlap; 冲突则**排队**(非拒绝), 待冲突预留
      分段释放后自动落锁. 这对应真实"进路请求可越占用、联锁待轨道清空才锁"(实测 2.2% 请求越占用).

overlap(信号后紧邻一段安全余量)取自数据: 进路 r 之后(从 r.end_signal 出发)各进路的首 TC 集
—— 754/912 进路边界止于信号, 故"信号后紧邻 TC"明确可得. 作已声明近似(真实 overlap 长度不一).
安全(无碰撞)由"只有预留者能进其 TC"结构性保证(D6, 不进 reward).

**预留身份 = (route_id, 车) —— 2026-07-31 修的结构性 bug**
    原实现用 `route_id` 单独做身份(`route_train: route_id→train`, `route_fp: route_id→足迹`)。
    但**同一条进路在一个窗口里会被十几辆车先后使用**(实测 `RTD5045B-1(M)` 在一个 3h 窗口内被
    15 辆车锁过)。于是: A 车还在该进路上时 B 车又锁了它 → `route_train` 被覆盖 →
    **A 车跑完时 `release_route` 释放掉的是 B 车的预留** → B 车脚下的锁凭空消失 →
    第三辆车锁进来 → **碰撞**。
    全量核查(80 个真实拥堵窗口)实测: 2 个窗口共 6 次碰撞, 且发生在**信号员自己的决策**下 ——
    即 D6 "无碰撞由预留结构性保证"当时并**不成立**。
    M10 复审 R3 曾观察到同一现象, 但归因为"合成 clone 车复用 route_id"而砍掉加车功能;
    真相是**身份键本身错了**, 不需要合成车, 真实车流即可触发。
    修法: 一次"办路"= 一个独立预留实例 `(route_id, train)`; 谁办的谁持有、谁释放。
"""
from __future__ import annotations

LOCKED = "LOCKED"
QUEUED = "QUEUED"


class RouteReservation:
    def __init__(self, net, overlap_depth: int = 1) -> None:
        self.net = net
        self.overlap_depth = overlap_depth   # C4 敏感性: 0=无 overlap, 1=信号后一段(默认), 2=两段
        # tc -> (route_id, train) —— **键含车**: 同一进路可被不同车先后/并发使用
        self.reserved: dict[str, tuple[str, str]] = {}
        self.holder_fp: dict[tuple[str, str], frozenset] = {}   # (route_id, train) -> 预留足迹
        self.queue: list[tuple[str, str]] = []                   # FIFO 排队请求 (route_id, train)
        self._overlap_cache: dict[str, frozenset] = {}

    # ---- overlap / footprint ----
    def overlap(self, route) -> frozenset:
        rid = route.route_id
        if rid in self._overlap_cache:
            return self._overlap_cache[rid]
        tcs: set[str] = set()
        frontier = [route.end_signal] if route.end_signal is not None else []
        for _ in range(self.overlap_depth):          # 逐段向信号后延伸 overlap_depth 段
            nxt = []
            for sig in frontier:
                for r in self.net.routes_from(sig):
                    if r.tc_sequence:
                        tcs.add(r.tc_sequence[0])
                        if r.end_signal is not None:
                            nxt.append(r.end_signal)
            frontier = nxt
        tcs -= set(route.tc_sequence)                # overlap 不含进路自身 TC
        fs = frozenset(tcs)
        self._overlap_cache[rid] = fs
        return fs

    def footprint(self, route) -> frozenset:
        return frozenset(route.tc_sequence) | self.overlap(route)

    # ---- 冲突判定 / 落锁 ----
    def _conflict_tc(self, tc: str, train: str) -> bool:
        """tc 是否被**别的车**的进路预留(同车自己的预留不算冲突)."""
        holder = self.reserved.get(tc)
        return holder is not None and holder[1] != train

    def can_lock(self, route, train: str) -> bool:
        return not any(self._conflict_tc(tc, train) for tc in self.footprint(route))

    def _lock(self, route, train: str) -> None:
        fp = self.footprint(route)
        key = (route.route_id, train)
        for tc in fp:
            self.reserved[tc] = key
        self.holder_fp[key] = fp

    def request(self, route, train: str) -> str:
        """办理一条进路: 可锁→立即预留(LOCKED); 冲突→入队(QUEUED)."""
        if self.can_lock(route, train):
            self._lock(route, train)
            return LOCKED
        self.queue.append((route.route_id, train))
        return QUEUED

    # ---- 分段释放 ----
    def release_tc(self, tc: str, key) -> None:
        """车尾清出 tc → 释放该 TC 的预留(**仅当由该 (进路,车) 持有**), 再尝试放行排队。"""
        key = tuple(key)
        if self.reserved.get(tc) == key:
            del self.reserved[tc]
            self.holder_fp[key] = self.holder_fp.get(key, frozenset()) - {tc}
        self._pump_queue()

    def release_route(self, route_id: str, train: str) -> None:
        """整条进路一次性释放(该车已出区/该进路作废)。**必须带车**, 否则会释放别人的锁。"""
        key = (route_id, train)
        for tc in [t for t, k in self.reserved.items() if k == key]:
            del self.reserved[tc]
        self.holder_fp.pop(key, None)
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
    def is_locked(self, route_id: str, train: str) -> bool:
        """**该车**的这条进路是否已落锁(不是"这条进路被任何车锁着")。"""
        return (route_id, train) in self.holder_fp

    def holder_of(self, tc: str):
        """tc 当前的持有者 (route_id, train); 无则 None。"""
        return self.reserved.get(tc)

    def pending(self) -> list[tuple[str, str]]:
        return list(self.queue)
