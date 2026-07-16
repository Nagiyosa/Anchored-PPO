"""M9c 死锁: 等待图找环(检测) + 一步前瞻屏蔽(预防). 退役 M8 的 MAX_WAIT 超时兜底.

CS 视角: 资源(轨道段)受限调度中的循环等待 = 死锁。
    - **检测**: 建**等待图**(wait-for graph)——一列车 t 有**排队中**的进路请求, 其足迹某 TC 被**别的车 u**
      的进路预留 → 连边 t→u(t 等 u 释放)。图中有环 = 真死锁(涉环各车互相等、谁也进不了)。
    - **处理**: episode **判负终止**(M9 定; 由环境据 has_deadlock 结束并计失败)。
    - **预防**: 一步前瞻——若"办某条进路"会**当即形成环**, 从动作候选集剔除(前瞻屏蔽)。

为何等待只在"排队请求"层: 全物理锁下, 已落锁的进路其 TC(+overlap)已独占预留, 该车物理必可前进;
故阻塞只发生在"请求尚未获准(排队)"处 → 死锁必现于排队请求构成的环(见 M9_DESIGN §2.4)。
"""

from __future__ import annotations

from collections import defaultdict


def wait_for_graph(res) -> dict[str, set[str]]:
    """由预留引擎当前态建等待图: 边 t→u 表示"车 t 的排队请求被车 u 的预留挡住"."""
    edges: dict[str, set[str]] = defaultdict(set)
    for rid, t in res.queue:
        r = res.net.routes.get(rid)
        if r is None:
            continue
        for tc in res.footprint(r):
            holder_route = res.reserved.get(tc)
            if holder_route is None:
                continue
            u = res.route_train.get(holder_route)
            if u is not None and u != t:
                edges[t].add(u)
    return edges


def find_cycle(edges: dict[str, set[str]]) -> list[str] | None:
    """有向图找环(DFS). 返回环上的结点列表(闭合), 无环返回 None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = defaultdict(int)
    stack: list[str] = []

    def dfs(u: str):
        color[u] = GREY
        stack.append(u)
        for v in edges.get(u, ()):  # type: ignore[union-attr]
            if color[v] == GREY:                      # 回边 → 环
                i = stack.index(v)
                return stack[i:] + [v]
            if color[v] == WHITE:
                c = dfs(v)
                if c:
                    return c
        stack.pop()
        color[u] = BLACK
        return None

    nodes = set(edges) | {v for vs in edges.values() for v in vs}
    for n in nodes:
        if color[n] == WHITE:
            c = dfs(n)
            if c:
                return c
    return None


def has_deadlock(res) -> bool:
    return find_cycle(wait_for_graph(res)) is not None


def deadlock_cycle(res) -> list[str] | None:
    return find_cycle(wait_for_graph(res))


def would_create_deadlock(res, route, train: str) -> bool:
    """一步前瞻: 若此刻给 train 办 route, 会否当即形成死锁环? (在快照上试办后回滚, 不改真实态.)"""
    snap = (dict(res.reserved), dict(res.route_train),
            dict(res.route_fp), list(res.queue))
    try:
        res.request(route, train)
        return has_deadlock(res)
    finally:
        res.reserved, res.route_train, res.route_fp, res.queue = (
            snap[0], snap[1], snap[2], snap[3])
