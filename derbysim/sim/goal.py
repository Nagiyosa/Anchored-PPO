"""M9b 目标与可达性: 出区口方向组 + 可达性过滤(硬约束) + 客运站台服务.

CS 视角: episode 的**终止判据**与**动作有效性约束**——
    - 每列车的目标 = 到达其**去向一致的出区口组**中任一(方向组), 客运还须途中**服务一个站台**。
    - 动作有效性(硬约束, 非学习项): 在决策点选一条进路, 必须**仍能到达**目标; 会把车导入死胡同
      (无法再到达所需出区口/未服务站台就出区)的进路, 从候选集**剔除**。这不是"决策风格"而是无效动作。

方向组由拓扑推得(数据驱动, 非人工): 出区口若由**同一批区内信号**可达(approach 集相交), 归为一组
—— 平行同向出口(如 5093/5095/5097)自然成组。安全/占用不在本模块(见 reservation.py)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from derbysim import config


def region_exit_signals(net) -> set[int]:
    """通向区外的出区口信号(in-scope 进路的 end_signal 落在区外)."""
    insc = set(config.IN_SCOPE_SIGNALS)
    return {r.end_signal for r in net.routes.values()
            if getattr(r, "in_scope", True) and r.end_signal is not None
            and r.end_signal not in insc}


def _exit_approaches(net) -> dict[int, set[int]]:
    """出区口 -> 直接通向它的区内起始信号集."""
    insc = set(config.IN_SCOPE_SIGNALS)
    out: dict[int, set[int]] = {}
    for r in net.routes.values():
        if (getattr(r, "in_scope", True) and r.end_signal is not None
                and r.end_signal not in insc):
            out.setdefault(r.end_signal, set()).add(r.signal)
    return out


def exit_direction_groups(net) -> dict[int, frozenset[int]]:
    """出区口 -> 其所属方向组(frozenset). 规则: approach 集相交的出口连通即同组."""
    appr = _exit_approaches(net)
    exits = sorted(appr)
    parent = {e: e for e in exits}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, e1 in enumerate(exits):
        for e2 in exits[i + 1:]:
            if appr[e1] & appr[e2]:
                parent[find(e1)] = find(e2)
    comp: dict[int, set[int]] = {}
    for e in exits:
        comp.setdefault(find(e), set()).add(e)
    groups = {r: frozenset(v) for r, v in comp.items()}
    return {e: groups[find(e)] for e in exits}


def platform_end_signals(net) -> set[int]:
    return {s for p in net.platforms.values() for s in p.end_signals.values()}


def _reach(net, start: int, insc: set[int]) -> tuple[set[int], set[int]]:
    """从 start 沿 adjacency BFS(只在区内扩展); 返回(可达区内信号, 触及的出区口).

    触及出区口即边界终止, 不越区外扩展(避免经区外网络回环污染可达性).
    """
    seen = {start}
    q = deque([start])
    hit_exits: set[int] = set()
    while q:
        u = q.popleft()
        for v in net.adjacency.get(u, ()):  # type: ignore[attr-defined]
            if v not in insc:
                hit_exits.add(v)             # 边界出区口, 不扩展
            elif v not in seen:
                seen.add(v)
                q.append(v)
    return seen, hit_exits


@dataclass
class Goal:
    exit_group: frozenset[int]        # 去向一致的出区口组(空=区内终止,如 ECS 入库)
    must_serve_platform: bool         # 客运(停站)须服务一个站台


def train_goal(route_chain, net, stops) -> Goal:
    """从列车的进路链推目标: 观测出区口所属方向组 + 是否停站(须服务站台)."""
    groups = exit_direction_groups(net)
    exit_grp: frozenset[int] = frozenset()
    for rid in route_chain:                # 取链中通向区外的进路, 其 end_signal 的方向组
        r = net.routes.get(rid)
        if r and r.end_signal in groups:
            exit_grp = groups[r.end_signal]
            break
    return Goal(exit_group=exit_grp, must_serve_platform=bool(stops))


class Reachability:
    """封装可达性判定, 缓存 BFS 结果(拓扑静态)."""

    def __init__(self, net) -> None:
        self.net = net
        self.insc = set(config.IN_SCOPE_SIGNALS)
        self.plat_sigs = platform_end_signals(net)
        self._cache: dict[int, tuple[set[int], set[int]]] = {}
        self.pair: dict[int, int] = {}                # 站台两端互为配对(反向出站)
        for _p in net.platforms.values():
            _e = list(_p.end_signals.values())
            if len(_e) == 2:
                self.pair[_e[0]] = _e[1]
                self.pair[_e[1]] = _e[0]

    def _r(self, start: int):
        if start not in self._cache:
            self._cache[start] = _reach(self.net, start, self.insc)
        return self._cache[start]

    def can_reach_goal(self, from_signal: int, goal: Goal, platform_served: bool) -> bool:
        """从 from_signal 出发, 是否仍能达成目标(可达出区口组; 若需且未服务, 先可达站台再出区)."""
        if from_signal is None:
            return True
        seen, exits = self._r(from_signal)
        need_plat = goal.must_serve_platform and not platform_served
        if not goal.exit_group:               # 区内终止(无出区口目标): 只需(若需)可达站台
            return (not need_plat) or bool((seen | {from_signal}) & self.plat_sigs)
        at_exit = from_signal in goal.exit_group   # 进路已止于出区口本身 = 已达
        if not need_plat:
            return at_exit or bool(exits & goal.exit_group)
        # 需先服务站台再出区: 存在可达站台 p, 且从 p 可达该出口组
        cand_plats = (seen | {from_signal}) & self.plat_sigs
        for p in cand_plats:
            _, pexits = self._r(p)
            if pexits & goal.exit_group:
                return True
        return False

    def can_reach_exit(self, from_signal: int | None, exit_group) -> bool:
        """仅"出区口方向可达"(含站台反向: 站台端可从配对端出). 空 exit_group(区内终止)恒 True."""
        if from_signal is None or not exit_group:
            return True
        sigs = {from_signal}
        if from_signal in self.pair:
            sigs.add(self.pair[from_signal])          # 站台可反向出站
        for s in sigs:
            if s in exit_group:
                return True
            _, exits = self._r(s)
            if exits & exit_group:
                return True
        return False

    def route_keeps_goal(self, route, goal: Goal, platform_served: bool) -> bool:
        """动作掩码硬约束 = **仅出区口方向可达**(防真死胡同). 站台服务=**软目标**(转 M10 奖励,
        客运没停站出区=扣分, 不禁止选路) —— 见 M9_DESIGN §10 修正(原站台硬筛致 22.4% false-negative,
        违站台决策 §1). platform_served 保留供调用方/奖励用, 此处不作硬筛."""
        return self.can_reach_exit(route.end_signal, goal.exit_group)
