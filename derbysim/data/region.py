"""区域过滤: 判定一条 TD 事件是否落在 48信号/63berth 实验区域, 并解析 TD id.

TD id 形态(实测):
    Signal          'STD5054'         = S + 前缀(2) + 信号号
    Route/Panel     'RTD5053G(M)'     = R + 前缀(2) + 信号号 + 出口 + (类别)
    CA/CB/CC        id='0', 用 from/to berth
    TRTS            'LPLAT6ATRS(S)'   = 站台就绪, 含站台号
    Track           'TDMV' 等 4 字轨道电路码
"""

from __future__ import annotations

import re

from derbysim import config

# S(信号) 或 R(进路) + 2字前缀 + 数字信号号
_SIG_RE = re.compile(r"^[SR][A-Z]{2}(\d+)")
# TRTS 站台号: LPLAT<数字>
_PLAT_RE = re.compile(r"LPLAT(\d+)")


def parse_signal(td_id: str) -> int | None:
    """从 Signal / Route / Panel_Request 的 id 解析信号号."""
    m = _SIG_RE.match(td_id or "")
    return int(m.group(1)) if m else None


def parse_platform(td_id: str) -> int | None:
    """从 TRTS id 解析站台号."""
    m = _PLAT_RE.search(td_id or "")
    return int(m.group(1)) if m else None


def event_in_region(row: dict, region_tc_set: frozenset[str]) -> bool:
    """一条 TD 事件是否在实验区域内(按事件类型分别判定).

    row: 含 type/id/from_berth/to_berth 的 TD 行(str 值).
    region_tc_set: 区域内轨道电路集合(来自在范围进路的 TC 序列并集).
    """
    etype = row["type"]
    if etype in ("Signal", "Route", "Panel_Request"):
        sig = parse_signal(row["id"])
        return sig is not None and sig in config.IN_SCOPE_SIGNALS
    if etype == "CA":
        return (
            row["from_berth"].strip() in config.REGION_BERTHS
            or row["to_berth"].strip() in config.REGION_BERTHS
        )
    if etype == "CB":
        return row["from_berth"].strip() in config.REGION_BERTHS
    if etype == "CC":
        return row["to_berth"].strip() in config.REGION_BERTHS
    if etype == "Track":
        return row["id"].strip() in region_tc_set
    if etype == "TRTS":
        p = parse_platform(row["id"])
        return p is not None and p in config.DERBY_PLATFORMS
    return False


def region_tc_set_from_network(net) -> frozenset[str]:
    """区域轨道电路集 = 在范围进路的 TC 序列并集."""
    tcs: set[str] = set()
    for r in net.routes.values():
        if r.in_scope:
            tcs.update(r.tc_sequence)
    return frozenset(tcs)
