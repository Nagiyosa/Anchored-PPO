"""区域过滤: 判定一条 TD 事件是否落在 48信号/63berth 实验区域, 并解析 TD id.

TD id 形态(实测):
    Signal          'STD5054'         = S + 前缀(2) + 信号号
    Route/Panel     'RTD5053G(M)'     = R + 前缀(2) + 信号号 + 出口 + (类别)
    CA/CB/CC        id='0', 用 from/to berth; 站台 berth 形如 'PLT3'
    TRTS            'LPLAT6ATRS(S)'   = 站台就绪, 含站台号
    Track           'TDMV' 等 4 字轨道电路码
"""

from __future__ import annotations

import re

from derbysim import config

_SIG_RE = re.compile(r"^[SR][A-Z]{2}(\d+)")     # S(信号)/R(进路) + 前缀 + 信号号
_PLAT_TRTS_RE = re.compile(r"LPLAT(\d+)")        # TRTS 站台号
_PLAT_BERTH_RE = re.compile(r"^PLT(\d+)$")       # 站台 berth PLT<n>


def parse_signal(td_id: str) -> int | None:
    m = _SIG_RE.match(td_id or "")
    return int(m.group(1)) if m else None


def parse_platform(td_id: str) -> int | None:
    """从 TRTS id 解析站台号."""
    m = _PLAT_TRTS_RE.search(td_id or "")
    return int(m.group(1)) if m else None


def parse_platform_berth(berth: str) -> int | None:
    """从站台 berth 'PLT3' 解析站台号."""
    m = _PLAT_BERTH_RE.match((berth or "").strip())
    return int(m.group(1)) if m else None


def is_region_berth(berth: str) -> bool:
    """63-berth 区域 或 德比站台 berth(PLT<n>)."""
    b = (berth or "").strip()
    if b in config.REGION_BERTHS:
        return True
    p = parse_platform_berth(b)
    return p is not None and p in config.DERBY_PLATFORMS


def event_in_region(row: dict, region_tc_set: frozenset[str]) -> bool:
    etype = row["type"]
    if etype in ("Signal", "Route", "Panel_Request"):
        sig = parse_signal(row["id"])
        return sig is not None and sig in config.IN_SCOPE_SIGNALS
    if etype == "CA":
        return is_region_berth(row["from_berth"]) or is_region_berth(row["to_berth"])
    if etype == "CB":
        return is_region_berth(row["from_berth"])
    if etype == "CC":
        return is_region_berth(row["to_berth"])
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
