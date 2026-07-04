"""设施实体(不含时间/仿真态): Signal / TrackCircuit / Route / Platform.

设计要点:
- 信号身份 = 数字号(int). TD 前缀(DC/TD/DW/DY/EC) 仅区域码, 号在全区唯一(已实测0冲突).
- 进路身份 = route 字段全名(如 RTD5045A(M)); 信号号/出口/类别从中解析(start/end 列含
  X-berth 等非信号起点, 不可作信号身份).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrackCircuit:
    """轨道电路(闭塞时间的占用原语)."""

    tc_id: str
    routes_using: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Route:
    """一条进路(信号出口): 从某信号经一串 TC 到下一节点."""

    route_id: str          # RTD5045A(M) —— 全名, 唯一
    signal: int            # 起始信号号 5045
    prefix: str            # TD / DC / DW / DY / EC
    exit: str              # 出口标记, 如 "A" 或 "B-1"
    route_class: str       # M(主) / C(副/call-on) / S(调车)
    start_node: str        # route_to_tc_all.start 原始值(可能是 X-berth/TC)
    end_node: str          # route_to_tc_all.end 原始值
    end_signal: int | None # 若 end 为数字信号号则填, 否则 None(区域外/非信号终点)
    tc_sequence: tuple[str, ...]  # 途经轨道电路序列(占用建模用)
    in_scope: bool         # 起始信号 ∈ 48 白名单
    # 物理属性(来自 Derby_info, 可缺): 供 M3 占用/闭塞时间
    length_m: float | None = None
    ave_speed_ms: float | None = None
    ave_grad: float | None = None
    gap_time_s: float | None = None
    berth_path: tuple[str, ...] = ()

    @property
    def exit_key(self) -> tuple[int, str]:
        """(信号, 出口) —— 忽略 M/C/S 类别的物理走向标识."""
        return (self.signal, self.exit)


@dataclass
class Signal:
    """信号(决策点或途经)."""

    number: int
    prefixes: set[str] = field(default_factory=set)
    in_scope: bool = False
    is_decision_point: bool = False
    route_ids: list[str] = field(default_factory=list)      # 出口进路(全名)
    platform_end: tuple[int, str] | None = None             # (platform_id, 'N'/'S') 若为站台端

    @property
    def n_routes(self) -> int:
        return len(self.route_ids)


@dataclass(frozen=True)
class Platform:
    """站台: 由若干轨道电路子段构成, 两端各有站台端信号."""

    platform_id: int
    tc_sections: dict[str, str]          # tc_id -> sub_section(A/B/middle)
    end_signals: dict[str, int]          # 'N'/'S' -> 信号号
