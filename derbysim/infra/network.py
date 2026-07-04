"""InfrastructureNetwork —— 从静态 CSV 建设施网(信号/进路/TC/站台 + 邻接图).

用法:
    from derbysim.infra import InfrastructureNetwork
    net = InfrastructureNetwork.load()
    net.self_check()            # 返回自检字典
    net.routes_from(5045)       # 5045 的全部出口进路(动作空间雏形)
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass, field

from derbysim import config
from derbysim.infra.entities import Platform, Route, Signal, TrackCircuit

# route 全名解析: R + 前缀(2) + 信号号 + 出口字母(+可选 -变体) + (类别)
_ROUTE_RE = re.compile(r"^R([A-Z]{2})(\d+)([A-Z])(?:-(\d+))?\(([MCS])\)$")


def _parse_tc_list(raw: str) -> tuple[str, ...]:
    """解析形如 "['TDAB', 'TDAC']" 的列表字面量."""
    raw = (raw or "").strip()
    if not raw:
        return ()
    try:
        val = ast.literal_eval(raw)
        return tuple(str(x) for x in val)
    except (ValueError, SyntaxError):
        return ()


def _to_float(x: str) -> float | None:
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _canonical_key(r: Route) -> tuple[int, int]:
    """同名进路选规范行的排序键: 优先 start 恰为信号号, 其次 TC 序列更长."""
    start_is_signal = r.start_node.isdigit() and int(r.start_node) == r.signal
    return (1 if start_is_signal else 0, len(r.tc_sequence))


@dataclass
class InfrastructureNetwork:
    signals: dict[int, Signal] = field(default_factory=dict)
    routes: dict[str, Route] = field(default_factory=dict)
    track_circuits: dict[str, TrackCircuit] = field(default_factory=dict)
    platforms: dict[int, Platform] = field(default_factory=dict)
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "InfrastructureNetwork":
        net = cls()
        physical = net._load_derby_info_physical()
        net._load_routes(physical)
        net._load_platforms()
        net._finalize_signals()
        net._build_adjacency()
        return net

    def _load_derby_info_physical(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not config.DATA_DERBY_INFO.exists():
            return out
        with open(config.DATA_DERBY_INFO, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = "R" + row["routeid"].strip()
                out[rid] = {
                    "length_m": _to_float(row.get("length")),
                    "ave_speed_ms": _to_float(row.get("ave_speed(m/s)")),
                    "ave_grad": _to_float(row.get("ave_grad")),
                    "gap_time_s": _to_float(row.get("gap_time(s)")),
                    "berth_path": _parse_tc_list(row.get("path", "")),
                }
        return out

    def _load_routes(self, physical: dict[str, dict]) -> None:
        with open(config.DATA_ROUTE_TC, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row["route"].strip()
                m = _ROUTE_RE.match(rid)
                if not m:
                    continue
                prefix, num_s, letter, variant, cls_ = m.groups()
                signal = int(num_s)
                exit_ = letter if not variant else f"{letter}-{variant}"
                end_raw = row["end"].strip()
                start_raw = row["start"].strip()
                tc_seq = _parse_tc_list(row.get("track", ""))
                phys = physical.get(rid, {})
                new = Route(
                    route_id=rid,
                    signal=signal,
                    prefix=prefix,
                    exit=exit_,
                    route_class=cls_,
                    start_node=start_raw,
                    end_node=end_raw,
                    end_signal=int(end_raw) if end_raw.isdigit() else None,
                    tc_sequence=tc_seq,
                    in_scope=signal in config.IN_SCOPE_SIGNALS,
                    length_m=phys.get("length_m"),
                    ave_speed_ms=phys.get("ave_speed_ms"),
                    ave_grad=phys.get("ave_grad"),
                    gap_time_s=phys.get("gap_time_s"),
                    berth_path=phys.get("berth_path", ()),
                )
                old = self.routes.get(rid)
                if old is None or _canonical_key(new) > _canonical_key(old):
                    self.routes[rid] = new

        for r in self.routes.values():
            for tc in r.tc_sequence:
                tcs = self.track_circuits.get(tc)
                using = (tcs.routes_using if tcs else frozenset()) | {r.route_id}
                self.track_circuits[tc] = TrackCircuit(tc_id=tc, routes_using=using)

    def _load_platforms(self) -> None:
        tc_by_platform: dict[int, dict[str, str]] = {}
        if config.DATA_PLATFORM_TC.exists():
            with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid_s = (row.get("platform_id") or "").strip()
                    if not pid_s.isdigit():
                        continue
                    pid = int(pid_s)
                    tc_by_platform.setdefault(pid, {})[row["tc_id"].strip()] = (
                        row.get("sub_section", "").strip()
                    )
        ends_by_platform: dict[int, dict[str, int]] = {}
        if config.DATA_PLATFORM_END_SIGNALS.exists():
            with open(config.DATA_PLATFORM_END_SIGNALS, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid_s = (row.get("platform_id") or "").strip()
                    sig_s = (row.get("signal_id") or "").strip()
                    if not (pid_s.isdigit() and sig_s.isdigit()):
                        continue
                    pid, sig = int(pid_s), int(sig_s)
                    ends_by_platform.setdefault(pid, {})[row["direction"].strip()] = sig
        all_pids = set(tc_by_platform) | set(ends_by_platform)
        for pid in sorted(all_pids):
            self.platforms[pid] = Platform(
                platform_id=pid,
                tc_sections=tc_by_platform.get(pid, {}),
                end_signals=ends_by_platform.get(pid, {}),
            )

    def _get_signal(self, number: int) -> Signal:
        sig = self.signals.get(number)
        if sig is None:
            sig = Signal(number=number, in_scope=number in config.IN_SCOPE_SIGNALS)
            self.signals[number] = sig
        return sig

    def _finalize_signals(self) -> None:
        for r in self.routes.values():
            sig = self._get_signal(r.signal)
            sig.prefixes.add(r.prefix)
            sig.route_ids.append(r.route_id)
            if r.end_signal is not None:
                self._get_signal(r.end_signal)
        for pid, plat in self.platforms.items():
            for direction, sig_num in plat.end_signals.items():
                self._get_signal(sig_num).platform_end = (pid, direction)
        for sig in self.signals.values():
            sig.is_decision_point = sig.in_scope and sig.n_routes > 1

    def _build_adjacency(self) -> None:
        for r in self.routes.values():
            if r.end_signal is not None:
                self.adjacency.setdefault(r.signal, set()).add(r.end_signal)

    def routes_from(self, signal: int) -> list[Route]:
        sig = self.signals.get(signal)
        if not sig:
            return []
        return [self.routes[rid] for rid in sig.route_ids]

    def decision_points(self) -> set[int]:
        return {s for s, sig in self.signals.items() if sig.is_decision_point}

    def in_scope_signals(self) -> set[int]:
        return {s for s, sig in self.signals.items() if sig.in_scope}

    def self_check(self) -> dict:
        derived_dp = self.decision_points()
        const_dp = set(config.DECISION_POINTS)
        in_scope = self.in_scope_signals()
        selectivity = sorted(
            ((sig.n_routes, s) for s, sig in self.signals.items() if sig.is_decision_point),
            reverse=True,
        )
        isolated = sorted(
            s for s in config.IN_SCOPE_SIGNALS if self.signals.get(s, Signal(s)).n_routes == 0
        )
        boundary_routes = [
            r.route_id
            for r in self.routes.values()
            if r.in_scope and r.end_signal is not None
            and r.end_signal not in config.IN_SCOPE_SIGNALS
        ]
        return {
            "n_signals_total": len(self.signals),
            "n_signals_in_scope": len(in_scope),
            "n_routes_total": len(self.routes),
            "n_routes_in_scope": sum(1 for r in self.routes.values() if r.in_scope),
            "n_track_circuits": len(self.track_circuits),
            "n_platforms": len(self.platforms),
            "n_decision_points_derived": len(derived_dp),
            "decision_points_match_const": derived_dp == const_dp,
            "dp_in_const_not_derived": sorted(const_dp - derived_dp),
            "dp_derived_not_in_const": sorted(derived_dp - const_dp),
            "top_selectivity": selectivity[:6],
            "isolated_in_scope_signals": isolated,
            "n_boundary_exit_routes": len(boundary_routes),
        }
