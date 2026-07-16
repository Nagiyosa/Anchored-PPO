"""M9c 测试: 等待图找环(检测) + 一步前瞻屏蔽. 构造真实 2 车循环等待."""

from __future__ import annotations

import pytest

from derbysim.infra import InfrastructureNetwork
from derbysim.sim.reservation import RouteReservation, LOCKED, QUEUED
from derbysim.sim.deadlock import (
    find_cycle, has_deadlock, deadlock_cycle, would_create_deadlock)


@pytest.fixture(scope="module")
def net():
    return InfrastructureNetwork.load()


def _disjoint_pair(net, R):
    """取两条足迹不相交的进路 P,Q(可被两车分别独立锁)."""
    P = net.routes["RDC5061A(M)"]
    fpP = R.footprint(P)
    for r in net.routes.values():
        if r.route_id != P.route_id and not (R.footprint(r) & fpP):
            return P, r
    raise RuntimeError("no disjoint route found")


def test_find_cycle_basic():
    assert find_cycle({"a": {"b"}, "b": {"a"}}) is not None      # 2-环
    assert find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) is not None  # 3-环
    assert find_cycle({"a": {"b"}, "b": {"c"}}) is None          # 无环


def test_no_deadlock_single_wait(net):
    R = RouteReservation(net)
    P, _ = _disjoint_pair(net, R)
    assert R.request(P, "A") == LOCKED
    assert R.request(P, "B") == QUEUED       # B 单向等 A, 无环
    assert not has_deadlock(R)


def test_deadlock_two_cycle(net):
    R = RouteReservation(net)
    P, Q = _disjoint_pair(net, R)
    assert R.request(P, "A") == LOCKED
    assert R.request(Q, "B") == LOCKED
    assert R.request(Q, "A") == QUEUED       # A 等 B (Q 被 B 占)
    assert R.request(P, "B") == QUEUED       # B 等 A (P 被 A 占) → 环
    assert has_deadlock(R)
    cyc = deadlock_cycle(R)
    assert "A" in cyc and "B" in cyc


def test_lookahead_prevents_deadlock(net):
    R = RouteReservation(net)
    P, Q = _disjoint_pair(net, R)
    R.request(P, "A")
    R.request(Q, "B")
    R.request(Q, "A")                        # A→B 边已成, 尚无环
    assert not has_deadlock(R)
    qlen = len(R.queue)
    # 此刻给 B 办 P 会成环 → 前瞻报 True, 且不改真实态(快照回滚)
    assert would_create_deadlock(R, P, "B")
    assert len(R.queue) == qlen and not has_deadlock(R)
