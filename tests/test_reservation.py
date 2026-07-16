"""M9a 预留引擎测试: 空闲即锁 / 冲突排队 / overlap 致冲突 / 分段释放 pump 队列.

用真实进路: RDC5061A(body TDAB,TDAC; overlap TDAE) 与 RDC5075A(body TDAE,TDAF)
—— 车体不相交, 仅因 overlap 相邻而冲突, 正好验 overlap 的作用.
"""

from __future__ import annotations

import pytest

from derbysim.infra import InfrastructureNetwork
from derbysim.sim.reservation import RouteReservation, LOCKED, QUEUED


@pytest.fixture(scope="module")
def net():
    return InfrastructureNetwork.load()


def test_lock_when_free(net):
    R = RouteReservation(net)
    r = net.routes["RDC5061A(M)"]
    assert R.request(r, "TX") == LOCKED
    assert R.is_locked("RDC5061A(M)")
    for tc in R.footprint(r):
        assert R.reserved[tc] == "RDC5061A(M)"


def test_body_conflict_queues(net):
    R = RouteReservation(net)
    a = net.routes["RTD5045A(M)"]
    b = net.routes["RTD5045A(C)"]          # 同信号出口, 共享车体 TC
    assert set(a.tc_sequence) & set(b.tc_sequence)          # 前提: 车体确有交集
    assert R.request(a, "TX") == LOCKED
    assert R.request(b, "TY") == QUEUED    # 别的车想办 → 冲突排队
    assert ("RTD5045A(C)", "TY") in R.pending()


def test_overlap_causes_conflict(net):
    R = RouteReservation(net)
    r1 = net.routes["RDC5061A(M)"]
    r2 = net.routes["RDC5075A(M)"]
    # 前提: 车体不相交, 冲突纯由 overlap(TDAE) 引入
    assert not (set(r1.tc_sequence) & set(r2.tc_sequence))
    assert "TDAE" in R.overlap(r1) and "TDAE" not in r1.tc_sequence
    assert "TDAE" in r2.tc_sequence
    assert R.request(r1, "TX") == LOCKED
    assert R.request(r2, "TY") == QUEUED   # 若无 overlap 本不冲突 → 证明 overlap 生效


def test_sectional_release_pumps_queue(net):
    R = RouteReservation(net)
    r1 = net.routes["RDC5061A(M)"]
    r2 = net.routes["RDC5075A(M)"]
    R.request(r1, "TX")
    assert R.request(r2, "TY") == QUEUED
    # 分段释放 TDAE(r1 的 overlap TC 清出) → r2 应自动落锁
    R.release_tc("TDAE", "RDC5061A(M)")
    assert R.is_locked("RDC5075A(M)")
    assert ("RDC5075A(M)", "TY") not in R.pending()


def test_release_route_frees_and_pumps(net):
    R = RouteReservation(net)
    r1 = net.routes["RDC5061A(M)"]
    r2 = net.routes["RDC5075A(M)"]
    R.request(r1, "TX")
    R.request(r2, "TY")                     # QUEUED
    R.release_route("RDC5061A(M)")          # 整条释放
    assert not R.is_locked("RDC5061A(M)")
    assert R.is_locked("RDC5075A(M)")       # pump 后排队路落锁


def test_same_train_no_self_conflict(net):
    R = RouteReservation(net)
    r = net.routes["RDC5061A(M)"]
    R.request(r, "TX")
    assert R.can_lock(r, "TX")              # 同车对自己已预留的路不算冲突
