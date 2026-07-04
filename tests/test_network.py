"""M1 验收测试: 设施网建对了吗."""

import pytest

from derbysim import config
from derbysim.infra import InfrastructureNetwork


@pytest.fixture(scope="module")
def net():
    return InfrastructureNetwork.load()


def test_decision_points_equal_35_and_match_const(net):
    dp = net.decision_points()
    assert len(dp) == 35
    assert dp == set(config.DECISION_POINTS)


def test_all_48_in_scope_signals_present(net):
    # 48 白名单信号都应在网中出现(作起点或终点)
    missing = config.IN_SCOPE_SIGNALS - set(net.signals)
    assert not missing, f"缺失在范围信号: {sorted(missing)}"


def test_no_isolated_in_scope_signal_among_decision_points(net):
    # 决策点必有 >1 出口
    for s in config.DECISION_POINTS:
        assert net.signals[s].n_routes > 1


def test_high_selectivity_signals(net):
    # 最高选择度决策点: 5045/5076/5486/5488 各应有多条出口进路
    for s in (5045, 5076, 5486, 5488):
        assert net.signals[s].n_routes >= 9, f"{s} 出口进路过少: {net.signals[s].n_routes}"


def test_routes_have_tc_sequences(net):
    # 在范围进路应带非空 TC 序列(占用建模基础)
    empty = [r.route_id for r in net.routes.values() if r.in_scope and not r.tc_sequence]
    assert not empty, f"在范围进路缺 TC 序列: {empty[:5]}"


def test_routes_from_5045_are_all_from_5045(net):
    for r in net.routes_from(5045):
        assert r.signal == 5045


def test_platforms_have_end_signals(net):
    assert net.platforms, "无站台"
    # 站台端信号应落在信号集中
    for plat in net.platforms.values():
        for sig in plat.end_signals.values():
            assert sig in net.signals
