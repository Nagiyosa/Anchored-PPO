"""M9b 测试: 出区口方向组 + 可达性硬约束 + 客运站台服务."""

from __future__ import annotations

import pytest

from derbysim.infra import InfrastructureNetwork
from derbysim.sim.goal import (
    Goal, Reachability, exit_direction_groups, region_exit_signals, train_goal)


@pytest.fixture(scope="module")
def net():
    return InfrastructureNetwork.load()


def test_exit_direction_groups(net):
    g = exit_direction_groups(net)
    assert g[5093] == g[5095] == g[5097]        # 平行北向出口同组
    assert g[5323] == g[5329] == g[5331]        # 平行南向出口同组
    assert g[5093] != g[5323]                    # 不同方向不同组
    assert {5093, 5323} <= region_exit_signals(net)


def test_reachability_directional(net):
    g = exit_direction_groups(net)
    R = Reachability(net)
    north = Goal(g[5093], must_serve_platform=False)
    south = Goal(g[5323], must_serve_platform=False)
    assert R.can_reach_goal(5319, south, platform_served=False)       # 5319 通南向
    assert not R.can_reach_goal(5319, north, platform_served=False)   # 5319 到不了北向


def test_platform_service_reachability(net):
    g = exit_direction_groups(net)
    R = Reachability(net)
    north_stop = Goal(g[5093], must_serve_platform=True)
    south_stop = Goal(g[5323], must_serve_platform=True)
    # 5319 直通线无站台可达 → 停站车到不了(需先服务站台)
    assert not R.can_reach_goal(5319, north_stop, platform_served=False)
    # 已服务站台后, 只需可达出口
    assert R.can_reach_goal(5319, south_stop, platform_served=True)
    # 5045 能到站台 + 北向出口 → 停站客运可达
    assert R.can_reach_goal(5045, north_stop, platform_served=False)


def test_route_keeps_goal_filters_deadend(net):
    g = exit_direction_groups(net)
    R = Reachability(net)
    north = Goal(g[5093], must_serve_platform=False)
    south = Goal(g[5323], must_serve_platform=False)
    # 找一条 end 只能到南向的进路: 对北向目标应剔除, 对南向保留
    r_south = next(r for r in net.routes.values()
                   if r.end_signal is not None
                   and not R.can_reach_goal(r.end_signal, north, False)
                   and R.can_reach_goal(r.end_signal, south, False))
    assert not R.route_keeps_goal(r_south, north, platform_served=False)
    assert R.route_keeps_goal(r_south, south, platform_served=False)


def test_train_goal_derivation(net):
    exr = next(r for r in net.routes.values() if r.end_signal == 5093)
    goal = train_goal([exr.route_id], net, stops={4})
    assert 5093 in goal.exit_group and goal.must_serve_platform
    goal2 = train_goal([exr.route_id], net, stops=set())
    assert not goal2.must_serve_platform                  # 无停站 → 不须服务站台
