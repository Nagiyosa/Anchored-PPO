"""D33 Step 9 驱动器测试: 扰动回放的等价性与 reroute 注入正确性。

用一个 val 运营日验证: ①无注入时扰动回放逐位等于信号员回放(不引入偏差); ②reroute 确实换了
一条不同的合法进路且安全项仍为 0; ③普查报告的可注入点非空且自洽。
"""
from __future__ import annotations

import pytest

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy, run_episode
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.learn.step9 import census_signaller, run_injection
from derbysim.sim.reward import RewardedDynamicEnv


@pytest.fixture(scope="module")
def assets():
    return load_assets()


def _day():
    return operating_days("val")[0]


def _env(assets):
    net, params, dwell, ptc = assets
    return RewardedDynamicEnv(net, params, ptc, dwell), PanelAdapter(net, Featurizer(net, ptc))


def test_no_injection_equals_signaller_replay(assets):
    """inject_at 永不触发时, 扰动回放必须与纯信号员回放逐位相同(completion/delay/steps)。"""
    net, params, dwell, ptc = assets
    day = _day()
    env, adapter = _env(assets)
    base = run_episode(env, adapter, make_signaller_replay_policy(env),
                       build_day_trains(day, inject=False), horizon=day_horizon(day)).metrics
    env2 = RewardedDynamicEnv(net, params, ptc, dwell)
    inj = run_injection(env2, adapter, build_day_trains(day, inject=False),
                        day_horizon(day), inject_at=10**9, mode="reroute")
    assert not inj["injected"]
    for k in ("completion_rate", "weighted_secondary_delay_s", "n_steps",
              "n_collisions", "n_reached_exit"):
        assert base[k] == inj["metrics"][k], k


def test_reroute_forces_a_different_legal_route(assets):
    """在第一个可分叉点强制替代路: 记录 forced≠logged, 安全项仍 0, 偏离车归因齐全。"""
    day = _day()
    env, adapter = _env(assets)
    census = census_signaller(env, adapter, build_day_trains(day, inject=False), day_horizon(day))
    assert census["n_branchable"] > 0, "该 val 日应存在可分叉决策点"
    k = census["branchable_idx"][0]

    env2, adapter2 = _env(assets)
    inj = run_injection(env2, adapter2, build_day_trains(day, inject=False),
                        day_horizon(day), inject_at=k, mode="reroute")
    info = inj["info"]
    assert info is not None and info["mode"] == "reroute"
    assert info["forced_route"] != info["logged_route"]
    assert inj["metrics"]["n_collisions"] == 0
    assert inj["deviated"] is not None and inj["deviated"]["train_id"] == info["deviated_train"]


def test_census_branchable_is_subset_of_set_panels(assets):
    """普查自洽: 可分叉点 ⊆ set panel, 且 baseline completion 为 1.000。"""
    day = _day()
    env, adapter = _env(assets)
    c = census_signaller(env, adapter, build_day_trains(day, inject=False), day_horizon(day))
    assert set(c["branchable_idx"]).issubset(set(c["set_idx"]))
    assert c["n_set"] == c["n_set_panels"]
    assert c["baseline_metrics"]["completion_rate"] == 1.0
