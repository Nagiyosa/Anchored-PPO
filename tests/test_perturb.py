"""M10c 扰动测试(复审后, 2026-07-23): 只保留**安全且成立**的断言。

复审发现(见 docs/M10_DESIGN.md §12):
    - 延误注入在常态稀疏窗口**惰性**(Δ=0, 无 overlap 可扰) —— 不主张它增冲突, 只验安全+确定性。
    - `add_resampled_trains`(clone route_chain)与预留模型**不兼容**: 预留按 route_id 记身份, clone 复用
      源车 route_id → 并发同 route 时 route_train 被覆盖 → **破坏无碰撞不变式**。故其相关测试撤下, 待重设计。
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import PanelAction
from derbysim.sim.reward import RewardConfig, RewardedDynamicEnv, to_dyn_trains, realized_weighted_delay
from derbysim.sim.perturb import (DelaySampler, load_derby_lateness, inject_primary_delays,
                                  rank_high_disruption_windows, build_clean_window_dyn_trains,
                                  PRIMARY_FRACTION, REAL_FRAC_LATE, CLIP_MAX_MIN)

WIN = ("2023-11-06 07:00:00", "2023-11-06 10:00:00")


def _load():
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    win = td[(td["ts"] >= WIN[0]) & (td["ts"] < WIN[1])].copy()
    built = build_env_trains_for_window(win, net, dwell, ptc)
    return net, params, dwell, ptc, built


def _replay(env, dyn):
    obs, info = env.reset(dyn); p = info["decision"]
    while p is not None:
        act = None
        for tid, rid in p.logged.items():
            act = PanelAction("set", tid, rid); break
        if act is None:
            act = PanelAction("wait")
        obs, r, done, info = env.step(act); p = info["decision"]
        if done:
            break
    return env


# ---- 采样器数据锚定 ----
def test_lateness_pool_is_real_and_clipped():
    pool = load_derby_lateness()
    assert len(pool) > 1000
    assert pool.min() > 0 and pool.max() <= CLIP_MAX_MIN
    assert 0.20 <= PRIMARY_FRACTION <= 0.50
    assert abs(REAL_FRAC_LATE - 0.50) < 1e-9


def test_sampler_alpha_scaling_and_determinism():
    pool = np.array([2.0, 4.0, 10.0, 20.0])
    s = DelaySampler(pool, alpha=0.36)
    assert s.sample_seconds(np.random.default_rng(7)) == s.sample_seconds(np.random.default_rng(7))
    v = s.sample_seconds(np.random.default_rng(3))
    assert any(abs(v - 0.36 * m * 60.0) < 1e-9 for m in pool)


# ---- 注入: 确定性 + 只后推不改路由 ----
def test_inject_deterministic_and_route_preserved():
    net, params, dwell, ptc, built = _load()
    dyn = to_dyn_trains(built)
    s = DelaySampler(load_derby_lateness())
    a = inject_primary_delays(dyn, s, seed=42)
    b = inject_primary_delays(dyn, s, seed=42)
    assert [t.entry_time for t in a] == [t.entry_time for t in b]
    base = {t.train_id: t for t in dyn}
    for t in a:
        assert t.route_chain == base[t.train_id].route_chain
        assert t.entry_time >= base[t.train_id].entry_time - 1e-6


# ---- 真实高扰动窗口(冲突来源, 替代不安全的合成加车) ----
def test_high_disruption_windows_are_real_and_ranked():
    ws = rank_high_disruption_windows(win_hours=3, top_k=5)
    assert len(ws) == 5
    lat = [w["total_lateness_min"] for w in ws]
    assert lat == sorted(lat, reverse=True)              # 按真实晚点降序
    assert lat[0] > 500 and ws[0]["n_late"] > 0          # top 窗口确有真实扰动
    assert ws[0]["end"] > ws[0]["start"]                 # 真实时间窗


# ---- 真实拥堵窗口必须链清洗才无碰撞(R3 附带发现) ----
def test_clean_window_build_no_collision_on_busy_window():
    # 11-14 09-12 是真实高扰动窗口; raw build 含 H6 脏车 replay 会撞车, 链清洗后 0 碰撞
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    win = td[(td["ts"] >= "2023-11-14 09:00:00") & (td["ts"] < "2023-11-14 12:00:00")].copy()
    dyn = build_clean_window_dyn_trains(win, net, dwell, ptc)
    assert len(dyn) > 10
    e = _replay(RewardedDynamicEnv(net, params, ptc, dwell), dyn)
    assert not e.engine.collisions              # 链清洗 → 保无碰撞不变式


# ---- 延误注入: 结构性无碰撞 + 确定性(不主张增冲突, 常态窗口惰性) ----
def test_delay_injection_safe_and_deterministic():
    net, params, dwell, ptc, built = _load()
    dyn = to_dyn_trains(built); cfg = RewardConfig()
    s = DelaySampler(load_derby_lateness())
    for seed in range(4):
        e1 = _replay(RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg),
                     inject_primary_delays(dyn, s, seed=seed, p_late=0.8))
        e2 = _replay(RewardedDynamicEnv(net, params, ptc, dwell, reward_config=cfg),
                     inject_primary_delays(dyn, s, seed=seed, p_late=0.8))
        assert not e1.engine.collisions
        assert realized_weighted_delay(e1.engine, cfg) == realized_weighted_delay(e2.engine, cfg)
