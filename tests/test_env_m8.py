"""M8 测试: 防泄露状态 + 物理触发 + 联锁掩码 + D18 泄露自检.

配合 test_env.py 的 M7 逐位等价(用 advance_cfg 默认加载, 已证触发提前不破坏等价).
"""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import DerbyEnv, build_env_trains_for_window

WIN_START = "2023-11-06 07:00:00"
WIN_END = "2023-11-06 10:00:00"


def _load():
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    plat_tc_to_pid = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                plat_tc_to_pid[row["tc_id"].strip()] = int(row["platform_id"])
    win = td[(td["ts"] >= WIN_START) & (td["ts"] < WIN_END)].copy()
    return win, net, params, dwell, plat_tc_to_pid


def _collect(win, net, params, dwell, plat_tc_to_pid, policy="replay"):
    built = build_env_trains_for_window(win, net, dwell, plat_tc_to_pid)
    trains = [b[0] for b in built]
    byid = {t.train_id: t for t in trains}
    env = DerbyEnv(net, params, plat_tc_to_pid, dwell)
    env.reset(trains)
    rows = []
    while not env.done:
        ctx = env.decision()
        t = byid[ctx.train_id]
        adv = (len(t.path) - 1) - t.pos
        rows.append({"ctx": ctx, "adv": adv})
        if policy == "replay":
            env.step(ctx.logged_route_id)
        else:                                   # blind: 忽略状态, 选第一条合法进路
            env.step(ctx.legal_route_ids[0] if ctx.legal_route_ids else ctx.logged_route_id)
    return env, rows


def test_state_has_no_leak():
    """D18: obs 因果先于动作 — 日志所选进路(答案)绝不出现在 obs; 无 chosen/logged 字段;
    候选 TC 占用值只能是列车号或 None(不是进路号)."""
    win, net, params, dwell, ptc = _load()
    env, rows = _collect(win, net, params, dwell, ptc)
    for r in rows:
        ctx = r["ctx"]
        obs = ctx.obs
        blob = json.dumps(obs, default=str)
        assert ctx.logged_route_id is None or ctx.logged_route_id not in blob, \
            f"泄露: 日志进路 {ctx.logged_route_id} 出现在 obs"
        assert "chosen" not in obs and "logged" not in obs
        for tc, occv in obs["candidate_tc_occupancy"].items():
            assert occv is None or not str(occv).startswith("R"), \
                f"候选占用值疑似进路号(泄露): {occv}"


def test_mask_is_interlocking_subset():
    """掩码 ⊆ 该信号出口进路(或被记录的 mask_miss 补回)."""
    win, net, params, dwell, ptc = _load()
    env, rows = _collect(win, net, params, dwell, ptc)
    miss = {(m["train"], m["signal"], m["logged"]) for m in env.engine.mask_misses}
    for r in rows:
        ctx = r["ctx"]
        if ctx.signal is None:
            continue
        exits = {rt.route_id for rt in net.routes_from(ctx.signal)}
        for rid in ctx.legal_route_ids:
            assert rid in exits or (ctx.train_id, ctx.signal, rid) in miss, \
                f"掩码含非本信号出口且未记录: {rid} @ {ctx.signal}"


def test_trigger_fires_early_on_real_decisions():
    """真实多选决策应在真实决策时刻(信号上游/停站中)触发, 而非贴信号."""
    win, net, params, dwell, ptc = _load()
    env, rows = _collect(win, net, params, dwell, ptc)
    real = [r for r in rows if len(r["ctx"].legal_route_ids) > 1]
    assert len(real) > 10
    kinds = {r["ctx"].trigger_kind for r in real}
    assert "dwell" in kinds and "upstream_n" in kinds, f"缺触发类型: {kinds}"
    early = [r for r in real if r["adv"] > 0 or r["ctx"].trigger_kind == "dwell"]
    frac_early = len(early) / len(real)
    assert frac_early >= 0.6, f"提前触发比例过低 {frac_early:.2f}(应≥0.6)"


def test_env_accepts_arbitrary_legal_action():
    """RL 就绪 + 无强制: env 不强推日志进路; 盲策略(选首条合法)能跑完且选择与回放不同."""
    win, net, params, dwell, ptc = _load()
    _, rows_replay = _collect(win, net, params, dwell, ptc, policy="replay")
    _, rows_blind = _collect(win, net, params, dwell, ptc, policy="blind")
    assert len(rows_blind) > 0
    chose_replay = [r["ctx"].logged_route_id for r in rows_replay]
    chose_blind = [r["ctx"].legal_route_ids[0] if r["ctx"].legal_route_ids else None
                   for r in rows_blind]
    assert chose_replay != chose_blind, "盲策略与回放完全相同, 环境疑似强制日志进路"


if __name__ == "__main__":
    import sys
    sys.exit(0 if pytest.main([__file__, "-s", "-q"]) == 0 else 1)
