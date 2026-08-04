"""M12a 测试(纯 numpy, 沙盒可跑): 状态适配层 + 面板逐选项特征化 + overlap_depth 透传.

**诚实边界**: 这里**不**声称"适配层输出 == M8 `_build_state` 的运行时输出" —— M9 是自由时序,
决策时刻本就与 M8 不同, 逐时刻比对不成立(M9_DESIGN §4)。这里验的是:
  (a) **构造真值**: 手工构造的面板态下, 重建结果与手算逐键相等(键名/语义对齐 env.py::_build_state);
  (b) **纪律②**: occ 只可能来自各车 cur_tc, 锁态污染会被测出;
  (c) **纪律③**: 带 live prior 的特征器会被拒绝;
  (d) **真实窗口上的结构不变量 + 确定性**(选项逐位对齐、掩码、两遍逐位一致)。
"""

from __future__ import annotations

from functools import lru_cache

import csv
import json

import numpy as np
import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter, collate_panels, occupancy_from_panel
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.env_dynamic import DynamicDerbyEnv, DynTrain, PanelAction
from derbysim.sim.occupation import load_params
from derbysim.sim.reward import RewardedDynamicEnv

WIN = ("2023-11-06 07:00:00", "2023-11-06 09:00:00")


class _StubRoute:
    def __init__(self, tcs):
        self.tc_sequence = list(tcs)


class _StubNet:
    def __init__(self, routes):
        self.routes = {k: _StubRoute(v) for k, v in routes.items()}


class _StubFz:
    route_prior: dict = {}


@lru_cache(maxsize=1)
def _load_static():
    """重的 I/O(路网/标定/parquet/建车)只做一次 —— 全套测试要快, 且不影响确定性。"""
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
    specs = tuple((b[0].train_id, tuple(b[0].route_chain), b[0].entry_time,
                   frozenset(b[0].stop_pids), b[0].booked_platform) for b in built)
    return net, params, dwell, ptc, specs


def _load():
    """每次调用给**全新** DynTrain(引擎会推进 actual_chain/path/pos), 静态资产走缓存。"""
    net, params, dwell, ptc, specs = _load_static()
    dyn = [DynTrain(tid, list(rc), et, set(sp), bp) for tid, rc, et, sp, bp in specs]
    return net, params, dwell, ptc, dyn


def _fresh(dyn):
    """DynTrain 是有状态的(actual_chain/path/pos 被引擎推进) → 每次 reset 必须给全新对象。"""
    return [DynTrain(t.train_id, list(t.route_chain), t.entry_time,
                     set(t.stop_pids), t.booked_platform) for t in dyn]


def _fz(net, ptc):
    return Featurizer(net, ptc)                      # route_prior 默认空 = no-prior(纪律③)


# ---------- (a) 构造真值: 逐键手算 ----------
def test_rebuild_context_matches_hand_computed_truth():
    net = _StubNet({"R1": ["a", "b"], "R2": ["b", "c"]})
    ad = PanelAdapter(net, _StubFz())
    obs = {
        "now": 100.0, "tod_hour": 7,
        "reservations": {"a": "RX", "c": "RY"},          # 锁态: 若被误用, occ 会出现 RX/RY
        "n_reserved_tc": 2, "queue_len": 1,
        "trains": [
            {"train_id": "T0", "cur_tc": "z", "pos_idx": 3, "delay_min": 2.0,
             "booked_platform": 4, "service_class": "1", "rem_path": ["z", "b"],
             "dec_signal": "S5045", "awaiting": True, "has_pending": False,
             "served_platform": False, "pending_route": None},
            {"train_id": "T1", "cur_tc": "c", "pos_idx": 0, "delay_min": -1.0,
             "booked_platform": 3, "service_class": "2", "rem_path": ["c"],
             "dec_signal": None, "awaiting": False, "has_pending": False,
             "served_platform": True, "pending_route": None},
            {"train_id": "T2", "cur_tc": "q", "pos_idx": 1, "delay_min": None,
             "booked_platform": None, "service_class": "6", "rem_path": ["q", "w"],
             "dec_signal": None, "awaiting": False, "has_pending": False,
             "served_platform": False, "pending_route": None},
        ],
    }
    ctx = ad.rebuild_context(obs, "T0", ["R1", "R2"])

    # 候选 TC = a,b,c(按进路顺序去重); 占用只来自 cur_tc → 只有 c 被 T1 占
    assert ctx["candidate_tc_occupancy"] == {"a": None, "b": None, "c": "T1"}
    # 冲突车: T1(占 c 且 rem 交 c), T2 无交集不占 → 不在内
    assert [c["train_id"] for c in ctx["conflict_trains"]] == ["T1"]
    assert ctx["conflict_trains"][0]["contends_tc"] == "c"
    assert ctx["deciding"] == {"train_id": "T0", "cur_tc": "z", "pos_idx": 3,
                               "delay_min": 2.0, "booked_platform": 4, "service_class": "1"}
    assert ctx["decision_signal"] == "S5045"
    assert set(ctx) == {"now", "tod_hour", "decision_signal", "deciding",
                        "candidate_tc_occupancy", "conflict_trains"}    # 与 M8 契约同键


def test_occupancy_never_reads_reservations():
    """纪律②: 锁态(tc->route_id) 绝不进 occ; 若误用, route_id 会出现在值里."""
    obs = {"trains": [{"train_id": "T0", "cur_tc": "a"}, {"train_id": "T1", "cur_tc": None}],
           "reservations": {"a": "RX", "b": "RY"}}
    occ = occupancy_from_panel(obs)
    assert occ == {"a": "T0"}
    assert "RX" not in occ.values() and "b" not in occ


def test_live_route_prior_is_rejected():
    """纪律③: 热启动必须 no-prior; 带 live prior 的特征器构造即拒。"""
    net, _p, _d, ptc, _t = _load()
    fz = Featurizer(net, ptc, route_prior={("S1", "R1"): 0.9})
    try:
        PanelAdapter(net, fz)
        raise AssertionError("PanelAdapter should reject a live route prior")
    except ValueError as e:
        assert "no-prior" in str(e)
    PanelAdapter(net, fz, require_no_prior=False)          # 显式放行仅供对照实验


# ---------- (d) 真实窗口: 结构不变量 + 确定性 ----------
def _rollout_encode(net, params, ptc, dwell, dyn, adapter, max_dec=60):
    env = DynamicDerbyEnv(net, params, ptc, dwell)
    obs, info = env.reset(dyn)
    out = []
    n = 0
    while not env.done and n < max_dec:
        dec = env.decision()
        if dec is None:
            break
        out.append((adapter.encode(obs, dec), dec))
        act = next((o for o in dec.options if o.kind == "set"), PanelAction("wait"))
        obs, _r, done, info = env.step(act)
        n += 1
        if done:
            break
    return out


def test_panel_encoding_aligns_with_env_options_on_real_window():
    net, params, dwell, ptc, dyn = _load()
    ad = PanelAdapter(net, _fz(net, ptc))
    seen = _rollout_encode(net, params, ptc, dwell, dyn, ad)
    assert len(seen) >= 10                                   # 真实窗口确实产生了决策
    for enc, dec in seen:
        # 选项与环境 PanelDecision 逐位对齐(动作索引可直接回指 options)
        assert enc["n_options"] == len(dec.options)
        for slot, opt in zip(enc["slots"], dec.options):
            assert slot.kind == opt.kind and slot.train_id == opt.train_id
            if opt.kind == "set":
                assert slot.route_id == opt.route_id
                assert enc["cand_mask"][slot.row, slot.col] == 1.0     # 指向有效候选位
            elif opt.kind == "cancel":
                assert enc["cancel_mask"][slot.row] == 1.0
        assert (enc["x_dec"].shape[0] == enc["x_cand"].shape[0]
                == enc["conf_mask"].shape[0] == len(enc["train_rows"]))
        assert enc["x_cand"].shape[2] == ad.cand_dim
        assert enc["x_cancel"].shape[1] == ad.cancel_dim
        assert enc["x_panel"].shape[1] == ad.train_dim
        assert enc["x_global"].shape == (ad.global_dim,)
        assert np.isfinite(enc["x_cand"]).all() and np.isfinite(enc["x_dec"]).all()


def test_panel_encoding_is_deterministic():
    net, params, dwell, ptc, dyn = _load()
    ad = PanelAdapter(net, _fz(net, ptc))

    def run():
        _n, _p, _d, _c, d2 = _load()
        return _rollout_encode(net, params, ptc, dwell, d2, ad, max_dec=25)
    a, b = run(), run()
    assert len(a) == len(b)
    for (ea, da), (eb, db) in zip(a, b):
        assert [s.__dict__ for s in ea["slots"]] == [s.__dict__ for s in eb["slots"]]
        for k in ("x_dec", "x_conf", "x_cand", "x_cancel", "x_panel", "x_global"):
            assert np.array_equal(ea[k], eb[k])              # 逐位一致


def test_collate_panels_shapes_and_alignment():
    net, params, dwell, ptc, dyn = _load()
    ad = PanelAdapter(net, _fz(net, ptc))
    seen = _rollout_encode(net, params, ptc, dwell, dyn, ad, max_dec=12)
    batch = collate_panels([e for e, _d in seen])
    B = len(seen)
    assert batch["x_dec"].shape[0] == B and batch["opt_kind"].shape[0] == B
    assert batch["opt_mask"].sum() == sum(e["n_options"] for e, _d in seen)
    for b, (enc, _dec) in enumerate(seen):
        assert batch["opt_mask"][b, :enc["n_options"]].all()
        assert not batch["opt_mask"][b, enc["n_options"]:].any()


# ---------- overlap_depth 透传(M10 F6) ----------
def test_overlap_depth_passthrough():
    net, params, dwell, ptc, dyn = _load()
    for depth in (0, 1, 2):
        env = DynamicDerbyEnv(net, params, ptc, dwell, overlap_depth=depth)
        env.reset(_fresh(dyn)[:4])
        assert env.engine.res.overlap_depth == depth
    r = RewardedDynamicEnv(net, params, ptc, dwell, overlap_depth=2)
    r.reset(_fresh(dyn)[:4])
    assert r.engine.res.overlap_depth == 2
    d = DynamicDerbyEnv(net, params, ptc, dwell)                 # 默认仍是 1(原行为不变)
    d.reset(_fresh(dyn)[:4])
    assert d.engine.res.overlap_depth == 1


def test_option_gather_index_arithmetic_is_correct():
    """torch 侧 `flat_set[b, row*Ma+col]` 的索引算术在 numpy 上先验一遍.

    沙盒装不上 torch → `panel_models.py` 无法运行(诚实标注); 但**最易错的一步是这个展平索引**,
    它是纯算术, 可用 numpy 等价验证: 对每个 set 选项, 展平索引取回的值必须等于 x_cand 的该行该列。
    """
    net, params, dwell, ptc, dyn = _load()
    ad = PanelAdapter(net, _fz(net, ptc))
    seen = _rollout_encode(net, params, ptc, dwell, dyn, ad, max_dec=12)
    batch = collate_panels([e for e, _d in seen])
    Ma = batch["x_cand"].shape[2]
    # 用"每个候选位的唯一标记"当探针: flat[row*Ma+col] 必须还原 (row, col)
    B, T = batch["x_cand"].shape[0], batch["x_cand"].shape[1]
    probe = (np.arange(T)[None, :, None] * Ma + np.arange(Ma)[None, None, :]).repeat(B, 0)
    flat = probe.reshape(B, T * Ma)
    idx = batch["opt_row"] * Ma + batch["opt_col"]
    got = np.take_along_axis(flat, idx, axis=1)
    for b, (enc, _dec) in enumerate(seen):
        for i, slot in enumerate(enc["slots"]):
            if slot.kind == "set":
                assert got[b, i] == slot.row * Ma + slot.col
                assert batch["cand_mask"][b, slot.row, slot.col] == 1.0
