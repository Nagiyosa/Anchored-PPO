"""M3 占用模型测试: 纯函数数学 + 冻结黄金样例回归(真实 traversal)."""

from derbysim.sim.occupation import occupancy_intervals, TCOccupation


def test_pure_chaining_math():
    # A: occ100→110; advance5→B occ105→117; advance6→C occ111→119
    params = {"A": (5, 10), "B": (6, 12), "C": (7, 8)}
    out = occupancy_intervals(["A", "B", "C"], 100.0, params)
    assert out == [
        TCOccupation("A", 100.0, 110.0),
        TCOccupation("B", 105.0, 117.0),
        TCOccupation("C", 111.0, 119.0),
    ]


def test_defaults_when_param_missing():
    out = occupancy_intervals(["X", "Y"], 0.0, {}, default_advance=8, default_occ=18)
    assert out[0] == TCOccupation("X", 0.0, 18.0)
    assert out[1] == TCOccupation("Y", 8.0, 26.0)


def test_empty_sequence():
    assert occupancy_intervals([], 0.0, {}) == []


# ── 冻结黄金样例: 真实车次 1V66 走 RDW5315A(M) (由 M3 标定脚本抽取) ──
_GOLDEN = {
    "tc_sequence": ["TYVJ", "TYVK", "TYVL", "TYVM", "TYVN", "TYVR"],
    "obs_occ": [0.0, 11.0, 18.0, 24.0, 28.0, 33.0],
    "params": {
        "TYVJ": (12.0, 20.0), "TYVK": (9.0, 15.0), "TYVL": (6.0, 12.0),
        "TYVM": (5.0, 10.0), "TYVN": (7.0, 12.0), "TYVR": (17.0, 22.0),
    },
}


def test_golden_sample_regression():
    seq = _GOLDEN["tc_sequence"]
    out = occupancy_intervals(seq, 0.0, _GOLDEN["params"])
    pred = [o.t_occupied for o in out]
    obs = _GOLDEN["obs_occ"]
    # 顺序完全一致
    assert [o.tc for o in out] == seq
    errs = [abs(p - a) for p, a in zip(pred, obs)]
    mae = sum(errs) / len(errs)
    assert mae < 8.0, f"golden MAE {mae:.1f}s 超阈"
    assert max(errs) < 12.0, f"golden 最大误差 {max(errs):.1f}s 超阈"
