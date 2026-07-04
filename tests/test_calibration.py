"""M5 标定测试: 事件驱动配对 + per-TC估参(合成TD, 确定性)."""

from dataclasses import dataclass

import pandas as pd

from derbysim.validation.calibration import (
    pair_route_occupations, calibrate_tc_params, _in_split,
)


@dataclass
class R:
    route_id: str
    tc_sequence: tuple
    signal: int = 0


class FakeNet:
    def __init__(self, routes):
        self.routes = {r.route_id: r for r in routes}


def _row(t, typ, id_, state, hc="T1"):
    return {"time": f"2023-09-01 10:00:{t:02d}", "type": typ, "id": id_,
            "state": state, "trainid_filled": hc, "from_berth": "", "to_berth": ""}


def _synthetic_td():
    rows = [
        _row(0, "Panel_Request", "R", "1"),   # 锁路
        _row(10, "Track", "X", "1"),          # 占X
        _row(18, "Track", "Y", "1"),          # 占Y (advance 8)
        _row(25, "Track", "X", "0"),          # 出清X (occ 15)
        _row(26, "Track", "Z", "1"),          # 占Z (advance 8)
        _row(36, "Track", "Y", "0"),          # 出清Y (occ 18)
        _row(40, "Track", "Z", "0"),          # 出清Z (occ 14)
        _row(45, "Route", "R", "0"),          # 解锁
    ]
    return pd.DataFrame(rows)


def test_in_split():
    assert _in_split(pd.Timestamp("2023-09-15"), ("2023-09-01", "2023-10-31"))
    assert not _in_split(pd.Timestamp("2023-11-15"), ("2023-09-01", "2023-10-31"))
    assert _in_split(pd.Timestamp("2023-11-15"), None)


def test_event_driven_pairing():
    net = FakeNet([R("R", ("X", "Y", "Z"))])
    pairs = list(pair_route_occupations(_synthetic_td(), net))
    assert len(pairs) == 1
    rid, train, occ, rel, seq = pairs[0]
    assert rid == "R" and train == "T1"
    assert set(occ) == {"X", "Y", "Z"}
    # 占用时长: X=15, Y=18, Z=14
    assert round(rel["X"] - occ["X"]) == 15
    assert round(rel["Z"] - occ["Z"]) == 14


def test_calibrate_params_from_pairing():
    net = FakeNet([R("R", ("X", "Y", "Z"))])
    params, n = calibrate_tc_params(_synthetic_td(), net, min_samples=1)
    assert n == 1
    # X: advance=8(占Y-占X), occ=15
    assert params["X"] == (8.0, 15.0)
    assert params["Y"][1] == 18.0
