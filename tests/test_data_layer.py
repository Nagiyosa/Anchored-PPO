"""M2 数据层单元测试(纯逻辑, 不依赖大数据 parquet)."""

import pandas as pd

from derbysim import config
from derbysim.data import region as R
from derbysim.data.cif_parser import parse_cif_time, ScheduleLayer, resolve
from derbysim.data.trust_loader import signed_variation
import datetime as dt


def test_parse_signal():
    assert R.parse_signal("STD5054") == 5054
    assert R.parse_signal("RTD5053G(M)") == 5053
    assert R.parse_signal("TDMV") is None


def test_platform_berth():
    assert R.parse_platform_berth("PLT3") == 3
    assert R.is_region_berth("PLT6")
    assert R.is_region_berth("5045")
    assert R.is_region_berth("X302")
    assert not R.is_region_berth("ZZZZ")


def test_region_membership():
    tc = frozenset({"TDMV"})
    assert R.event_in_region({"type": "Panel_Request", "id": "RTD5045A(M)",
                              "from_berth": "", "to_berth": ""}, tc)
    assert not R.event_in_region({"type": "Panel_Request", "id": "RTD9999A(M)",
                                  "from_berth": "", "to_berth": ""}, tc)
    assert R.event_in_region({"type": "CC", "id": "0", "from_berth": "", "to_berth": "PLT4"}, tc)
    assert R.event_in_region({"type": "Track", "id": "TDMV", "from_berth": "", "to_berth": ""}, tc)


def test_signed_variation():
    var = pd.Series(["19", "13", "0", "5"])
    st = pd.Series(["EARLY", "LATE", "ON TIME", "OFF ROUTE"])
    out = signed_variation(var, st)
    assert out.iloc[0] == -19
    assert out.iloc[1] == 13
    assert out.iloc[2] == 0
    assert pd.isna(out.iloc[3])


def test_parse_cif_time():
    assert parse_cif_time("0751") == 7 * 3600 + 51 * 60
    assert parse_cif_time("0709H") == 7 * 3600 + 9 * 60 + 30
    assert parse_cif_time("0000") is None
    assert parse_cif_time("") is None


def test_stp_resolution():
    P = ScheduleLayer("U1", "P", dt.date(2023, 9, 1), dt.date(2023, 12, 9), "1111100",
                      "1C01", "x", derby_rec="LI", derby_dep=100)
    O = ScheduleLayer("U1", "O", dt.date(2023, 9, 4), dt.date(2023, 9, 4), "1000000",
                      "1C01", "x", derby_rec="LI", derby_dep=200)
    C = ScheduleLayer("U1", "C", dt.date(2023, 9, 11), dt.date(2023, 9, 11), "1000000",
                      "1C01", "x")
    layers = [P, O, C]
    assert resolve(layers, dt.date(2023, 9, 4)).stp == "O"    # overlay wins
    assert resolve(layers, dt.date(2023, 9, 5)).stp == "P"    # tue -> permanent
    assert resolve(layers, dt.date(2023, 9, 11)) is None      # cancelled
    assert resolve(layers, dt.date(2023, 9, 2)) is None       # sat, P not run
