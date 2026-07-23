"""M11d 匹配率指标单测(纯 python): 宏平均/长尾/优先级/基线 —— 合成数据验真值."""
from __future__ import annotations
import numpy as np
from derbysim.learn.evaluate import (match_metrics, fit_most_frequent, predict_most_frequent,
                                     expected_random_acc)


def _rec(sig, sc, chosen, pred, legal):
    return {"signal": sig, "service_class": sc, "chosen_route": chosen,
            "pred_route": pred, "n_legal": len(legal), "legal_route_ids": legal}


def test_macro_vs_micro_and_priority():
    # 信号A: 100 条全对; 信号B: 2 条全错 → 微平均 100/102≈0.98, 宏平均 (1.0+0.0)/2=0.5
    recs = []
    for i in range(100):
        recs.append(_rec("A", "1", "r1", "r1", ["r1", "r2"]))
    for i in range(2):
        recs.append(_rec("B", "2", "r3", "r9", ["r3", "r4"]))
    m = match_metrics(recs)
    assert abs(m["overall_acc"] - 100 / 102) < 1e-9
    assert abs(m["macro_signal_acc"] - 0.5) < 1e-9          # 宏平均防头部虚高
    assert m["per_priority_acc"]["1"] == 1.0 and m["per_priority_acc"]["2"] == 0.0


def test_singlechoice_excluded_from_headline():
    recs = [_rec("A", "1", "r1", "rX", ["r1"])] * 50 + [_rec("B", "1", "r2", "r2", ["r2", "r3"])]
    m = match_metrics(recs)
    assert m["n_multi"] == 1                                 # 单选(n_legal=1)不计入
    assert m["macro_signal_acc"] == 1.0


def test_longtail_subset():
    recs = [_rec("A", "1", "common", "common", ["common", "rare"]) for _ in range(10)]
    recs += [_rec("A", "1", "rare", "common", ["common", "rare"])]   # 稀有进路被判错
    rc = {"common": 10, "rare": 1}
    m = match_metrics(recs, route_counts=rc, tail_le=5)
    assert m["n_longtail"] == 1 and m["longtail_acc"] == 0.0


def test_most_frequent_baseline():
    train = [_rec("A", "1", "r1", "r1", ["r1", "r2"]) for _ in range(9)]
    train += [_rec("A", "1", "r2", "r2", ["r1", "r2"])]
    fitted = fit_most_frequent(train)
    assert fitted["A"] == "r1"                               # A 最频=r1
    assert predict_most_frequent(fitted, _rec("A", "1", "r2", "?", ["r1", "r2"])) == "r1"
    # 未知信号 → 回退首个合法
    assert predict_most_frequent(fitted, _rec("Z", "1", "x", "?", ["rp", "rq"])) == "rp"


def test_expected_random_acc():
    recs = [_rec("A", "1", "r", "r", ["a", "b"]), _rec("B", "1", "r", "r", ["a", "b", "c", "d"])]
    assert abs(expected_random_acc(recs) - (0.5 + 0.25) / 2) < 1e-9
