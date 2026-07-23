"""M11b 特征化测试: 维度一致 / 标签有效 / delay 裁剪 / 候选可区分(可达性) / 确定性."""
from __future__ import annotations
import csv, json
import numpy as np
import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.features import Featurizer, featurize_frame


def _load():
    net = InfrastructureNetwork.load()
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    return net, ptc


def _sample_multi(n=200):
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all.parquet")
    return df[df["n_legal"] > 1].head(n)


def test_dims_consistent_and_label_valid():
    net, ptc = _load()
    fz = Featurizer(net, ptc)
    samples = featurize_frame(_sample_multi(300), fz)
    assert len(samples) > 100
    for s in samples:
        assert s["x_dec"].shape == (fz.dec_dim,)
        assert s["x_cand"].shape == (s["n_legal"], fz.cand_dim)
        assert s["x_conf"].shape[1] == fz.conf_dim
        assert 0 <= s["y"] < s["n_legal"]            # 被选动作在 legal 内


def test_delay_clip_applied():
    net, ptc = _load()
    fz = Featurizer(net, ptc, delay_clip=120.0)
    rid = next(iter(net.routes))
    obs = {"tod_hour": 8, "deciding": {"delay_min": -1131.0, "service_class": "1",
           "booked_platform": 4, "pos_idx": 3}, "conflict_trains": [],
           "candidate_tc_occupancy": {}}
    rec = {"obs": json.dumps(obs), "legal_route_ids": [rid], "chosen_route": rid,
           "signal": 5045, "service_class": "1"}
    s = fz.featurize_record(rec)
    assert s["x_dec"][0] == -1.0                      # -1131 裁剪到 -120 → 归一 -1.0


def test_candidates_are_discriminable():
    # 多选决策的候选特征不应全同(可达性/占用能区分) —— pointer 才有意义
    net, ptc = _load()
    fz = Featurizer(net, ptc)
    samples = featurize_frame(_sample_multi(300), fz)
    multi = [s for s in samples if s["n_legal"] >= 3]
    assert multi
    n_distinct = 0
    for s in multi:
        rows = {tuple(np.round(r, 4)) for r in s["x_cand"]}
        if len(rows) > 1:
            n_distinct += 1
    assert n_distinct / len(multi) > 0.8             # 绝大多数决策候选可区分


def test_reachability_features_present():
    net, ptc = _load()
    fz = Featurizer(net, ptc)
    samples = featurize_frame(_sample_multi(300), fz)
    # 至少部分候选通向某出区组(group one-hot 命中) 或可达站台
    any_group = any(s["x_cand"][:, :fz.n_groups].sum() > 0 for s in samples)
    any_plat = any(s["x_cand"][:, fz.n_groups:fz.n_groups + 6].sum() > 0 for s in samples)
    assert any_group and any_plat


def test_featurize_deterministic():
    net, ptc = _load()
    df = _sample_multi(120)
    a = featurize_frame(df, Featurizer(net, ptc))
    b = featurize_frame(df, Featurizer(net, ptc))
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.array_equal(sa["x_dec"], sb["x_dec"])
        assert np.array_equal(sa["x_cand"], sb["x_cand"])
        assert sa["y"] == sb["y"]
