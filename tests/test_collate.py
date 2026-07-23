"""M11c collate 单测(纯 numpy): 形状/mask/标签正确, 变长 padding."""
from __future__ import annotations
import csv
import numpy as np
import pandas as pd
from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.features import Featurizer, featurize_frame
from derbysim.learn.collate import collate


def _load():
    net = InfrastructureNetwork.load(); ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    return net, ptc


def test_collate_shapes_and_masks():
    net, ptc = _load(); fz = Featurizer(net, ptc)
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all.parquet")
    samples = featurize_frame(df[df["n_legal"] > 1].head(64), fz)
    b = collate(samples, fz.dec_dim, fz.conf_dim, fz.cand_dim)
    B = len(samples); Ma = max(s["n_legal"] for s in samples)
    assert b["x_dec"].shape == (B, fz.dec_dim)
    assert b["x_cand"].shape == (B, Ma, fz.cand_dim)
    assert b["cand_mask"].shape == (B, Ma)
    # cand_mask 有效数 == n_legal
    for i, s in enumerate(samples):
        assert b["cand_mask"][i].sum() == s["n_legal"]
        assert b["cand_mask"][i, s["n_legal"]:].sum() == 0        # 尾部是 pad
    # y 落在有效候选内
    assert all(0 <= b["y"][i] < samples[i]["n_legal"] for i in range(B))
    # pad 行是 0
    for i, s in enumerate(samples):
        assert np.all(b["x_cand"][i, s["n_legal"]:] == 0)


def test_collate_conf_mask_matches_count():
    net, ptc = _load(); fz = Featurizer(net, ptc)
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all.parquet")
    samples = featurize_frame(df[df["n_legal"] > 1].head(48), fz)
    b = collate(samples, fz.dec_dim, fz.conf_dim, fz.cand_dim)
    for i, s in enumerate(samples):
        assert b["conf_mask"][i].sum() == s["x_conf"].shape[0]
