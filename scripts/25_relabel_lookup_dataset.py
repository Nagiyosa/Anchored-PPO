#!/usr/bin/env python3
"""把 BC 数据集的**目标**重标成 lookup(每信号最频路)的选择, 用于蒸馏一个"lookup 策略网络"。

动机(D45→根本变化): PPO 锚到 BC(闭环 232k)被困在坏盆地出不来; 改为锚到 lookup(闭环 109k)。
lookup 是一张表, 要当热启动 + KL 参照需先蒸馏成 PointerBC。做法: 复用 scripts/12 全套训练, 只把
每行的 `chosen_route` 换成 **lookup 在该决策会选的进路**(最频路 top[signal]; 若不在该行合法集里,
退回第一条合法 —— 与 `baselines.make_lookup_policy` 的实际行为一致)。

**不动真 BC**: 只另存 `bc_dataset_lookup_d35.parquet`(scripts/12 会据名给 `_lookup` 后缀,
产物 `bc_<enc>_noprior_nw_lookup_d35.pt`, 与 `bc_*_nw_d35.pt` 不撞)。lookup 表只用 train 段拟合(D43)。

用法: python scripts/25_relabel_lookup_dataset.py
后续: python scripts/12_train_bc.py mlp deepsets settransformer gnn --epochs 30 --no-prior \
         --no-route-weight --dataset bc_dataset_lookup_d35.parquet
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config

SRC = "bc_dataset_all_d35.parquet"
DST = "bc_dataset_lookup_d35.parquet"


def main() -> None:
    path = config.ARTIFACTS / "dataset" / SRC
    df = pd.read_parquet(path)

    # lookup 表: train 段每信号最频进路(与 baselines.lookup_table 一致)
    tr = df[df["split"] == "train"]
    top: dict = {}
    cnt: dict = {}
    for sig, rid in zip(tr["signal"], tr["chosen_route"]):
        cnt.setdefault(sig, Counter())[rid] += 1
    for s, c in cnt.items():
        top[s] = c.most_common(1)[0][0]

    def lookup_pick(row):
        legal = list(row["legal_route_ids"]) if row["legal_route_ids"] is not None else []
        want = top.get(row["signal"])
        if want is not None and want in legal:
            return want
        return legal[0] if legal else row["chosen_route"]

    new_target = df.apply(lookup_pick, axis=1)
    changed = int((new_target != df["chosen_route"]).sum())
    multi = df["n_legal"] >= 2
    changed_multi = int((new_target[multi] != df["chosen_route"][multi]).sum())

    out = df.copy()
    out["chosen_route"] = new_target.values
    dst = config.ARTIFACTS / "dataset" / DST
    out.to_parquet(dst, index=False)

    print(f"[relabel] {SRC} -> {DST}")
    print(f"  rows={len(df)}  signals in lookup={len(top)}")
    print(f"  target changed (=信号员偏离最频路, 现改回最频): {changed} ({changed/len(df):.1%})")
    print(f"    其中多选决策(n_legal>=2): {changed_multi} ({changed_multi/multi.sum():.1%})")
    print(f"  wrote {dst}")
    print("  next: python scripts/12_train_bc.py mlp deepsets settransformer gnn "
          "--epochs 30 --no-prior --no-route-weight --dataset bc_dataset_lookup_d35.parquet")


if __name__ == "__main__":
    main()
