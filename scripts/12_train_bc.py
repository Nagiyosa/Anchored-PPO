"""M11c/d BC 训练 + 决策匹配率评估 CLI(本地 GPU 运行; 设备无关).

用法: python scripts/12_train_bc.py [mlp deepsets settransformer gnn] --epochs 15
产出: artifacts/reports/bc_<encoder>.json (holdout 匹配率: 宏平均/长尾/优先级 + 基线)。
"""
from __future__ import annotations

import os
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 使 derbysim 可导入(从任意 cwd 运行)

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.features import Featurizer, featurize_frame, fit_route_prior
from derbysim.learn.train_bc import train_bc, predict_records, get_device, route_inverse_freq_weights
from derbysim.learn.evaluate import match_metrics, fit_most_frequent, predict_most_frequent, expected_random_acc


def _ptc():
    d = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                d[row["tc_id"].strip()] = int(row["platform_id"])
    return d


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("encoders", nargs="*", default=["mlp"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--no-prior", action="store_true",
                   help="不用候选先验特征(热启动版: 保留偏离信号、长尾>随机, M11d 决定A)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--no-save-ckpt", action="store_true",
                   help="不保存权重(默认保存 .pt 供 M12 热启动/KL 参照)")
    args = p.parse_args(argv)
    encoders = args.encoders or ["mlp"]
    epochs = args.epochs
    use_prior = not args.no_prior
    net = InfrastructureNetwork.load(); ptc = _ptc()
    # 默认用 **D31 重建版**(断链切分 + CIF 按趟匹配); 要跑旧口径对照设 BC_DATASET=bc_dataset_all.parquet
    _ds = os.environ.get("BC_DATASET", "bc_dataset_all_d31.parquet")
    print(f"[data] {_ds}")
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / _ds)
    tr_df = df[df["split"] == "train"]; ho_df = df[df["split"] == "holdout"]
    prior = fit_route_prior([{"signal": r["signal"], "legal_route_ids": r["legal_route_ids"],
                              "chosen_route": r["chosen_route"], "n_legal": r["n_legal"]}
                             for r in tr_df.to_dict("records")]) if use_prior else None  # 只训练集拟合(无泄露)
    fz = Featurizer(net, ptc, route_prior=prior)                        # None → 不含先验(热启动版)
    print(f"use_prior={use_prior}")
    dims = (fz.dec_dim, fz.conf_dim, fz.cand_dim)
    tr = featurize_frame(tr_df, fz)
    ho = featurize_frame(ho_df, fz)
    device = get_device()
    print(f"device={device} dims={dims} train={len(tr)} holdout={len(ho)}")

    # 基线(D10)
    from collections import Counter
    rc = Counter(s["chosen_route"] for s in ho if s["n_legal"] > 1)
    ho_recs_base = [{"signal": s["signal"], "service_class": s["service_class"],
                     "chosen_route": s["chosen_route"], "n_legal": s["n_legal"],
                     "legal_route_ids": s["legal_route_ids"]} for s in ho]
    fitted = fit_most_frequent([{"signal": s["signal"], "chosen_route": s["chosen_route"],
                                 "n_legal": s["n_legal"]} for s in tr])
    for r in ho_recs_base:
        r["pred_route"] = predict_most_frequent(fitted, r)
    baseline = {"most_frequent_macro": match_metrics(ho_recs_base, dict(rc))["macro_signal_acc"],
                "random_legal_expected": expected_random_acc(ho_recs_base)}
    print("baselines:", baseline)

    weights = route_inverse_freq_weights(tr)
    for enc in encoders:
        model, hist = train_bc(enc, tr, dims, val_samples=ho, epochs=epochs,
                               route_weight=weights, seed=0, device=device, hidden=args.hidden)
        recs = predict_records(model, ho, dims, device)
        m = match_metrics(recs, dict(rc))
        rep = {"encoder": enc, "epochs": epochs, "dims": dims,
               "holdout_macro_signal_acc": m["macro_signal_acc"],
               "holdout_overall_acc": m["overall_acc"],
               "holdout_longtail_acc": m.get("longtail_acc"),
               "per_priority_acc": m["per_priority_acc"],
               "n_multi_holdout": m["n_multi"], "baselines": baseline, "history": hist}
        suffix = "" if use_prior else "_noprior"
        outp = config.ARTIFACTS / "reports" / f"bc_{enc}{suffix}.json"
        outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        if not args.no_save_ckpt:
            # **权重 + 元信息**: M12 要用它做热启动和 KL 参照, 元信息用于在 scripts/13 里
            # 硬校验(编码器/维度/hidden 必须一致; use_prior 必须 False —— M12_DESIGN §10-A③)。
            import torch
            ck = config.ARTIFACTS / "reports" / f"bc_{enc}{suffix}.pt"
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "encoder": enc, "dims": list(dims), "hidden": args.hidden,
                        "use_prior": bool(use_prior), "epochs": epochs,
                        "holdout_macro": m["macro_signal_acc"],
                        "holdout_longtail": m.get("longtail_acc")}, ck)
            print(f"[{enc}] saved checkpoint -> {ck}")
        print(f"[{enc}] holdout macro-signal acc={m['macro_signal_acc']:.3f} "
              f"(overall={m['overall_acc']:.3f}, longtail={m.get('longtail_acc')}); wrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1:])
