"""M11c/d BC 训练 + 决策匹配率评估 CLI(本地 GPU 运行; 设备无关).

用法: python scripts/12_train_bc.py [mlp deepsets settransformer gnn] --epochs 15
产出: artifacts/reports/bc_<encoder>.json (holdout 匹配率: 宏平均/长尾/优先级 + 基线)。
"""
from __future__ import annotations
import csv, json, sys
import numpy as np
import pandas as pd

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.learn.features import Featurizer, featurize_frame
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
    encoders = [a for a in argv if not a.startswith("--")] or ["mlp"]
    epochs = 15
    if "--epochs" in argv:
        epochs = int(argv[argv.index("--epochs") + 1])
    net = InfrastructureNetwork.load(); ptc = _ptc()
    fz = Featurizer(net, ptc)
    dims = (fz.dec_dim, fz.conf_dim, fz.cand_dim)
    df = pd.read_parquet(config.ARTIFACTS / "dataset" / "bc_dataset_all.parquet")
    tr = featurize_frame(df[df["split"] == "train"], fz)
    ho = featurize_frame(df[df["split"] == "holdout"], fz)
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
                               route_weight=weights, seed=0, device=device)
        recs = predict_records(model, ho, dims, device)
        m = match_metrics(recs, dict(rc))
        rep = {"encoder": enc, "epochs": epochs, "dims": dims,
               "holdout_macro_signal_acc": m["macro_signal_acc"],
               "holdout_overall_acc": m["overall_acc"],
               "holdout_longtail_acc": m.get("longtail_acc"),
               "per_priority_acc": m["per_priority_acc"],
               "n_multi_holdout": m["n_multi"], "baselines": baseline, "history": hist}
        outp = config.ARTIFACTS / "reports" / f"bc_{enc}.json"
        outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"[{enc}] holdout macro-signal acc={m['macro_signal_acc']:.3f} "
              f"(overall={m['overall_acc']:.3f}, longtail={m.get('longtail_acc')}); wrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1:])
