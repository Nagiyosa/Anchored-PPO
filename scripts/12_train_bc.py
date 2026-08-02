"""M11c/d BC 训练 + 决策匹配率评估 CLI(本地 GPU 运行; 设备无关).

用法: python scripts/12_train_bc.py [mlp deepsets settransformer gnn] --epochs 15
产出: artifacts/reports/bc_<encoder>.json (holdout 匹配率: 宏平均/长尾/优先级 + 基线)。
"""
from __future__ import annotations

import os
import time
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
    p.add_argument("--no-route-weight", action="store_true",
                   help="关掉长尾逆频加权。实测(D31 训练段, cap=10): 加权把**头部进路的梯度占比从 "
                        "97.4% 压到 47.3%**, 11 条 ≤5 次的罕见进路(占样本 0.21%)独吞 13.8% 的权重, "
                        "而长尾在 holdout 上只有 52 条(0.56%) —— 为 0.56% 的数据牺牲 97.4% 的拟合")
    p.add_argument("--early-stop-on", default="overall", choices=["overall", "macro"],
                   help="早停判据(默认 overall)。macro 被 4 个 n=1 的信号绑架, 逐 epoch 在 0.63-0.76 "
                        "间跳 —— 用它早停等于用噪声叫停。**报告仍报 macro**, 口径不变")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="从 train 段**末尾**按日期切出的 val 比例(早停/选最佳用); 与 RL 侧 "
                        "scenarios.operating_days 的 val_frac 同约定。holdout 全程不参与选择")
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
    # **三段划分(2026-08-01 修, 方法学硬伤)**: 早停/选最佳 checkpoint 此前用的就是 `holdout`,
    # 报出来的 `holdout macro` 恰好等于 `best_val_macro` —— **在测试集上做模型选择**, 数字系统性
    # 偏乐观(逐 epoch 的 val macro 在 0.63–0.76 之间震荡, 取其最大值本身就是一次挑选)。
    # 改法与 RL 侧的划分约定一致(`scenarios.operating_days`: val 取 train 段**末尾** 15%):
    #   train段 = 9-10 月 → 前 85% 日期训练 / 后 15% 日期做 val(早停+选最佳)
    #   holdout = 11-12 月, **训练全程不碰**, 只在最后报一次。
    full_tr = df[df["split"] == "train"]; ho_df = df[df["split"] == "holdout"]
    _dates = sorted(full_tr["date"].astype(str).unique())
    _n_val = max(1, int(round(len(_dates) * args.val_frac)))
    _val_dates = set(_dates[len(_dates) - _n_val:])
    tr_df = full_tr[~full_tr["date"].astype(str).isin(_val_dates)]
    va_df = full_tr[full_tr["date"].astype(str).isin(_val_dates)]
    print(f"[split] train {len(tr_df)} 决策 / {len(_dates)-_n_val} 天 | "
          f"val {len(va_df)} / {_n_val} 天({min(_val_dates)}~{max(_val_dates)}) | "
          f"holdout {len(ho_df)} —— **holdout 只在最后报一次, 不参与早停**")
    # 先验只在**缩减后的 train** 上拟合 → val 上的先验特征是样本外的, 早停才诚实。
    prior = fit_route_prior([{"signal": r["signal"], "legal_route_ids": r["legal_route_ids"],
                              "chosen_route": r["chosen_route"], "n_legal": r["n_legal"]}
                             for r in tr_df.to_dict("records")]) if use_prior else None
    fz = Featurizer(net, ptc, route_prior=prior)                        # None → 不含先验(热启动版)
    print(f"use_prior={use_prior}")
    dims = (fz.dec_dim, fz.conf_dim, fz.cand_dim)
    tr = featurize_frame(tr_df, fz)
    va = featurize_frame(va_df, fz)
    ho = featurize_frame(ho_df, fz)
    device = get_device()
    print(f"device={device} dims={dims} train={len(tr)} val={len(va)} holdout={len(ho)}")

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

    weights = None if args.no_route_weight else route_inverse_freq_weights(tr)
    print(f"[train] 长尾逆频加权: {'关' if weights is None else '开'} | 早停判据: val {args.early_stop_on}")
    for enc in encoders:
        model, hist = train_bc(enc, tr, dims, val_samples=va, epochs=epochs,   # **val, 不是 holdout**
                               route_weight=weights, seed=0, device=device, hidden=args.hidden,
                               early_stop_metric=args.early_stop_on)
        recs = predict_records(model, ho, dims, device)
        m = match_metrics(recs, dict(rc))
        va_m = match_metrics(predict_records(model, va, dims, device),
                             dict(Counter(s["chosen_route"] for s in va if s["n_legal"] > 1)))
        rep = {"encoder": enc, "epochs": epochs, "dims": dims,
               "holdout_macro_signal_acc": m["macro_signal_acc"],
               "holdout_overall_acc": m["overall_acc"],
               "holdout_longtail_acc": m.get("longtail_acc"),
               "per_priority_acc": m["per_priority_acc"],
               "val_macro_signal_acc": va_m["macro_signal_acc"],     # 早停用的那个集(供对照乐观偏差)
               "val_overall_acc": va_m["overall_acc"],
               "n_multi_holdout": m["n_multi"], "baselines": baseline,
               "dataset": _ds, "val_frac": args.val_frac,
               "route_weight": not args.no_route_weight, "early_stop_on": args.early_stop_on,
               "history": hist}
        # **每一版都留下来**(用户 2026-07-31): 带时间戳另存一份, 同名文件仍写(供 scripts/13 直接引用)
        suffix = ("" if use_prior else "_noprior") + ("" if not args.no_route_weight else "_nw")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (config.ARTIFACTS / "reports" / f"bc_{enc}{suffix}_{stamp}.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
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
        print(f"[{enc}] **holdout(未参与早停)** macro={m['macro_signal_acc']:.3f} "
              f"overall={m['overall_acc']:.3f} longtail={m.get('longtail_acc')} | "
              f"val macro={va_m['macro_signal_acc']:.3f}(早停用) | "
              f"最频基线 {baseline['most_frequent_macro']:.3f}; wrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1:])
