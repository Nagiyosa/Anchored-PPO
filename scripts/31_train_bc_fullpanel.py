"""⑦ full-panel BC 训练(D57/D58): 克隆信号员**完整面板决策** `chosen_idx` ∈ {WAIT/SET/CANCEL}。

训 `PanelOptionPolicy`(M12a 面板-选项指针; set 支=M11 PointerBC 原件, 整体可作 RL 热启动)使
`forward()` 输出的选项 log 概率在信号员选的那个选项上最大(NLL)。取代旧"只克隆选路"的 BC。
数据来自 `scripts/30`(narrow proactive 逐 epoch (obs,options,chosen_idx))。held-out 用 D34 split。

**纪律**: 拟合型模型评估用 held-out(D40/D43); 本地只 smoke, 全量训练在服务器(72核+A100)。
device 无关、不 pin 版本。

用法: PYTHONPATH=. python scripts/31_train_bc_fullpanel.py [encoder] [epochs] [--smoke N]
产出: artifacts/dataset/bc_fullpanel_<encoder>.pt + 训练报告(时间戳)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter, collate_panels
from derbysim.learn.panel_models import PanelOptionPolicy, pick_device, to_tensors
from derbysim.sim.env_dynamic import PanelAction, PanelDecision
from derbysim.infra import InfrastructureNetwork


def _dec_from_options(opts_json):
    opts = [PanelAction(k, tid, rid) for k, tid, rid in json.loads(opts_json)]
    return PanelDecision(0.0, opts)


def _encode_row(adapter, row):
    obs = json.loads(row["obs"])
    return adapter.encode(obs, _dec_from_options(row["options"]))


def _batch(adapter, rows):
    encs = [_encode_row(adapter, r) for _, r in rows.iterrows()]
    batch = collate_panels(encs)
    y = torch.as_tensor(rows["chosen_idx"].to_numpy(), dtype=torch.long)
    return batch, y


def _acc_loss(model, batch_t, y, device):
    logits, _v, _d = model(batch_t)                       # [B,O] 已是 log 概率
    y = y.to(device).clamp(min=0, max=logits.shape[1] - 1)
    nll = -logits.gather(1, y.unsqueeze(1)).squeeze(1).mean()
    acc = (logits.argmax(1) == y).float().mean().item()
    return nll, acc


def main(argv):
    encoder = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else "deepsets"
    epochs = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 8
    smoke = None
    if "--smoke" in argv:
        smoke = int(argv[argv.index("--smoke") + 1])

    allp = config.ARTIFACTS / "dataset" / "bc_fullpanel_all.parquet"
    df = pd.read_parquet(allp)
    if smoke:
        df = df.sample(min(smoke, len(df)), random_state=0).reset_index(drop=True)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    ho = df[df["split"] == "holdout"].reset_index(drop=True)
    if len(tr) == 0:                                       # smoke 数据全 holdout → 行切分凑合
        n = int(len(df) * 0.8); tr, ho = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
    print(f"[data] train={len(tr)} holdout={len(ho)} encoder={encoder} epochs={epochs}")

    net = InfrastructureNetwork.load()
    ptc = {}
    import csv
    with open(config.DATA_PLATFORM_TC, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("platform_id") or "").strip().isdigit():
                ptc[r["tc_id"].strip()] = int(r["platform_id"])
    adapter = PanelAdapter(net, Featurizer(net, ptc))

    e0 = _encode_row(adapter, tr.iloc[0])
    dims = (e0["x_dec"].shape[1], e0["x_conf"].shape[2], e0["x_cand"].shape[2],
            e0["x_cancel"].shape[1], e0["x_panel"].shape[1], e0["x_global"].shape[0])
    device = pick_device()
    model = PanelOptionPolicy(encoder, *dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    rng = np.random.default_rng(0)
    bs = 128
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), bs):
            rows = tr.iloc[order[i:i + bs]]
            batch, y = _batch(adapter, rows)
            batch_t = to_tensors(batch, device)
            nll, _ = _acc_loss(model, batch_t, y, device)
            opt.zero_grad(); nll.backward(); opt.step()
            tot += float(nll) * len(rows)
        # holdout 评估
        model.eval()
        with torch.no_grad():
            accs = []
            for i in range(0, len(ho), 256):
                rows = ho.iloc[i:i + 256]
                b, y = _batch(adapter, rows); accs.append(_acc_loss(model, to_tensors(b, device), y, device)[1] * len(rows))
            ho_acc = sum(accs) / max(len(ho), 1)
        hist.append({"epoch": ep + 1, "train_nll": round(tot / len(tr), 4), "holdout_acc": round(ho_acc, 4)})
        print(f"  ep{ep+1}: train_nll={hist[-1]['train_nll']} holdout_acc={hist[-1]['holdout_acc']}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt = config.ARTIFACTS / "dataset" / f"bc_fullpanel_{encoder}.pt"
    torch.save({"state_dict": model.state_dict(), "encoder": encoder, "dims": dims}, ckpt)
    (config.ARTIFACTS_REPORTS / f"bc_fullpanel_train_{stamp}.json").write_text(json.dumps(
        {"encoder": encoder, "epochs": epochs, "n_train": len(tr), "n_holdout": len(ho),
         "smoke": smoke, "history": hist}, indent=2, ensure_ascii=False))
    print(f"-> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
