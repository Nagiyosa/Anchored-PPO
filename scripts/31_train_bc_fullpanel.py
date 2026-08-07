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
from derbysim.learn.features import Featurizer, fit_route_prior
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
    va = df[df["split"] == "val"].reset_index(drop=True)   # D43: 选型/早停**只用 val**
    # **test 段不加载、不碰**(D43: test 全程不碰); 终局评估用 scripts/29/32 在 test 上另跑一次。
    if len(tr) == 0 or len(va) == 0:                       # smoke(仅部分天)兜底: 行切分, 明确非正式协议
        n = int(len(df) * 0.8); tr, va = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
        print("[smoke] this partial dataset lacks train/val split; row-split fallback (NOT the D34 protocol)")
    print(f"[data] train={len(tr)} val={len(va)} (test isolated) encoder={encoder} epochs={epochs}")

    net = InfrastructureNetwork.load()
    ptc = {}
    import csv
    with open(config.DATA_PLATFORM_TC, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("platform_id") or "").strip().isdigit():
                ptc[r["tc_id"].strip()] = int(r["platform_id"])
    # route_prior 特征只从 **train 段**拟合(held-out 纪律 D40; 不碰 holdout/test)。用新数据集的
    # signal + 该决策车的合法 set 路(从 options 取)。缺 signal 的行(entry/WAIT)不进先验。
    prior_recs = []
    for _, r in tr[tr["chosen_kind"] == "set"].iterrows():
        if r.get("signal") is None or (isinstance(r.get("signal"), float) and r["signal"] != r["signal"]):
            continue
        legal = [rid for k, tid, rid in json.loads(r["options"])
                 if k == "set" and tid == r["chosen_train"]]
        if len(legal) > 1:
            prior_recs.append({"signal": r["signal"], "legal_route_ids": legal,
                               "chosen_route": r["chosen_route"], "n_legal": len(legal)})
    route_prior = fit_route_prior(prior_recs) if prior_recs else {}
    print(f"[route_prior] fitted on {len(prior_recs)} multi-choice SET rows -> {len(route_prior)} (sig,route) keys")
    adapter = PanelAdapter(net, Featurizer(net, ptc, route_prior=route_prior))

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
        # val 评估(选型/早停口径; test 不碰)
        model.eval()
        with torch.no_grad():
            accs = []
            for i in range(0, len(va), 256):
                rows = va.iloc[i:i + 256]
                b, y = _batch(adapter, rows); accs.append(_acc_loss(model, to_tensors(b, device), y, device)[1] * len(rows))
            val_acc = sum(accs) / max(len(va), 1)
        hist.append({"epoch": ep + 1, "train_nll": round(tot / len(tr), 4), "val_acc": round(val_acc, 4)})
        print(f"  ep{ep+1}: train_nll={hist[-1]['train_nll']} val_acc={hist[-1]['val_acc']}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt = config.ARTIFACTS / "dataset" / f"bc_fullpanel_{encoder}.pt"
    # 存 route_prior: 评估(scripts/29)必须用**同一份** route_prior 建 featurizer, 否则特征对不上、闭环虚低。
    torch.save({"state_dict": model.state_dict(), "encoder": encoder, "dims": dims,
                "route_prior": route_prior}, ckpt)
    (config.ARTIFACTS_REPORTS / f"bc_fullpanel_train_{stamp}.json").write_text(json.dumps(
        {"encoder": encoder, "epochs": epochs, "n_train": len(tr), "n_val": len(va),
         "smoke": smoke, "history": hist}, indent=2, ensure_ascii=False))
    print(f"-> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
