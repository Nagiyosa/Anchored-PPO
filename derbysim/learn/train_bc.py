"""M11c BC 训练/推理(torch, 设备无关). **沙盒未运行**(装不上 torch); 本地 GPU 用 scripts/12 / smoke 验证.

device = cuda if available else cpu(可 override); 不硬编码 .cuda()/.cpu(); seed 可复现。
掩码交叉熵(非法候选 logits=-inf → softmax 0); 长尾可选按 route 逆频加权。
"""
from __future__ import annotations
import numpy as np

from derbysim.learn.collate import collate
from derbysim.learn.models import PointerBC


def get_device(pref=None):
    import torch
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensors(samples, dims, device):
    import torch
    b = collate(samples, *dims)
    return {k: torch.as_tensor(v).to(device) for k, v in b.items()}


def route_inverse_freq_weights(samples, cap=10.0) -> dict:
    from collections import Counter
    c = Counter(s["chosen_route"] for s in samples if s["n_legal"] > 1)
    if not c:
        return {}
    med = np.median(list(c.values()))
    return {r: float(min(cap, med / n)) for r, n in c.items()}


def train_bc(encoder, train_samples, dims, val_samples=None, epochs=10, batch_size=256,
             lr=1e-3, route_weight=None, seed=0, device=None, hidden=64, log=print):
    import torch
    import torch.nn.functional as F
    device = device or get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    model = PointerBC(encoder, *dims, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    train_samples = [s for s in train_samples if s["n_legal"] > 1]   # 只训真多选
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_samples))
        tot, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            s = [train_samples[j] for j in order[i:i + batch_size]]
            t = _to_tensors(s, dims, device)
            logits = model(t["x_dec"], t["x_conf"], t["conf_mask"], t["x_cand"], t["cand_mask"])
            if route_weight:
                w = torch.as_tensor([route_weight.get(x["chosen_route"], 1.0) for x in s],
                                    dtype=torch.float32).to(device)
                loss = (F.cross_entropy(logits, t["y"], reduction="none") * w).mean()
            else:
                loss = F.cross_entropy(logits, t["y"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(s); n += len(s)
        rec = {"epoch": ep, "train_loss": round(tot / max(n, 1), 4)}
        if val_samples is not None:
            rec["val_macro_acc"] = round(_macro_acc(model, val_samples, dims, device), 4)
        hist.append(rec); log(rec)
    return model, hist


def predict_records(model, samples, dims, device, batch_size=512):
    """→ records(含 pred_route) 供 evaluate.match_metrics。"""
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            s = samples[i:i + batch_size]
            t = _to_tensors(s, dims, device)
            logits = model(t["x_dec"], t["x_conf"], t["conf_mask"], t["x_cand"], t["cand_mask"])
            pred = logits.argmax(1).cpu().numpy()
            for k, samp in enumerate(s):
                legal = samp["legal_route_ids"]
                pi = int(pred[k]); pi = pi if pi < len(legal) else 0
                out.append({"signal": samp["signal"], "service_class": samp["service_class"],
                            "chosen_route": samp["chosen_route"], "pred_route": legal[pi],
                            "n_legal": samp["n_legal"], "legal_route_ids": legal})
    return out


def _macro_acc(model, samples, dims, device):
    from derbysim.learn.evaluate import match_metrics
    return match_metrics(predict_records(model, samples, dims, device))["macro_signal_acc"]
