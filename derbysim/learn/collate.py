"""M11c collate(纯 numpy, 无 torch, 可单测): featurized 样本 → padding 批数组 + mask.

批: x_dec[B,Dd] · x_conf[B,Mc,Dc]+conf_mask[B,Mc] · x_cand[B,Ma,Da]+cand_mask[B,Ma] · y[B].
候选/冲突车变长 → 右侧 0 padding, mask 标有效(1)/pad(0)。y=被选候选下标(在有效候选内)。
torch 侧只需 torch.as_tensor(...).to(device) —— 设备无关, 不与本地 GPU torch 冲突。
"""
from __future__ import annotations
import numpy as np


def collate(samples: list, dec_dim: int, conf_dim: int, cand_dim: int) -> dict:
    B = len(samples)
    Mc = max((s["x_conf"].shape[0] for s in samples), default=0)
    Mc = max(Mc, 1)                                   # 至少 1 列(全 pad)避免空维
    Ma = max(s["x_cand"].shape[0] for s in samples)
    x_dec = np.zeros((B, dec_dim), np.float32)
    x_conf = np.zeros((B, Mc, conf_dim), np.float32)
    conf_mask = np.zeros((B, Mc), np.float32)
    x_cand = np.zeros((B, Ma, cand_dim), np.float32)
    cand_mask = np.zeros((B, Ma), np.float32)
    y = np.zeros((B,), np.int64)
    for i, s in enumerate(samples):
        x_dec[i] = s["x_dec"]
        nc = s["x_conf"].shape[0]
        if nc:
            x_conf[i, :nc] = s["x_conf"]; conf_mask[i, :nc] = 1.0
        na = s["x_cand"].shape[0]
        x_cand[i, :na] = s["x_cand"]; cand_mask[i, :na] = 1.0
        y[i] = s["y"]
    return {"x_dec": x_dec, "x_conf": x_conf, "conf_mask": conf_mask,
            "x_cand": x_cand, "cand_mask": cand_mask, "y": y}
