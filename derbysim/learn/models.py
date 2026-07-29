"""M11c BC 模型: 四编码器(独立可插拔) + 共享 pointer 打分头.

设备无关(cuda if available else cpu; 不 pin torch 版本、不硬编码 .cuda()/.cpu()); 标准 nn API。
**沙盒未运行**(装不上 torch); 本地 GPU 用 tests/test_bc_smoke.py / scripts/12 验证。

M11d 修(§11): 加 dropout 正则(防过拟合, 配合 train_bc weight_decay + 早停 + features 候选先验/信号id)。
编码器统一: forward(x_dec[B,Dd], x_conf[B,Mc,Dc], conf_mask[B,Mc]) -> state_emb[B,H]。
"""
from __future__ import annotations
import torch
import torch.nn as nn


def masked_mean(x, mask):
    m = mask.unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class MLPEncoder(nn.Module):
    def __init__(self, dec_dim, conf_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dec_dim + conf_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, x_dec, x_conf, conf_mask):
        return self.net(torch.cat([x_dec, masked_mean(x_conf, conf_mask)], dim=-1))


class DeepSetsEncoder(nn.Module):
    def __init__(self, dec_dim, conf_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(conf_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(dec_dim + hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, x_dec, x_conf, conf_mask):
        h = self.phi(x_conf)
        return self.rho(torch.cat([x_dec, masked_mean(h, conf_mask)], dim=-1))


class SetTransformerEncoder(nn.Module):
    def __init__(self, dec_dim, conf_dim, hidden=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.dec_proj = nn.Linear(dec_dim, hidden)
        self.conf_proj = nn.Linear(conf_dim, hidden)
        self.blocks = nn.ModuleList([nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
                                     for _ in range(layers)])
        self.ffs = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
                                  for _ in range(layers)])
        self.out_dim = hidden

    def forward(self, x_dec, x_conf, conf_mask):
        q = self.dec_proj(x_dec).unsqueeze(1)
        kv = self.conf_proj(x_conf)
        key_pad = conf_mask < 0.5
        key_pad = key_pad & (~key_pad.all(dim=1, keepdim=True))     # 无冲突车时避免全屏蔽 NaN
        h = q
        for att, ff in zip(self.blocks, self.ffs):
            a, _ = att(h, kv, kv, key_padding_mask=key_pad)
            h = ff(h + a)
        return h.squeeze(1)


class GNNEncoder(nn.Module):
    def __init__(self, dec_dim, conf_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(conf_dim, hidden)
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.nbr_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(dec_dim + hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, x_dec, x_conf, conf_mask):
        h = torch.relu(self.proj(x_conf))
        for sw, nw in zip(self.self_w, self.nbr_w):
            agg = masked_mean(h, conf_mask).unsqueeze(1)
            h = self.drop(torch.relu(sw(h) + nw(agg)))
        return self.readout(torch.cat([x_dec, masked_mean(h, conf_mask)], dim=-1))


ENCODERS = {"mlp": MLPEncoder, "deepsets": DeepSetsEncoder,
            "settransformer": SetTransformerEncoder, "gnn": GNNEncoder}


class PointerBC(nn.Module):
    """编码器(状态) + pointer 打分头(逐候选). 输出 legal 候选上的掩码 logits。dropout 正则(§11)。"""
    def __init__(self, encoder: str, dec_dim, conf_dim, cand_dim, hidden=64, dropout=0.2):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder}; choose {list(ENCODERS)}")
        self.encoder = ENCODERS[encoder](dec_dim, conf_dim, hidden, dropout=dropout)
        self.score = nn.Sequential(nn.Linear(self.encoder.out_dim + cand_dim, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_dec, x_conf, conf_mask, x_cand, cand_mask):
        s = self.encoder(x_dec, x_conf, conf_mask)
        B, Ma, _ = x_cand.shape
        s_exp = s.unsqueeze(1).expand(B, Ma, s.shape[-1])
        logits = self.score(torch.cat([s_exp, x_cand], dim=-1)).squeeze(-1)
        return logits.masked_fill(cand_mask < 0.5, float("-inf"))
