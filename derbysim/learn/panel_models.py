"""M12a panel-option pointer + critic (torch, 设备无关).

把 M11 的"单车选路 pointer"推广到**整个信号员面板**的选项集
`{WAIT} ∪ {set(t,r)} ∪ {cancel(t)}` 上逐选项打分 + 一个 critic。

**热启动可迁移性是本模块的第一约束**(docs/M12_DESIGN §10-A/D):
    set 分支**原样复用 M11 `PointerBC`**(同一编码器 + 同一打分头 + 同一输入布局: 一行=一辆待决车),
    故 no-prior BC 的权重可**整体**载入(不只候选特征) —— `load_bc_state_dict()`。
    WAIT / cancel 是 BC 里不存在的子决策 → **另起小头、从头学**(§10-D), 不伪造时机先验。

**KL-to-BC 只对齐选路**(§10-B): `routing_log_probs()` 给出 π(r | 车 t) —— 与 BC 在同一输入上的
分布**逐元素可比**; WAIT/cancel 完全不进 KL 项 = "最小安全偏离"。

**安全不由网络负责**(D6): 选项集来自环境 `PanelDecision.options`(联锁几何 ∩ 目标可达 ∩ 死锁前瞻),
本模块的 mask 只屏蔽 padding。网络永远只给环境**已提供**的选项打分。

设备无关: `cuda if available else cpu`, 不 pin 版本、不硬编码 `.cuda()/.cpu()`。模型控小(hidden=64)。
**沙盒装不上 torch → 本模块沙盒未运行**; 由 `tests/test_panel_smoke.py` 在本地 GPU 验证。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from derbysim.learn.models import ENCODERS, PointerBC, masked_mean

NEG = -1e9          # 掩码用有限大负数(非 -inf): 全屏蔽行 softmax 不出 NaN


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PanelOptionPolicy(nn.Module):
    """面板选项打分(actor) + 状态值(critic), 共享 M11 编码器族(4 种可插拔, D9 消融)."""

    def __init__(self, encoder: str, dec_dim: int, conf_dim: int, cand_dim: int,
                 cancel_dim: int, train_dim: int, global_dim: int,
                 hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder}; choose {list(ENCODERS)}")
        self.encoder_name = encoder
        self.hidden = hidden
        # set 分支 = M11 PointerBC 原件(权重整体可从 BC 载入)
        self.pointer = PointerBC(encoder, dec_dim, conf_dim, cand_dim, hidden, dropout)
        # 面板摘要(全区在读车 + 全局标量) —— critic / WAIT / cancel 的共同上下文
        self.train_enc = nn.Sequential(nn.Linear(train_dim, hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.panel_mix = nn.Sequential(nn.Linear(hidden + global_dim, hidden), nn.ReLU(),
                                       nn.Dropout(dropout))
        # 从头学的两个头(§10-D)
        self.cancel_head = nn.Sequential(nn.Linear(hidden + cancel_dim, hidden), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.wait_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self._init_new_heads()

    def _init_new_heads(self) -> None:
        """WAIT/cancel 头小随机 + WAIT 轻微负偏置 = 温和的"该办就办"先验(§10-D), 不是硬编码策略."""
        for head in (self.cancel_head, self.wait_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.wait_head[-1].bias, -0.5)

    # ---- 热启动 ----
    def load_bc_state_dict(self, bc_state: dict, strict: bool = True) -> None:
        """载入 M11 **no-prior** BC(`PointerBC`)权重 → 编码器 + 选路打分头整体迁移(§10-A)."""
        self.pointer.load_state_dict(bc_state, strict=strict)

    # ---- 前向 ----
    def panel_summary(self, x_panel, panel_mask, x_global):
        h = self.train_enc(x_panel)
        return self.panel_mix(torch.cat([masked_mean(h, panel_mask), x_global], dim=-1))

    def set_logits(self, x_dec, x_conf, conf_mask, x_cand, cand_mask):
        """[B,T,Ma] 选路 logits —— 把 (B,T) 折叠成 M11 的批维, 保证与 BC 数值路径完全一致."""
        B, T = x_dec.shape[0], x_dec.shape[1]
        flat = self.pointer(x_dec.reshape(B * T, -1),
                            x_conf.reshape(B * T, x_conf.shape[2], -1),
                            conf_mask.reshape(B * T, -1),
                            x_cand.reshape(B * T, x_cand.shape[2], -1),
                            cand_mask.reshape(B * T, -1))
        return torch.nan_to_num(flat, neginf=NEG).reshape(B, T, -1)

    def routing_log_probs(self, x_dec, x_conf, conf_mask, x_cand, cand_mask):
        """π(r | 车 t) 的 log 分布 [B,T,Ma] —— KL-to-BC **只**用这个(§10-B)."""
        return torch.log_softmax(self.set_logits(x_dec, x_conf, conf_mask, x_cand, cand_mask), dim=-1)

    def forward(self, batch: dict):
        """batch = `panel.collate_panels()` 的张量版 → (option_logits[B,O], value[B], 诊断项)."""
        s_log = self.set_logits(batch["x_dec"], batch["x_conf"], batch["conf_mask"],
                                batch["x_cand"], batch["cand_mask"])              # [B,T,Ma]
        summary = self.panel_summary(batch["x_panel"], batch["panel_mask"], batch["x_global"])
        B, C = batch["x_cancel"].shape[0], batch["x_cancel"].shape[1]
        c_in = torch.cat([summary.unsqueeze(1).expand(B, C, summary.shape[-1]),
                          batch["x_cancel"]], dim=-1)
        c_log = self.cancel_head(c_in).squeeze(-1)                                # [B,C]
        c_log = c_log.masked_fill(batch["cancel_mask"] < 0.5, NEG)
        w_log = self.wait_head(summary).squeeze(-1)                               # [B]
        value = self.critic(summary).squeeze(-1)                                  # [B]

        flat_set = s_log.reshape(B, -1)                                           # [B,T*Ma]
        Ma = s_log.shape[2]
        idx = (batch["opt_row"] * Ma + batch["opt_col"]).clamp(min=0, max=flat_set.shape[1] - 1)
        gathered_set = torch.gather(flat_set, 1, idx)
        gathered_cancel = torch.gather(c_log, 1, batch["opt_row"].clamp(min=0, max=C - 1))
        kind = batch["opt_kind"]
        logits = torch.where(kind == 1, gathered_set,
                             torch.where(kind == 2, gathered_cancel,
                                         w_log.unsqueeze(1).expand_as(gathered_set)))
        logits = logits.masked_fill(batch["opt_mask"] < 0.5, NEG)
        return logits, value, {"set_logits": s_log, "summary": summary}


def to_tensors(batch: dict, device) -> dict:
    """numpy 批 → torch 批(设备无关): 浮点 float32, opt_* 索引 int64."""
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v)
        out[k] = t.to(device) if k.startswith("opt_") and t.dtype == torch.int64 \
            else t.float().to(device)
    return out
