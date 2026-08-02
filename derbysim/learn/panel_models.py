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

# 动作类别的**初始**边际分布(kind 头的偏置由它反解)。
# 依据: 信号员回放在 6 个真实运营日 2857 步上的实测构成 = WAIT 10.2% / set 89.8% / cancel 0.0%
# (回放策略永不撤销, 故 cancel 实测为 0)。撤销不能初始化成 0 概率 —— 它必须可达才谈得上探索,
# 故按"真实撤销率 10.44%(92534 请求 / 9664 撤销)"给一个小而非零的份额。
INIT_KIND_MIX = (0.15, 0.80, 0.05)      # (wait, set, cancel)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid_routing_rows(cand_mask):
    """[B,T] bool: 该行(=一辆待决车)是否真有候选进路.

    **全 padding 行的选路分布是均匀的**(该行 logits 全为 NEG → log_softmax 出均匀), 无意义。
    `opt_row` 永远不会指向这种行, 故不影响动作分布; 但 **KL-to-BC 以及任何按行的选路损失
    必须用本掩码筛掉它们**(§10-B), 否则会把 1/Ma 的噪声当成"策略偏离 BC"算进目标。
    """
    return cand_mask.sum(-1) > 0


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
        # 从头学的两个头(§10-D): cancel 的**组内**打分 + **动作类别**头
        self.cancel_head = nn.Sequential(nn.Linear(hidden + cancel_dim, hidden), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 1))
        # kind_head 取代原来的 wait_head(§17 修): 输出 3 个类别 logit (wait/set/cancel)。
        # 原设计把 WAIT/cancel/set 的 logit 丢进**同一个 softmax**, 而 set 那支的 logit 尺度来自
        # 只含进路的 BC softmax, 与两个随机初始化的新头**不可比** → 热启动被稀释成近似均匀。
        self.kind_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self._init_new_heads()

    def _init_new_heads(self) -> None:
        """新头初始化: cancel 组内打分小随机; **kind 头末层权重置零、偏置 = log(目标构成)**。

        末层权重置零 ⇒ 初始时类别分布**与状态无关地**恰等于 `INIT_KIND_MIX`(见该常量的依据),
        而不是由随机权重决定。这条是 §17 的直接修复: 上一轮冒烟里 argmax 的动作构成是
        **WAIT 0% / set 42–45% / cancel 55–58%**(真实撤销率只有 10.44%), 策略从第一步起就在刷
        cancel, 而所有单点指标(KL≈0 / EV 0.97 / 维度对上 / 权重载入成功)**全都正常**。
        """
        for m in self.cancel_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        for m in self.kind_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        last = self.kind_head[-1]
        nn.init.zeros_(last.weight)                       # 权重置零 → 输出只由偏置决定
        mix = torch.tensor(INIT_KIND_MIX, dtype=last.bias.dtype)
        with torch.no_grad():
            last.bias.copy_(torch.log(mix / mix.sum()))

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
        value = self.critic(summary).squeeze(-1)                                  # [B]

        flat_set = s_log.reshape(B, -1)                                           # [B,T*Ma]
        Ma = s_log.shape[2]
        idx = (batch["opt_row"] * Ma + batch["opt_col"]).clamp(min=0, max=flat_set.shape[1] - 1)
        gathered_set = torch.gather(flat_set, 1, idx)
        gathered_cancel = torch.gather(c_log, 1, batch["opt_row"].clamp(min=0, max=C - 1))

        # ---- **因子化动作分布(§17 根治)**: log p(选项) = log p(类别) + log p(选项 | 类别) ----
        # 为什么必须分层: BC pointer 的 logit 尺度**只在"路与路之间"有意义**(它是在只含进路的
        # softmax 上训出来的); 把它和两个新头的 logit 丢进同一个 softmax, 联合分布就没有标定,
        # 热启动带来的选路知识会被稀释。分层之后:
        #   · 选路条件分布 p(选项 | set) 完全由 pointer 决定 → **热启动 100% 生效**, 且这正是
        #     KL-to-BC 唯一作用的对象(§10-B), 两者从此在同一个数学对象上;
        #   · 何时办/是否撤 由 kind 头单独学, 尺度自洽, 初始构成可标定(INIT_KIND_MIX)。
        kind = batch["opt_kind"]                                                  # 0=wait 1=set 2=cancel
        valid = batch["opt_mask"] > 0.5
        m_wait, m_set, m_cancel = (kind == 0) & valid, (kind == 1) & valid, (kind == 2) & valid

        def _within(scores, m):
            """组内 log-softmax(组外置 NEG); 空组返回全 0, 因其类别 logit 会被屏蔽掉。"""
            x = scores.masked_fill(~m, NEG)
            return torch.where(m, torch.log_softmax(x, dim=-1), torch.zeros_like(x))

        lp_in = (_within(torch.zeros_like(gathered_set), m_wait)      # WAIT 组只有一个选项 → log1=0
                 + _within(gathered_set, m_set)
                 + _within(gathered_cancel, m_cancel))

        avail = torch.stack([m_wait.any(-1), m_set.any(-1), m_cancel.any(-1)], dim=-1)   # [B,3]
        k_log = self.kind_head(summary).masked_fill(~avail, NEG)                 # [B,3]
        k_lp = torch.log_softmax(k_log, dim=-1)
        lp_kind = torch.gather(k_lp, 1, kind.clamp(min=0, max=2))                # [B,O]

        logits = (lp_kind + lp_in).masked_fill(~valid, NEG)                      # 已是 log 概率
        return logits, value, {"set_logits": s_log, "summary": summary, "kind_log_probs": k_lp}


def to_tensors(batch: dict, device) -> dict:
    """numpy 批 → torch 批(设备无关): 浮点 float32, opt_* 索引 int64."""
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v)
        out[k] = t.to(device) if k.startswith("opt_") and t.dtype == torch.int64 \
            else t.float().to(device)
    return out
