"""IQL(Implicit Q-Learning, Kostrikov 2021)—— 离线 RL,给面板-选项策略用。

**为什么选 IQL**(D64):我们的专家(信号员)近乎完美、数据干净,但 online PPO 几周只降不升。
IQL 从**固定的离线转移数据**里学,**训练时从不查"数据里没出现过的动作"的 Q** → 天然回避 offline RL
的外推错误。三步:
  · **V 用 expectile 回归**逼近"数据内动作的 Q"(τ→1 越像 max, 但不碰 OOD 动作);
  · **Q 用 Bellman**: Q(s,a) ← r + γ(1−done)·V(s');
  · **策略用 advantage-weighted 回归(AWR)**: 照抄数据 option, 权重 exp((Q−V)/β) 截断。

**actor 复用 `PanelOptionPolicy`** → 评估口径与 PPO **完全一致**(同一 `TorchPanelDriver` 贪心)。
critic(Q/V)用**独立**的编码器,不与 actor 纠缠(标准 IQL: actor≠critic)。
只在数据 option 上算 Q → 回避外推。局限(诚实): 数据里**没有"从坏状态恢复"的样本**,故 IQL 学的是
**避免**进坏状态(≈好放行), 不是恢复; 恢复靠后续 AWAC 的在线相(D64)。

设备无关(torch); 沙盒无 torch → 本模块由服务器 smoke 验证。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from derbysim.learn.models import masked_mean
from derbysim.learn.panel_models import PanelOptionPolicy
from derbysim.learn.ppo import masked_categorical


class IQL(nn.Module):
    """actor = PanelOptionPolicy(评估同 PPO); + 独立 critic 编码器 + V 头 + Q(s,a_data) 头。"""

    def __init__(self, encoder: str, dec_dim: int, conf_dim: int, cand_dim: int,
                 cancel_dim: int, train_dim: int, global_dim: int,
                 hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.actor = PanelOptionPolicy(encoder, dec_dim, conf_dim, cand_dim, cancel_dim,
                                       train_dim, global_dim, hidden, dropout)
        self.cand_dim = cand_dim
        # 独立 critic 摘要编码器(与 actor 的 panel_summary 同构, 但**不共享参数**)
        self.c_train_enc = nn.Sequential(nn.Linear(train_dim, hidden), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.c_mix = nn.Sequential(nn.Linear(hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.v_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Q(s, a_data): [critic 摘要 ++ 取到的 option 特征(kind onehot 3 + set 候选特征)]
        self.q_head = nn.Sequential(nn.Linear(hidden + 3 + cand_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))

    # ---- critic 侧 ----
    def _csummary(self, b: dict):
        h = self.c_train_enc(b["x_panel"])
        return self.c_mix(torch.cat([masked_mean(h, b["panel_mask"]), b["x_global"]], dim=-1))

    def _opt_feat(self, b: dict, a):
        """取被选 option 的特征 [B, 3+cand_dim] = kind onehot ++ (set 的候选特征, 否则 0)."""
        B = a.shape[0]
        ar = torch.arange(B, device=a.device)
        kind = b["opt_kind"][ar, a]                                  # [B] 0=wait 1=set 2=cancel
        row = b["opt_row"][ar, a].clamp(min=0)
        col = b["opt_col"][ar, a].clamp(min=0)
        kind_oh = torch.zeros(B, 3, device=a.device)
        kind_oh[ar, kind.clamp(0, 2)] = 1.0
        cand = b["x_cand"][ar, row, col]                             # [B, cand_dim](set 的候选特征)
        is_set = (kind == 1).float().unsqueeze(-1)
        return torch.cat([kind_oh, cand * is_set], dim=-1)

    def v(self, b: dict):
        return self.v_head(self._csummary(b)).squeeze(-1)           # [B]

    def q(self, b: dict, a):
        return self.q_head(torch.cat([self._csummary(b), self._opt_feat(b, a)], dim=-1)).squeeze(-1)  # [B]

    # ---- actor 侧(评估/AWR 用)----
    def logp(self, b: dict, a):
        logits, _v, _aux = self.actor(b)
        return masked_categorical(logits, b["opt_mask"]).log_prob(a)

    def critic_parameters(self):
        for m in (self.c_train_enc, self.c_mix, self.v_head, self.q_head):
            yield from m.parameters()


def expectile_loss(u, tau: float):
    """expectile 回归损失: u>0 权 τ, u<0 权 1−τ(τ→1 → 逼近上分位=数据内 max)."""
    w = torch.where(u > 0, torch.as_tensor(tau, device=u.device), torch.as_tensor(1 - tau, device=u.device))
    return (w * u.pow(2)).mean()


def iql_update(model: IQL, opt_critic, opt_actor, batch: dict, next_batch: dict,
               actions, rewards, dones, gamma: float = 1.0, tau: float = 0.7,
               beta: float = 3.0, w_max: float = 100.0) -> dict:
    """一步 IQL 更新。batch/next_batch = to_tensors(collate_panels(...)); actions/rewards/dones [B] tensor。"""
    # ---- critic: Q(Bellman) + V(expectile) ----
    with torch.no_grad():
        v_next = model.v(next_batch)
        q_target = rewards + gamma * (1.0 - dones) * v_next
    q_sa = model.q(batch, actions)
    loss_q = (q_sa - q_target).pow(2).mean()
    v_s = model.v(batch)
    loss_v = expectile_loss(q_sa.detach() - v_s, tau)
    opt_critic.zero_grad()
    (loss_q + loss_v).backward()
    opt_critic.step()

    # ---- actor: AWR(照抄数据 option, 优势加权)----
    with torch.no_grad():
        adv = model.q(batch, actions) - model.v(batch)
        w = torch.clamp(torch.exp(adv / beta), max=w_max)
    logp = model.logp(batch, actions)
    loss_pi = -(w * logp).mean()
    opt_actor.zero_grad()
    loss_pi.backward()
    opt_actor.step()

    return {"loss_q": float(loss_q.item()), "loss_v": float(loss_v.item()),
            "loss_pi": float(loss_pi.item()), "adv_mean": float(adv.mean().item()),
            "v_mean": float(v_s.mean().detach().item()), "w_mean": float(w.mean().item())}
