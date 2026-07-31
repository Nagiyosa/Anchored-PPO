"""M12b maskable PPO + KL-to-BC（torch，设备无关）。

- **动作分布**: 只在环境给的 `PanelDecision.options` 上做 softmax(padding 屏蔽) → 结构性安全(D6)
  不由网络负责; 策略输出的是**选项下标**, 不构造动作。
- **SMDP**: 优势用 `smdp.compute_gae` 按**实际推进的仿真时长**折扣; 默认 β=0(γ=1, §10-G)。
- **KL-to-BC(§10-B/F)**: 只对齐**选路**子决策 —— 拿 `routing_log_probs()`(π(r|车 t)) 与冻结的
  no-prior BC 在**同一输入**上的分布算 KL, `WAIT`/`cancel` 完全不进这一项 = "最小安全偏离"。
  方向取 `KL(π ‖ π_BC)`(mode-seeking, 把策略摁在 BC 支撑集内 = 安全先验的本意)。
  系数 `kl_beta` 是**固定档位**(手动扫描), 不是自适应 —— β 本身就是论文的自变量。
  **必须按 `valid_routing_rows` 掩行**: 全 padding 行的分布是均匀的, 混进去等于把 1/Ma 的噪声
  当成"策略偏离 BC"(M12a GPU 冒烟发现)。

**沙盒装不上 torch → 本模块未运行**; 数学部分(GAE/折扣)在 `smdp.py` 已用 numpy 单测,
本模块由 `tests/test_ppo_smoke.py` 在本地 GPU 验证。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from derbysim.learn.panel import collate_panels
from derbysim.learn.panel_models import to_tensors, valid_routing_rows
from derbysim.learn.smdp import prepare_batch  # noqa: F401  (纯 numpy, 从 smdp 再导出)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    update_epochs: int = 4
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # GAE λ。**默认 1.0**(= 蒙特卡洛优势, 无偏), 由实测决定而非惯例:
    #   episode 长 143–642 步、奖励集中在末尾(最后一步就占 15–16%); λ=0.95 的有效回溯长度
    #   只有 1/(1−λ)=20 步 → 第 0 步"看到"的总回报仅 **0.2%**(λ=0.99 也只有 4%)。
    #   即**早期决策几乎拿不到学习信号** —— 实测中表现为曲线全平、KL 恒 ~0.004、
    #   连续两次评估逐位相同。λ=1 让信用完整回溯; 方差由优势归一化 + critic 承担。
    lam: float = 1.0
    half_life_s: float | None = None       # None = γ≡1(不打折, §10-G); M12e 扫 300/900/3600
    kl_beta: float = 0.1                   # KL-to-BC 固定档位(手动扫描: 0/0.01/0.1/1/10)
    normalise_adv: bool = True
    # 纯数值缩放: 整局回报量级 ~10³, 让 critic 去拟合 −2000 这种目标数值上很吃力。
    # 缩放**不改变**最优策略(优势还要归一化), 只让值损失/学习率处在正常量级。
    reward_scale: float = 0.01


def masked_categorical(logits, opt_mask):
    """只在有效选项上的分类分布(padding 已是大负数, 这里再保一道)。"""
    logits = logits.masked_fill(opt_mask < 0.5, float("-inf"))
    return torch.distributions.Categorical(logits=logits)


class TorchPanelDriver:
    """把 torch 策略包成 `rollout.run_episode` 要的 `act_fn(enc) -> (idx, logp, value)`。"""

    def __init__(self, policy, device, generator=None, greedy: bool = False) -> None:
        self.policy = policy
        self.device = device
        self.generator = generator
        self.greedy = greedy

    def __call__(self, enc):
        batch = to_tensors(collate_panels([enc]), self.device)
        self.policy.eval()
        with torch.no_grad():
            logits, value, _aux = self.policy(batch)
            dist = masked_categorical(logits, batch["opt_mask"])
            if self.greedy:
                idx = logits.argmax(dim=-1)
            else:
                probs = dist.probs
                idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            logp = dist.log_prob(idx)
        return int(idx.item()), float(logp.item()), float(value.item())


def routing_kl(policy, bc_ref, batch) -> torch.Tensor:
    """KL(π_routing ‖ π_BC) —— 只在**有效选路行**上平均(§10-B; 全 pad 行必须掩掉)。"""
    x = (batch["x_dec"], batch["x_conf"], batch["conf_mask"], batch["x_cand"], batch["cand_mask"])
    pi_lp = policy.routing_log_probs(*x)                                   # [B,T,Ma]
    B, T = pi_lp.shape[0], pi_lp.shape[1]
    with torch.no_grad():
        ref = bc_ref(batch["x_dec"].reshape(B * T, -1),
                     batch["x_conf"].reshape(B * T, batch["x_conf"].shape[2], -1),
                     batch["conf_mask"].reshape(B * T, -1),
                     batch["x_cand"].reshape(B * T, batch["x_cand"].shape[2], -1),
                     batch["cand_mask"].reshape(B * T, -1))
        bc_lp = torch.log_softmax(torch.nan_to_num(ref, neginf=-1e9), dim=-1).reshape(B, T, -1)
    valid = batch["cand_mask"] > 0.5
    term = torch.where(valid, pi_lp.exp() * (pi_lp - bc_lp), torch.zeros_like(pi_lp))
    kl_row = term.sum(-1)                                                  # [B,T]
    row_mask = valid_routing_rows(batch["cand_mask"]).float()
    return (kl_row * row_mask).sum() / row_mask.sum().clamp(min=1.0)


def explained_variance(pred, target) -> float:
    """EV = 1 − Var(target−pred)/Var(target)。**critic 是否有用的直接诊断**。

    EV≈0 → critic 等于常数预测 ⇒ 优势 ≈ 回报本身。γ=1 下回报全负、G_t 随 t 单调趋近 0,
    于是优势几乎只编码"这一步在 episode 里的位置"(实测与步序相关 0.77–0.96), 早期动作一律被压、
    晚期一律被抬 —— 与动作好坏无关。**必须监控**: 这个数不起来, 训练就是在跟噪声较劲。
    """
    t = np.asarray(target, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    var = t.var()
    return float(1.0 - ((t - p).var() / var)) if var > 1e-12 else 0.0


def ppo_update(policy, bc_ref, optimizer, data: dict, cfg: PPOConfig, device,
               value_only: bool = False) -> dict:
    """一轮 PPO 更新(多 epoch × minibatch)。返回诊断量(损失/KL/熵/裁剪比例/**EV**)。

    `value_only=True` = **critic 预热**: 只训值头, 策略完全不动。用于训练开局 ——
    critic 没学会之前, 优势是"位置噪声", 让策略照它更新只会把策略推坏(实测: λ 放开后
    KL 起来了, 完成率却从 0.567 掉到 0.388、比随机还差)。
    """
    n = len(data["encs"])
    adv = data["adv"].copy()
    if cfg.normalise_adv and n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "kl_to_bc": 0.0, "clip_frac": 0.0, "n_minibatch": 0, "explained_variance": 0.0}
    preds = np.zeros(n, dtype=np.float64)
    policy.train()
    rng = np.random.default_rng(0)
    for _ in range(cfg.update_epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.minibatch):
            sel = order[start:start + cfg.minibatch]
            batch = to_tensors(collate_panels([data["encs"][i] for i in sel]), device)
            a = torch.as_tensor(data["actions"][sel], device=device)
            old_lp = torch.as_tensor(data["old_logp"][sel], device=device)
            mb_adv = torch.as_tensor(adv[sel], device=device)
            mb_ret = torch.as_tensor(data["ret"][sel], device=device)

            logits, value, _aux = policy(batch)
            dist = masked_categorical(logits, batch["opt_mask"])
            logp = dist.log_prob(a)
            ratio = (logp - old_lp).exp()
            unclipped = ratio * mb_adv
            clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * mb_adv
            pol_loss = -torch.min(unclipped, clipped).mean()
            val_loss = ((value - mb_ret) ** 2).mean()
            ent = dist.entropy().mean()
            kl = routing_kl(policy, bc_ref, batch) if cfg.kl_beta > 0 else torch.zeros((), device=device)
            if value_only:
                loss = val_loss                       # critic 预热: 策略不动
            else:
                loss = pol_loss + cfg.vf_coef * val_loss - cfg.ent_coef * ent + cfg.kl_beta * kl
            preds[sel] = value.detach().to("cpu").numpy().astype("float64")

            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(pol_loss.item())
            stats["value_loss"] += float(val_loss.item())
            stats["entropy"] += float(ent.item())
            stats["kl_to_bc"] += float(kl.item())
            stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean().item())
            stats["n_minibatch"] += 1
    k = max(stats["n_minibatch"], 1)
    for key in ("policy_loss", "value_loss", "entropy", "kl_to_bc", "clip_frac"):
        stats[key] /= k
    stats["explained_variance"] = explained_variance(preds, data["ret"])
    stats["value_only"] = bool(value_only)
    return stats


def frozen_bc(encoder: str, dec_dim: int, conf_dim: int, cand_dim: int, hidden: int,
              state_dict: dict | None, device):
    """冻结的 no-prior BC 参照(KL 的另一端)。无 checkpoint 时用随机初始化(仅供冒烟)。"""
    from derbysim.learn.models import PointerBC
    bc = PointerBC(encoder, dec_dim, conf_dim, cand_dim, hidden=hidden)
    if state_dict is not None:
        bc.load_state_dict(state_dict)
    bc.to(device).eval()
    for p in bc.parameters():
        p.requires_grad_(False)
    return bc
