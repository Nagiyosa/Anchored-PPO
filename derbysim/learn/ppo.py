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
from derbysim.learn.smdp import RunningMoments, prepare_batch  # noqa: F401  (纯 numpy, 再导出)


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
    # **回报标准化**(2026-08-01 取代写死的 reward_scale, 理由见 `smdp.RunningMoments`):
    #   运营日 episode 的单步 |r| 中位 0.00 / p99 826 / 最大 8968, 末步独占 29%;
    #   整局回报信号员 +40 vs "有路就办" −31966。写死的常数只对一个场景+一个策略水平成立。
    #   critic 改预测**归一化值**, GAE 仍在原始单位上算(driver 负责还原)。
    #   `False` = 回退到固定常数(消融对照)。
    normalise_returns: bool = True
    # 仅在 `normalise_returns=False` 时起作用的固定缩放(旧行为 = 0.01, 3h 窗口上标的)。
    reward_scale: float = 1.0
    # PPO 标准护栏: 一次更新内策略走太远就提前停 epoch 循环。实测三轮训练都在解冻后约 20 轮见顶
    # 然后塌(KL-to-BC 0.028→0.044, 熵 0.888→0.769) —— 更新步长失控是主要嫌疑。
    target_kl: float | None = 0.02
    # **时机锚(2026-08-02)**: 对 p(动作类别) 到信号员实测构成的 KL。§10-B 的设计只约束**选路**,
    # `kind_head`(何时办/是否撤)完全没有锚 —— 当初假设"奖励会管住它", 实测**不会**: 撤销能立刻
    # 释放预留、降低其它车的 held_time(即时正奖励), 而代价(完成率下降)是分摊到全局、延后到终局的
    # 一笔总账; 即使 γ=1/λ=1 让信用完整回溯, 那笔账摊到 ~350 步上单步优势里几乎看不见。
    # 实测后果: cancel 占比 0%→58%, completion 从 0.306 退回 0.241(真实撤销率只有 10.44%)。
    # 与 KL-to-BC 完全同构, 锚点是**数据**(信号员 2857 步实测 WAIT 10.2%/set 89.8%/cancel 0%,
    # 撤销按真实 10.44% 给非零下限 = `INIT_KIND_MIX`)。0 = 关掉(消融对照)。
    kind_anchor_beta: float = 0.05


def masked_categorical(logits, opt_mask):
    """只在有效选项上的分类分布(padding 已是大负数, 这里再保一道)。"""
    logits = logits.masked_fill(opt_mask < 0.5, float("-inf"))
    return torch.distributions.Categorical(logits=logits)


class TorchPanelDriver:
    """把 torch 策略包成 `rollout.run_episode` 要的 `act_fn(enc) -> (idx, logp, value)`。

    `ret_norm` 给定时, critic 输出的是**归一化值**, 这里**必须还原成原始单位**再交给 GAE ——
    否则 `δ = r + γV' − V` 里 r 是原始单位而 V 是归一化单位, 两边量纲不同, 优势直接是垃圾。
    """

    def __init__(self, policy, device, generator=None, greedy: bool = False,
                 ret_norm=None) -> None:
        # **D32/方案 B**: rollout 期一律 batch=1, GPU 上每次前向的 kernel 启动往返实测 ≈4.06 ms,
        # 占 rollout 时间的 **73%**(纯 CPU 侧仿真+编码只要 1.5 ms/步); 而网络只有 hidden=64 的
        # deepsets, 真算是微秒级 —— GPU 全程在等指令, 换 A100 也一样。故 driver 不接受"必须用
        # 传入 device"的约束: 调用方给什么就用什么, `scripts/13` 在 rollout 期传 CPU、更新期用
        # GPU。副作用同样重要: 每次训练变成纯 CPU 单核任务 → 实验矩阵可以 N 路并发而不抢 GPU。
        self.policy = policy
        self.device = device
        self.generator = generator
        self.greedy = greedy
        self.ret_norm = ret_norm

    def __call__(self, enc):
        batch = to_tensors(collate_panels([enc]), self.device)
        self.policy.eval()
        with torch.no_grad():
            logits, value, aux = self.policy(batch)
            dist = masked_categorical(logits, batch["opt_mask"])
            if self.greedy:
                idx = greedy_option(logits, batch, aux)
            else:
                probs = dist.probs
                idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            logp = dist.log_prob(idx)
        v = float(value.item())
        if self.ret_norm is not None:
            v = float(self.ret_norm.denormalise(v))
        return int(idx.item()), float(logp.item()), v


def greedy_option(logits, batch, aux):
    """因子化策略的**贪心动作 = 先选类别、再在类别内选** —— 不是在扁平 logits 上取 argmax。

    **为什么不能扁平 argmax(2026-08-01 实测抓到)**: 分层之后
    `log p(选项) = log p(类别) + log p(选项 | 类别)`。WAIT 类只有 **1 个**选项, 概率全给它;
    set 类的 0.80 要摊到 N 条候选进路上, 每条只有 0.80/N。于是 **N > 5.3 时任何单条进路的概率都
    低于 WAIT**, 扁平 argmax 必然选 WAIT —— 而 WAIT 又让车堆积、下个决策点候选更多, **自我放大**
    成 WAIT 100%(实测初始体检: WAIT 100% / completion 0.028, 与"永远WAIT"红队策略同级)。
    这**不是策略偏好, 是"把混合分布的众数当成决策"这个操作本身的偏差** —— 采样(训练时)完全没有
    这个问题, 只有贪心评估会中招。正确的贪心是**按类别的边际取最大, 再在该类别内取最大**。
    """
    k_lp = aux.get("kind_log_probs") if isinstance(aux, dict) else None
    if k_lp is None:
        return logits.argmax(dim=-1)
    kind, valid = batch["opt_kind"], batch["opt_mask"] > 0.5
    best_kind = k_lp.argmax(dim=-1, keepdim=True)                 # [B,1] 不可用类别已是 NEG
    sel = (kind == best_kind) & valid
    sel = torch.where(sel.any(-1, keepdim=True), sel, valid)      # 兜底: 该类别恰好无有效选项
    return logits.masked_fill(~sel, float("-inf")).argmax(dim=-1)


def kind_anchor_kl(kind_log_probs, batch) -> torch.Tensor:
    """KL(p(类别) ‖ 信号员实测构成) —— **按可用性重新归一化后**再比。

    某类别在该决策点没有选项时它的概率恒为 0; 若不把锚也按同样的掩码重新归一化, 锚就会去推
    一个**不可能的动作**, 那是纯噪声。
    """
    from derbysim.learn.panel_models import INIT_KIND_MIX
    kind, valid = batch["opt_kind"], batch["opt_mask"] > 0.5
    avail = torch.stack([((kind == j) & valid).any(-1) for j in range(3)], dim=-1)      # [B,3]
    a = torch.as_tensor(INIT_KIND_MIX, dtype=kind_log_probs.dtype,
                        device=kind_log_probs.device).unsqueeze(0) * avail
    a = a / a.sum(-1, keepdim=True).clamp(min=1e-8)
    p = kind_log_probs.exp()
    term = torch.where(avail, p * (kind_log_probs - torch.log(a.clamp(min=1e-8))),
                       torch.zeros_like(p))
    return term.sum(-1).mean()


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
             "kl_to_bc": 0.0, "kind_anchor_kl": 0.0, "clip_frac": 0.0, "n_minibatch": 0, "explained_variance": 0.0,
             "approx_kl": 0.0, "kl_early_stops": 0}
    preds = np.zeros(n, dtype=np.float64)
    # **必须 eval() 而不是 train() —— dropout 会污染重要性比(2026-08-01 实测)。**
    # `old_logp` 是 rollout 时采的, 而 driver 里调的是 `policy.eval()`(dropout 关)。若这里开 train(),
    # `logp` 带 dropout 噪声 ⇒ `ratio = exp(logp - old_lp)` 比较的是**两个不同的网络**, 与参数是否
    # 更新无关。后果实测: `approx_kl` 在**任何学习发生之前**就有 0.023–0.081, 高于 `target_kl=0.02`
    # ⇒ 每个 epoch 的第一个 minibatch 就 break。1M 步那轮 240 迭代共触发 **895 次早停 ≈ 每轮 3.7 次**
    # (update_epochs=4) ⇒ 每轮只做约 4 个 minibatch 更新, 而本该做 (4096/64)×4 = 256 个 —— **实际
    # 只执行了 1.5% 的更新**, 且这 1.5% 用的还是被噪声污染的 ratio。曲线因此从头到尾平坦
    # (it9 0.284 → it239 0.274, 最佳 0.299 全在噪声带内)。
    # 模型无 BatchNorm, eval() 只关 dropout, 梯度照常回传。RL 策略网络不用 dropout 是通行做法;
    # BC 那边是监督学习, dropout 保留不变。
    policy.eval()
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
            kkl = (kind_anchor_kl(_aux["kind_log_probs"], batch)
                   if (cfg.kind_anchor_beta > 0 and isinstance(_aux, dict)
                       and "kind_log_probs" in _aux) else torch.zeros((), device=device))
            if value_only:
                loss = val_loss                       # critic 预热: 策略不动
            else:
                loss = (pol_loss + cfg.vf_coef * val_loss - cfg.ent_coef * ent
                        + cfg.kl_beta * kl + cfg.kind_anchor_beta * kkl)
            preds[sel] = value.detach().to("cpu").numpy().astype("float64")

            with torch.no_grad():                     # 本次更新把策略推了多远(PPO 标准诊断)
                approx_kl = float((old_lp - logp).mean().item())
            stats["approx_kl"] = max(stats.get("approx_kl", 0.0), abs(approx_kl))
            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(pol_loss.item())
            stats["value_loss"] += float(val_loss.item())
            stats["entropy"] += float(ent.item())
            stats["kl_to_bc"] += float(kl.item())
            stats["kind_anchor_kl"] += float(kkl.item())
            stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean().item())
            stats["n_minibatch"] += 1
            # **护栏放在更新之后**(SB3 口径, 2026-08-01 修): 原先在 step 之前就 break —— 只要第一个
            # minibatch 的 approx_kl 超阈, 整轮**一次更新都不做**且 `n_minibatch=0`, 训练静默空转。
            # 行为策略与当前策略差得远时(如用随机策略采的数据热启动)必然触发。
            if cfg.target_kl is not None and abs(approx_kl) > cfg.target_kl and not value_only:
                stats["kl_early_stops"] = stats.get("kl_early_stops", 0) + 1
                break
    k = max(stats["n_minibatch"], 1)
    for key in ("policy_loss", "value_loss", "entropy", "kl_to_bc", "kind_anchor_kl", "clip_frac"):
        stats[key] /= k
    stats["explained_variance"] = explained_variance(preds, data["ret"])
    # 护栏若长期主导, 训练就是在空转 —— 必须能一眼看见(而不是埋在 JSON 里)。
    stats["kl_stop_rate"] = stats["kl_early_stops"] / max(cfg.update_epochs, 1)
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
