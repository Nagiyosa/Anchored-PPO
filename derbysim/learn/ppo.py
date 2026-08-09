"""M12b maskable PPO + 选路 KL-to-rule（torch，设备无关）。

- **动作分布**: 只在环境给的 `PanelDecision.options` 上做 softmax(padding 屏蔽) → 结构性安全(D6)
  不由网络负责; 策略输出的是**选项下标**, 不构造动作。
- **SMDP**: 优势用 `smdp.compute_gae` 按**实际推进的仿真时长**折扣; 默认 β=0(γ=1, §10-G)。
- **选路 KL(D59, §10-B)**: 只对齐**选路**子决策, 锚到**规则参照**(schedule 计划路 / lookup 最频路,
  由 `run_episode` 的 ref_builder 逐行算好、随 enc 流入; 见 `routing_kl`)——按 RR"跟计划走、偏离靠
  reward 挣", 取代旧的冻结神经 BC 锚(该锚会把长尾偏离拽回、稀释 warm-start)。`WAIT`/`cancel` 不进
  KL = 时机放开由 RL 自学。**必须按 `valid_routing_rows ∩ ref_valid` 掩行**(全 pad / 无参照行不算)。
- **热启动**(scripts/13): 从 full-panel BC(整个 `PanelOptionPolicy`, D9 最优 gnn)**整网**载入 —— init
  与 KL 锚解耦(init=BC 给能跑的起点避免冷启死锁; 锚=schedule)。

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
    # **选路 KL 锚系数**(D59, 手动扫描: 0/0.01/0.1/1)。锚点是**规则参照**(schedule 计划路 /
    # lookup 最频路, 见 routing_kl + baselines.make_route_reference), 不再是冻结神经 BC ——
    # 按 RR: 把选路拉向计划/常规路, 偏离靠 reward 自己挣。β 先松(0/小)看 EV 能否自恢复(D45 作废,
    # 不预设强锚)。只约束选路; WAIT/cancel 时机放开由 RL 自学。
    kl_beta: float = 0.1
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
    # **时机锚**: 对 p(动作类别) 到某参照构成的 KL。**D59 默认关(0)** —— 主线要 RL **自由学时机**
    # (何时办/是否撤), 硬锚会摁死这个杠杆; 且旧的锚点构成(信号员 2857 步 WAIT 10.2% = `INIT_KIND_MIX`)
    # 是**旧 reactive 引擎**实测, 已作废。若将来新 proactive 引擎上 WAIT/cancel 塌了再考虑: 必须先在
    # 新引擎重测信号员的类别构成, 不得沿用旧数。>0 仅作消融对照。
    kind_anchor_beta: float = 0.0
    # D69 时机锚A(用户批准, 与上面的边际锚不同): **逐状态** KL(π(类别|s) ‖ π_BC冻结(类别|s))。
    # 治 D68.1 的"灾难日梯度把时机头逐状态推向 WAIT 的慢性棘轮" —— 每个状态都有回复力, 但 β 小,
    # 有持续奖励支持的改进仍推得动。选路锚管"走哪条", 本锚管"何时办"(D57 框架的对称一对)。
    bc_anchor_beta: float = 0.0


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


def bc_kind_anchor_kl(kind_log_probs, ref_kind_log_probs, batch) -> torch.Tensor:
    """D69 时机锚A: KL(p_cur(类别|s) ‖ p_BC冻结(类别|s)) —— **逐状态**行为正则(非边际比例锚)。

    两个分布都先在**可用类别**掩码内重归一化(同 `kind_anchor_kl` 的理由: 不可用类别概率恒 0,
    不归一化就会去推不可能的动作)。ref 来自冻结的热启动策略, no-grad。"""
    kind, valid = batch["opt_kind"], batch["opt_mask"] > 0.5
    avail = torch.stack([((kind == j) & valid).any(-1) for j in range(3)], dim=-1)      # [B,3]
    p = kind_log_probs.exp() * avail
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-8)
    q = ref_kind_log_probs.exp() * avail
    q = q / q.sum(-1, keepdim=True).clamp(min=1e-8)
    term = torch.where(avail, p * (torch.log(p.clamp(min=1e-8)) - torch.log(q.clamp(min=1e-8))),
                       torch.zeros_like(p))
    return term.sum(-1).mean()


def routing_kl(policy, batch, eps: float = 0.02) -> torch.Tensor:
    """D59 — 选路 KL **锚到规则参照**(schedule 计划路 / lookup 最频路), 取代旧的冻结神经 BC。

    参照目标由 `run_episode` 的 `ref_builder` 逐行算好、随 `enc` 流进 batch(`ref_col`/`ref_valid`):
    每个有效行给一条**目标进路列**。这里把它做成 ε 平滑的 one-hot 目标分布 q, 取
    **KL(q ‖ π_routing)**(= 目标进路的交叉熵 + 常数) —— 即"把策略往计划/常规路拉", 对 one-hot 目标
    数值稳定(mode-seeking 的 KL(π‖q) 对 one-hot 目标会发散)。

    只在 `ref_valid`(参照命中) ∩ `valid_routing_rows`(该行真有候选) 上平均; WAIT/cancel 不进(§10-B)。
    无参照(batch 不含 ref_col, 或全 0)→ 返回 0(等价关掉 KL)。
    """
    if "ref_col" not in batch:
        return torch.zeros((), device=batch["x_dec"].device)
    x = (batch["x_dec"], batch["x_conf"], batch["conf_mask"], batch["x_cand"], batch["cand_mask"])
    pi_lp = policy.routing_log_probs(*x)                                   # [B,T,Ma] log π(r|t)
    valid = batch["cand_mask"] > 0.5                                       # [B,T,Ma] 有效候选
    n_valid = valid.float().sum(-1, keepdim=True).clamp(min=1.0)          # 每行候选数
    ref_col = batch["ref_col"].long().unsqueeze(-1)                        # [B,T,1] 目标列
    tgt = torch.zeros_like(pi_lp).scatter(-1, ref_col.clamp(min=0), 1.0)  # 目标 one-hot
    tgt = tgt * valid.float()                                              # 目标必落在有效候选上
    q = (1.0 - eps) * tgt + eps * valid.float() / n_valid                 # ε 平滑到有效候选
    q = torch.where(valid, q, torch.zeros_like(q))
    q = q / q.sum(-1, keepdim=True).clamp(min=1e-8)
    term = torch.where(valid, q * (torch.log(q.clamp(min=1e-8)) - pi_lp), torch.zeros_like(q))
    kl_row = term.sum(-1)                                                  # [B,T] KL(q‖π)
    row_mask = (valid_routing_rows(batch["cand_mask"]).float()
                * batch["ref_valid"].float())                             # 参照命中且有候选
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


def ppo_update(policy, optimizer, data: dict, cfg: PPOConfig, device,
               value_only: bool = False, anchor_policy=None) -> dict:
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
             "routing_kl": 0.0, "kind_anchor_kl": 0.0, "bc_anchor_kl": 0.0, "clip_frac": 0.0,
             "n_minibatch": 0, "explained_variance": 0.0,
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
            kl = routing_kl(policy, batch) if cfg.kl_beta > 0 else torch.zeros((), device=device)
            kkl = (kind_anchor_kl(_aux["kind_log_probs"], batch)
                   if (cfg.kind_anchor_beta > 0 and isinstance(_aux, dict)
                       and "kind_log_probs" in _aux) else torch.zeros((), device=device))
            bkl = torch.zeros((), device=device)      # D69 时机锚A: 逐状态 KL 到冻结热启动
            if (cfg.bc_anchor_beta > 0 and anchor_policy is not None
                    and isinstance(_aux, dict) and "kind_log_probs" in _aux):
                with torch.no_grad():
                    _, _, _ref_aux = anchor_policy(batch)
                if isinstance(_ref_aux, dict) and "kind_log_probs" in _ref_aux:
                    bkl = bc_kind_anchor_kl(_aux["kind_log_probs"],
                                            _ref_aux["kind_log_probs"], batch)
            if value_only:
                loss = val_loss                       # critic 预热: 策略不动
            else:
                loss = (pol_loss + cfg.vf_coef * val_loss - cfg.ent_coef * ent
                        + cfg.kl_beta * kl + cfg.kind_anchor_beta * kkl
                        + cfg.bc_anchor_beta * bkl)
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
            stats["routing_kl"] += float(kl.item())
            stats["kind_anchor_kl"] += float(kkl.item())
            stats["bc_anchor_kl"] += float(bkl.item())
            stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean().item())
            stats["n_minibatch"] += 1
            # **护栏放在更新之后**(SB3 口径, 2026-08-01 修): 原先在 step 之前就 break —— 只要第一个
            # minibatch 的 approx_kl 超阈, 整轮**一次更新都不做**且 `n_minibatch=0`, 训练静默空转。
            # 行为策略与当前策略差得远时(如用随机策略采的数据热启动)必然触发。
            if cfg.target_kl is not None and abs(approx_kl) > cfg.target_kl and not value_only:
                stats["kl_early_stops"] = stats.get("kl_early_stops", 0) + 1
                break
    k = max(stats["n_minibatch"], 1)
    for key in ("policy_loss", "value_loss", "entropy", "routing_kl", "kind_anchor_kl",
                "bc_anchor_kl", "clip_frac"):
        stats[key] /= k
    stats["explained_variance"] = explained_variance(preds, data["ret"])
    # 护栏若长期主导, 训练就是在空转 —— 必须能一眼看见(而不是埋在 JSON 里)。
    stats["kl_stop_rate"] = stats["kl_early_stops"] / max(cfg.update_epochs, 1)
    stats["value_only"] = bool(value_only)
    return stats
