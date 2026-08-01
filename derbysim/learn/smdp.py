"""M12b 事件驱动 SMDP 的折扣与优势估计（纯 numpy，沙盒可单测）。

**为什么按时间折扣而不是按步数（D24 精化点②）**: `WAIT` 是**时间延展宏动作** —— 一步可能推进
3 秒，也可能推进 8 分钟。按步数折扣会把这两者算成一样重，那是实打实的错。故折扣写成
`γ(Δt) = exp(−β·Δt)`，Δt = 该转移实际推进的**仿真秒数**（set/cancel 瞬时 → Δt≈0 → γ≈1）。

**默认 β = 0（即 γ = 1，不打折）—— §10-G 定案，理由记在设计文档，这里复述要点**:
  ① M10 的延误项是**每步差分**，整局求和恰等于"整局总加权次生延误"(M10_DESIGN §2.1)。
     **该等式只在 γ=1 成立**；γ<1 时训练优化的就不是 M13 汇报的那个数(D23 的价值锚点)。
  ② 短半衰期让"把麻烦往后推"变便宜(20 分钟后 150s 在 5min 半衰期下只值 ~9.4s)，
     正是 M10 用搁浅/未完成 terminal 惩罚在堵的"晾着难车"——一边堵洞一边挖洞。
方差由 **GAE 的 λ** 承担(那才是干这活的旋钮)。半衰期作 M12e 灵敏度扫描(5min/15min/60min/∞)。
"""
from __future__ import annotations

import math

import numpy as np


def beta_from_half_life(half_life_s: float | None) -> float:
    """半衰期(仿真秒) → 折扣率 β。`None`/`<=0` → **β=0 即 γ≡1(不打折)**。"""
    if half_life_s is None or half_life_s <= 0:
        return 0.0
    return math.log(2.0) / float(half_life_s)


def smdp_gamma(dt, beta: float):
    """γ(Δt) = exp(−β·Δt)。β=0 → 恒为 1；Δt=0(瞬时 set/cancel) → 1。"""
    dt = np.asarray(dt, dtype=np.float64)
    if beta == 0.0:
        return np.ones_like(dt)
    return np.exp(-beta * np.maximum(dt, 0.0))


def compute_gae(rewards, values, dts, dones, last_value: float = 0.0,
                half_life_s: float | None = None, lam: float = 0.95):
    """SMDP 版 GAE(λ)。返回 (advantages, returns)，均为 float64 一维数组。

    δ_t = r_t + γ(Δt_t)·V(s_{t+1})·(1−done_t) − V(s_t)
    A_t = δ_t + γ(Δt_t)·λ·(1−done_t)·A_{t+1}
    β=0 时退化为**无折扣**有限期 GAE —— 正是我们要的(训练目标 ≡ 汇报指标)。
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    g = smdp_gamma(dts, beta_from_half_life(half_life_s))
    n = len(r)
    if not (len(v) == len(d) == len(g) == n):
        raise ValueError("rewards/values/dts/dones must have equal length")
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in range(n - 1, -1, -1):
        nv = last_value if t == n - 1 else v[t + 1]
        nonterminal = 1.0 - d[t]
        delta = r[t] + g[t] * nv * nonterminal - v[t]
        running = delta + g[t] * lam * nonterminal * running
        adv[t] = running
    return adv, adv + v


def discounted_return(rewards, dts, half_life_s: float | None = None) -> float:
    """整局折扣回报(诊断/单测用)。β=0 时 = 奖励直接求和 = 我们真正要优化的量。"""
    r = np.asarray(rewards, dtype=np.float64)
    g = smdp_gamma(dts, beta_from_half_life(half_life_s))
    out, disc = 0.0, 1.0
    for t in range(len(r)):
        out += disc * r[t]
        disc *= g[t]
    return float(out)


class RunningMoments:
    """回报的运行均值/方差(Welford)——**取代写死的 `reward_scale`**。

    为什么需要(2026-08-01 实测, 运营日 episode):
      单步奖励在**一局之内**就横跨 5 个数量级 —— |r| 中位 0.00 / p99 826 / 最大 8968,
      末尾那一记终局罚独占全局 |r| 的 **29%**；整局回报从信号员的 **+40** 到"有路就办"的
      **−31966**。写死的 `reward_scale=0.01` 是在 3h 窗口上标的(那时"有路就办" ≈ −613),
      换成运营日就差 50 倍: 值目标 ~10², `vf_coef·(V−R)²` 初值 ~10⁴–10⁵, 而策略损失是 O(1)
      → 梯度被值损失整个吃掉(正是第 3 轮 "critic 无用、优势与步序相关 0.77–0.96" 的同一病根)。
      更根本的是: **策略一变强, 回报量级就跟着变**(−3 万 → −100), 任何常数都只对一个阶段成立。

    做法(PopArt 的简化版): critic 预测**归一化值** v̂; 真值 v = v̂·σ + μ。GAE 全程在**原始单位**
    上算(所以 `TorchPanelDriver` 必须还原), 只有喂给值损失的目标被归一化。优势另有自己的归一化。
    **诚实记下的缺陷**: 不做 PopArt 的末层权重补偿, 故 (μ,σ) 更新时旧的归一化目标会整体平移,
    critic 需重新适应。因每轮只更新一次**运行**统计量(非逐批), 漂移很慢; 且留了
    `normalise_returns=False` 作消融对照。EV 对仿射变换不变, 诊断口径不受影响。
    """

    __slots__ = ("mean", "var", "count")

    def __init__(self, mean: float = 0.0, var: float = 1.0, count: float = 1e-4) -> None:
        self.mean, self.var, self.count = float(mean), float(var), float(count)

    @property
    def std(self) -> float:
        return float(max(math.sqrt(max(self.var, 0.0)), 1e-6))

    def update(self, x) -> "RunningMoments":
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return self
        bm, bv, bn = float(x.mean()), float(x.var()), float(x.size)
        delta, tot = bm - self.mean, self.count + bn
        self.mean += delta * bn / tot
        m_a, m_b = self.var * self.count, bv * bn
        self.var = (m_a + m_b + delta * delta * self.count * bn / tot) / tot
        self.count = tot
        return self

    def normalise(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def denormalise(self, x):
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> "RunningMoments":
        self.mean, self.var, self.count = float(d["mean"]), float(d["var"]), float(d["count"])
        return self


def prepare_batch(transitions, cfg, last_value: float = 0.0, ret_norm=None) -> dict:
    """转移列表 → numpy 批 + 优势/回报(SMDP GAE)。**纯 numpy** —— 放在这里以便沙盒单测。

    `transitions[i].value` 必须是**原始单位**的 V(还原过的, 见 `RunningMoments`),
    GAE 因此整段在原始单位上算。`ret_norm` 给定时: 先用本批 returns 更新运行统计量,
    再把 `ret` 归一化后交给值损失; `ret_raw` 保留原值供诊断/日志。
    `cfg.reward_scale` 退为**可选的额外常数**(默认 1.0, 只在 `normalise_returns=False` 的
    消融里用)。`cfg.lam` 默认 1.0, 理由见该字段注释(λ=0.95 时第 0 步只看到 0.2% 回报)。
    """
    sc = getattr(cfg, "reward_scale", 1.0)
    adv, ret = compute_gae([t.reward * sc for t in transitions], [t.value for t in transitions],
                           [t.dt for t in transitions], [t.done for t in transitions],
                           last_value=last_value, half_life_s=cfg.half_life_s, lam=cfg.lam)
    out = {
        "encs": [t.enc for t in transitions],
        "actions": np.asarray([t.option_idx for t in transitions], np.int64),
        "old_logp": np.asarray([t.logp for t in transitions], np.float32),
        "adv": adv.astype(np.float32),
        "ret_raw": ret.astype(np.float64),
    }
    if ret_norm is not None:
        ret_norm.update(ret)
        out["ret"] = ret_norm.normalise(ret).astype(np.float32)
    else:
        out["ret"] = ret.astype(np.float32)
    return out
