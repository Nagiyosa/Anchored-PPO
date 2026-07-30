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
