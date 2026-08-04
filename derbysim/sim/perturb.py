"""M10c 扰动: 数据/真实惯例锚定(docs/M10_DESIGN.md §4/§12). 复审定案(2026-07-23):

**冲突来源 = 真实拥堵窗口**(真实车/真实时刻), **延误注入 = 拥堵下的随机增广**。
    - 复审 R2: 延误注入在常态稀疏窗口惰性(Δ=0, 无 overlap 可扰), 只在已拥堵时起效(可 de-bunch)。
    - 复审 R3: 合成加车(clone route_chain)**与预留模型不兼容**——预留按 route_id 记身份, clone 复用源车
      route_id → 并发同 route 时 route_train 被覆盖 → 破坏无碰撞不变式。**故放弃合成加车**(除非重构预留
      按 (train,route) 记身份=动 M9 已验证核心)。改用真实拥堵窗口造冲突。

真实数据(德比 TRUST): LATE 50%/晚点重尾。primary 延误注入见下(α=0.36 Palmqvist 出处, §4.1)。

**D28(2026-08-01): 合成延误注入已退出主线。** `DelaySampler` / `inject_primary_delays` /
`PRIMARY_FRACTION(α)` **不再用于训练、验证或测试** —— 全部改用真实历史。保留代码仅供**可选的**
鲁棒性消融, 使用时必须在论文里同时声明 α 未在德比数据上验证。
撤下的理由(逐条):
  ① 入区时刻(TD 观测)**本身已含当天真实的 primary + secondary 晚点**; 我们要问的反事实是
     "同样这批车、同样时刻到达区界, 换一套进路决策, 区内延误会不会更少" —— 这是良定义且
     审稿人一听就懂的问题, 注入把它搅浑成"在一个从未发生过、按瑞典系数缩放的世界里……"。
  ② α=0.36 出自 Palmqvist 的**瑞典**案例(SSRN 4168505, 区间 20–50%), **从未在德比数据上验证**;
     我们验证过的只是注入**幅度的边际分布**(bootstrap 自真实德比晚点), 从未验证注入后产生的
     **拥堵形态**与现实一致 —— 这两件事不是一回事。
  ③ 实测(3 个 val 运营日, 信号员回放): 注入把加权次生延误从 **6544s 抬到 16487s(2.5 倍)**,
     而回放的人**没有机会重新决策** ⇒ 人类对照锚被系统性压低, 我们跟它比偏易。
  ④ "扰动下是否 robust"用真实数据里**本来就有的乱日子**分层报告即可(`congested_windows` 的
     n_late 分位), 比合成扰动更强。
  ⑤ 若将来 val 出现过拟合(环境确定 → 65 个训练日各被重放约 90 次), 该加的随机性是
     **M5 标定过的逐 TC 占用时分残差**(我们自己的德比数据, 验收过中位残差无偏/中位绝对值<5s),
     不是 α。
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from derbysim import config

PRIMARY_FRACTION = 0.36          # α: Palmqvist 瑞典案例实测 primary 占比(SSRN 4168505); 区间 20–50%
REAL_FRAC_LATE = 0.50            # 真实德比 LATE 占比(trust_moves)
CLIP_MAX_MIN = 120.0             # 剔极端噪声(headcode 复用错配), 保留真实重灾 ≤2hr


def load_derby_lateness(path=None) -> np.ndarray:
    """真实德比晚点(min>0)数组, 供 bootstrap; 剔 0/负(早/准)与 >CLIP_MAX_MIN 噪声。"""
    p = path or (config.ARTIFACTS_SUBSET / "trust_moves.parquet")
    tm = pd.read_parquet(p)
    tm = tm[(tm["valid"] == True) & (tm["is_derby_station"] == True)]
    lat = tm["lateness"].dropna().astype(float).values
    return np.array([x for x in lat if 0.0 < x <= CLIP_MAX_MIN], dtype=float)


class DelaySampler:
    """经验 primary 延误采样器(秒). 从真实德比晚点分布 bootstrap, 按 α 缩放(只注入 primary 份额)。"""

    def __init__(self, lateness_min: np.ndarray, alpha: float = PRIMARY_FRACTION) -> None:
        if lateness_min is None or len(lateness_min) == 0:
            raise ValueError("empty lateness pool")
        self.pool = np.asarray(lateness_min, dtype=float)
        self.alpha = float(alpha)

    def sample_seconds(self, rng: np.random.Generator) -> float:
        """一次 bootstrap 抽样 → primary 延误秒数 = α × 经验晚点(min) × 60。"""
        base_min = float(rng.choice(self.pool))
        return self.alpha * base_min * 60.0


def inject_primary_delays(trains, sampler: DelaySampler, seed: int,
                          p_late: float = REAL_FRAC_LATE):
    """给 trains 副本注入 primary 入区延误: 每车以 p_late 概率 entry_time += α×经验采样(秒)。

    返回**新列表**(深拷贝, 不改原); 路由不变 → 目标/可达性天然保持。确定性: 同 seed 同结果。
    **注意(R2)**: 常态稀疏窗口下此注入惰性(无 overlap 可扰); 配合真实拥堵窗口用。
    """
    rng = np.random.default_rng(seed)
    out = []
    for t in trains:
        t2 = copy.deepcopy(t)
        if rng.random() < p_late:
            t2.entry_time = t2.entry_time + sampler.sample_seconds(rng)
        out.append(t2)
    out.sort(key=lambda x: x.entry_time)
    return out


def rank_high_disruption_windows(win_hours: float = 3.0, top_k: int = 8, path=None):
    """真实高扰动窗口(真实车/真实时刻)—— 冲突来源(替代不安全的合成加车, R3)。

    扫 TRUST 德比站事件, 按固定 win_hours 时窗聚合真实 lateness, 返回 top_k:
      [{start, end, total_lateness_min, n_late, n_events}], 按总晚点降序。
    这些窗口**真实发生过**(答辩无可辩驳), 用作训练冲突来源 + M13 固定评估场景。
    """
    p = path or (config.ARTIFACTS_SUBSET / "trust_moves.parquet")
    tm = pd.read_parquet(p)
    tm = tm[(tm["valid"] == True) & (tm["is_derby_station"] == True)].copy()
    ts = pd.to_datetime(tm["actual_timestamp"], errors="coerce")
    tm = tm[ts.notna()].copy(); tm["ts"] = ts[ts.notna()]
    tm["lateness"] = pd.to_numeric(tm["lateness"], errors="coerce").fillna(0.0)
    freq = f"{int(win_hours * 60)}min"
    tm["bin"] = tm["ts"].dt.floor(freq)
    g = tm.groupby("bin").agg(total_lateness_min=("lateness", "sum"),
                              n_late=("lateness", lambda s: int((s > 0).sum())),
                              n_events=("lateness", "size")).reset_index()
    # **全序 + 稳定排序**(2026-07-31 修): 原用 pandas 默认 quicksort(不稳定), 而池中存在并列的
    # total_lateness_min → 并列谁进前 top_k 随版本/数据顺序而变 ⇒ **train/val/test 成员在不同机器上不同**
    # (实测: 同代码同数据, 一台机器 val=9 个窗口、另一台 10 个)。留出集必须跨机器可复现, 故加时间戳作末位
    # 决胜键并指定 mergesort(稳定)。
    g = g.sort_values(["total_lateness_min", "bin"], ascending=[False, True],
                      kind="mergesort").head(top_k)
    out = []
    for r in g.itertuples():
        out.append({"start": r.bin, "end": r.bin + pd.Timedelta(hours=win_hours),
                    "total_lateness_min": float(r.total_lateness_min),
                    "n_late": int(r.n_late), "n_events": int(r.n_events)})
    return out


def build_clean_window_dyn_trains(td_win, net, dwell_by_platform, plat_tc_to_pid, cif_map=None):
    """构造 DynTrain，按观测位置复序预办进路并切开真断链——**保无碰撞不变式**。

    复审 R3 附带发现: 真实繁忙窗口的 raw build 含 H6 链断裂脏车, replay 会撞车(如 11-14 09-12 raw
    coll=1); 用 dataset.py 同一链连续性过滤(丢脏车)后 coll=0。故真实拥堵窗口造冲突**必须**走此清洗路径,
    与 BC 数据集(dataset.py)同一份干净车。
    """
    from derbysim.sim.env import build_env_trains_for_window
    from derbysim.sim.dataset import _chain_continuity, split_broken_chains
    from derbysim.sim.reward import to_dyn_trains
    built = build_env_trains_for_window(td_win, net, dwell_by_platform, plat_tc_to_pid, cif_map=cif_map)
    succ, pair = _chain_continuity(net)
    clean = split_broken_chains(built, net, succ, pair, plat_tc_to_pid)   # D31: 切分不丢弃
    return to_dyn_trains(clean)
