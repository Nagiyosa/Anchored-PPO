"""M12b 训练/评估场景构造：真实拥堵窗口 + primary 延误注入（§10-H，无 torch）。

**为什么只取拥堵窗口**: M10 复审实测——常态窗口注入延误 **Δ=0**(稀疏咽喉无 overlap 可扰),
常态 episode 近似白跑。冲突(=学习信号)只在真实拥堵时段有。故训练场景 =
`rank_high_disruption_windows`(真实最堵时段, 真车真时刻) → `build_clean_window_dyn_trains`
(链清洗, 保无碰撞不变式, 与 BC 数据集同一份干净车) → 每 episode 按种子注入 primary 延误(α=0.36)。

**诚实代价(必须写进论文)**: 训练分布 ≠ 真实运营分布, agent 在常态时段是 out-of-distribution。
M13 **必须常态/扰动分开报**(§14.0); 常态若不如基线, 如实写。

**时间划分(§10-E)**: train ≤ 2023-11-15 / test ≥ 2023-11-16(~3 周完全留出, 最后报一次);
val = 从 train 窗口里切一部分(调超参/早停), 不碰 test。
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache

import pandas as pd

from derbysim import config
from derbysim.sim.dataset import build_cif_map
from derbysim.sim.occupation import load_params
from derbysim.sim.perturb import (DelaySampler, build_clean_window_dyn_trains,
                                  inject_primary_delays, load_derby_lateness,
                                  rank_high_disruption_windows)

TRAIN_END = pd.Timestamp("2023-11-16")        # train: < 11-16
TEST_START = pd.Timestamp("2023-11-16")       # test:  >= 11-16 (完全留出)


@lru_cache(maxsize=1)
def load_assets():
    """路网 / 占用参数 / 停站 / 站台TC —— 重 I/O 只做一次。"""
    from derbysim.infra import InfrastructureNetwork
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")
    dwell = {int(k): v for k, v in
             json.loads((config.ARTIFACTS_CALIB / "dwell.json").read_text()).items()}
    ptc = {}
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("platform_id") or "").strip().isdigit():
                ptc[row["tc_id"].strip()] = int(row["platform_id"])
    return net, params, dwell, ptc


@lru_cache(maxsize=1)
def _td_events():
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    return td


@lru_cache(maxsize=1)
def _occurrences():
    return pd.read_parquet(config.ARTIFACTS_SUBSET / "train_occurrences.parquet")


@lru_cache(maxsize=8)
def _ranked(win_hours: float, pool: int):
    return tuple((w["start"], w["end"], w["total_lateness_min"], w["n_late"])
                 for w in rank_high_disruption_windows(win_hours=win_hours, top_k=pool))


@lru_cache(maxsize=1)
def _cif_index():
    """按趟的 CIF 索引(全量一次建好, 就近匹配) —— 跨午夜与同日多趟都正确。"""
    from derbysim.sim.dataset import build_cif_index
    return build_cif_index(_occurrences())


@lru_cache(maxsize=1)
def _sampler():
    return DelaySampler(load_derby_lateness())


def congested_windows(split: str = "train", win_hours: float = 3.0, top_k: int = 6,
                      pool: int = 60, val_frac: float = 0.25,
                      rank_by: str = "n_late") -> list[dict]:
    """真实最拥堵的窗口, 按时间划分取 split ∈ {train, val, test, all}。

    **排序统计量 = `n_late`(窗口内晚点事件数), 不是 `total_lateness_min`** —— 实测(2026-07-30,
    对 60 个候选窗口逐个建车 + 回放计决策数):

    | 排序方式 | top1 窗口 | 清洗后车数 | 面板决策数 | 真多选决策 | CIF planned_ref 覆盖 |
    |---|---|---|---|---|---|
    | total_lateness_min | 10-15 00:00 | **10** | **8** | **7** | **0/10** |
    | n_late | 11-27 06:00 | 31 | 580 | 49 | 26/31 |

    总晚点分钟数会被**少数极端晚点车**主导 → 排到"半夜几辆车晚得离谱"的窗口, 车少、决策少、
    CIF 覆盖为 0, 学不到东西(同 M10 对采样器 clip 120min 的理由)。`n_late` 数的是**多少趟车受影响**,
    与"忙 + 乱"都相关: 按它排的 top10 全部为 30–43 车 / 522–1558 决策 / 33–83 真多选。
    两种排序都保留(`rank_by`), 默认 n_late。

    val 从 train 段**末尾**切 `val_frac`(时间上更靠后 = 更贴近 test 的调参用途, 但仍 < 11-16,
    不碰留出集)。
    """
    # 排序键必须是**全序**(末位用窗口起始时刻决胜), 否则并列项的先后随实现而变 → 划分不可复现。
    key = {"n_late": lambda w: (-w["n_late"], -w["total_lateness_min"], w["start"]),
           "total_lateness": lambda w: (-w["total_lateness_min"], w["start"])}.get(rank_by)
    if key is None:
        raise ValueError(f"unknown rank_by {rank_by!r}; use n_late/total_lateness")
    ranked = sorted(({"start": s, "end": e, "total_lateness_min": tl, "n_late": nl}
                     for s, e, tl, nl in _ranked(win_hours, pool)), key=key)
    if split == "all":
        return ranked[:top_k]
    if split == "test":
        return [w for w in ranked if w["start"] >= TEST_START][:top_k]
    # **必须先切掉 test 段**(< TRAIN_END), 再在其内部分 train/val。
    # (2026-07-30 修: train 分支曾只排除 val 未排除 test → 留出集窗口漏进训练集。单测锁死。)
    tr = [w for w in ranked if w["start"] < TRAIN_END]
    tr_sorted = sorted(tr, key=lambda w: w["start"])
    n_val = max(1, int(round(len(tr_sorted) * val_frac))) if tr_sorted else 0
    val_ids = {w["start"] for w in tr_sorted[len(tr_sorted) - n_val:]} if n_val else set()
    if split == "val":
        return [w for w in tr if w["start"] in val_ids][:top_k]
    if split == "train":
        return [w for w in tr if w["start"] not in val_ids][:top_k]
    raise ValueError(f"unknown split {split!r}; use train/val/test/all")


def build_episode_trains(window: dict, seed: int | None = None, inject: bool = True,
                         p_late: float = 0.50):
    """一个窗口 → 一 episode 的 DynTrain 列表(链清洗 + 可选 primary 延误注入)。

    `seed=None` 或 `inject=False` → 不注入(用于**固定评估**: 真车真时刻, 可复现, 答辩无可辩驳)。
    """
    net, _params, dwell, ptc = load_assets()
    td = _td_events()
    win = td[(td["ts"] >= window["start"]) & (td["ts"] < window["end"])].copy()
    date = pd.Timestamp(window["start"]).strftime("%Y-%m-%d")
    cif_map = build_cif_map(_occurrences(), date)
    trains = build_clean_window_dyn_trains(win, net, dwell, ptc, cif_map=cif_map)
    if inject and seed is not None:
        trains = inject_primary_delays(trains, _sampler(), seed=seed, p_late=p_late)
    return trains


def episode_horizon(window: dict, drain_s: float = 1200.0) -> float:
    """episode 的**共同时间地平线** = 窗口末 + 排空期(epoch 秒)。

    所有策略都被结算到同一时刻 → ①被晾着的车按"一直站到地平线"计延误, 少干活不能把欠账截短;
    ②跨策略比较的时间跨度一致(否则对比本身无意义)。drain 依据见 `RewardConfig.drain_s`。
    """
    return float(pd.Timestamp(window["end"]).timestamp() + drain_s)


# ============ 运营日 episode（2026-07-31 定案，取代"只取拥堵窗口"） ============
#
# **为什么不再切拥堵窗口**(更正 §10-H / D26 的那半句, 连同其错误依据):
#   环境是时间上的**连续过程**。原先切 3h 窗口、且只取最堵的 60 个(= 全部 758 个窗口的 7.9%,
#   门槛 n_late>=57 ≈ 全体 p92), 有三重问题:
#     ① agent 只见"已经很糟"的状态, **看不到拥堵的形成与恢复** —— 而那正是调度本事的主要部分;
#     ② 每个窗口都从"区内空无一车"开始, 对拥堵时段是不真实的初始状态; 被边界切碎的车正是那批
#        `goal.exit_group` 为空、结构上不可能完成的车;
#     ③ "只取拥堵"的原始依据是 M10 复审"常态窗口注入延误 Δ=0" —— 那是**在一个窗口上**测的。
#        全量分层实测推翻了它的推论: **中位窗口(n_late=36)有 23–32 车、35–70 个真多选决策,
#        "有路就办"只完成 0.633–0.950 且延误达 225k** ⇒ 中位窗口的学习信号不比 p90 少。
#        (注入在清淡窗口确实无效, 这条仍成立 —— 但"注入无效" ≠ "窗口没冲突", 我当时混为一谈。)
#
# **切在 03:00**: 判据不是"哪小时请求最少", 而是**"在哪切会把车拦腰切断"**。绕开按自然日分组的
# `train_occurrences`(该表建表时就把跨午夜趟次劈成两行, 结构上测不出 00:00 的切断), 从原始 TD
# 按 90min 间隙重切趟后实测(98 天):
#     03:00 **0.80 趟/天**(最少) · 02:00 0.90 · 12:00 1.41 · **00:00 自然日 2.49** · 06:00 4.11(最差)
# ⇒ 03:00 比自然日好 3.1 倍。这是**新约定**(项目原先只有自然日 `.dt.date` 与 90min 切趟两条规则),
#   故跨午夜必须用 `dataset.build_cif_map_days` 的 (headcode, 日期) 双键, 否则午夜后的车会拿到
#   差 24 小时的 `planned_ref`(实测确认)。
OPERATING_DAY_CUT_HOUR = 3


def operating_days(split: str = "train", val_frac: float = 0.15) -> list[dict]:
    """按时间顺序切运营日(03:00→次日 03:00); 划分沿用 §10-E 的日期界, 不做拥堵度筛选。"""
    td = _td_events()
    lo = (td["ts"].min() - pd.Timedelta(hours=OPERATING_DAY_CUT_HOUR)).normalize()
    hi = (td["ts"].max() - pd.Timedelta(hours=OPERATING_DAY_CUT_HOUR)).normalize()
    days = [{"start": d + pd.Timedelta(hours=OPERATING_DAY_CUT_HOUR),
             "end": d + pd.Timedelta(hours=OPERATING_DAY_CUT_HOUR) + pd.Timedelta(days=1)}
            for d in pd.date_range(lo, hi, freq="D")]
    days = [d for d in days if d["end"] > td["ts"].min() and d["start"] < td["ts"].max()]
    if split == "all":
        return days
    tr = [d for d in days if d["start"] < TRAIN_END]
    te = [d for d in days if d["start"] >= TEST_START]
    if split == "test":
        return te
    n_val = max(1, int(round(len(tr) * val_frac)))
    if split == "val":
        return tr[len(tr) - n_val:]          # 取 train 段**末尾**(时间上更靠近 test)
    if split == "train":
        return tr[:len(tr) - n_val]
    raise ValueError(f"unknown split {split!r}; use train/val/test/all")


def build_day_trains(day: dict, seed: int | None = None, inject: bool = True,
                     p_late: float = 0.50):
    """一个运营日 → DynTrain 列表(链清洗 + 可选 primary 延误注入)。跨午夜, 故用双键 CIF map。"""
    net, _params, dwell, ptc = load_assets()
    td = _td_events()
    win = td[(td["ts"] >= day["start"]) & (day["end"] > td["ts"])].copy()
    trains = build_clean_window_dyn_trains(win, net, dwell, ptc, cif_map=_cif_index())
    if inject and seed is not None:
        trains = inject_primary_delays(trains, _sampler(), seed=seed, p_late=p_late)
    return trains


def day_horizon(day: dict, drain_s: float = 1200.0) -> float:
    """运营日的共同时间地平线 = 日末 + 排空期(epoch 秒)。理由同 `episode_horizon`。"""
    return float(pd.Timestamp(day["end"]).timestamp() + drain_s)
