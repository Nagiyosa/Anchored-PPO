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
    key = {"n_late": lambda w: (-w["n_late"], -w["total_lateness_min"]),
           "total_lateness": lambda w: -w["total_lateness_min"]}.get(rank_by)
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


def episode_horizon(window: dict, drain_s: float = 900.0) -> float:
    """episode 的**共同时间地平线** = 窗口末 + 排空期(epoch 秒)。

    所有策略都被结算到同一时刻 → ①被晾着的车按"一直站到地平线"计延误, 少干活不能把欠账截短;
    ②跨策略比较的时间跨度一致(否则对比本身无意义)。drain 依据见 `RewardConfig.drain_s`。
    """
    return float(pd.Timestamp(window["end"]).timestamp() + drain_s)
