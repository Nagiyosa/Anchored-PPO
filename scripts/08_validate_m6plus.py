"""M6+ 命门加强验证(诚实版): 描述性有效 + 冲突机制 + 现象刻画 + 局限.

不追全程秒级(差分目标里车速方差为共模, 已论证无必要)。四块:
  1. run-time/占用: 分段(计时点间)已验 ~2s (见 M6 validation_holdout, 标准做法)。
  2. 联锁+反应式机制: 单元测试(被占则等, 次生延误因果涌现)。
  3. 现象刻画(留出集): 区域内"被扣时间"分布 + 冲突归因(TC冲突 vs 外生)。
  4. 局限: 仿真器建模 TC冲突引起的次生延误; 外生延误是输入, 不生成; 反事实不可全验。
运行: python scripts/08_validate_m6plus.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.validation.calibration import pair_route_occupations


def main() -> int:
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    net = InfrastructureNetwork.load()
    params = load_params(config.ARTIFACTS_CALIB / "params.json")

    def focc(tc):
        return params.get(tc, (8, 18))[1]

    plat_tc = set()
    with open(config.DATA_PLATFORM_TC, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("platform_id") or "").strip().isdigit():
                plat_tc.add(r["tc_id"].strip())

    # 3a. 区域内被扣时间分布(留出集, 超额占用=停在段上不动, 排除站台停站)
    held = []
    for rid, tr, o, rel, seq in pair_route_occupations(td, net, split=config.HOLDOUT_SPLIT):
        ko = sum(max(0.0, rel[tc] - o[tc] - focc(tc))
                 for tc in seq if tc in o and tc in rel and tc not in plat_tc)
        held.append(ko)
    held = np.array(held)
    held_dist = {
        "n": int(len(held)), "median_s": round(float(np.median(held)), 1),
        "mean_s": round(float(held.mean()), 1), "p90_s": round(float(np.percentile(held, 90)), 1),
        "p99_s": round(float(np.percentile(held, 99)), 1),
        "frac_gt30s": round(float((held > 30).mean()), 3),
        "frac_gt60s": round(float((held > 60).mean()), 3),
    }

    # 3b. 冲突归因: 被扣>30s 中, 前方TC是否被他车占(=TC冲突, 仿真器可复现)
    win = td[(td["ts"] >= "2023-11-06") & (td["ts"] < "2023-11-13")].copy()
    tr = win[win["type"] == "Track"].copy()
    tr["id"] = tr["id"].str.strip()
    tr = tr.sort_values("ts")
    intervals = defaultdict(list)
    opent = {}
    for x in tr.itertuples():
        k = (x.id, x.trainid_filled)
        if x.state == "1":
            opent[k] = x.ts
        elif x.state == "0" and k in opent:
            intervals[x.id].append((opent.pop(k), x.ts, x.trainid_filled))

    def occ_other(tc, t, me):
        for a, b, who in intervals.get(tc, []):
            if who != me and a <= t <= b:
                return True
        return False

    occ_by = defaultdict(list)
    for tc, ivs in intervals.items():
        for a, b, who in ivs:
            if who not in ("0", "****", ""):
                occ_by[who].append((a, b, tc))
    for who in occ_by:
        occ_by[who].sort()
    conflict = exo = 0
    for who, ivs in occ_by.items():
        for i in range(len(ivs) - 1):
            a, b, tc = ivs[i]
            _, _, ntc = ivs[i + 1]
            if (b - a).total_seconds() - focc(tc) > 30:
                if occ_other(ntc, b - pd.Timedelta(seconds=5), who):
                    conflict += 1
                else:
                    exo += 1
    tot = conflict + exo
    attribution = {
        "n_holds_gt30s": tot,
        "tc_conflict_induced": conflict,
        "tc_conflict_frac": round(conflict / tot, 3) if tot else 0,
        "exogenous": exo,
        "exogenous_frac": round(exo / tot, 3) if tot else 0,
    }

    report = {
        "run_time_validation": "见 M6 validation_holdout: 分段(计时点间)占用时刻 中位~2s, 标准做法",
        "interlocking_and_reactive": "见 tests/test_reactive.py: 被占则等, 次生延误因果涌现(B等A 13s), 联锁0双占",
        "held_time_distribution_holdout": held_dist,
        "hold_attribution_peak_week": attribution,
        "conclusion": (
            f"区域内被扣时间中位 {held_dist['median_s']}s, {held_dist['frac_gt30s']:.0%} 超30s; "
            f"其中仅 {attribution['tc_conflict_frac']:.0%} 由TC冲突引起(仿真器可复现), "
            f"{attribution['exogenous_frac']:.0%} 为外生(仿真器不生成, 属输入)。"
        ),
    }
    print("=" * 64)
    print("M6+ 命门加强验证(诚实版)")
    print("=" * 64)
    for k, v in report.items():
        print(f"{k}:\n  {v}")

    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_REPORTS / "validation_m6plus.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# M6+ 命门加强 — 诚实验证报告

## 立场
不追"全程逐车秒级"。论证: 目标是降**次生延误(差分量)**, 每车车速方差(±10-22s/route)为**共模**,
在决策差值里抵消; 全程秒级是给错问题的精度(见 §5b)。故按**正确粒度**验证。

## 1. 运行时分/占用模型(描述性有效)
分段(计时点间)前推, 留出集占用时刻**中位 ~2s**(见 `validation_holdout.md`)。此为 RailSys/OpenTrack
类微观仿真的**标准校验粒度**(区间运行时分), 非全程逐车。

## 2. 联锁 + 反应式闭塞机制(构造正确)
`tests/test_reactive.py`: 下一段被占→停等, 次生延误**因果涌现**(B 因 A 等 13s 精确); 门禁按真实
进路时序串行化; 联锁 0 双占。

## 3. 现象刻画(留出集 11-12月)
- **区域内被扣时间**(超额占用=停在段上不动): 中位 {held_dist['median_s']}s, 均值 {held_dist['mean_s']}s,
  p90 {held_dist['p90_s']}s, p99 {held_dist['p99_s']}s; **{held_dist['frac_gt30s']:.0%} 超30s, {held_dist['frac_gt60s']:.0%} 超60s**。
- **冲突归因**(高峰周 {attribution['n_holds_gt30s']} 次>30s 被扣): **仅 {attribution['tc_conflict_frac']:.0%} 由 TC 冲突引起**
  (前方被他车占, 仿真器可复现), **{attribution['exogenous_frac']:.0%} 为外生**(下游/站台/接续/司机, 属输入)。

## 4. 结论与局限(诚实)
- 仿真器**描述性有效**(运行时分~2s, 联锁正确), 冲突机制**构造正确**。
- 它建模 **TC冲突引起的次生延误**; **外生延误是输入, 不由仿真器生成**——故复现不了全部被扣分布**是正确性质, 非缺陷**。
- 正常运营下咽喉 TC 冲突**罕见**(占被扣仅~{attribution['tc_conflict_frac']:.0%}); RL 的价值在**扰动下避免会发生的冲突**(反事实), 这一层**无法从数据完全验证**(§8⑤), 仅在专家校验+物理约束下作保守结论。
- **数据优势**: 借 TD 秒级轨道电路数据, 我们能把次生延误刻画到**秒级**, 多数延误管理研究(仅分钟级站点数据)做不到。
"""
    (config.ARTIFACTS_REPORTS / "validation_m6plus.md").write_text(md, encoding="utf-8")
    print("\n报告已写入 artifacts/reports/validation_m6plus.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
