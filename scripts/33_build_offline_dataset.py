"""D64: 生成离线转移数据集(供 IQL / AWAC)—— **多核并行 + 分片写盘 + 进度 + 幂等续跑**。

跑规则策略(signaller 专家 + lookup/greedy/fcfs/fsfs 坏状态样本)过 proactive-wide 环境,逐决策记
`(enc, option_idx, reward, done)`,**每个 (策略,日) 一个分片 pkl**(不堆内存)。**torch-free**,可本地跑。
`scripts/34 --dataset` 直接读这个**分片目录**。

优化理由(旧版极慢的原因): 旧版**串行** + 把全部 ~1GB enc 堆内存最后一次性 pickle。现在: 每个
(策略,日)独立 → **进程池并行**(72 核); 各自写自己的分片(内存只留一局); tqdm 进度; 分片已存则跳过(续跑)。

用法: PYTHONPATH=. python scripts/33_build_offline_dataset.py [split] [n_days] [n_workers]
产出: artifacts/dataset/offline_shards_<split>/<policy>_<date>.pkl (一堆分片)
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config

_NET = _PARAMS = _DWELL = _PTC = _AD = _POLICIES = _DAYS = None
_OUTDIR: Path | None = None


def _init_worker(split: str, outdir: str) -> None:
    """每个 worker 只加载一次网络/适配器/策略/日历(避免每任务重载)。"""
    global _NET, _PARAMS, _DWELL, _PTC, _AD, _POLICIES, _DAYS, _OUTDIR
    from derbysim.learn.features import Featurizer
    from derbysim.learn.panel import PanelAdapter
    from derbysim.learn.baselines import (lookup_table, make_fcfs_policy, make_fsfs_policy,
                                          make_lookup_policy)
    from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy
    from derbysim.learn.scenarios import load_assets, operating_days
    _NET, _PARAMS, _DWELL, _PTC = load_assets()
    _AD = PanelAdapter(_NET, Featurizer(_NET, _PTC))
    top = lookup_table("train")
    _POLICIES = {                                        # 都是 factory(env)->act
        "signaller": make_signaller_replay_policy,
        "lookup": make_lookup_policy(top),
        "greedy": lambda e: make_greedy_set_policy(),
        "fcfs": make_fcfs_policy(top),
        "fsfs": make_fsfs_policy(top),
    }
    _DAYS = list(operating_days(split))
    _OUTDIR = Path(outdir)


def _run_one(policy_name: str, day_idx: int) -> tuple:
    """跑一局 (策略,日) → 写分片 pkl,返回 (tag, n_steps)。"""
    from derbysim.learn.rollout import run_episode
    from derbysim.learn.scenarios import build_day_trains, day_horizon
    from derbysim.sim.reward import RewardedDynamicEnv
    w = _DAYS[day_idx]
    date = str(w["start"])[:10]
    shard = _OUTDIR / f"{policy_name}_{date}.pkl"
    if shard.exists():
        return (f"{policy_name}/{date}", -1)             # 已有 → 跳过(续跑)
    env = RewardedDynamicEnv(_NET, _PARAMS, _PTC, _DWELL, proactive=True, reserve_window="wide")
    res = run_episode(env, _AD, _POLICIES[policy_name](env), build_day_trains(w, inject=False),
                      max_decisions=300000, horizon=day_horizon(w))
    ep = {"policy": policy_name, "day": date,
          "steps": [{"enc": t.enc, "opt_idx": int(t.option_idx),
                     "reward": float(t.reward), "done": bool(t.done)} for t in res.transitions]}
    tmp = shard.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(ep, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, shard)                                # 原子写(并发安全)
    return (f"{policy_name}/{date}", len(ep["steps"]))


def main(argv) -> int:
    split = argv[1] if len(argv) > 1 else "train"
    n_days = int(argv[2]) if len(argv) > 2 else 24
    n_workers = int(argv[3]) if len(argv) > 3 else min(70, os.cpu_count() or 8)
    from derbysim.learn.scenarios import operating_days
    days = operating_days(split)[:n_days]
    pols = ["signaller", "lookup", "greedy", "fcfs", "fsfs"]
    outdir = config.ARTIFACTS / "dataset" / f"offline_shards_{split}"
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = [(p, i) for p in pols for i in range(len(days))]
    todo = [(p, i) for (p, i) in tasks
            if not (outdir / f"{p}_{str(days[i]['start'])[:10]}.pkl").exists()]
    print(f"[offline] split={split} days={len(days)} policies={pols} workers={n_workers} "
          f"| 任务 {len(tasks)} 待跑 {len(todo)}(已存跳过 {len(tasks)-len(todo)}) -> {outdir}", flush=True)
    if not todo:
        print("[offline] 全部已生成。"); return 0

    bar = None
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc="offline")
    except Exception:
        bar = None

    started, n, n_steps = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(split, str(outdir))) as ex:
        futs = {ex.submit(_run_one, p, i): (p, i) for (p, i) in todo}
        for fut in as_completed(futs):
            tag, k = fut.result()
            n += 1; n_steps += max(k, 0)
            if bar is not None:
                bar.update(1); bar.set_postfix_str(f"{tag} {k}步")
            elif n % 10 == 0 or n == len(todo):
                el = time.time() - started
                print(f"  {n}/{len(todo)} | {tag} {k}步 | eta {el/n*(len(todo)-n):.0f}s", flush=True)
    if bar is not None:
        bar.close()
    print(f"[offline] DONE 分片 {len(list(outdir.glob('*.pkl')))} 个 | 累计 {n_steps} 转移 | "
          f"{time.time()-started:.0f}s | 目录 {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
