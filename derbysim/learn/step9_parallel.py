"""D33 Step 9 并行扫描(服务器 / Jupyter 版): ProcessPoolExecutor + 断点重续 + 进度条。

与 `scripts/23_step9_scan.py`(CLI 日分片)口径完全一致, 但为**多核服务器 + notebook** 设计:
  - **点级**任务分发(不是日分片)→ 72 核负载更均衡;
  - worker 用 `initializer` **只加载一次**网络/参数/适配器;
  - 主进程**集中写 JSONL**(无多进程文件竞争)+ tqdm 进度条;
  - **断点重续**: 重跑自动跳过 JSONL 里已完成的 (day, k)。

用法(notebook 一格):
    from derbysim.learn.step9_parallel import run_scan_parallel, summarize_scan
    run_scan_parallel("reroute", n_workers=70)          # ① 延误级联, ~31k 点
    summarize_scan("reroute")                            # 汇总
    run_scan_parallel("wait", n_workers=70)              # ② 时机轴, ~42k 点
    summarize_scan("wait")
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.scenarios import (build_day_trains, day_horizon, load_assets,
                                      operating_days)
from derbysim.learn.step9 import census_signaller, run_injection
from derbysim.sim.reward import RewardedDynamicEnv

# --- worker 全局(每个进程 initializer 载一次) ---
_NET = _PARAMS = _DWELL = _PTC = _ADAPTER = None


def _init_worker():
    global _NET, _PARAMS, _DWELL, _PTC, _ADAPTER
    _NET, _PARAMS, _DWELL, _PTC = load_assets()
    _ADAPTER = PanelAdapter(_NET, Featurizer(_NET, _PTC))


def _run_point(day, k, mode):
    env = RewardedDynamicEnv(_NET, _PARAMS, _PTC, _DWELL)
    res = run_injection(env, _ADAPTER, build_day_trains(day, inject=False),
                        day_horizon(day), inject_at=k, mode=mode)
    m = res["metrics"]
    return {"k": int(k), "injected": res["injected"], "info": res["info"],
            "deviated": res["deviated"], "delay_s": m["weighted_secondary_delay_s"],
            "completion": m["completion_rate"], "n_reached": m["n_reached_exit"],
            "n_completable": m["n_completable"], "n_collisions": m["n_collisions"],
            "n_deadlocks": int(m["deadlocked"])}


def _days(split):
    if split == "trainval":
        return list(operating_days("train")) + list(operating_days("val"))
    return list(operating_days(split))


def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add((r["day"], r["k"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_scan_parallel(mode="reroute", split="trainval", n_workers=None, out=None,
                      limit_days=0, limit_points=0, progress=True):
    """并行跑单点扰动扫描, 断点重续。mode: 'reroute'(①) | 'wait'(②)。"""
    n_workers = int(n_workers or os.cpu_count() or 8)
    days = _days(split)
    if limit_days:
        days = days[:limit_days]
    out = Path(out) if out else (config.ARTIFACTS_REPORTS / f"d33_step9_{mode}_scan.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)

    # 主进程做 census(顺序, 便宜): 每天基线延误 + 可注入点; 组装未完成任务列表。
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    baselines, tasks = {}, []
    for day in days:
        dstr = str(day["start"])[:10]
        env = RewardedDynamicEnv(net, params, ptc, dwell)
        c = census_signaller(env, adapter, build_day_trains(day, inject=False), day_horizon(day))
        b = c["baseline_metrics"]
        baselines[dstr] = {"delay": b["weighted_secondary_delay_s"],
                           "reached": b["n_reached_exit"], "able": b["n_completable"]}
        pts = c["branchable_idx"] if mode == "reroute" else c["set_idx"]
        if limit_points:
            pts = pts[:limit_points]
        for k in pts:
            if (dstr, k) not in done:
                tasks.append((day, dstr, k))

    total = len(tasks)
    print(f"mode={mode} split={split} workers={n_workers} days={len(days)} "
          f"tasks={total} (resume_skip={len(done)})", flush=True)
    if not total:
        print("nothing to do (all points already scanned)")
        return out

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, desc=f"step9-{mode}")
        except Exception:
            bar = None

    started, n = time.time(), 0
    with out.open("a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        futs = {ex.submit(_run_point, day, k, mode): (dstr, k) for (day, dstr, k) in tasks}
        for fut in as_completed(futs):
            dstr, k = futs[fut]
            r = fut.result()
            base = baselines[dstr]
            row = {"day": dstr, "mode": mode, **r, "base_delay_s": base["delay"],
                   "delta_delay_s": r["delay_s"] - base["delay"],
                   "base_n_reached": base["reached"], "base_n_completable": base["able"]}
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            n += 1
            if bar is not None:
                bar.update(1)
            elif n % 500 == 0:
                el = time.time() - started
                print(f"  {n}/{total}  eta={_fmt(el / n * (total - n))}", flush=True)
    if bar is not None:
        bar.close()
    print(f"DONE mode={mode} rows_written={n} file={out}", flush=True)
    return out


def summarize_scan(mode_or_glob="reroute", write=True):
    """读回分片/单个 JSONL 汇总: 延误级联分布 + 完成率掉点 + 安全。"""
    glob = mode_or_glob if mode_or_glob.endswith((".jsonl", "*")) else f"d33_step9_{mode_or_glob}_scan*.jsonl"
    files = sorted(Path(config.ARTIFACTS_REPORTS).glob(glob))
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no rows for", glob)
        return None
    deltas = sorted(r["delta_delay_s"] for r in rows)
    worse = [d for d in deltas if d > 0]
    better = [d for d in deltas if d < 0]
    comp_drop = [r for r in rows if r["n_reached"] < r["base_n_reached"]]

    def pct(q):
        pos = (len(deltas) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(deltas) - 1)
        return deltas[lo] * (1 - (pos - lo)) + deltas[hi] * (pos - lo)

    mode = rows[0]["mode"]
    summary = {
        "audit": f"D33 step 9 {mode} scan summary", "mode": mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_files": len(files), "n_points": len(rows),
        "delta_delay_s": {"mean": sum(deltas) / len(deltas), "median": pct(0.5),
                          "p05": pct(0.05), "p95": pct(0.95), "p99": pct(0.99),
                          "min": deltas[0], "max": deltas[-1],
                          "n_worsened": len(worse), "n_improved": len(better),
                          "frac_worsened": len(worse) / len(deltas),
                          "total_worsening_s": sum(worse)},
        "completion": {"n_with_drop": len(comp_drop),
                       "frac_with_drop": len(comp_drop) / len(rows)},
        "safety": {"n_collisions": sum(r["n_collisions"] for r in rows),
                   "n_deadlocks": sum(r["n_deadlocks"] for r in rows)},
    }
    if write:
        outp = config.ARTIFACTS_REPORTS / f"d33_step9_{mode}_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
        outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print("wrote", outp)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
