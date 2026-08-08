"""D60 RL 训练矩阵启动器(服务器 72 核 / notebook 版): 子进程池 + 幂等续跑 + 进度/ETA。

为什么是**子进程**(不是 step9 那样的进程池函数): 每个 config 是一次完整的 `scripts/13` 训练
(torch、长时、要 --resume 断点续训), 用子进程隔离最干净; 每个子进程**强制单核**(线程数=1)以贴合
"72 个单核 run 并发"的模型(net 只有 hidden=64, GPU 无收益; 见 ppo.py D32)。

两层断点续跑, 对齐服务器单次 12h 上限:
  · **矩阵级(幂等)**: 已完成的 config(其 `<tag>.json` 已写)重跑本函数时**自动跳过**;
  · **run 级**: 未写完(12h 被杀)的 config 带 `--resume` 重启, 从 `<tag>_last.pt` 续训。
故 12h 到点后**重跑同一格**即可把整个矩阵接着跑完。

用法(notebook 一格):
    from derbysim.learn.rl_matrix import default_grid, run_rl_matrix, summarize_rl_matrix
    run_rl_matrix(default_grid(encoders=["gnn"]), n_workers=70, iters=40)
    summarize_rl_matrix()
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from derbysim import config

REPO = Path(__file__).resolve().parents[2]
REPORTS = config.ARTIFACTS_REPORTS
LOGDIR = REPORTS / "rl_logs"


def _tag(enc, anchor, beta, seed) -> str:
    """稳定 run 名(无时间戳): none 锚下 β 无意义 → 不进名字, 天然去重。"""
    if anchor == "none":
        return f"ppo_{enc}_none_s{seed}"
    return f"ppo_{enc}_{anchor}_b{beta:g}_s{seed}"


def default_grid(encoders=("gnn",), anchors=("schedule", "lookup", "none"),
                 betas=(0.0, 0.01, 0.1, 1.0), seeds=(0, 1, 2)) -> list:
    """展开 encoder × kl-anchor × β × seed。规则:
    · anchor='none' → β 无关, 只留一格/seed(β=0);
    · anchor∈{schedule,lookup} 且 β=0 等价 none(锚被关) → 跳过, 避免重复。
    """
    grid = []
    seen = set()
    for enc in encoders:
        for anchor in anchors:
            bs = [0.0] if anchor == "none" else [b for b in betas if b > 0]
            for beta in bs:
                for seed in seeds:
                    t = _tag(enc, anchor, beta, seed)
                    if t in seen:
                        continue
                    seen.add(t)
                    grid.append({"encoder": enc, "kl_anchor": anchor,
                                 "kl_beta": float(beta), "seed": int(seed), "tag": t})
    return grid


def _done(tag: str) -> bool:
    """完成判据 = 最终报告 <tag>.json 已写(scripts/13 只在正常跑完/早停后写)。"""
    return (REPORTS / f"{tag}.json").exists()


def _cmd(cfg: dict, py: str, common: dict) -> list:
    bc = common.get("bc_ckpt_fmt", "artifacts/dataset/bc_fullpanel_{enc}.pt").format(enc=cfg["encoder"])
    cmd = [py, "scripts/13_train_ppo.py",
           "--encoder", cfg["encoder"], "--kl-anchor", cfg["kl_anchor"],
           "--kl-beta", str(cfg["kl_beta"]), "--seed", str(cfg["seed"]),
           "--bc-ckpt", bc, "--run-tag", cfg["tag"], "--resume",
           "--reserve-window", common.get("reserve_window", "wide"),
           "--iters", str(common.get("iters", 40)),
           "--steps-per-iter", str(common.get("steps_per_iter", 2048)),
           "--eval-every", str(common.get("eval_every", 5))]
    for k in ("half_life", "hidden", "critic_warmup_iters", "early_stop_patience", "days", "eval_days"):
        if k in common:
            cmd += [f"--{k.replace('_', '-')}", str(common[k])]
    return cmd


def _run_one(cfg: dict, py: str, common: dict, env: dict) -> dict:
    tag = cfg["tag"]
    if _done(tag):
        return {"tag": tag, "status": "skip", "elapsed_s": 0.0}
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{tag}.log"
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launch {tag} =====\n")
        fh.flush()
        rc = subprocess.run(_cmd(cfg, py, common), cwd=str(REPO), env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    return {"tag": tag, "status": ("ok" if rc == 0 else f"fail(rc={rc})"),
            "elapsed_s": round(time.time() - t0, 1)}


def _fmt(sec: float) -> str:
    sec = int(sec); h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_rl_matrix(grid: list, n_workers=None, py=None, progress=True, **common) -> dict:
    """并发跑训练矩阵, 幂等续跑。**每个子进程强制单核**(线程数=1)以支持 72 路并发。

    common 里可放 scripts/13 的公共旋钮: iters / steps_per_iter / eval_every / reserve_window /
    half_life / hidden / days / eval_days / early_stop_patience / bc_ckpt_fmt。
    """
    n_workers = int(n_workers or os.cpu_count() or 8)
    py = py or sys.executable
    # **预检**: 子进程用的 python 必须能 import torch。notebook 内核的 sys.executable 常常没装 torch
    # (torch 只在 venv 里) → 全部 run 会秒挂在 `ModuleNotFoundError: torch`。这里一次性快速失败并提示。
    chk = subprocess.run([py, "-c", "import torch"], capture_output=True, text=True)
    if chk.returncode != 0:
        raise SystemExit(
            f"[rl-matrix] 选定的 python 无法 import torch:\n  py = {py}\n  {chk.stderr.strip()}\n"
            f"→ 传 py=PY(venv 里那个有 torch 的 python), 例如 run_rl_matrix(grid, py=PY, ...)。")
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",           # 每 run 单核
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    pending = [c for c in grid if not _done(c["tag"])]
    skipped = len(grid) - len(pending)
    print(f"[rl-matrix] configs={len(grid)} 待跑={len(pending)} 已完成跳过={skipped} "
          f"workers={n_workers} py={py}", flush=True)
    if not pending:
        print("[rl-matrix] 全部已完成。")
        return {"ok": 0, "fail": 0, "skip": skipped}
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(pending), desc="rl-matrix")
        except Exception:
            bar = None
    res = {"ok": 0, "fail": 0, "skip": skipped, "runs": []}
    started, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one, c, py, common, env): c["tag"] for c in pending}
        for fut in as_completed(futs):
            r = fut.result(); res["runs"].append(r); n += 1
            res["ok"] += (r["status"] == "ok"); res["fail"] += r["status"].startswith("fail")
            msg = f"  [{n}/{len(pending)}] {r['tag']}: {r['status']} ({_fmt(r['elapsed_s'])})"
            if r["status"].startswith("fail"):
                msg += f"  ← 见 {LOGDIR / (r['tag'] + '.log')}"
            if bar is not None:
                bar.update(1); bar.write(msg)
            else:
                el = time.time() - started
                print(msg + f"  eta={_fmt(el / n * (len(pending) - n))}", flush=True)
    if bar is not None:
        bar.close()
    print(f"[rl-matrix] DONE ok={res['ok']} fail={res['fail']} skip={res['skip']}", flush=True)
    return res


def summarize_rl_matrix(glob="ppo_*.json", write=True) -> dict:
    """读回各 run 的最终报告, 按 val best reward 排名, 附基线参照线。"""
    files = sorted(REPORTS.glob(glob))
    rows = []
    baselines = None
    for f in files:
        if f.name.endswith(("_best.pt", "_last.pt")) or "summary" in f.name:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "best" not in d or "args" not in d:
            continue
        a, b = d["args"], d["best"]
        if "kl_anchor" not in a:          # 旧 reactive 时代的 ppo_*.json 无此键 → 排除, 不污染排名(D60+)
            continue
        baselines = baselines or d.get("baselines")
        rows.append({"tag": d.get("run_id", f.stem),
                     "encoder": a.get("encoder"), "kl_anchor": a.get("kl_anchor"),
                     "kl_beta": a.get("kl_beta"), "seed": a.get("seed"),
                     "window": a.get("reserve_window"),
                     "best_iter": b.get("iter"), "reward": b.get("reward"),
                     "completion": b.get("completion"), "delay": b.get("delay")})
    rows.sort(key=lambda r: (r["reward"] is None, -(r["reward"] or -1e18)))
    lk = (baselines or {}).get("lookup_most_frequent", {})
    hu = (baselines or {}).get("signaller_replay", {})
    summary = {"audit": "D60 RL matrix summary", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "n_runs": len(rows),
               "baseline_lookup_reward": lk.get("eval_episode_reward"),
               "baseline_signaller_reward": hu.get("eval_episode_reward"),
               "baseline_lookup_delay_s": lk.get("eval_weighted_secondary_delay_s"),
               "ranking": rows}
    if write and rows:
        outp = REPORTS / f"rl_matrix_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
        outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", outp)
    for r in rows[:20]:
        beat = ("✅" if (r["reward"] is not None and lk.get("eval_episode_reward") is not None
                        and r["reward"] > lk["eval_episode_reward"]) else "  ")
        print(f"{beat} [{str(r['window'])[:6]:<6}] {r['tag']:<36} reward {str(r['reward'])[:9]:>9} "
              f"comp {str(r['completion'])[:5]:>5} delay {str(r['delay'])[:9]:>9} "
              f"(it{r['best_iter']})")
    if lk:
        print(f"   靶子 lookup reward {lk.get('eval_episode_reward')} / "
              f"人类 {hu.get('eval_episode_reward')}")
    return summary
