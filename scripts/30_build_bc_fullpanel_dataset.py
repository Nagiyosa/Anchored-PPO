"""⑦ full-panel BC 数据集(D57/D58): 在 **narrow proactive** 上回放信号员, 逐决策 epoch 记
`(obs, options, chosen_idx)` —— 克隆信号员**完整面板决策** {WAIT/SET/CANCEL}(时机+选路+入区放行),
取代旧"只克隆选路"。存 obs+options(训练时重编码, 与 featurizer 版本解耦, 同 scripts/11 口径)。

**口径**: 环境 narrow proactive(人类锚, D57.1); 每个 yield 的面板都是"办/等/撤"的真决策(引擎只在
有 set/cancel 时才开面板), 全部保留; 记 chosen 动作(WAIT/SET/CANCEL)+ 决策车 service_class(分层)。
held-out 划分沿 D34(train=9-10月 / holdout=11-12月; test 全程不碰)。**reward 与本数据无关**(BC 学动作)。

用法: PYTHONPATH=. python scripts/30_build_bc_fullpanel_dataset.py gen [split] [n_days]
      PYTHONPATH=. python scripts/30_build_bc_fullpanel_dataset.py combine
产出: artifacts/dataset/bc_fullpanel/decisions_<date>.parquet(逐日) ; combine → all + report。
"""
from __future__ import annotations
import hashlib, json, sys, time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import RewardedDynamicEnv

OUT = config.ARTIFACTS / "dataset" / "bc_fullpanel"


def _record_day(day, net, params, dwell, ptc, adapter, cap=300000):
    """narrow proactive 回放, 逐 epoch 记样本。"""
    env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="narrow")
    obs, _ = env.reset(build_day_trains(day, inject=False), horizon=day_horizon(day))
    act = make_signaller_replay_policy(env)
    rows = []
    while not env.done and len(rows) < cap:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        idx, _lp, _v = act(enc)
        opts = dec.options
        chosen = opts[idx]
        # 决策车 = 被选动作的车(WAIT 无车)。service_class 从 obs.trains 查, 供分层/加权。
        cls = ""
        if chosen.train_id:
            for tr in obs.get("trains", []):
                if tr.get("train_id") == chosen.train_id:
                    cls = (tr.get("service_class") or "")[:1]; break
        rows.append({
            "obs": json.dumps(obs, default=str, ensure_ascii=False),
            "options": json.dumps([[o.kind, o.train_id, o.route_id] for o in opts], ensure_ascii=False),
            "chosen_idx": int(idx),
            "chosen_kind": chosen.kind,
            "chosen_train": chosen.train_id,
            "chosen_route": chosen.route_id,
            "n_options": len(opts),
            "n_set": sum(1 for o in opts if o.kind == "set"),
            "service_class": cls,
        })
        obs, _r, _done, _ = env.step(chosen)
    return rows


def gen(split, n_days):
    OUT.mkdir(parents=True, exist_ok=True)
    net, params, dwell, ptc = load_assets()
    adapter = PanelAdapter(net, Featurizer(net, ptc))
    days = operating_days(split)[:n_days]
    t0 = time.time()
    for day in days:
        d = str(day["start"])[:10]
        outp = OUT / f"decisions_{d}.parquet"
        if outp.exists():
            print(f"skip {d}"); continue
        rows = _record_day(day, net, params, dwell, ptc, adapter)
        pd.DataFrame(rows).to_parquet(outp, index=False)
        kinds = Counter(r["chosen_kind"] for r in rows)
        print(f"{d}: n={len(rows)} set={kinds['set']} wait={kinds['wait']} cancel={kinds['cancel']}")
    print(f"[gen {split}] {len(days)} days in {time.time()-t0:.1f}s")


def combine():
    files = sorted(OUT.glob("decisions_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no per-day files under {OUT}; run gen first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = df.index.map(lambda i: None)  # placeholder; date from filename below
    # 由文件名带日期: 重读时逐文件加 date 列更稳
    parts = []
    for f in files:
        p = pd.read_parquet(f); p["date"] = f.stem.replace("decisions_", ""); parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    mon = pd.to_datetime(df["date"]).dt.month
    df["split"] = mon.map(lambda m: "train" if m in (9, 10) else "holdout")   # D34
    allp = config.ARTIFACTS / "dataset" / "bc_fullpanel_all.parquet"
    df.to_parquet(allp, index=False)
    fp = hashlib.sha256(pd.util.hash_pandas_object(
        df[["chosen_idx", "chosen_kind", "chosen_train", "chosen_route", "date"]],
        index=False).values.tobytes()).hexdigest()[:16]
    rep = {"n_days": int(df["date"].nunique()), "n_samples": int(len(df)),
           "by_split": df.groupby("split").size().to_dict(),
           "by_kind": df["chosen_kind"].value_counts().to_dict(),
           "kind_by_split": df.groupby(["split", "chosen_kind"]).size().unstack(fill_value=0).to_dict(),
           "service_class_dist": df[df["chosen_kind"] == "set"]["service_class"].value_counts().to_dict(),
           "fingerprint": fp}
    (config.ARTIFACTS / "dataset" / "bc_fullpanel_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "gen"
    if cmd == "combine":
        combine(); return 0
    split = argv[2] if len(argv) > 2 else "train"
    n_days = int(argv[3]) if len(argv) > 3 else 999
    gen(split, n_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
