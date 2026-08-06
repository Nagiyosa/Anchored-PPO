"""M-analysis: 量测信号员"并发预留深度"(D49 想要、但上个会话沙盒内联算飞的那个量).

问题(精确定义): 在为某条进路 k 发 Panel_Request 的时刻 t_k, **该车**已有多少条进路
处于"已办(已请求)但首 TC 尚未被占用"的状态 —— 即真正意义上的"提前排在前面的一串"。

  depth_excl_self(t_k) = #{ j != k : req_time[j] <= t_k 且 occ_first[first_tc(j)] > t_k }
  depth_incl_self(t_k) = depth_excl_self + 1   (把刚办的 k 也算进"手上已办未占用")

"已办未占用" ⇔ 一定在车前方(车还没占到该进路首 TC)。用 req_time 比大小(不是链序),
因为 D35 证实 Panel_Request 可先办下游后办上游, 请求时序 != 物理序。

数据源与口径: 完全复用 BC/RL 的车集构造 —— `build_env_trains_for_window` + D31/D35
`split_broken_chains`(Etches Park 进出库本就是两次独立调动, 不能跨库计并发)。
per-train `route_chain`/`req_times` 来自 TD Panel_Request(state=1);
`occ_first[tc]` = 该 TC 首个 Track(state=1) 观测占用时刻(同 build_env_trains_for_window)。

用法(本地只 smoke, 全量在服务器跑):
    PYTHONPATH=. python scripts/26_measure_reservation_depth.py smoke [n_days]   # 前 n_days 天(默认 3)
    PYTHONPATH=. python scripts/26_measure_reservation_depth.py full             # 全 98 天

产出(时间戳 artifact, 可复跑):
    artifacts/reports/reservation_depth_<mode>_<YYYYmmdd-HHMMSS>.json
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 从任意 cwd 可导入 derbysim

from derbysim import config
from derbysim.infra import InfrastructureNetwork
from derbysim.sim.occupation import load_params
from derbysim.sim.env import build_env_trains_for_window
from derbysim.sim.dataset import split_broken_chains


def _load_common():
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


def _all_days(td):
    return sorted({str(d) for d in pd.to_datetime(td["time"]).dt.date})


def _first_tc(net, rid):
    r = net.routes.get(rid)
    return r.tc_sequence[0] if (r and r.tc_sequence) else None


def _depths_for_train(net, et, occ_first):
    """逐 Panel_Request 计算 depth_excl_self / depth_incl_self, 以及本车链长。返回 list[dict]。"""
    chain, req = et.route_chain, et.req_times
    n = len(chain)
    first_tcs = [_first_tc(net, rid) for rid in chain]
    # 每条进路首 TC 的占用时刻(缺观测 → +inf = 窗口内从未占用, 视为"尚未占用")
    occ_t = [occ_first.get(ft, float("inf")) if ft is not None else float("inf")
             for ft in first_tcs]
    rows = []
    for k in range(n):
        tk = req[k]
        excl = 0
        excl_strict = 0          # 严格变体: 只数"最终确有观测占用"的进路(排除窗口边缘截断)
        for j in range(n):
            if j == k:
                continue
            if req[j] <= tk and occ_t[j] > tk:
                excl += 1
                if occ_t[j] != float("inf"):
                    excl_strict += 1
        rows.append({
            "train_id": et.train_id,
            "service_class": (et.service_class or "")[:1],
            "k": k,
            "depth_excl_self": excl,
            "depth_incl_self": excl + 1,
            "depth_excl_self_strict": excl_strict,
            "chain_len": n,
        })
    return rows


def _dist(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "frac_ge1": round(float((a >= 1).mean()), 4),
        "frac_ge2": round(float((a >= 2).mean()), 4),
        "frac_ge3": round(float((a >= 3).mean()), 4),
    }


def run(mode, n_days):
    net, params, dwell, ptc = _load_common()
    td = pd.read_parquet(config.ARTIFACTS_SUBSET / "td_events.parquet")
    td["ts"] = pd.to_datetime(td["time"])
    days = _all_days(td)
    if mode == "smoke":
        days = days[:n_days]

    all_rows = []
    n_trains = 0
    t0 = time.time()
    for d in days:
        win = td[(td["ts"] >= f"{d} 00:00:00") & (td["ts"] < f"{d} 23:59:59")].copy()
        if win.empty:
            continue
        built = build_env_trains_for_window(win, net, dwell, ptc)
        trains = split_broken_chains(built, net, plat_tc_to_pid=ptc)
        for et, occ_first in trains:
            n_trains += 1
            all_rows.extend(_depths_for_train(net, et, occ_first))
    dt = time.time() - t0

    df = pd.DataFrame(all_rows)
    per_train_len = df.groupby("train_id")["chain_len"].first().values if len(df) else []
    rep = {
        "_meta": {
            "script": "scripts/26_measure_reservation_depth.py",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": mode,
            "n_days": len(days),
            "n_trains": n_trains,
            "n_requests": int(len(df)),
            "elapsed_s": round(dt, 1),
            "source": "TD Panel_Request(state=1); occ=first Track(state=1); "
                      "train set = build_env_trains_for_window + split_broken_chains (D31/D35)",
            "definition": "depth = # routes with req_time<=t_k and first-TC occ_time>t_k (excl/incl the just-set route k)",
        },
        "depth_excl_self": _dist(df["depth_excl_self"]) if len(df) else {},
        "depth_incl_self": _dist(df["depth_incl_self"]) if len(df) else {},
        "depth_excl_self_strict": _dist(df["depth_excl_self_strict"]) if len(df) else {},
        "chain_len_per_train": _dist(per_train_len),
        "depth_excl_self_by_class": {
            cls: _dist(g["depth_excl_self"])
            for cls, g in df.groupby("service_class")
        } if len(df) else {},
    }

    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outp = config.ARTIFACTS_REPORTS / f"reservation_depth_{mode}_{stamp}.json"
    outp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    d0 = rep["depth_excl_self"]
    print(f"[{mode}] days={len(days)} trains={n_trains} requests={rep['_meta']['n_requests']} "
          f"in {dt:.1f}s")
    if d0:
        print(f"  depth_excl_self: median={d0['median']} p90={d0['p90']} p95={d0['p95']} "
              f"max={d0['max']} ge2={d0['frac_ge2']} ge3={d0['frac_ge3']}")
    cl = rep["chain_len_per_train"]
    if cl:
        print(f"  chain_len/train: median={cl['median']} p90={cl['p90']} max={cl['max']}")
    print(f"  -> {outp}")
    return 0


def main(argv):
    mode = argv[1] if len(argv) > 1 else "full"
    n_days = int(argv[2]) if (mode == "smoke" and len(argv) > 2) else 3
    if mode not in ("smoke", "full"):
        print(__doc__); return 2
    return run(mode, n_days)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
