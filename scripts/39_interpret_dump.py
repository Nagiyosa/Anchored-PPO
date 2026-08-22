"""D77 可解释性数据导出(服务器跑一次, 产出 json; 画图/统计在本地做)。

四个 mode(--mode):
  cumgap    日内累计加权延误曲线(每决策后采样 realized_weighted_delay)—— 「差距一点点拉开」主图数据。
  trace     逐决策日志(时刻/选了谁/什么动作/选路是否=计划路/=最频路/入区等待人数)+ 目标车的完整
            offer 历史 —— 案例剖析(11-24 病理 / 11-27 悬崖 / 日常技艺)与选路一致率、动作构成。
  leadtime  逐车记录(entry/entered/exit/加权延误/完成/**完成时刻相对计划的晚点分钟**)——
            放行提前量分布 + PPM(晚点≤5min 占比)。
  permimport 特征组置零消融(val): 逐组把特征列置 0 后整套评估 —— 策略到底在看什么(重点验证 fv2 两特征)。

策略集 --policies rl,fcfs,signaller(默认); 不含 rl 时不 import torch(本地可冒烟)。
产出: artifacts/reports/interpret_<mode>_<标签>.json(全部新文件, 零覆盖)。
用法示例(服务器):
  PYTHONPATH=. python scripts/39_interpret_dump.py --mode cumgap --ckpt artifacts/reports/ppo_d73_AB_s0_best.pt \
      --days 2023-10-23,2023-11-13,2023-11-27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from derbysim import config
from derbysim.learn.baselines import lookup_table, make_fcfs_policy
from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter
from derbysim.learn.rollout import make_signaller_replay_policy
from derbysim.learn.scenarios import build_day_trains, day_horizon, load_assets, operating_days
from derbysim.sim.reward import (RewardConfig, RewardedDynamicEnv, can_complete,
                                 has_reached_goal, realized_weighted_delay)

RCFG = RewardConfig()


def find_day(date):
    for sp in ("train", "val", "test"):
        for w in operating_days(sp):
            if str(w["start"])[:10] == date:
                return w, sp
    raise SystemExit(f"day {date} not found")


def load_rl(ckpt):
    """懒加载 torch 与策略(--policies 含 rl 时才调)。返回 act_factory。"""
    import torch
    from derbysim.learn.panel_models import PanelOptionPolicy
    from derbysim.learn.ppo import RunningMoments, TorchPanelDriver
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck = st.get("args", {})
    fv = int(ck.get("feature_version", 2))
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc, fv=fv))
    pol = PanelOptionPolicy(ck.get("encoder", "gnn"), ad.dec_dim, ad.conf_dim, ad.cand_dim,
                            ad.cancel_dim, ad.train_dim, ad.global_dim,
                            hidden=int(ck.get("hidden", 64)))
    pol.load_state_dict(st["model"])
    pol.eval()
    rn = RunningMoments()
    if st.get("ret_norm"):
        rn.load_state_dict(st["ret_norm"])
    drv = TorchPanelDriver(pol, torch.device("cpu"), greedy=True, ret_norm=rn)
    return (lambda env: drv), fv


def run_instrumented(env, adapter, act, trains, horizon, on_step=None, on_decision=None):
    """自定义回放循环(带钩子): on_decision(env, enc, dec, idx) / on_step(env)。返回终态 engine。"""
    obs, _ = env.reset(trains, horizon=horizon)
    n = 0
    while not env.done and n < 20000:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        idx, _lp, _v = act(enc)
        if on_decision is not None:
            on_decision(env, enc, dec, int(idx))
        obs, _r, _d, _ = env.step(dec.options[int(idx)])
        if on_step is not None:
            on_step(env)
        n += 1
    return env.engine


def train_rows(engine):
    out = {}
    for t in engine.trains.values():
        if t.background or not can_complete(t):
            continue
        occ = list(getattr(t, "sim_occ", {}).values())
        out[t.train_id] = {
            "cls": str(t.service_class), "entry": float(t.entry_time),
            "entered": (min(occ) if occ else None), "exit": (max(occ) if occ else None),
            "wdelay": RCFG.lam(t.service_class) * float(getattr(t, "held_time", 0.0)),
            "stops": int(getattr(t, "n_stops", 0)), "done": bool(has_reached_goal(t)),
            "final_delay_min": engine._delay_min(t),   # 完成时刻相对 CIF 计划的签名晚点(分钟)
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["cumgap", "trace", "leadtime", "permimport", "sensitivity"])
    ap.add_argument("--ckpt", default="artifacts/reports/ppo_d73_AB_s0_best.pt")
    ap.add_argument("--days", default="", help="逗号分隔日期(cumgap/trace 用)")
    ap.add_argument("--split", default="val", help="leadtime/permimport 用")
    ap.add_argument("--max-days", type=int, default=999)
    ap.add_argument("--policies", default="rl,fcfs,signaller")
    ap.add_argument("--tag", default="", help="输出文件名附加标签")
    args = ap.parse_args()

    pols = [p.strip() for p in args.policies.split(",") if p.strip()]
    net, params, dwell, ptc = load_assets()
    fv = 1
    facs = {}
    if "rl" in pols:
        facs["rl"], fv = load_rl(args.ckpt)
    adapter = PanelAdapter(net, Featurizer(net, ptc, fv=fv))
    top = lookup_table()
    if "fcfs" in pols:
        facs["fcfs"] = make_fcfs_policy(top)
    if "signaller" in pols:
        facs["signaller"] = make_signaller_replay_policy
    envf = lambda: RewardedDynamicEnv(net, params, ptc, dwell, proactive=True,
                                      reserve_window="wide", control_classes={"1", "2"})
    outdir = config.ARTIFACTS_REPORTS

    # ---------------- cumgap ----------------
    if args.mode == "cumgap":
        days = [d for d in args.days.split(",") if d]
        out = {}
        for date in days:
            w, sp = find_day(date)
            out[date] = {"split": sp}
            for name, fac in facs.items():
                env = envf()
                series = []
                def on_step(env_, _s=series):
                    _s.append((float(env_.engine.now),
                               float(realized_weighted_delay(env_.engine, RCFG))))
                run_instrumented(env, adapter, fac(env), build_day_trains(w, inject=False),
                                 day_horizon(w), on_step=on_step)
                if len(series) > 800:                       # 降采样
                    idx = np.linspace(0, len(series) - 1, 800).astype(int)
                    series = [series[i] for i in idx]
                out[date][name] = series
                print(f"  cumgap {date} {name}: {len(series)} 点, 终值 {series[-1][1]:.0f}", flush=True)
        f = outdir / f"interpret_cumgap{('_'+args.tag) if args.tag else ''}.json"
        f.write_text(json.dumps(out), encoding="utf-8")
        print("[out]", f)

    # ---------------- trace ----------------
    elif args.mode == "trace":
        days = [d for d in args.days.split(",") if d]
        for date in days:
            w, sp = find_day(date)
            trains0 = build_day_trains(w, inject=False)
            planned = {t.train_id: list(t.route_chain) for t in trains0}
            for name, fac in facs.items():
                env = envf()
                recs = []
                def on_dec(env_, enc, dec, idx, _r=recs):
                    s = enc["slots"][idx]
                    eng = env_.engine
                    row = {"t": float(eng.now), "kind": s.kind,
                           "train": getattr(s, "train_id", None),
                           "route": getattr(s, "route_id", None),
                           "n_opt": len(dec.options),
                           "await": sum(1 for x in eng.trains.values()
                                        if x.awaiting_entry and not x.background),
                           "pend": sum(1 for x in eng.trains.values()
                                       if x.pending_request is not None and not x.entered
                                       and not x.background)}
                    if s.kind == "set" and s.train_id in planned:
                        t_ = eng.trains.get(s.train_id)
                        i_ = len(t_.actual_chain) if t_ else 0
                        pc = planned[s.train_id]
                        row["is_planned"] = bool(i_ < len(pc) and pc[i_] == s.route_id)
                        sig = eng._decision_signal(t_) if (t_ and t_.actual_chain) else None
                        row["is_lookup"] = bool(top.get(sig) == s.route_id) if sig else None
                    _r.append(row)
                eng = run_instrumented(env, adapter, fac(env),
                                       build_day_trains(w, inject=False), day_horizon(w),
                                       on_decision=on_dec)
                rows = train_rows(eng)
                f = outdir / f"interpret_trace_{date}_{name}.json"
                f.write_text(json.dumps({"date": date, "split": sp, "policy": name,
                                         "decisions": recs, "trains": rows}), encoding="utf-8")
                print(f"[out] {f} ({len(recs)} 决策)", flush=True)

    # ---------------- leadtime ----------------
    elif args.mode == "leadtime":
        days = operating_days(args.split)[:args.max_days]
        out = {}
        for w in days:
            date = str(w["start"])[:10]
            out[date] = {}
            for name, fac in facs.items():
                env = envf()
                eng = run_instrumented(env, adapter, fac(env),
                                       build_day_trains(w, inject=False), day_horizon(w))
                out[date][name] = train_rows(eng)
            print(f"  leadtime {date} done", flush=True)
        f = outdir / f"interpret_leadtime_{args.split}.json"
        f.write_text(json.dumps(out), encoding="utf-8")
        print("[out]", f)

    # ---------------- sensitivity(D79.3, 取代粗糙的置零) ----------------
    # 方法学修正: 置零不是中性值(delay_min=0 = "正点", 是有含义的分布内值!)。改用**批内置换**
    # (把该特征列在收集到的真实状态间随机重排 = 破坏该特征与状态的关联但保持边际分布)+
    # **动作层敏感度**(argmax 翻转率 / 选中动作 logit 变化)—— 比终局延误敏感得多(环境冗余会吸收
    # 小决策差, 造成"看不出效果"的假象)。
    elif args.mode == "sensitivity":
        assert "rl" in facs, "sensitivity 需要 rl 策略"
        import torch
        from derbysim.learn.panel import collate_panels
        from derbysim.learn.panel_models import to_tensors
        encs = []
        days = operating_days(args.split)[:args.max_days]
        rl_fac = facs["rl"]
        for w in days:
            env = envf()
            act = rl_fac(env)
            def on_dec(env_, enc, dec, idx, _e=encs):
                if len(_e) < 6000:
                    _e.append(enc)
            run_instrumented(env, adapter, act, build_day_trains(w, inject=False),
                             day_horizon(w), on_decision=on_dec)
            if len(encs) >= 6000:
                break
        print(f"[sensitivity] 收集 {len(encs)} 个真实决策状态")
        pol = rl_fac(None).policy if hasattr(rl_fac(None), "policy") else None
        # TorchPanelDriver 持有 policy; 取其内部网络
        drv = rl_fac(None)
        net_pol = getattr(drv, "policy", None) or getattr(drv, "model", None)
        assert net_pol is not None, "拿不到 driver 内部网络"
        base_c = adapter.fz.n_groups + 6
        groups = {
            "lateness(dec0+panel0)": {"x_dec": [0], "x_panel": [0]},
            "fine_time_v2(dec-1)": {"x_dec": [adapter.dec_dim - 1]},
            "bg_foresight_v2(cand-1)": {"x_cand": [adapter.cand_dim - 1]},
            "occupancy(occ+first)": {"x_cand": [base_c + 3, base_c + 4]},
            "route_prior": {"x_cand": [base_c + 6]},
            "tod(sin,cos)": {"x_dec": [1 + 8 + 7 + 1, 1 + 8 + 7 + 2]},
        }
        rng = np.random.default_rng(0)
        B = 256
        def logits_of(enc_list, cols=None, perm=None):
            outs = []
            for i0 in range(0, len(enc_list), B):
                chunk = enc_list[i0:i0 + B]
                if cols:
                    chunk2 = []
                    for j, e in enumerate(chunk):
                        e2 = dict(e)
                        for key, idxs in cols.items():
                            src = enc_list[perm[i0 + j]][key]
                            arr = np.array(e2[key], copy=True)
                            sarr = np.asarray(src)
                            for c in idxs:
                                # 用置换来源状态该列的均值填充(行数可能不同 → 按均值移植)
                                arr[..., c] = float(sarr[..., c].mean()) if sarr.size else 0.0
                            e2[key] = arr
                        chunk2.append(e2)
                    chunk = chunk2
                b = to_tensors(collate_panels(chunk), torch.device("cpu"))
                with torch.no_grad():
                    lg, _v, _aux = net_pol(b)
                    lg = lg.masked_fill(b["opt_mask"] < 0.5, float("-inf"))
                outs.append(lg)
            return outs
        base_out = logits_of(encs)
        base_act = [int(l.argmax(dim=1)[j]) for l in base_out for j in range(l.shape[0])]
        results = {}
        for g, cols in groups.items():
            perm = rng.permutation(len(encs))
            out = logits_of(encs, cols, perm)
            acts = [int(l.argmax(dim=1)[j]) for l in out for j in range(l.shape[0])]
            flip = float(np.mean([a != b_ for a, b_ in zip(acts, base_act)]))
            results[g] = {"flip_rate": flip}
            print(f"  [{g}] argmax 翻转率 {flip:.3f}")
        f = outdir / f"interpret_sensitivity_{args.split}.json"
        f.write_text(json.dumps(results), encoding="utf-8")
        print("[out]", f)

    # ---------------- permimport ----------------
    elif args.mode == "permimport":
        assert "rl" in facs, "permimport 需要 rl 策略"
        from derbysim.learn.rollout import as_factory, evaluate_policy
        fz = adapter.fz
        base_c = fz.n_groups + 6
        groups = {
            "none(基准)": {},
            "延误特征(dec0+panel0)": {"x_dec": [0], "x_panel": [0]},
            "细时刻fv2(dec末)": {"x_dec": [adapter.dec_dim - 1]},
            "背景前瞻fv2(cand末)": {"x_cand": [adapter.cand_dim - 1]},
            "占用(occ_frac+first_occ)": {"x_cand": [base_c + 3, base_c + 4]},
            "候选先验(prior)": {"x_cand": [base_c + 6]},
        }
        eps = [(str(w["start"])[:16], (lambda w=w: build_day_trains(w)), day_horizon(w))
               for w in operating_days(args.split)]
        rl_fac = facs["rl"]
        results = {}
        for gname, cols in groups.items():
            def make_masked(inner_fac, _cols=cols):
                def fac(env):
                    inner = inner_fac(env)
                    def act(enc):
                        if _cols:
                            enc = dict(enc)
                            for key, idxs in _cols.items():
                                arr = np.array(enc[key], copy=True)
                                arr[..., idxs] = 0.0
                                enc[key] = arr
                        return inner(enc)
                    return act
                return fac
            ev = evaluate_policy(envf, adapter, make_masked(rl_fac), eps)
            ev.pop("eval_episodes", None)
            results[gname] = ev
            print(f"  [{gname}] 完成 {ev['eval_completion_rate']:.4f} | "
                  f"延误 {ev['eval_weighted_secondary_delay_s']:>10.0f} | "
                  f"WAIT {ev['eval_frac_wait']:.0%}", flush=True)
        f = outdir / f"interpret_permimport_{args.split}.json"
        f.write_text(json.dumps(results), encoding="utf-8")
        print("[out]", f)


if __name__ == "__main__":
    main()
