"""Phase 1 BC 决策数据集 —— DerbyEnv 的副产物 (§14.6/§14.9).

回放模式逐决策点吐出 (状态, 掩码, 被选动作=日志进路, 弱标签). 与 RL 环境**同一份状态契约**
(同一个 DerbyEnv._build_state), 从源头杜绝 BC 训练态 vs RL 部署态失配.

弱标签(D11, 弱监督唯一路径): 无 ARS/人工显式标签 → 用可得信号推断"是否偏离计划":
    platform_deviation = 所选进路导向站台 != CIF booked 站台 (booked 可得时).
仅作弱信号, 噪声靠多信号交叉+保守阈值(此处只落一个可得项, 后续可加 offroute/延误).
"""

from __future__ import annotations

import json

import pandas as pd

from derbysim.sim.env import DerbyEnv, build_env_trains_for_window


def build_cif_map(occ_df: pd.DataFrame, date: str) -> dict:
    """{headcode: {booked_platform, planned_ref(epoch s), service_class}} for a given date."""
    sub = occ_df[occ_df["date"].astype(str) == date]
    out: dict[str, dict] = {}
    for r in sub.to_dict("records"):
        hc = r["headcode"]
        bp = r.get("cif_platform")
        try:
            bp = int(str(bp).replace("PLT", "").strip()) if bp not in (None, "", "None") else None
        except Exception:
            bp = None
        ref = None
        for c in ("cif_arr", "cif_dep", "cif_pass"):
            v = r.get(c)
            if v is not None and str(v) not in ("None", "nan", ""):
                try:
                    ref = pd.to_datetime(date + " " + str(v)).timestamp()
                    break
                except Exception:
                    ref = None
        out[hc] = {"booked_platform": bp, "planned_ref": ref,
                   "service_class": str(hc)[0] if hc else ""}
    return out


def _route_platform(net, plat_tc_to_pid: dict, route_id: str):
    """所选进路导向的站台(其 TC 序列命中的第一个站台 TC 的 pid), 无则 None."""
    r = net.routes.get(route_id)
    if not r:
        return None
    for tc in r.tc_sequence:
        if tc in plat_tc_to_pid:
            return plat_tc_to_pid[tc]
    return None


def generate_decisions(net, params, plat_tc_to_pid, dwell_by_platform, td_win,
                       cif_map=None) -> tuple[list[dict], dict]:
    """回放窗口, 逐决策点收集 BC 记录. 返回 (records, stats).

    records[i] = {date/train/signal/time/trigger_kind, n_legal, legal_route_ids,
                  chosen_route(=BC 标签), obs(json), weak_platform_deviation}.
    """
    built = build_env_trains_for_window(td_win, net, dwell_by_platform, plat_tc_to_pid,
                                        cif_map=cif_map)
    trains = [b[0] for b in built]
    byid = {t.train_id: t for t in trains}
    env = DerbyEnv(net, params, plat_tc_to_pid, dwell_by_platform)
    env.reset(trains)
    records: list[dict] = []
    n_real = 0
    while not env.done:
        ctx = env.decision()
        t = byid[ctx.train_id]
        chosen = ctx.logged_route_id                 # 回放: 被选动作 = 日志真实进路
        booked = t.booked_platform
        chosen_plat = _route_platform(net, plat_tc_to_pid, chosen) if chosen else None
        weak_dev = None
        if booked is not None and chosen_plat is not None:
            weak_dev = int(chosen_plat != booked)
        rec = {
            "train_id": ctx.train_id,
            "signal": ctx.signal,
            "time": ctx.now,
            "trigger_kind": ctx.trigger_kind,
            "n_legal": len(ctx.legal_route_ids),
            "legal_route_ids": list(ctx.legal_route_ids),
            "chosen_route": chosen,
            "chosen_platform": chosen_plat,
            "booked_platform": booked,
            "weak_platform_deviation": weak_dev,
            "obs": json.dumps(ctx.obs, default=str),
        }
        records.append(rec)
        if len(ctx.legal_route_ids) > 1:
            n_real += 1
        env.step(chosen)
    stats = {
        "n_decisions": len(records),
        "n_real_choices": n_real,
        "n_mask_misses": len(env.engine.mask_misses) if env.engine else 0,
        "n_trains": len(trains),
    }
    return records, stats
