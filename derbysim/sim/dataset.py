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


def _chain_continuity(net):
    from collections import defaultdict
    succ = defaultdict(set)
    for r in net.routes.values():
        for i in range(1, len(r.tc_sequence)):
            succ[r.tc_sequence[i - 1]].add(r.tc_sequence[i])
    pair = {}
    for pl in net.platforms.values():
        e = list(pl.end_signals.values())
        if len(e) == 2:
            pair[e[0]] = e[1]; pair[e[1]] = e[0]
    return succ, pair


def chain_is_continuous(route_chain, net, succ=None, pair=None):
    """物理连续 = 每段进路首TC 是上段末TC 的后继(tc_succ), 或站台配对端折返. 断链=H6乱序/缺段(数据脏)."""
    if succ is None:
        succ, pair = _chain_continuity(net)
    last_tc = None; last_end = None
    for rid in route_chain:
        r = net.routes.get(rid)
        if not r or not r.tc_sequence:
            continue
        if last_tc is not None:
            cont = (r.signal == last_end                            # 常规边界: 下段起始信号=上段end
                    or r.tc_sequence[0] in succ.get(last_tc, ())    # 物理续接(信号名不同但TC连续)
                    or (last_end is not None and pair.get(last_end) == r.signal))  # 站台配对折返
            if not cont:
                return False
        last_tc = r.tc_sequence[-1]; last_end = r.end_signal
    return True


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


def build_cif_index(occ_df: pd.DataFrame) -> dict:
    """**按趟**的 CIF 索引: `{headcode: [(td_first_epoch, {...}), ...]}`, 由调用方用该趟车自己的
    首事件时刻**就近匹配**。这是 `build_cif_map` 的正确替代, 一次修掉两个缺陷:

    ① **同日多趟互相覆盖(既有缺陷, 3h 窗口路径同样中招)**: `build_cif_map` 是 `out[hc] = {...}`,
       同一 headcode 同一天的多趟后者覆盖前者 → 早班那趟拿到晚班的计划时刻。全量实测:
       (headcode,date) 组合 3.0% 是同日多趟, 涉及 **6.0% 的趟次(1573 趟)**, 其中 **814 趟**
       拿到的是同日最后一趟的 `planned_ref`(如 1M00 11-06 09:31 那趟拿到 22:43, 偏 13.2h)。
       影响 `delay_min` / on-time, 不影响路径/占用/安全。
    ② **跨午夜**: 运营日 03:00→次日 03:00 会跨两个自然日; 本索引覆盖全部日期, 按时刻就近匹配,
       天然无需调用方指定日期。

    **尊重既有真值标记**: `alignment` 已给出 `cif_reliable`(= `cif_time_gap_min <= 30min`;
    False 多为 H6 headcode 复用导致的错服务匹配, 如 1F82 首事件 00:22 却匹到 23:09 的 CIF)。
    不可靠时**不给 `planned_ref`**(置 None), 宁缺勿错 —— 拿错 22.8h 的计划时刻比没有更糟。
    """
    out: dict = {}
    for r in occ_df.to_dict("records"):
        hc = r.get("headcode")
        if not hc:
            continue
        ds = str(r.get("date"))
        bp = r.get("cif_platform")
        try:
            bp = int(str(bp).replace("PLT", "").strip()) if bp not in (None, "", "None") else None
        except Exception:
            bp = None
        ref = None
        if bool(r.get("cif_reliable")):                 # 不可靠就不给 planned_ref
            for c in ("cif_arr", "cif_dep", "cif_pass"):
                v = r.get(c)
                if v is not None and str(v) not in ("None", "nan", ""):
                    try:
                        ref = pd.to_datetime(ds + " " + str(v)).timestamp()
                        break
                    except Exception:
                        ref = None
        t0 = pd.to_datetime(r.get("td_first"), errors="coerce")
        if pd.isna(t0):
            continue
        out.setdefault(hc, []).append(
            (t0.timestamp(), {"booked_platform": bp, "planned_ref": ref,
                              "service_class": str(hc)[0] if hc else ""}))
    for hc in out:
        out[hc].sort(key=lambda kv: kv[0])
    return out


def build_cif_map_days(occ_df: pd.DataFrame, dates) -> dict:
    """**跨日安全**的 CIF map: `{(headcode, 'YYYY-MM-DD'): {...}}`。

    `build_cif_map(occ, date)` 只按 headcode 键、且把 `planned_ref` 锚在**那一个自然日**上。
    运营日若跨午夜(如 03:00→次日 03:00), 午夜后那段车会拿到**差 24 小时**的计划时刻,
    `delay_min` / on-time 全错(2026-07-31 实测确认)。故按 (headcode, 日期) 双键,
    由调用方用**该趟车自己首个事件的日期**去查。
    """
    out: dict = {}
    for d in dates:
        ds = str(d)
        for hc, v in build_cif_map(occ_df, ds).items():
            out[(hc, ds)] = v
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
    succ, pair = _chain_continuity(net)
    dropped = [b[0].train_id for b in built
               if not chain_is_continuous(b[0].route_chain, net, succ, pair)]
    trains = [b[0] for b in built
              if chain_is_continuous(b[0].route_chain, net, succ, pair)]   # 丢断链(H6乱序)车
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
        "n_dropped_broken_chain": len(dropped),
    }
    return records, stats
