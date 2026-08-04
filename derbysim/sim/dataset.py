"""Phase 1 BC 决策数据集 —— DerbyEnv 的副产物 (§14.6/§14.9).

回放模式逐决策点吐出 (状态, 掩码, 被选动作=日志进路, 弱标签). 与 RL 环境**同一份状态契约**
(同一个 DerbyEnv._build_state), 从源头杜绝 BC 训练态 vs RL 部署态失配.

弱标签(D11, 弱监督唯一路径): 无 ARS/人工显式标签 → 用可得信号推断"是否偏离计划":
    platform_deviation = 所选进路导向站台 != CIF booked 站台 (booked 可得时).
仅作弱信号, 噪声靠多信号交叉+保守阈值(此处只落一个可得项, 后续可加 offroute/延误).
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import median

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


def position_order_indices(et, occ_first, net, succ=None, pair=None) -> tuple[int, ...]:
    """Conservatively recover traversal order from observed Track occupations.

    ``Panel_Request`` records request time, not passage order: a downstream route
    may be preset before its upstream feeder.  Reordering is accepted only when
    route ids are unique, every route has an observed TC unique to that route,
    and the resulting order strictly reduces physical discontinuities.  Repeated
    shunt moves or insufficient observations remain unchanged rather than being
    guessed and are handled by the existing D31 boundary split.
    """
    if succ is None:
        succ, pair = _chain_continuity(net)
    n = len(et.route_chain)
    original = tuple(range(n))
    if n < 2 or len(set(et.route_chain)) != n:
        return original
    routes = [net.routes.get(rid) for rid in et.route_chain]
    if any(r is None or not r.tc_sequence for r in routes):
        return original
    tc_counts = Counter(tc for r in routes for tc in r.tc_sequence)
    keys = []
    for i, route in enumerate(routes):
        times = [occ_first[tc] for tc in route.tc_sequence
                 if tc_counts[tc] == 1 and tc in occ_first]
        if not times:
            return original
        keys.append((float(median(times)), min(times), max(times), i))
    order = tuple(k[-1] for k in sorted(keys))
    if order == original:
        return original

    def n_breaks(indices):
        chain = [routes[i].route_id for i in indices]
        return sum(not chain_is_continuous(chain[j:j + 2], net, succ, pair)
                   for j in range(len(chain) - 1))

    return order if n_breaks(order) < n_breaks(original) else original


def split_broken_chains(built, net, succ=None, pair=None, plat_tc_to_pid=None):
    """先按观测位置保守复序，再把真断链切成独立车辆段(D31+D35)。

    **为什么改**: 原做法 `chain_is_continuous` 为假就丢整车, 10 天实测丢 **303/2056 = 14.7%**,
    且**丢得极不均匀** —— 1x/2x 客运只丢 5.5%/5.6%, 而 **5x ECS 丢 45.8%**、0x 单机 32.3%、
    6x 货运 17.5%。诊断: 被丢的 ECS 里 **159/169** 的断点形如 `TEDJ→TEDG`, 即车进 **Etches Park
    车辆段**、在我们**没有建模**的库内区段停留、再出库。**这是建模边界不是脏数据** ——
    进库和出库本来就是**两次独立的调动**, 把它们当成一条断链丢掉, 等于系统性删掉近一半的 ECS,
    而 ECS 正是终点站咽喉冲突的重要来源(冲突强度被低估, BC 数据集与 RL 场景同时受影响)。
    自动补链不可行(303 个断点: 唯一可补 1 / 多解 216 / 无解 86 —— 多解等于我们替信号员编走行)。

    **D35 / D33 step 5 (2026-08-04)**: Panel_Request 可先办下游后办上游，请求时间序不是
    物理通过序。旧 D31 会把这类合法预办误切成碎片。切分前调用 `position_order_indices`，只有
    “route id 唯一 + 每段有独占 Track 观测 + 复序严格减少物理断点”三条同时成立才复序；重复调车
    或证据不足时不猜，仍走原 D31 边界切分。请求时间跟随各自 route 移动，保留预办时机真值。

    切分规则(与 `build_env_trains_for_window` 的过滤规则保持一致, 否则两条构造路径会分叉):
      - 每段自成一辆 `EnvTrain`, id 记作 `<原id>@<段序>`(原车不断链时**保持原 id 不变**);
      - 段内 `entry` = 该段首 TC 的**观测占用时刻**; 无观测占用或 `len(path)<2` 的段丢弃;
      - `planned_ref` 只给**第一段**(计划时刻属于整趟, 给后续段会凭空造出准点/晚点);
      - `stop_pids` 按站台 TC 落在哪一段就归哪一段; `booked_platform`/`service_class` 全段继承。
    """
    from derbysim.sim.env import EnvTrain
    if succ is None:
        succ, pair = _chain_continuity(net)
    out = []
    for et, occ_first in built:
        order = position_order_indices(et, occ_first, net, succ, pair)
        reordered = order != tuple(range(len(et.route_chain)))
        route_chain = [et.route_chain[i] for i in order]
        req_times = [et.req_times[i] for i in order]
        cuts = [0]
        last_tc = last_end = None
        for i, rid in enumerate(route_chain):
            r = net.routes.get(rid)
            if not r or not r.tc_sequence:
                continue
            if last_tc is not None and not (
                    r.signal == last_end
                    or r.tc_sequence[0] in succ.get(last_tc, ())
                    or (last_end is not None and pair.get(last_end) == r.signal)):
                cuts.append(i)
            last_tc = r.tc_sequence[-1]; last_end = r.end_signal
        if len(cuts) == 1 and not reordered:
            out.append((et, occ_first))                       # 本来就连续 → 原样, id 不变
            continue
        cuts.append(len(route_chain))
        for k in range(len(cuts) - 1):
            lo, hi = cuts[k], cuts[k + 1]
            chain = list(route_chain[lo:hi])
            times = list(req_times[lo:hi])
            path = []
            for rid in chain:
                r = net.routes.get(rid)
                for tc in (r.tc_sequence if r else ()):
                    if not path or path[-1] != tc:
                        path.append(tc)
            if len(path) < 2 or path[0] not in occ_first:     # 同 build_env_trains_for_window
                continue
            # 站台停站按站台 TC 落在哪一段就归哪一段(段外的停站不属于本段)。
            # 无 plat_tc_to_pid 时保守起见全段继承(宁可多停不可漏停 —— 漏停会被 −P_nostop 误罚)。
            pids = set(getattr(et, "stop_pids", set()))
            if plat_tc_to_pid is not None:
                seen = {plat_tc_to_pid[tc] for tc in path if tc in plat_tc_to_pid}
                pids &= seen
            seg_id = et.train_id if len(cuts) == 2 else f"{et.train_id}@{k}"
            seg = EnvTrain(seg_id, chain, times, occ_first[path[0]], pids)
            seg.service_class = et.service_class
            seg.booked_platform = getattr(et, "booked_platform", None)
            seg.planned_ref = getattr(et, "planned_ref", None) if k == 0 else None
            out.append((seg, occ_first))
    return out


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
    # D31(2026-08-01): 断链**切分成多段独立的车**, 不再整车丢弃(丢法对 ECS 有 45.8% 的选择性偏差,
    # 成因是 Etches Park 库内未建模 —— 进库/出库本就是两次独立调动)。见 `split_broken_chains`。
    dropped = [b[0].train_id for b in built
               if not chain_is_continuous(b[0].route_chain, net, succ, pair)]
    reordered = [b[0].train_id for b in built
                 if position_order_indices(b[0], b[1], net, succ, pair)
                 != tuple(range(len(b[0].route_chain)))]
    trains = [b[0] for b in split_broken_chains(built, net, succ, pair, plat_tc_to_pid)]
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
        "n_position_reordered": len(reordered),
    }
    return records, stats
