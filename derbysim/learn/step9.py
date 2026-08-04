"""D33 Step 9 诊断驱动器: 单点偏离可恢复性(①) / 时机轴(②) / 卡住车再提供审计(③).

设计口径(D40): 结论一律在 train+val 上算, test 不碰。单点偏离后**其余决策仍按信号员日志
回放** —— 这是"单个扰动后环境/区域能否自愈"的最干净因果读数(用户定)。

本模块只提供可复用的驱动器与普查; `scripts/22_step9_diagnostics.py` 负责编排与落时间戳 JSON。
"""
from __future__ import annotations

from collections import Counter

from derbysim.learn.rollout import (episode_metrics, make_signaller_replay_policy,
                                    run_episode)
from derbysim.sim.deadlock import would_create_deadlock


def _reset(env, trains, horizon):
    try:
        return env.reset(trains, horizon=horizon)
    except TypeError:
        return env.reset(trains)


def _set_options_by_train(enc) -> dict:
    """panel 里每辆车的 goal-safe∩deadlock-safe set 选项(引擎已过滤) → {train_id: [route_id,...]}."""
    out: dict = {}
    for s in enc["slots"]:
        if s.kind == "set":
            out.setdefault(s.train_id, []).append(s.route_id)
    return out


def _geometry_routes(engine, train) -> list[str]:
    """复刻 DynamicEngine._legal 的几何阶段(信号出口 ∪ 物理续接), 供 ③ 漏斗分层。"""
    routes = []
    signal = engine._decision_signal(train)
    if signal is not None:
        sigs = {signal}
        if signal in engine._pair:
            sigs.add(engine._pair[signal])
        for s in sigs:
            routes.extend(engine.net.routes_from(s))
    if train.path:
        for tc in engine._tc_succ.get(train.path[-1], ()):
            routes.extend(engine._routes_by_first.get(tc, ()))
    seen, out = set(), []
    for r in routes:
        if r.route_id not in seen:
            seen.add(r.route_id)
            out.append(r.route_id)
    return out


# ----------------------------------------------------------------------------
# 扰动回放策略: 信号员回放 + 在第 inject_at 个 panel 注入一次扰动
# ----------------------------------------------------------------------------
def make_perturbed_replay_policy(env, inject_at: int, mode: str, alt_rank: int = 0):
    """信号员回放, 但在第 `inject_at` 个 panel(从 0 计)注入一次扰动, 之后照常按日志回放。

    mode='reroute': 把信号员本该办的车改办**另一条 goal-safe 替代路**(按 route_id 排序取
                    第 alt_rank+1 条非日志); 无替代路时不注入, 退回日志回放。
    mode='wait'   : 该 panel 强制 WAIT(options[0] 恒为 WAIT)。
    注入信息记在 `act.state['info']`, 供上层归因。
    """
    base = make_signaller_replay_policy(env)
    state = {"n": 0, "injected": False, "info": None}

    def act(enc):
        i = state["n"]
        state["n"] += 1
        if i == inject_at and not state["injected"]:
            base_idx, _, _ = base(enc)
            base_slot = enc["slots"][base_idx]
            if mode == "wait":
                state["injected"] = True
                state["info"] = {"panel": i, "mode": "wait",
                                 "displaced_train": base_slot.train_id if base_slot.kind == "set" else None,
                                 "displaced_route": base_slot.route_id if base_slot.kind == "set" else None}
                return 0, 0.0, 0.0
            if mode == "reroute" and base_slot.kind == "set":
                opts = sorted({s.route_id for s in enc["slots"]
                               if s.kind == "set" and s.train_id == base_slot.train_id})
                alts = [r for r in opts if r != base_slot.route_id]
                if alts:
                    chosen = alts[min(alt_rank, len(alts) - 1)]
                    for j, s in enumerate(enc["slots"]):
                        if s.kind == "set" and s.train_id == base_slot.train_id and s.route_id == chosen:
                            state["injected"] = True
                            state["info"] = {"panel": i, "mode": "reroute",
                                             "deviated_train": base_slot.train_id,
                                             "logged_route": base_slot.route_id,
                                             "forced_route": chosen,
                                             "n_alternatives": len(alts)}
                            return j, 0.0, 0.0
            # 无可注入(base 是 WAIT 或无替代路) → 退回日志回放
        return base(enc)

    act.state = state
    return act


def run_injection(env, adapter, trains, horizon, inject_at: int, mode: str,
                  alt_rank: int = 0, max_decisions: int = 200000) -> dict:
    """跑一局带单点扰动的信号员回放, 返回闭环指标 + 注入归因 + 偏离车是否完成。"""
    pol = make_perturbed_replay_policy(env, inject_at, mode, alt_rank)
    res = run_episode(env, adapter, pol, trains, horizon=horizon, max_decisions=max_decisions)
    info = pol.state["info"]
    eng = env.engine
    deviated = None
    if info is not None:
        tid = info.get("deviated_train") or info.get("displaced_train")
        t = eng.trains.get(tid) if tid else None
        if t is not None:
            deviated = {"train_id": tid,
                        "completed": bool(t.reached_exit or t.reached_terminal),
                        "done": bool(t.done),
                        "termination_reason": t.termination_reason}
    return {"metrics": res.metrics, "info": info, "deviated": deviated,
            "injected": bool(info is not None)}


# ----------------------------------------------------------------------------
# 普查: 一遍无扰动回放, 统计可注入点规模(定 ①② 扫描规模) + baseline 指标
# ----------------------------------------------------------------------------
def census_signaller(env, adapter, trains, horizon, max_decisions: int = 200000) -> dict:
    """无扰动信号员回放一遍, 记录每个 set panel 的 goal-safe 选项数。

    返回: baseline 闭环指标 + 分叉规模(branchable=信号员 set 且该车 ≥2 goal-safe 选项 → ① 注入点;
    set_idx=所有 set panel → ② WAIT 注入点)。索引可复现, 供后续按 (day,k) 逐点扫描。
    """
    obs, _ = _reset(env, trains, horizon)
    base = make_signaller_replay_policy(env)
    n_set = n_wait = 0
    setopt = []
    branchable_idx: list[int] = []
    set_idx: list[int] = []
    i = 0
    while not env.done and i < max_decisions:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        by_train = _set_options_by_train(enc)
        idx, _, _ = base(enc)
        slot = enc["slots"][idx]
        if slot.kind == "set":
            n_set += 1
            nopt = len(by_train.get(slot.train_id, []))
            setopt.append(nopt)
            set_idx.append(i)
            if nopt >= 2:
                branchable_idx.append(i)
        else:
            n_wait += 1
        obs, _r, done, _ = env.step(dec.options[idx])
        i += 1
        if done:
            break
    metrics = episode_metrics(env.engine, env.cfg, env.plat_tc_to_pid,
                              getattr(env, "horizon", None))
    return {
        "n_panels": i, "n_set": n_set, "n_wait": n_wait,
        "n_branchable": len(branchable_idx), "n_set_panels": len(set_idx),
        "setopt_hist": dict(sorted(Counter(setopt).items())),
        "branchable_idx": branchable_idx, "set_idx": set_idx,
        "baseline_metrics": metrics,
    }


def census_stuck(env, adapter, trains, horizon, policy_factory,
                 max_decisions: int = 200000) -> dict:
    """③ 卡住车再提供审计: 每个 panel 对每辆 awaiting 车走几何→目标→死锁前瞻漏斗, 分类无-set 成因。

    注: 被别车预留占用的进路**仍会作为 set 选项提供**(请求会排队, 不算卡住), 故"排队未重试"
    在新引擎里不构成无-set 成因; 若 has_set 但闭环仍差, 属策略质量而非建模缺陷。
    """
    obs, _ = _reset(env, trains, horizon)
    pol = policy_factory(env)
    tot = {"awaiting_rows": 0, "no_geometry": 0, "goal_filtered": 0,
           "deadlock_filtered": 0, "has_set": 0, "n_panels": 0}
    i = 0
    while not env.done and i < max_decisions:
        dec = env.decision()
        if dec is None or obs is None:
            break
        enc = adapter.encode(obs, dec)
        eng = env.engine
        for tr in eng._actionable():
            tot["awaiting_rows"] += 1
            geom = _geometry_routes(eng, tr)
            goal = [r for r in geom if eng.reach.route_keeps_goal(
                eng.net.routes[r], tr.goal, tr.served_platform)]
            safe = [r for r in goal if not would_create_deadlock(
                eng.res, eng.net.routes[r], tr.train_id)]
            if not geom:
                tot["no_geometry"] += 1
            elif not goal:
                tot["goal_filtered"] += 1
            elif not safe:
                tot["deadlock_filtered"] += 1
            else:
                tot["has_set"] += 1
        idx, _, _ = pol(enc)
        obs, _r, done, _ = env.step(dec.options[idx])
        i += 1
        if done:
            break
    tot["n_panels"] = i
    return tot
