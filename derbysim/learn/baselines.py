"""规则/启发式派单基线(免训练), 供 scripts/13 基线阶梯与 scripts/24 对照复用(SSOT)。

统一口径: 除 `signaller_replay`(会策略性 WAIT)外, 这些启发式**不做策略性 WAIT** —— 有可办的车就办;
差别只在**选路**(lookup 最频 / booked 计划路)与**服务顺序**(first / FCFS 到达序 / FSFS 计划时刻序)。
所有候选都来自面板已过滤的安全选项(mask ∩ 可达 ∩ in-scope, D41), 故"安全通过"结构性保证, 与基线无关。

对照定位(D26/D30 基线阶梯): 打不过这些零学习基线的学习型方法没有价值; 尤其 `lookup` 是 held-out
最强安全基线(≈信号员, +8.7% 延误, D42), Step 10 的 PPO 必须打败它。

每个 `make_*` 返回 `factory(env) -> act(enc) -> (option_idx, logp, value)`, 与 rollout 的策略约定一致。
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from derbysim import config

_INF = float("inf")


def lookup_table(split: str = "train", dataset: str = "bc_dataset_all_d35.parquet") -> dict:
    """每信号最频进路 —— 只用数据集的 `train` 段拟合(holdout=test 不碰, D43)。默认 D35。"""
    bc = pd.read_parquet(config.ARTIFACTS / "dataset" / dataset)
    tr = bc[bc["split"] == split]
    cnt: dict = {}
    for sig, rid in zip(tr["signal"], tr["chosen_route"]):
        cnt.setdefault(sig, Counter())[rid] += 1
    return {s: c.most_common(1)[0][0] for s, c in cnt.items()}


def _set_slots_by_train(enc) -> dict:
    """{train_id: [面板里该车的 set 选项下标, ...]}(只含 set, 已是安全候选)。"""
    out: dict = {}
    for i, s in enumerate(enc["slots"]):
        if s.kind == "set":
            out.setdefault(s.train_id, []).append(i)
    return out


def _lookup_slot_for_train(enc, env, tid, slot_idxs, top):
    """该车若有'最频路'选项就返回它, 否则返回它的第一条 set 选项下标。"""
    t = env.engine.trains.get(tid)
    sig = env.engine._decision_signal(t) if t is not None else None
    for i in slot_idxs:
        if sig is not None and top.get(sig) == enc["slots"][i].route_id:
            return i
    return slot_idxs[0]


def make_lookup_policy(top):
    """lookup 最频路: 办面板里第一个'其最频路可选'的车; 都不匹配则第一条合法。即时, 不策略性等。
    与 scripts/24 的 `make_lookup_factory` 逐位一致, 保证 scripts/13 的 lookup 基线 = 已报的 +8.7%。
    """
    def factory(env):
        def act(enc):
            first = None
            for i, s in enumerate(enc["slots"]):
                if s.kind != "set":
                    continue
                if first is None:
                    first = i
                t = env.engine.trains.get(s.train_id)
                sig = env.engine._decision_signal(t) if t is not None else None
                if sig is not None and top.get(sig) == s.route_id:
                    return i, 0.0, 0.0
            return (first, 0.0, 0.0) if first is not None else (0, 0.0, 0.0)
        return act
    return factory


def _ordered_policy(top, keyfn):
    """按 keyfn(train) 升序挑'最靠前'的可办车, 走它的最频路。FCFS/FSFS 共用。"""
    def factory(env):
        def act(enc):
            by = _set_slots_by_train(enc)
            if not by:
                return 0, 0.0, 0.0
            trains = env.engine.trains

            def k(tid):
                t = trains.get(tid)
                return keyfn(t) if t is not None else _INF
            tid = min(by, key=k)
            return _lookup_slot_for_train(enc, env, tid, by[tid], top), 0.0, 0.0
        return act
    return factory


def make_fcfs_policy(top):
    """FCFS 先到先服务: 多辆可办时先办**入区最早**的车(entry_time), 走其最频路。"""
    return _ordered_policy(top, lambda t: getattr(t, "entry_time", _INF))


def make_fsfs_policy(top):
    """FSFS 先计划先服务: 按**计划参考时刻**(planned_ref)升序, 缺失退回 entry_time。"""
    def key(t):
        pr = getattr(t, "planned_ref", None)
        return pr if pr is not None else getattr(t, "entry_time", _INF)
    return _ordered_policy(top, key)


def make_route_reference(top, mode: str = "schedule"):
    """D59 — RL 选路 KL 的**规则参照**(取代旧的冻结神经 BC 作锚)。按 RR: 跟计划走, 只为冲突偏离。

    对面板每个 set-row(一辆待决车 t)给一个**参照进路目标列** + 有效位:
      · mode='schedule'(默认, RR faithful): 优先取该车**计划/booked 进路**(`dec.logged[t]`,
        = 历史逐条请求的那条路)命中的候选列; 若计划路不在本决策候选里, 退回 lookup 最频路 `top[sig]`。
      · mode='lookup': 只用 lookup 最频路(消融)。
    命中不了任何参照 → ref_valid=0(该行不进 KL, 让 RL 自由), 不硬塞。

    返回 `build(enc, dec, env) -> (ref_col[T] int64, ref_valid[T] float32)`, 与 `enc['train_rows']`
    逐行对齐。**纯 numpy / 无 torch / 与引擎无关**: 只读 slots(route_id/col) + dec.logged + lookup 表,
    proactive 面板同样成立(计划进路稳定, 提前预办与到点办用的是同一条 booked 路)。
    """
    use_plan = mode in ("schedule", "both")
    use_lookup = mode in ("schedule", "both", "lookup")

    def build(enc, dec, env):
        rows = enc.get("train_rows", [])
        T = len(rows)
        ref_col = np.zeros(max(T, 1), np.int64)
        ref_valid = np.zeros(max(T, 1), np.float32)
        col_of: dict = {}                                  # tid -> {route_id: col}
        for s in enc["slots"]:
            if s.kind == "set":
                col_of.setdefault(s.train_id, {})[s.route_id] = s.col
        for i, tid in enumerate(rows):
            cols = col_of.get(tid, {})
            if not cols:
                continue
            target = None
            if use_plan and dec is not None:               # 计划/booked 进路(RR: 跟计划走)
                plan = dec.logged.get(tid)
                if plan in cols:
                    target = cols[plan]
            if target is None and use_lookup:              # 兜底: 逐信号最频路(常规走法)
                t = env.engine.trains.get(tid)
                sig = env.engine._decision_signal(t) if t is not None else None
                lk = top.get(sig) if sig is not None else None
                if lk in cols:
                    target = cols[lk]
            if target is not None:
                ref_col[i] = int(target)
                ref_valid[i] = 1.0
        return ref_col, ref_valid

    return build


def make_ars_proxy_policy(top):
    """ARS 代理(自动选路): 一有可办车就办其**计划(booked)进路**(=位置匹配的历史请求), 即时、不策略性等;
    无 booked 选项则退回最频路 / 第一条合法。区别于 lookup(最频路)与 signaller(会策略性 WAIT)。
    """
    def factory(env):
        def act(enc):
            dec = env.decision()
            logged = dec.logged if dec is not None else {}
            first = None
            for i, s in enumerate(enc["slots"]):
                if s.kind != "set":
                    continue
                if first is None:
                    first = i
                if logged.get(s.train_id) == s.route_id:       # booked/计划进路
                    return i, 0.0, 0.0
            if first is None:
                return 0, 0.0, 0.0                             # 无可办 → WAIT
            # 无 booked 选项: 不策略性等, 用最频路兜底
            by = _set_slots_by_train(enc)
            tid = next(iter(by))
            return _lookup_slot_for_train(enc, env, tid, by[tid], top), 0.0, 0.0
        return act
    return factory
