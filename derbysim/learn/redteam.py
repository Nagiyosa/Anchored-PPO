"""M12b 红队：一组**故意钻空子**的脚本策略，把"发现 reward hacking"变成常规检查。

**为什么需要**: 2026-07-30 首次真训, agent 的评估回报逼近人类而完成率只有 0.217 —— 它学到的是
漏洞不是调度。当时**所有单元测试全绿、determinism 全绿**: 这类目标错配只在**闭环长程交互**下显形。
逐个症状去修是打地鼠; 正确做法是**立不变式 + 常备红队**, 每次改奖励/环境都重跑一遍。

**不变式**(见 `tests/test_redteam.py`):
  1. **信号员回放**必须优于所有红队策略(人类是对照锚, D23)。
  2. **没有以钻空子为设计目的的策略可以优于参照线** —— 一个不好好派路的策略拿高分 = 奖励错了。
     (2026-08-01 措辞修正: 原文写"任何退化策略"。时间闸门修好后 `oscillate_set_cancel` 已**不再退化**
      —— 它实际做的是"撤掉卡住的请求再试一次", 那是粗糙但**正当**的调度行为; 全 11 个 val 运营日
      复核显示它确实**没有**赢过"有路就办"(0.294 vs 0.297)。不变式按**设计意图**判, 不按策略名单判。)
  3. 没有策略能把延误刷成 ~0 却完成率很低(时钟被冻住的征兆)。
  4. 没有策略能靠**提前结束**(死锁/截断)逃掉欠账。

**两条已立的结构性规则**(比逐个补丁更根本):
  - **终局结算与终止原因无关**: 自然跑完 / 死锁判负 / 截断, 同一套账都结; `p_deadlock` 是额外罚
    而非替代罚。→ 关掉整类"提前结束逃费"。
  - **零仿真时间内不得无限动作**: `cancel` 要求请求已排队 >0 仿真时间(现实里信号员也不可能
    同一瞬间排上又撤掉)。→ 关掉整类"原地空转"。
"""
from __future__ import annotations

import numpy as np

from derbysim.learn.rollout import make_greedy_set_policy, make_signaller_replay_policy


def _occupied_tcs(engine) -> set:
    return {t.path[t.pos] for t in engine.trains.values()
            if not t.done and t.pos < len(t.path)}


def always_wait_factory(_env):
    """拖延: 什么都不办。测"不作为是否比作为划算"。"""
    return lambda enc: (0, 0.0, 0.0)


def oscillate_factory(_env):
    """零时间 churn: 能撤就撤、否则就办 —— 曾靠此把时钟冻住、延误刷成 0。"""
    def act(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "cancel":
                return i, 0.0, 0.0
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act


def cancel_maximiser_factory(_env):
    """最大化撤销: 每有机会就撤, 但不主动补办 —— 测撤销本身是否成为逃避手段。"""
    def act(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "cancel":
                return i, 0.0, 0.0
        sets = [i for i, s in enumerate(enc["slots"]) if s.kind == "set"]
        return (sets[0], 0.0, 0.0) if sets else (0, 0.0, 0.0)
    return act


def priority_only_factory(env, keep=("1", "2")):
    """只服务高优先级车: 利用 λ 加权, 把低优先级(货/ECS)晾着换低加权延误。"""
    def act(enc):
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                t = env.engine.trains.get(s.train_id)
                if t is not None and str(t.service_class) in keep:
                    return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act


def easy_only_factory(env):
    """只跑好跑的车: 候选进路完全没被占用才办 —— 测"晾着难车"是否划算(M10 §1 就担心这个)。"""
    def act(enc):
        occ = _occupied_tcs(env.engine)
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                r = env.net.routes.get(s.route_id)
                if r and not (set(r.tc_sequence) & occ):
                    return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act


def skip_platform_factory(env):
    """尽量不经过站台: 测站台档位是否被"干脆不停"绕开(R4 把 −P_nostop gate 关了)。"""
    def act(enc):
        best = None
        for i, s in enumerate(enc["slots"]):
            if s.kind != "set":
                continue
            r = env.net.routes.get(s.route_id)
            has_plat = bool(r and (set(r.tc_sequence) & set(env.plat_tc_to_pid)))
            if not has_plat:
                return i, 0.0, 0.0
            best = best if best is not None else i
        return (best, 0.0, 0.0) if best is not None else (0, 0.0, 0.0)
    return act


def late_dispatch_factory(env, wait_frac: float = 0.5):
    """前半程只等、后半程猛办: 测"拖到窗口末尾"能否规避罚分(边界效应利用)。"""
    state = {"t0": None, "t_end": None}

    def act(enc):
        eng = env.engine
        if state["t0"] is None:
            entries = [t.entry_time for t in eng.trains.values()]
            state["t0"] = min(entries)
            state["t_end"] = max(entries)
        span = max(state["t_end"] - state["t0"], 1.0)
        if (eng.now - state["t0"]) / span < wait_frac:
            return 0, 0.0, 0.0
        for i, s in enumerate(enc["slots"]):
            if s.kind == "set":
                return i, 0.0, 0.0
        return 0, 0.0, 0.0
    return act


def random_factory(seed: int = 0):
    rng = np.random.default_rng(seed)

    def factory(_env):
        def act(enc):
            n = enc["n_options"]
            return int(rng.integers(n)), 0.0, 0.0
        return act
    return factory


def loop_the_train_factory(env):
    """**开环**: 优先选那条会把车带回它刚走过的 TC 的进路。

    2026-08-01 由随机策略偶然触发后补进红队。进路链是策略**在线**选的, 没有任何机制阻止策略把
    车导成环(`RDC5061A→RDC5075A→RDC5083A` = `TDAB→TDAC→TDAB→TDAC→TDAB→TDAC`); 而
    `_release_behind(idx-2)` 原先假设"路径不回头", 于是车把**自己脚下**的锁释放掉 → 别的车锁走
    → 本车再进 = **碰撞**(D6 红线)。这不是数据问题: 98 天 18650 辆车的原始链里 0 条有重复 TC。
    留作常备用例, 任何改动都要保证它跑出 0 碰撞。
    """
    def act(enc):
        best, best_rank = 0, -1
        for i, s in enumerate(enc["slots"]):
            if s.kind != "set":
                continue
            t = env.engine.trains.get(getattr(s, "train_id", None))
            rid = getattr(s, "route_id", None)
            r = env.net.routes.get(rid) if rid else None
            if t is None or r is None:
                continue
            back = len(set(r.tc_sequence) & set(t.path))   # 与已走过的路径重叠越多越"绕"
            if back > best_rank:
                best, best_rank = i, back
        return (best, 0.0, 0.0) if best_rank >= 0 else (0, 0.0, 0.0)
    return act


def greedy_factory(_env):
    return make_greedy_set_policy()


def signaller_factory(env):
    return make_signaller_replay_policy(env)


#: 红队 = 会钻空子的; 参照 = 人类(上界) 与 有路就办(称职基准)
RED_TEAM = {
    "always_wait": always_wait_factory,
    "oscillate_set_cancel": oscillate_factory,
    "cancel_maximiser": cancel_maximiser_factory,
    "priority_only": priority_only_factory,
    "easy_only": easy_only_factory,
    "skip_platform": skip_platform_factory,
    "late_dispatch": late_dispatch_factory,
    "random_legal": random_factory(0),
    "loop_the_train": loop_the_train_factory,
}
REFERENCES = {"greedy_set": greedy_factory, "signaller_replay": signaller_factory}
