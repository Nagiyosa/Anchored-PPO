"""M12a 状态适配层 + 面板逐选项特征化(纯 numpy, 无 torch, 沙盒可单测).

**为什么需要适配层**(docs/M12_DESIGN §10-A, 已查码):
    M11 的 `Featurizer` 吃 M8 键 `deciding` / `conflict_trains` / `candidate_tc_occupancy`;
    M9 `DynamicEngine._build_obs` 给的是**面板级** `trains` / `reservations` / `queue_len`,
    没有 `deciding`/`conflict_trains`。若不适配, BC 热启动只能迁移候选特征、迁不了编码器。

**三条纪律**(§10-A, 违反会**静默**毁掉热启动, 故在本模块用断言/结构锁死):
    ① **按待决车 t 重建, 非按选项**: M8 三块 context 只是 t 与 t 候选集的函数、与选哪条 r 无关
       (=M11 一 obs 配多候选)。每个待决车重建一次 → 天然对上 M11 的批布局(一车=一行)。
    ② **occ 语义 = 物理占用**: `candidate_tc_occupancy` 由各车 `cur_tc` 重建, **不得**用
       `reservations`(那是 tc→route_id 的**锁态**, 同槽换义)。M8 侧 `self.occupied` 每车恒占
       **一个** TC(enter 时置、离开时删, env.py:300/342), 故"cur_tc 重建"与 M8 语义**等价**, 非近似。
    ③ **热启动必须 `route_prior={}`**: no-prior BC 的 prior 槽是常量 0.5; 给 live prior 会让该槽
       变成 BC 从未见其变动的信号 → 劣化热启动, 且绕过 M11 §12 决定 A(不压制长尾偏离)。
       本模块**构造时断言**, 另有单测锁死。

**动作掩码来源**: 选项集**只**来自环境 `PanelDecision.options`(env 已过 联锁几何 ∩ 目标可达 ∩
死锁一步前瞻)。本模块不自行判合法性 —— 安全是结构性的(D6), 不由网络或特征层负责。
输出的 `option_mask` 只标 padding 有效位, 不是安全掩码。

语言: 面板 / Router 插槽 / Panel_Request / berth / 预留 —— 与 RSD-GTORL 完全独立。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PLATS = (1, 2, 3, 4, 5, 6)
PRIORITIES = ("1", "2", "3", "5", "6")


@dataclass(frozen=True)
class OptionSlot:
    """面板选项 → 张量位置的映射(与 PanelDecision.options **逐位对齐**)."""
    kind: str                    # 'wait' | 'set' | 'cancel'
    train_id: str | None = None
    route_id: str | None = None
    row: int = -1                # set: 待决车行号; cancel: 可撤车行号; wait: -1
    col: int = -1                # set: 该车候选列号; 否则 -1


def occupancy_from_panel(obs) -> dict:
    """物理占用 tc -> train_id, **只**由各在读车的 `cur_tc` 重建(纪律②).

    绝不读 obs['reservations'](锁态). 引擎侧每车至多占一个 TC, 与 M8 `self.occupied` 等价。
    """
    occ: dict = {}
    for tr in obs.get("trains", []):
        tc = tr.get("cur_tc")
        if tc is not None:
            occ[tc] = tr.get("train_id")
    return occ


class PanelAdapter:
    """面板 obs + PanelDecision → M11 式 per-train context + 逐选项特征.

    featurizer: M11 `derbysim.learn.features.Featurizer`, **必须 route_prior 为空**(纪律③)。
    """

    def __init__(self, net, featurizer, require_no_prior: bool = True) -> None:
        self.net = net
        self.fz = featurizer
        if require_no_prior and getattr(featurizer, "route_prior", None):
            raise ValueError(
                "M12 warm start requires a no-prior featurizer (route_prior={}): a live route prior "
                "turns a slot the no-prior BC only ever saw as the constant 0.5 into a live signal, "
                "which silently degrades the warm start and re-suppresses long-tail deviation "
                "(M11_DESIGN §12 decision A; M12_DESIGN §10-A discipline 3)."
            )

    # ---- 维度 ----
    @property
    def dec_dim(self) -> int:
        return self.fz.dec_dim

    @property
    def conf_dim(self) -> int:
        return self.fz.conf_dim

    @property
    def cand_dim(self) -> int:
        return self.fz.cand_dim

    @property
    def train_dim(self) -> int:
        return 1 + (len(PRIORITIES) + 1) + len(PLATS) + 1 + 5

    @property
    def cancel_dim(self) -> int:
        return self.train_dim + self.cand_dim

    @property
    def global_dim(self) -> int:
        return 7

    # ---- ① 按车重建 M8 式 context ----
    def rebuild_context(self, obs, train_id: str, legal_route_ids: list) -> dict:
        """从面板 obs 重建车 train_id 的 M8 式快照(键名/语义与 `env.py::_build_state` 对齐)."""
        cand_tcs: list = []
        for rid in legal_route_ids:
            r = self.net.routes.get(rid)
            if r:
                for tc in r.tc_sequence:
                    if tc not in cand_tcs:
                        cand_tcs.append(tc)
        cand_set = set(cand_tcs)
        occ = occupancy_from_panel(obs)                        # 纪律②: 物理占用, 非锁态
        cand_occ = {tc: occ.get(tc) for tc in cand_tcs}

        me = None
        conflicts = []
        for tr in obs.get("trains", []):
            if tr.get("train_id") == train_id:
                me = tr
                continue
            rem = set(tr.get("rem_path") or ())
            inter = rem & cand_set
            occ_here = tr.get("cur_tc") in cand_set            # 每车恒占一格 → 与 M8 any(...) 等价
            if inter or occ_here:
                conflicts.append({
                    "train_id": tr.get("train_id"),
                    "cur_tc": tr.get("cur_tc"),
                    "delay_min": tr.get("delay_min"),
                    "booked_platform": tr.get("booked_platform"),
                    "service_class": tr.get("service_class"),
                    "contends_tc": sorted(inter)[0] if inter else None,
                })
        me = me or {}
        return {
            "now": obs.get("now"),
            "tod_hour": obs.get("tod_hour", 0),
            "decision_signal": me.get("dec_signal"),
            "deciding": {
                "train_id": train_id,
                "cur_tc": me.get("cur_tc"),
                "pos_idx": me.get("pos_idx", 0),
                "delay_min": me.get("delay_min"),
                "booked_platform": me.get("booked_platform"),
                "service_class": me.get("service_class"),
            },
            "candidate_tc_occupancy": cand_occ,
            "conflict_trains": conflicts,
        }

    # ---- 面板级特征(critic / WAIT 用: 看全区, 不只待决车) ----
    def train_features(self, tr) -> list:
        dm = tr.get("delay_min")
        dm = 0.0 if dm is None else max(-120.0, min(120.0, float(dm))) / 120.0
        f = [dm]
        sc = str(tr.get("service_class", ""))
        prio = [0.0] * (len(PRIORITIES) + 1)
        prio[PRIORITIES.index(sc) if sc in PRIORITIES else -1] = 1.0
        f += prio
        bp = tr.get("booked_platform")
        bpoh = [0.0] * len(PLATS)
        if bp in PLATS:
            bpoh[bp - 1] = 1.0
        f += bpoh
        f.append(min(int(tr.get("pos_idx") or 0), 40) / 40.0)
        f += [1.0 if tr.get("awaiting") else 0.0,
              1.0 if tr.get("has_pending") else 0.0,
              1.0 if tr.get("served_platform") else 0.0,
              min(len(tr.get("rem_path") or ()), 20) / 20.0,
              1.0 if tr.get("cur_tc") is None else 0.0]
        return f

    def global_features(self, obs, n_await: int, n_cancel: int) -> list:
        h = int(obs.get("tod_hour", 0) or 0)
        return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                min(len(obs.get("trains", [])), 20) / 20.0,
                min(int(obs.get("n_reserved_tc", 0) or 0), 40) / 40.0,
                min(int(obs.get("queue_len", 0) or 0), 10) / 10.0,
                min(n_await, 10) / 10.0,
                min(n_cancel, 10) / 10.0]

    # ---- ② 逐选项特征化(整个面板) ----
    def encode(self, obs, decision) -> dict:
        """面板 obs + PanelDecision → 张量字典 + 与 options 逐位对齐的 OptionSlot 列表.

        set 选项按**车**分组成行(纪律①): 一行 = 一辆待决车的 M11 式 context, 其候选沿列排开
        → `PointerBC` 的批布局原样可用, BC 权重(编码器 + 打分头)整体迁移。
        """
        if hasattr(self.fz, "set_runtime"):        # D71 fv2: 注入 (now, 背景车 TC 时刻表); fv1 忽略
            self.fz.set_runtime(obs.get("now"), obs.get("bg_tc_times"))
        options = list(decision.options)
        set_by_train: dict = {}
        cancel_ids: list = []
        for a in options:
            if a.kind == "set":
                set_by_train.setdefault(a.train_id, []).append(a.route_id)
            elif a.kind == "cancel" and a.train_id not in cancel_ids:
                cancel_ids.append(a.train_id)

        train_rows = list(set_by_train.keys())               # 稳定序 = 选项出现序(确定性)
        row_of = {tid: i for i, tid in enumerate(train_rows)}
        col_of = {tid: {rid: j for j, rid in enumerate(rids)}
                  for tid, rids in set_by_train.items()}
        crow_of = {tid: i for i, tid in enumerate(cancel_ids)}
        by_id = {tr.get("train_id"): tr for tr in obs.get("trains", [])}

        T = len(train_rows)
        contexts = [self.rebuild_context(obs, tid, set_by_train[tid]) for tid in train_rows]
        x_dec = np.zeros((T, self.dec_dim), np.float32)
        conf_rows = []
        cand_rows = []
        for i, (tid, ctx) in enumerate(zip(train_rows, contexts)):
            x_dec[i] = np.asarray(self.fz.decision_features(ctx), np.float32)
            conf_rows.append(np.asarray(self.fz.conflict_features(ctx), np.float32)
                             .reshape(-1, self.conf_dim))
            occ = ctx["candidate_tc_occupancy"]
            bp = ctx["deciding"].get("booked_platform")
            sig = ctx.get("decision_signal")
            cand_rows.append(np.asarray(
                [self.fz.candidate_features(rid, occ, bp, sig) for rid in set_by_train[tid]],
                np.float32).reshape(-1, self.cand_dim))

        Mc = max([r.shape[0] for r in conf_rows] + [1])
        Ma = max([r.shape[0] for r in cand_rows] + [1])
        x_conf = np.zeros((T, Mc, self.conf_dim), np.float32)
        conf_mask = np.zeros((T, Mc), np.float32)
        x_cand = np.zeros((T, Ma, self.cand_dim), np.float32)
        cand_mask = np.zeros((T, Ma), np.float32)
        for i, (cf, ca) in enumerate(zip(conf_rows, cand_rows)):
            if cf.shape[0]:
                x_conf[i, :cf.shape[0]] = cf
                conf_mask[i, :cf.shape[0]] = 1.0
            x_cand[i, :ca.shape[0]] = ca
            cand_mask[i, :ca.shape[0]] = 1.0

        # cancel 选项 = 该车面板特征 ++ 其待办进路的候选特征(同一套进路编码, 语义一致)
        occ_all = occupancy_from_panel(obs)
        x_cancel = np.zeros((max(len(cancel_ids), 1), self.cancel_dim), np.float32)
        for i, tid in enumerate(cancel_ids):
            tr = by_id.get(tid, {})
            rid = tr.get("pending_route")
            cf = (self.fz.candidate_features(rid, occ_all, tr.get("booked_platform"),
                                             tr.get("dec_signal"))
                  if rid is not None else [0.0] * self.cand_dim)
            x_cancel[i] = np.asarray(self.train_features(tr) + list(cf), np.float32)
        cancel_mask = np.zeros((max(len(cancel_ids), 1),), np.float32)
        cancel_mask[:len(cancel_ids)] = 1.0

        panel_trains = obs.get("trains", [])
        N = max(len(panel_trains), 1)
        x_panel = np.zeros((N, self.train_dim), np.float32)
        panel_mask = np.zeros((N,), np.float32)
        for i, tr in enumerate(panel_trains):
            x_panel[i] = np.asarray(self.train_features(tr), np.float32)
            panel_mask[i] = 1.0
        x_global = np.asarray(self.global_features(obs, T, len(cancel_ids)), np.float32)

        slots = []
        for a in options:
            if a.kind == "set":
                slots.append(OptionSlot("set", a.train_id, a.route_id,
                                        row_of[a.train_id], col_of[a.train_id][a.route_id]))
            elif a.kind == "cancel":
                slots.append(OptionSlot("cancel", a.train_id, None, crow_of[a.train_id], -1))
            else:
                slots.append(OptionSlot("wait"))

        return {"slots": slots, "options": options, "train_rows": train_rows,
                "cancel_rows": cancel_ids,
                "x_dec": x_dec, "x_conf": x_conf, "conf_mask": conf_mask,
                "x_cand": x_cand, "cand_mask": cand_mask,
                "x_cancel": x_cancel, "cancel_mask": cancel_mask,
                "x_panel": x_panel, "panel_mask": panel_mask, "x_global": x_global,
                "n_options": len(options)}


KIND_CODE = {"wait": 0, "set": 1, "cancel": 2}


def collate_panels(encoded: list) -> dict:
    """把多个决策的 encode() 结果 padding 成批(纯 numpy), 供 PPO minibatch.

    形状: x_dec[B,T,Dd] · x_conf[B,T,Mc,Dc] · x_cand[B,T,Ma,Da] · x_cancel[B,C,Dx] ·
          x_panel[B,N,Dt] · x_global[B,Dg]; 选项侧 opt_kind/opt_row/opt_col/opt_mask [B,O]。
    `opt_*` 让"选项 → 张量位置"在批内保持逐位对齐 —— argmax 出来的下标可直接索引回
    `PanelDecision.options`(动作是环境给的选项本身, 不是网络自造的)。
    """
    B = len(encoded)
    Dd = encoded[0]["x_dec"].shape[1]
    Dc = encoded[0]["x_conf"].shape[2]
    Da = encoded[0]["x_cand"].shape[2]
    Dx = encoded[0]["x_cancel"].shape[1]
    Dt = encoded[0]["x_panel"].shape[1]
    Dg = encoded[0]["x_global"].shape[0]
    T = max(e["x_dec"].shape[0] for e in encoded) or 1
    Mc = max(e["x_conf"].shape[1] for e in encoded) or 1
    Ma = max(e["x_cand"].shape[1] for e in encoded) or 1
    C = max(e["x_cancel"].shape[0] for e in encoded) or 1
    N = max(e["x_panel"].shape[0] for e in encoded) or 1
    O = max(e["n_options"] for e in encoded) or 1

    out = {
        "x_dec": np.zeros((B, T, Dd), np.float32),
        "x_conf": np.zeros((B, T, Mc, Dc), np.float32),
        "conf_mask": np.zeros((B, T, Mc), np.float32),
        "x_cand": np.zeros((B, T, Ma, Da), np.float32),
        "cand_mask": np.zeros((B, T, Ma), np.float32),
        "x_cancel": np.zeros((B, C, Dx), np.float32),
        "cancel_mask": np.zeros((B, C), np.float32),
        "x_panel": np.zeros((B, N, Dt), np.float32),
        "panel_mask": np.zeros((B, N), np.float32),
        "x_global": np.zeros((B, Dg), np.float32),
        "opt_kind": np.zeros((B, O), np.int64),
        "opt_row": np.zeros((B, O), np.int64),
        "opt_col": np.zeros((B, O), np.int64),
        "opt_mask": np.zeros((B, O), np.float32),
    }
    # D59: 选路 KL 的规则参照目标(仅 RL 训练批携带; run_episode 的 ref_builder 挂上)。
    # 每行(一辆待决车)一个目标进路列 + 有效位 → routing_kl 只在有效行上把策略拉向计划/lookup 路。
    has_ref = "ref_col" in encoded[0]
    if has_ref:
        out["ref_col"] = np.zeros((B, T), np.int64)
        out["ref_valid"] = np.zeros((B, T), np.float32)
    for b, e in enumerate(encoded):
        t, mc, ma = e["x_dec"].shape[0], e["x_conf"].shape[1], e["x_cand"].shape[1]
        c, n, o = e["x_cancel"].shape[0], e["x_panel"].shape[0], e["n_options"]
        out["x_dec"][b, :t] = e["x_dec"]
        out["x_conf"][b, :t, :mc] = e["x_conf"]
        out["conf_mask"][b, :t, :mc] = e["conf_mask"]
        out["x_cand"][b, :t, :ma] = e["x_cand"]
        out["cand_mask"][b, :t, :ma] = e["cand_mask"]
        out["x_cancel"][b, :c] = e["x_cancel"]
        out["cancel_mask"][b, :c] = e["cancel_mask"]
        out["x_panel"][b, :n] = e["x_panel"]
        out["panel_mask"][b, :n] = e["panel_mask"]
        out["x_global"][b] = e["x_global"]
        if has_ref:
            rc = np.asarray(e["ref_col"]).reshape(-1)[:t]
            rv = np.asarray(e["ref_valid"]).reshape(-1)[:t]
            out["ref_col"][b, :rc.shape[0]] = rc
            out["ref_valid"][b, :rv.shape[0]] = rv
        for i, s in enumerate(e["slots"]):
            out["opt_kind"][b, i] = KIND_CODE[s.kind]
            out["opt_row"][b, i] = max(s.row, 0)
            out["opt_col"][b, i] = max(s.col, 0)
            out["opt_mask"][b, i] = 1.0
    return out
