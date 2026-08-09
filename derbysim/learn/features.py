"""M11b 特征化(纯 numpy): M8 防泄露快照 obs + legal 进路集 → 数值特征.

M11d 修(§11 根因): reward/BC 首轮 BC<查表(0.63 vs 0.93) → 过拟合。修法(特征侧):
    - **候选先验** prior[(signal,route)] = 训练集"该信号该进路被选/被提供"频率 → 把强基线(0.80)直接喂模型;
      **只用训练集拟合**(无泄露), holdout 用同一 prior。
    - **信号 one-hot** 进决策特征 → 模型知道身处哪个信号(学每信号先验)。
    (模型/训练侧的正则/早停/候选主导在 models.py/train_bc.py。)
"""
from __future__ import annotations
import json, math
from collections import defaultdict
import numpy as np

from derbysim.sim.goal import exit_direction_groups, Reachability

PRIORITIES = ["1", "2", "3", "5", "6"]
PLATS = [1, 2, 3, 4, 5, 6]


def _occupied(v) -> bool:
    return v is not None and str(v) not in ("", "None", "nan", "0")


def fit_route_prior(records) -> dict:
    """训练集拟合候选先验: P(被选 | 信号, 进路) = 被选次数/被提供次数(平滑). records 含 signal/legal_route_ids/chosen_route/n_legal。"""
    chosen = defaultdict(int); offered = defaultdict(int)
    for r in records:
        if r["n_legal"] <= 1:
            continue
        sig = r["signal"]
        for rid in list(r["legal_route_ids"]):
            offered[(sig, rid)] += 1
            if rid == r["chosen_route"]:
                chosen[(sig, rid)] += 1
    return {k: (chosen[k] + 0.5) / (offered[k] + 1.0) for k in offered}


class Featurizer:
    def __init__(self, net, plat_tc_to_pid, delay_clip: float = 120.0,
                 signal_index: dict | None = None, route_prior: dict | None = None,
                 fv: int = 1) -> None:
        self.net = net
        self.ptc = plat_tc_to_pid
        self.delay_clip = float(delay_clip)
        # D71 特征版本: 1=现状(默认, 全部旧 run/ckpt 逐位可复现); 2=加两个"冲突时刻"特征:
        #   dec +1  = 距计划时刻分钟数(±15min 细分辨率; delay_min 的 ±120min 太钝, 1min 只占 0.8%)
        #   cand +1 = 距最近一辆背景车需要本路轨道的分钟数(前瞻背景车流 = 调度员桌上的运行图, 合法信息;
        #             受控车自己的 req_times 仍按 D54 严格不进状态)
        self.fv = int(fv)
        self._rt_now = None                 # encode 期注入的运行时上下文(见 set_runtime)
        self._rt_bg = None
        self.reach = Reachability(net)
        self.groups = exit_direction_groups(net)
        gset = sorted({tuple(sorted(g)) for g in self.groups.values()})
        self.group_index = {g: i for i, g in enumerate(gset)}
        self.n_groups = len(gset)
        self.psig_pid = {s: pid for pid, p in net.platforms.items()
                         for s in p.end_signals.values()}
        # 信号索引(默认: 全网有出口的信号, 拓扑固定)
        self.signal_index = signal_index or {s: i for i, s in enumerate(
            sorted({r.signal for r in net.routes.values() if r.signal is not None}))}
        self.n_signals = len(self.signal_index)
        self.route_prior = route_prior or {}
        self._route_cache: dict = {}

    def set_runtime(self, now, bg_tc_times) -> None:
        """D71: encode 期注入 (now, 背景车 TC→未来使用时刻有序表)。fv=1 忽略; 缺省 None = 中性值。"""
        self._rt_now = now
        self._rt_bg = bg_tc_times

    def _bg_conflict_mins(self, tcs) -> float:
        """候选路 footprint 内, 最近一辆背景车要用任一 TC 的分钟数(0..15, 无冲突=15)。"""
        if not self._rt_bg or self._rt_now is None:
            return 15.0
        import bisect
        best = None
        for tc in tcs:
            lst = self._rt_bg.get(tc)
            if lst:
                j = bisect.bisect_left(lst, self._rt_now)
                if j < len(lst):
                    best = lst[j] if best is None else min(best, lst[j])
        if best is None:
            return 15.0
        return max(0.0, min(15.0, (best - self._rt_now) / 60.0))

    def _clip_delay(self, dm) -> float:
        if dm is None or (isinstance(dm, float) and math.isnan(dm)):
            return 0.0
        return max(-self.delay_clip, min(self.delay_clip, float(dm))) / self.delay_clip

    def _prio_oh(self, sc):
        v = [0.0] * (len(PRIORITIES) + 1)
        s = str(sc)
        v[PRIORITIES.index(s) if s in PRIORITIES else -1] = 1.0
        return v

    def _route_static(self, rid):
        if rid in self._route_cache:
            return self._route_cache[rid]
        r = self.net.routes.get(rid)
        if not r:
            f = {"group": -1, "plats": set(), "length": 0, "tcs": []}
            self._route_cache[rid] = f
            return f
        es = r.end_signal
        gid = -1
        if es in self.groups:
            gid = self.group_index[tuple(sorted(self.groups[es]))]
        elif es is not None:
            _, exits = self.reach._r(es)
            for e in exits:
                if e in self.groups:
                    gid = self.group_index[tuple(sorted(self.groups[e]))]
                    break
        seen = (self.reach._r(es)[0] | {es}) if es is not None else set()
        plats = {self.psig_pid[s] for s in seen if s in self.psig_pid}
        f = {"group": gid, "plats": plats, "length": len(r.tc_sequence), "tcs": list(r.tc_sequence)}
        self._route_cache[rid] = f
        return f

    def _signal_oh(self, signal):
        v = [0.0] * self.n_signals
        i = self.signal_index.get(signal)
        if i is not None:
            v[i] = 1.0
        return v

    def decision_features(self, obs) -> list:
        dec = obs.get("deciding", {})
        f = [self._clip_delay(dec.get("delay_min"))]
        f += self._prio_oh(dec.get("service_class"))
        bp = dec.get("booked_platform"); bpoh = [0.0] * 7
        bpoh[(bp - 1) if bp in PLATS else 6] = 1.0
        f += bpoh
        f.append(min(dec.get("pos_idx", 0) or 0, 40) / 40.0)
        h = obs.get("tod_hour", 0) or 0
        f += [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]
        f.append(min(len(obs.get("conflict_trains", [])), 20) / 20.0)
        f += self._signal_oh(obs.get("decision_signal"))          # 新增: 信号身份
        if self.fv >= 2:                                          # D71: ±15min 细分辨率计划时刻
            dm = float(dec.get("delay_min") or 0.0)
            f.append(max(-15.0, min(15.0, dm)) / 15.0)
        return f

    def conflict_features(self, obs) -> list:
        dbp = obs.get("deciding", {}).get("booked_platform")
        rows = []
        for c in obs.get("conflict_trains", []):
            r = [self._clip_delay(c.get("delay_min"))]
            r += self._prio_oh(c.get("service_class"))
            r.append(1.0 if (c.get("booked_platform") is not None
                             and c.get("booked_platform") == dbp) else 0.0)
            rows.append(r)
        return rows

    def candidate_features(self, rid, occ_map, booked_platform, signal=None) -> list:
        rs = self._route_static(rid)
        grp = [0.0] * self.n_groups
        if 0 <= rs["group"] < self.n_groups:
            grp[rs["group"]] = 1.0
        plat_oh = [0.0] * 6
        for p in rs["plats"]:
            if p in PLATS:
                plat_oh[p - 1] = 1.0
        reaches_booked = 1.0 if (booked_platform in rs["plats"]) else 0.0
        reaches_any = 1.0 if rs["plats"] else 0.0
        n_plat = len(rs["plats"]) / 6.0
        tcs = rs["tcs"]
        occ_cnt = sum(1 for tc in tcs if _occupied(occ_map.get(tc)))
        occ_frac = (occ_cnt / len(tcs)) if tcs else 0.0
        first_occ = 1.0 if (tcs and _occupied(occ_map.get(tcs[0]))) else 0.0
        length = min(rs["length"], 20) / 20.0
        prior = self.route_prior.get((signal, rid), 0.5)            # 新增: 训练集候选先验(强基线)
        out = grp + plat_oh + [reaches_booked, reaches_any, n_plat, occ_frac, first_occ, length, prior]
        if self.fv >= 2:                                            # D71: 前瞻背景车流(冲突预判)
            out.append(self._bg_conflict_mins(tcs) / 15.0)
        return out

    def featurize_record(self, rec) -> dict:
        obs = json.loads(rec["obs"]) if isinstance(rec["obs"], str) else rec["obs"]
        legal = list(rec["legal_route_ids"])
        occ = obs.get("candidate_tc_occupancy", {})
        bp = obs.get("deciding", {}).get("booked_platform")
        sig = rec.get("signal", obs.get("decision_signal"))
        x_cand = [self.candidate_features(rid, occ, bp, sig) for rid in legal]
        chosen = rec["chosen_route"]
        y = legal.index(chosen) if chosen in legal else -1
        return {
            "x_dec": np.asarray(self.decision_features(obs), dtype=np.float32),
            "x_conf": np.asarray(self.conflict_features(obs), dtype=np.float32).reshape(-1, self.conf_dim),
            "x_cand": np.asarray(x_cand, dtype=np.float32).reshape(-1, self.cand_dim),
            "y": int(y), "n_legal": len(legal), "signal": sig,
            "service_class": str(rec.get("service_class", "")),
            "chosen_route": chosen, "legal_route_ids": legal,
        }

    @property
    def dec_dim(self) -> int:
        return 1 + (len(PRIORITIES) + 1) + 7 + 1 + 2 + 1 + self.n_signals + (1 if self.fv >= 2 else 0)

    @property
    def conf_dim(self) -> int:
        return 1 + (len(PRIORITIES) + 1) + 1

    @property
    def cand_dim(self) -> int:
        # +7: reaches_booked/any/n_plat/occ_frac/first_occ/length/prior; fv2 +1: 背景车前瞻(D71)
        return self.n_groups + 6 + 7 + (1 if self.fv >= 2 else 0)


def featurize_frame(df, featurizer) -> list:
    out = []
    for rec in df.to_dict("records"):
        s = featurizer.featurize_record(rec)
        if s["y"] >= 0:
            out.append(s)
    return out
