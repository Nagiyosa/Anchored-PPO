"""M11b 特征化: 把 M8 防泄露快照 obs + legal 进路集 → 数值特征(纯 numpy, 无 torch).

设计(docs/M11_DESIGN.md §3):
    - 状态 = 决策车特征 + 冲突车集合(变长, 供 DeepSets/Transformer/GNN) + 时段。
    - 动作 = **逐候选进路特征**(pointer): 候选路的**可达性**特征(通向哪个出区方向组 / 可达哪些站台 /
      是否达 booked 台) + 其 TC 当前占用 + 长度。**用可达性算**(体检 P3: 朴素 TC 成员失败)。
    - delay 裁剪 ±120min(体检 P4/M11a: |dm|>120 仅 1.1% 脏点)。
输出每决策: (x_dec[Dd], x_conf[n_conf,Dc], x_cand[n_legal,Da], y=被选在 legal 的下标)。
"""
from __future__ import annotations
import json, math
import numpy as np

from derbysim.sim.goal import exit_direction_groups, Reachability

PRIORITIES = ["1", "2", "3", "5", "6"]     # TOPS: 特快/普客/包裹/ECS/货; 其它归 other
PLATS = [1, 2, 3, 4, 5, 6]


def _occupied(v) -> bool:
    return v is not None and str(v) not in ("", "None", "nan", "0")


class Featurizer:
    def __init__(self, net, plat_tc_to_pid, delay_clip: float = 120.0) -> None:
        self.net = net
        self.ptc = plat_tc_to_pid
        self.delay_clip = float(delay_clip)
        self.reach = Reachability(net)
        self.groups = exit_direction_groups(net)                 # exit_signal -> frozenset
        gset = sorted({tuple(sorted(g)) for g in self.groups.values()})
        self.group_index = {g: i for i, g in enumerate(gset)}
        self.n_groups = len(gset)
        self.psig_pid = {s: pid for pid, p in net.platforms.items()
                         for s in p.end_signals.values()}
        self._route_cache: dict = {}

    # ---- 标量/onehot 工具 ----
    def _clip_delay(self, dm) -> float:
        if dm is None or (isinstance(dm, float) and math.isnan(dm)):
            return 0.0
        return max(-self.delay_clip, min(self.delay_clip, float(dm))) / self.delay_clip

    def _prio_oh(self, sc):
        v = [0.0] * (len(PRIORITIES) + 1)
        s = str(sc)
        if s in PRIORITIES:
            v[PRIORITIES.index(s)] = 1.0
        else:
            v[-1] = 1.0
        return v

    # ---- 候选进路静态特征(缓存, 与占用无关) ----
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
        f = {"group": gid, "plats": plats, "length": len(r.tc_sequence),
             "tcs": list(r.tc_sequence)}
        self._route_cache[rid] = f
        return f

    # ---- 三类特征 ----
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

    def candidate_features(self, rid, occ_map, booked_platform) -> list:
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
        return grp + plat_oh + [reaches_booked, reaches_any, n_plat, occ_frac, first_occ, length]

    # ---- 每决策整合 ----
    def featurize_record(self, rec) -> dict:
        obs = json.loads(rec["obs"]) if isinstance(rec["obs"], str) else rec["obs"]
        legal = list(rec["legal_route_ids"])
        occ = obs.get("candidate_tc_occupancy", {})
        bp = obs.get("deciding", {}).get("booked_platform")
        x_cand = [self.candidate_features(rid, occ, bp) for rid in legal]
        chosen = rec["chosen_route"]
        y = legal.index(chosen) if chosen in legal else -1
        return {
            "x_dec": np.asarray(self.decision_features(obs), dtype=np.float32),
            "x_conf": np.asarray(self.conflict_features(obs), dtype=np.float32).reshape(-1, self.conf_dim),
            "x_cand": np.asarray(x_cand, dtype=np.float32).reshape(-1, self.cand_dim),
            "y": int(y),
            "n_legal": len(legal),
            "signal": rec.get("signal"),
            "service_class": str(rec.get("service_class", "")),
            "chosen_route": chosen,
            "legal_route_ids": legal,            # 供预测下标→进路映射(评估)
        }

    # ---- 维度(供模型构建) ----
    @property
    def dec_dim(self) -> int:
        return 1 + (len(PRIORITIES) + 1) + 7 + 1 + 2 + 1

    @property
    def conf_dim(self) -> int:
        return 1 + (len(PRIORITIES) + 1) + 1

    @property
    def cand_dim(self) -> int:
        return self.n_groups + 6 + 6


def featurize_frame(df, featurizer) -> list:
    """DataFrame → list[featurized dict]; 只保留 y>=0(被选在 legal 内)的样本。"""
    out = []
    for rec in df.to_dict("records"):
        s = featurizer.featurize_record(rec)
        if s["y"] >= 0:
            out.append(s)
    return out
