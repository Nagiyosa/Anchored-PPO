"""DerbyEnv (M8): gym 式环境 = 反应式引擎可暂停版 + 防泄露状态 + 物理触发 (D18/D22).

与 reactive.ReactiveEngine 的关系:
    ReactiveEngine 跑**预拼好的整条 path**(路由链已烘焙进 path/ready_at), 一次跑完.
    本模块把"整条 path"拆成**逐路由段增量构造**, 并在**真实决策时刻**(物理触发点)暂停问决策.

M8 落地 D22:
    - 触发(物理必然, 不碰日志时刻): 站台出站信号 → 车进入其停靠站台 TC(dwell 起点)触发;
      接近/通过信号 S → 车驶入 S **上游第 N[S] 格**触发(沿已物化路径回看 N 格, 可跨越更早进路段),
      N[S] 从 TD 标定(decision_advance.json); 回看越过入区 → 以入区截断.
    - 无泄露状态(靠构造, 非拍早): 快照拍在真实决策时刻, 剔除本决策后果(锁/信号绿/committed 位移/
      日志所选路), 只放可观测量(占用/逼近车位置/延误/CIF 目标/时段). 观测=决策车 + 所有冲突车.
    - 动作掩码 = 该信号**几何合法出口**(全部; D22 用户定案 request 层, 不按当下占用过滤——占用留状态,
      "会不会通"由策略学; 防撞靠引擎停等, 死锁留 M9).
    - 解耦(回放/建数据集): "问+拍快照"在物理触发点; "materialize/推进"仍照日志 req_time(ready_at 门禁),
      故列车推进时序不变 → M7 逐位等价续绿. 全物理锁留 M9.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from derbysim import config
from derbysim.sim.events import EventQueue
from derbysim.sim.reactive import DEFAULT_ADVANCE, DEFAULT_OCC, MAX_WAIT_S

_LPLAT_RE = re.compile(r"LPLAT(\d+)")
_RUN_GAP_S = 90 * 60   # H6: 同 headcode 事件间隔 >90min 判为新的一趟(与 data.alignment 一致)

_ADVANCE_CFG_PATH = config.ARTIFACTS_CALIB / "decision_advance.json"


def load_advance_cfg(path=_ADVANCE_CFG_PATH) -> dict:
    """加载每信号触发标定; 缺失则返回空(=N 回退 0, 贴信号触发, 等价于旧行为)."""
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@dataclass
class DecisionContext:
    """一次决策点情境(单智能体, 事件时序).

    obs = 因果先于动作、无泄露的状态快照(M8); legal_route_ids = 几何合法出口掩码;
    logged_route_id = 回放标签/BC 目标(只作 y, 不进 obs).
    """
    train_id: str
    signal: int | None
    route_ptr: int
    legal_route_ids: list[str]
    logged_route_id: str | None
    now: float
    obs: dict | None = None
    trigger_kind: str = ""        # 'dwell' | 'upstream_n' | 'entry_clamp' | 'fallback_endpath'


@dataclass
class EnvTrain:
    """增量式列车: 携带路由链(决策序列), path 由引擎逐段物化."""
    train_id: str
    route_chain: list[str]
    req_times: list[float]
    entry_time: float
    stop_pids: set = field(default_factory=set)
    # CIF 目标(状态用; 缺省 None → 状态相应字段为 None, 不影响等价):
    booked_platform: int | None = None
    planned_ref: float | None = None          # CIF 计划参考时刻(epoch s), 供 signed 延误
    service_class: str = ""                    # headcode 首位 = TOPS 优先级(D16)
    # 运行时:
    path: list[str] = field(default_factory=list)
    ready_at: list[float] = field(default_factory=list)
    route_ptr: int = 0
    dwell_idx: int | None = None
    dwell_s: float = 0.0
    pos: int = 0
    done: bool = False
    blocked_since: float | None = None
    wait_time: float = 0.0
    sim_occ: dict = field(default_factory=dict)
    # 触发调度(M8):
    next_trigger_idx: int | None = None
    next_decision_signal: int | None = None
    next_trigger_kind: str = ""

    def ready(self, idx: int) -> float:
        return self.ready_at[idx] if self.ready_at and idx < len(self.ready_at) else -1e18


class SteppableReactiveEngine:
    """反应式引擎生成器版: 到**真实决策时刻**(物理触发)yield DecisionContext, send(route_id) 续跑.

    物理/门禁公式与 reactive.ReactiveEngine 完全一致; 仅把"路由已知"改成"决策点问",
    且把"问"的时机从贴信号提前到标定的物理触发点(解耦: 锁路/推进仍照 req_time).
    """

    def __init__(self, net, params: dict | None = None,
                 plat_tc_to_pid: dict | None = None,
                 dwell_by_platform: dict | None = None,
                 advance_cfg: dict | None = None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.advance_cfg = advance_cfg if advance_cfg is not None else load_advance_cfg()
        self.q = EventQueue()
        self.occupied: dict[str, str] = {}
        self.waiting: dict[str, deque] = defaultdict(deque)
        self.trains: dict[str, EnvTrain] = {}
        self.now = 0.0
        self.deadlocks: list = []
        self.n_wait_events = 0
        self.n_decisions = 0
        self.mask_misses: list = []      # 日志真实进路被自己掩码误划掉的记录(数据质量自检)

    # ---- 物理参数(与 reactive 一致) ----
    def _adv(self, tc: str) -> float:
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    def _tail(self, tc: str) -> float:
        a, o = self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))
        return max(1.0, o - a)

    # ---- 增量物化(解耦: ready=log req_time, 保 M7 等价) ----
    def _materialize(self, t: EnvTrain, route_id: str) -> None:
        r = self.net.routes.get(route_id)
        rt = t.req_times[t.route_ptr] if t.route_ptr < len(t.req_times) else self.now
        seg_start = len(t.path)
        if r:
            for tc in r.tc_sequence:
                if not t.path or t.path[-1] != tc:
                    t.path.append(tc)
                    t.ready_at.append(rt)
                    if (t.dwell_idx is None and tc in self.plat_tc_to_pid
                            and self.plat_tc_to_pid[tc] in t.stop_pids):
                        t.dwell_idx = len(t.path) - 1
                        t.dwell_s = self.dwell_by_platform.get(self.plat_tc_to_pid[tc], 0.0)
        t.route_ptr += 1
        self._schedule_next_decision(t, min(seg_start, len(t.path) - 1))

    def _schedule_next_decision(self, t: EnvTrain, seg_start: int) -> None:
        """已物化 route_chain[route_ptr-1](终于信号 S); 若还有下一段, 计算其触发点."""
        if t.route_ptr >= len(t.route_chain):
            t.next_trigger_idx = None
            t.next_decision_signal = None
            t.next_trigger_kind = ""
            return
        # 决策信号 = 将要办理的这条 route 的**起始信号**(信号员在此办进路), 而非上一段 end_signal:
        # 站台两端配对(如 platform5 N=5065/S=5055)与 ECS 折返时二者不同, 用 route.signal 才正确.
        nxt_route = self.net.routes.get(t.route_chain[t.route_ptr])
        S = nxt_route.signal if nxt_route else None
        t.next_decision_signal = S
        seg_end = len(t.path) - 1
        cfg = self.advance_cfg.get(str(S), {}) if S is not None else {}
        is_pex = bool(cfg.get("is_platform_exit")) or cfg.get("trigger") == "dwell"
        if is_pex and t.dwell_idx is not None and seg_start <= t.dwell_idx <= seg_end:
            trig = t.dwell_idx                    # 站台出站: 车进入停靠站台 TC(dwell 起点)触发
            kind = "dwell"
        else:
            # 上游 N 格触发: 沿列车**已物化路径回看 N 格**(可跨越更早的进路段, 非本段截断);
            # 回看越过入区(idx<0) → 以入区(idx 0)截断. 短路由也能忠实反映真实提前量.
            n = int(cfg.get("n_berths", 0) or 0)
            trig = seg_end - n
            kind = "upstream_n"
            if trig <= 0:
                trig = 0
                kind = "entry_clamp" if t.route_ptr == 1 else "upstream_n"
        t.next_trigger_idx = max(0, trig)
        t.next_trigger_kind = kind

    # ---- 决策信号处的动作掩码 = 几何合法出口 (D22, 用户定案 request 层) ----
    def _legal_routes(self, S: int | None, train_id: str):
        """动作掩码 = 该信号**几何合法出口**(全部出口进路), 不按"当下 TC 空闲"过滤.

        定案理由(request-vs-lock): 我们学的是 Panel_Request(请求层); 实测 2.2% 真实请求发生在
        route TC 被别车占时(提前排在将清空的路上, 排队待清空才锁)。用"当下空闲"当硬掩码会砍掉这些
        真实决策、并把答案偷偷补回。占用信息已在状态 candidate_tc_occupancy 里, "会不会通、要不要
        提前排"是策略要学的判断(D21)。LOCK 层(不能锁占用)由引擎物理门禁(下一段被占则停等)体现;
        防撞结构性保证, 死锁避免留 M9(等待图找环+前瞻屏蔽)。几何掩码亦保证日志真进路恒在候选集内。
        """
        if S is None:
            return []
        return list(self.net.routes_from(S))

    # ---- 无泄露状态构造(D18/D22): 因果先于动作, 剔除本决策后果 ----
    def _delay_min(self, tr: EnvTrain):
        if tr.planned_ref is None:
            return None
        return round((tr.entry_time - tr.planned_ref) / 60.0, 1)   # 入区相对 CIF 参考的 signed 延误(因果先验)

    def _build_state(self, t: EnvTrain, S: int | None, legal_ids: list[str]) -> dict:
        cand_tcs: list[str] = []
        for rid in legal_ids:
            r = self.net.routes.get(rid)
            if r:
                for tc in r.tc_sequence:
                    if tc not in cand_tcs:
                        cand_tcs.append(tc)
        cand_set = set(cand_tcs)
        # 候选进路各 TC 当下物理占用(占者 train 或 None) —— 非"进路锁", 是物理占用
        cand_occ = {tc: self.occupied.get(tc) for tc in cand_tcs}
        # 所有冲突车: 剩余(已物化)路径与候选 TC 相交, 或当下占用候选 TC
        conflicts = []
        for tr in self.trains.values():
            if tr.train_id == t.train_id or tr.done:
                continue
            rem = set(tr.path[tr.pos:]) if tr.pos < len(tr.path) else set()
            inter = rem & cand_set
            occ_here = any(self.occupied.get(tc) == tr.train_id for tc in cand_tcs)
            if inter or occ_here:
                conflicts.append({
                    "train_id": tr.train_id,
                    "cur_tc": tr.path[tr.pos] if tr.pos < len(tr.path) else None,
                    "delay_min": self._delay_min(tr),
                    "booked_platform": tr.booked_platform,
                    "service_class": tr.service_class,
                    "contends_tc": sorted(inter)[0] if inter else None,
                })
        return {
            "now": self.now,
            "tod_hour": int((self.now // 3600) % 24),
            "decision_signal": S,
            "deciding": {
                "train_id": t.train_id,
                "cur_tc": t.path[t.pos] if t.pos < len(t.path) else None,
                "pos_idx": t.pos,
                "delay_min": self._delay_min(t),
                "booked_platform": t.booked_platform,
                "service_class": t.service_class,
            },
            "candidate_tc_occupancy": cand_occ,
            "conflict_trains": conflicts,
            # 明确不含(答案/后果): 日志所选进路、进路锁、信号变绿、req_time、committed 位移
        }

    # ---- 主循环(生成器) ----
    def run(self, trains: list[EnvTrain]):
        self.trains = {t.train_id: t for t in trains}
        # 入口段(route_chain[0]) 在 reset 物化以锚 entry(D14.8: 非决策); 并调度第一次决策触发
        for t in trains:
            if t.route_chain:
                self._materialize(t, t.route_chain[0])
        for t in trains:
            if t.path:
                self.q.push(t.entry_time, "enter", {"tr": t.train_id, "idx": 0})
        guard = 0
        limit = 500000 + 5000 * len(trains)
        while len(self.q):
            guard += 1
            if guard > limit:
                break
            ev = self.q.pop()
            self.now = ev.time
            if ev.kind == "enter":
                self._on_enter(ev.data)
                yield from self._maybe_decide(ev.data["tr"])
            elif ev.kind == "try_advance":
                need = self._try_advance(ev.data)
                if need is not None:                 # 兜底: 到段末仍未决策 → 强制决策
                    yield from self._maybe_decide(need.train_id, force=True)
                    self.q.push(self.now, "try_advance", {"tr": need.train_id, "idx": need.pos})
            elif ev.kind == "release":
                self._on_release(ev.data)
            elif ev.kind == "deadlock_check":
                self._on_deadlock_check(ev.data)

    def _maybe_decide(self, train_id: str, force: bool = False):
        """在物理触发点(或兜底)逐个 yield 决策; send(route_id) 后 materialize."""
        t = self.trains[train_id]
        while t.route_ptr < len(t.route_chain):
            trig = t.next_trigger_idx
            if not force and (trig is None or t.pos < trig):
                return
            S = t.next_decision_signal
            legal = self._legal_routes(S, train_id)
            legal_ids = [r.route_id for r in legal]
            logged = t.route_chain[t.route_ptr]
            if logged is not None and logged not in legal_ids:
                # 日志真实进路被自己的掩码误划掉 = 占用时序有洞; 记录(不静默), 并纳入掩码保回放可行
                self.mask_misses.append({"t": self.now, "train": train_id, "signal": S,
                                         "logged": logged, "legal": list(legal_ids)})
                legal_ids = legal_ids + [logged]
            obs = self._build_state(t, S, legal_ids)
            kind = t.next_trigger_kind or ("fallback_endpath" if force else "")
            ctx = DecisionContext(train_id, S, t.route_ptr, legal_ids, logged, self.now,
                                  obs=obs, trigger_kind=kind)
            self.n_decisions += 1
            chosen = yield ctx
            self._materialize(t, chosen)
            force = False

    def _on_enter(self, d: dict) -> None:
        t = self.trains[d["tr"]]
        idx = d["idx"]
        tc = t.path[idx]
        self.occupied[tc] = t.train_id
        t.pos = idx
        t.sim_occ[tc] = self.now
        t.blocked_since = None
        hold = t.dwell_s if (t.dwell_idx == idx and t.dwell_s > 0) else self._adv(tc)
        self.q.push(self.now + hold, "try_advance", {"tr": t.train_id, "idx": idx})

    def _try_advance(self, d: dict):
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.pos != idx or t.done:
            return None
        tc = t.path[idx]
        if idx + 1 >= len(t.path):
            if t.route_ptr < len(t.route_chain):     # 段末仍有决策未办(兜底)
                return t
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            t.done = True
            return None
        nxt = t.path[idx + 1]
        ready = t.ready(idx + 1)
        if self.now < ready:                         # 门禁1: 进路未办理(照 log req_time)
            self.q.push(ready, "try_advance", {"tr": t.train_id, "idx": idx})
            return None
        holder = self.occupied.get(nxt)
        if holder is None or holder == t.train_id:   # 门禁2: 下一段空 → 进
            if t.blocked_since is not None:
                t.wait_time += self.now - t.blocked_since
                t.blocked_since = None
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
        else:                                        # 被占 → 停等
            if t.blocked_since is None:
                t.blocked_since = self.now
                self.n_wait_events += 1
                self.q.push(self.now + MAX_WAIT_S, "deadlock_check", {"tr": t.train_id, "idx": idx})
            if t.train_id not in self.waiting[nxt]:
                self.waiting[nxt].append(t.train_id)
        return None

    def _on_release(self, d: dict) -> None:
        tc, tr = d["tc"], d["tr"]
        if self.occupied.get(tc) == tr:
            del self.occupied[tc]
        while self.waiting[tc]:
            w = self.waiting[tc].popleft()
            wt = self.trains.get(w)
            if wt and not wt.done and wt.pos < len(wt.path) - 1 and wt.path[wt.pos + 1] == tc:
                self.q.push(self.now, "try_advance", {"tr": w, "idx": wt.pos})
                break

    def _on_deadlock_check(self, d: dict) -> None:
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.done or t.pos != idx or t.blocked_since is None:
            return
        if self.now - t.blocked_since >= MAX_WAIT_S:
            self.deadlocks.append({"t": self.now, "train": t.train_id, "tc": t.path[idx],
                                   "blocked_next": t.path[idx + 1] if idx + 1 < len(t.path) else None})
            t.wait_time += self.now - t.blocked_since
            t.blocked_since = None
            self.q.push(self.now + self._tail(t.path[idx]), "release",
                        {"tc": t.path[idx], "tr": t.train_id})
            if idx + 1 < len(t.path):
                self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})


class DerbyEnv:
    """gym 式包装: reset(trains) → 跑到首个决策点; step(route_id) → 跑到下一个决策/终局.

    M8: obs = 防泄露状态快照(_build_state); reward=0.0(奖励留 M10). 单智能体, 每 step 决策一条进路(D7).
    """

    def __init__(self, net, params: dict | None = None,
                 plat_tc_to_pid: dict | None = None,
                 dwell_by_platform: dict | None = None,
                 advance_cfg: dict | None = None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.advance_cfg = advance_cfg
        self.engine: SteppableReactiveEngine | None = None
        self._gen = None
        self._pending: DecisionContext | None = None

    def reset(self, trains: list[EnvTrain]):
        self.engine = SteppableReactiveEngine(
            self.net, self.params, self.plat_tc_to_pid, self.dwell_by_platform, self.advance_cfg)
        self._gen = self.engine.run(trains)
        self._pending = next(self._gen, None)
        return self._obs(), self._info()

    def step(self, route_id: str | None):
        if self._gen is None or self._pending is None:
            raise RuntimeError("step() called on finished/uninitialized env")
        try:
            self._pending = self._gen.send(route_id)
            done = False
        except StopIteration:
            self._pending = None
            done = True
        return self._obs(), 0.0, done, self._info()

    @property
    def done(self) -> bool:
        return self._pending is None

    def decision(self) -> DecisionContext | None:
        return self._pending

    def _obs(self):
        return self._pending.obs if self._pending is not None else None

    def _info(self) -> dict:
        return {"decision": self._pending}


def build_env_trains_for_window(td_win, net, dwell_by_platform, plat_tc_to_pid, cif_map=None):
    """从窗口 TD 为每车构造 EnvTrain(route_chain/req_times/entry/stop_pids[/CIF 目标]).

    过滤规则与 replay.build_trains_for_window 严格一致(len(path)<2 跳过; path[0] 需有观测占用),
    保证两条构造路径产出同一列车集 → 等价回归才有意义.
    cif_map(可选): {headcode: {'booked_platform','planned_ref','service_class'}} 供状态字段; 缺省不影响等价.
    """
    td_win = td_win.sort_values("ts")
    out = []
    for hc, g in td_win.groupby("trainid_filled"):
        if hc in ("0", "****", ""):
            continue
        g = g.sort_values("ts")
        # H6 切趟: 同 headcode 日内可复用/多趟 → 按 90min 时间间隙切开, 免得把两趟(如 1M00
        # 09:40 与 22:51, 隔 13h)拼成一条非连续链。真折返(停站几分钟)间隔小, 不会被切。
        secs = [t.timestamp() for t in g["ts"]]
        breaks = [i + 1 for i in range(len(secs) - 1) if secs[i + 1] - secs[i] > _RUN_GAP_S]
        bounds = [0] + breaks + [len(g)]
        n_runs = len(bounds) - 1
        for ri in range(n_runs):
            gr = g.iloc[bounds[ri]:bounds[ri + 1]]
            tid = hc if n_runs == 1 else f"{hc}#{ri}"
            prs = gr[(gr["type"] == "Panel_Request") & (gr["state"] == "1")]
            route_chain, req_times, path = [], [], []
            for x in prs.itertuples():
                r = net.routes.get(x.id.strip())
                if not r:
                    continue
                route_chain.append(r.route_id)
                req_times.append(x.ts.timestamp())
                for tc in r.tc_sequence:
                    if not path or path[-1] != tc:
                        path.append(tc)
            if len(path) < 2:
                continue
            occ = gr[(gr["type"] == "Track") & (gr["state"] == "1")]
            occ_first: dict[str, float] = {}
            for x in occ.itertuples():
                tc = x.id.strip()
                if tc not in occ_first:
                    occ_first[tc] = x.ts.timestamp()
            if path[0] not in occ_first:
                continue
            entry = occ_first[path[0]]
            pids = set()
            for sid in gr[gr["type"] == "TRTS"]["id"]:
                m = _LPLAT_RE.search(str(sid))
                if m:
                    pids.add(int(m.group(1)))
            et = EnvTrain(tid, route_chain, req_times, entry, pids)
            et.service_class = str(hc)[0] if hc else ""
            if cif_map and hc in cif_map:
                c = cif_map[hc]
                et.booked_platform = c.get("booked_platform")
                et.planned_ref = c.get("planned_ref")
                if c.get("service_class"):
                    et.service_class = c["service_class"]
            out.append((et, occ_first))
    return out
