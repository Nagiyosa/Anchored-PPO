"""DerbyEnv (M7): gym 式 reset/step 环境骨架 —— 反应式引擎的可暂停版.

与 reactive.ReactiveEngine 的关系:
    ReactiveEngine 跑**预拼好的整条 path**(路由链已烘焙进 path/ready_at), 一次跑完.
    本模块把"整条 path"拆成**逐路由段增量构造**: 列车走到当前已物化 path 末端且
    路由链还有下一段时 → **暂停(yield 决策情境)** → step(所选进路) 追加该段 TC → 继续.

M7 只交付**骨架 + 等价回归**: 回放模式(每步选日志进路)必须与 ReactiveEngine 跑
build_trains 的整条 path 逐位一致(per-train wait_time / sim_occ)。
obs/reward 在 M7 是占位 —— 真正的**防泄露状态**留给 M8, 避免过早把泄露引进来.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from derbysim.sim.events import EventQueue
from derbysim.sim.reactive import DEFAULT_ADVANCE, DEFAULT_OCC, MAX_WAIT_S

_LPLAT_RE = re.compile(r"LPLAT(\d+)")


@dataclass
class DecisionContext:
    """一次决策点情境(单智能体, 事件时序). M7: mask 为初步(net 出口), 状态占位."""
    train_id: str
    signal: int | None            # 决策信号 = 上一段进路的 end_signal
    route_ptr: int                # 将要决定第 route_ptr 条进路
    legal_route_ids: list[str]    # 动作掩码(M7 初步: 该信号全部出口; 联锁掩码留待后续)
    logged_route_id: str | None   # 回放标签(日志真实进路), 也是 BC 目标
    now: float


@dataclass
class EnvTrain:
    """增量式列车: 携带路由链(决策序列), path 由引擎逐段物化."""
    train_id: str
    route_chain: list[str]                     # 有序 route_id(决策序列; 回放=日志 Panel_Request)
    req_times: list[float]                     # 平行: 各进路办理时刻(ready 门禁)
    entry_time: float
    stop_pids: set = field(default_factory=set)
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

    def ready(self, idx: int) -> float:
        return self.ready_at[idx] if self.ready_at and idx < len(self.ready_at) else -1e18


class SteppableReactiveEngine:
    """反应式引擎的生成器版: 到决策点 yield DecisionContext, send(route_id) 续跑.

    物理/门禁公式与 reactive.ReactiveEngine 完全一致, 仅把"路由已知"改成"决策点问".
    """

    def __init__(self, net, params: dict | None = None,
                 plat_tc_to_pid: dict | None = None,
                 dwell_by_platform: dict | None = None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.q = EventQueue()
        self.occupied: dict[str, str] = {}
        self.waiting: dict[str, deque] = defaultdict(deque)
        self.trains: dict[str, EnvTrain] = {}
        self.now = 0.0
        self.deadlocks: list = []
        self.n_wait_events = 0

    def _adv(self, tc: str) -> float:
        return self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))[0]

    def _tail(self, tc: str) -> float:
        a, o = self.params.get(tc, (DEFAULT_ADVANCE, DEFAULT_OCC))
        return max(1.0, o - a)

    def _materialize(self, t: EnvTrain, route_id: str) -> None:
        """把所选进路的 TC 序列追加到 path(跨段去重, ready=该进路办理时刻)."""
        r = self.net.routes.get(route_id)
        rt = t.req_times[t.route_ptr] if t.route_ptr < len(t.req_times) else self.now
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

    def _decision_signal(self, t: EnvTrain) -> int | None:
        prev = t.route_chain[t.route_ptr - 1] if t.route_ptr >= 1 else None
        r = self.net.routes.get(prev) if prev else None
        return r.end_signal if r else None

    def run(self, trains: list[EnvTrain]):
        """生成器: 逐决策点 yield DecisionContext; 外部 send(route_id)."""
        self.trains = {t.train_id: t for t in trains}
        # 入口段(route_chain[0]) 在 reset 物化, 使 entry 能锚在 path[0]; 非 step 决策(M7)
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
            elif ev.kind == "try_advance":
                sig = self._try_advance(ev.data)
                if sig is not None:                 # 需要下一段进路 → 暂停问决策
                    t = sig
                    dsig = self._decision_signal(t)
                    legal = [r.route_id for r in self.net.routes_from(dsig)] if dsig else []
                    logged = (t.route_chain[t.route_ptr]
                              if t.route_ptr < len(t.route_chain) else None)
                    ctx = DecisionContext(t.train_id, dsig, t.route_ptr, legal, logged, self.now)
                    chosen = yield ctx
                    self._materialize(t, chosen)
                    self.q.push(self.now, "try_advance", {"tr": t.train_id, "idx": t.pos})
            elif ev.kind == "release":
                self._on_release(ev.data)
            elif ev.kind == "deadlock_check":
                self._on_deadlock_check(ev.data)

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
        """返回 None=正常; 返回 EnvTrain=该车需要下一段进路(触发决策)."""
        t = self.trains[d["tr"]]
        idx = d["idx"]
        if t.pos != idx or t.done:
            return None
        tc = t.path[idx]
        if idx + 1 >= len(t.path):
            if t.route_ptr < len(t.route_chain):    # 还有下一段 → 决策
                return t
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            t.done = True
            return None
        nxt = t.path[idx + 1]
        ready = t.ready(idx + 1)
        if self.now < ready:                        # 门禁1: 进路未办理
            self.q.push(ready, "try_advance", {"tr": t.train_id, "idx": idx})
            return None
        holder = self.occupied.get(nxt)
        if holder is None or holder == t.train_id:  # 门禁2: 下一段空 → 进
            if t.blocked_since is not None:
                t.wait_time += self.now - t.blocked_since
                t.blocked_since = None
            self.q.push(self.now + self._tail(tc), "release", {"tc": tc, "tr": t.train_id})
            self.q.push(self.now, "enter", {"tr": t.train_id, "idx": idx + 1})
        else:                                       # 被占 → 停等
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

    M7: obs 占位, reward=0.0. 单智能体, 每 step 决策一条进路(D7).
    """

    def __init__(self, net, params: dict | None = None,
                 plat_tc_to_pid: dict | None = None,
                 dwell_by_platform: dict | None = None) -> None:
        self.net = net
        self.params = params or {}
        self.plat_tc_to_pid = plat_tc_to_pid or {}
        self.dwell_by_platform = dwell_by_platform or {}
        self.engine: SteppableReactiveEngine | None = None
        self._gen = None
        self._pending: DecisionContext | None = None

    def reset(self, trains: list[EnvTrain]):
        self.engine = SteppableReactiveEngine(
            self.net, self.params, self.plat_tc_to_pid, self.dwell_by_platform)
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
        return {}   # M7 占位; M8 填防泄露状态

    def _info(self) -> dict:
        return {"decision": self._pending}


def build_env_trains_for_window(td_win, net, dwell_by_platform, plat_tc_to_pid):
    """从窗口TD为每车构造 EnvTrain(route_chain/req_times/entry/stop_pids).

    过滤规则与 replay.build_trains_for_window 严格一致(len(path)<2 跳过; path[0] 需有观测占用),
    以保证两条构造路径产出**同一列车集** → 等价回归才有意义.
    """
    td_win = td_win.sort_values("ts")
    out = []
    for hc, g in td_win.groupby("trainid_filled"):
        if hc in ("0", "****", ""):
            continue
        prs = g[(g["type"] == "Panel_Request") & (g["state"] == "1")]
        route_chain, req_times, path = [], [], []
        for x in prs.itertuples():
            r = net.routes.get(x.id.strip())
            if not r:
                continue
            route_chain.append(r.route_id)
            req_times.append(x.ts.timestamp())
            for tc in r.tc_sequence:               # 仅为复现过滤/entry 的去重 path
                if not path or path[-1] != tc:
                    path.append(tc)
        if len(path) < 2:
            continue
        occ = g[(g["type"] == "Track") & (g["state"] == "1")]
        occ_first: dict[str, float] = {}
        for x in occ.itertuples():
            tc = x.id.strip()
            if tc not in occ_first:
                occ_first[tc] = x.ts.timestamp()
        if path[0] not in occ_first:
            continue
        entry = occ_first[path[0]]
        pids = set()
        for sid in g[g["type"] == "TRTS"]["id"]:
            m = _LPLAT_RE.search(str(sid))
            if m:
                pids.add(int(m.group(1)))
        out.append((EnvTrain(hc, route_chain, req_times, entry, pids), occ_first))
    return out
