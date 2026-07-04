"""M4 引擎/联锁测试: 确定性小场景(fake 网), 验证互斥/延迟/掩码/回放接口."""

from dataclasses import dataclass

from derbysim.sim.engine import Engine
from derbysim.sim.interlocking import Interlocking
from derbysim.sim.router import ReplayRouter


@dataclass
class R:
    route_id: str
    tc_sequence: tuple
    signal: int = 0


class FakeNet:
    def __init__(self, routes):
        self.routes = {r.route_id: r for r in routes}

    def routes_from(self, signal):
        return [r for r in self.routes.values() if r.signal == signal]


def test_interlocking_no_double_book_and_defer():
    # 两车争用共享TC 'X'
    net = FakeNet([R("A", ("X", "Y"), 1), R("B", ("X", "Z"), 2)])
    reqs = [
        {"train": "T1", "route_id": "A", "t_req": 0, "t_entry": 0},
        {"train": "T2", "route_id": "B", "t_req": 1, "t_entry": 1},
    ]
    res = Engine(net, {}).run(reqs)
    assert res.interlocking_violations == []
    assert res.n_locked == 2           # T2 等 X 空出后成功
    assert res.n_deferred >= 1
    assert res.persistent_conflicts == []


def test_nonconflicting_lock_immediately():
    net = FakeNet([R("A", ("X",), 1), R("B", ("Y",), 2)])
    res = Engine(net, {}).run([
        {"train": "T1", "route_id": "A", "t_req": 0, "t_entry": 0},
        {"train": "T2", "route_id": "B", "t_req": 0, "t_entry": 0},
    ])
    assert res.n_locked == 2 and res.n_deferred == 0
    assert res.interlocking_violations == []


def test_action_mask_excludes_conflicting():
    net = FakeNet([R("A", ("X", "Y"), 5), R("B", ("Y", "Z"), 5), R("C", ("W",), 5)])
    il = Interlocking(net)
    assert {r.route_id for r in il.legal_routes(5)} == {"A", "B", "C"}
    il.lock(net.routes["A"], "T1")     # 持有 X,Y
    assert {r.route_id for r in il.legal_routes(5)} == {"C"}   # B 共享Y被排除


def test_sim_occupation_generated():
    net = FakeNet([R("A", ("X", "Y", "Z"), 1)])
    res = Engine(net, {}).run([{"train": "T1", "route_id": "A", "t_req": 0, "t_entry": 100}])
    tcs = [o.tc for o in sorted(res.sim_occupations, key=lambda o: o.t_occupied)]
    assert tcs == ["X", "Y", "Z"]
    assert res.sim_occupations[0].t_occupied == 100


def test_replay_router():
    assert ReplayRouter().request_route({"logged_route": "RTD5045A(M)"}) == "RTD5045A(M)"
    assert ReplayRouter().request_route({}) is None
