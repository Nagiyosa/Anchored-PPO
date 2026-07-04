"""Cut2 反应式引擎测试: 闭塞机制正确性(被占则等, 次生延误因果涌现)."""

from derbysim.sim.reactive import ReactiveEngine, Train


def test_blocking_wait_emerges():
    # 默认 advance=8 occ=18 tail=10; A、B 共用中间段 M
    A = Train("A", ["X", "M", "Y"], entry_time=0)
    B = Train("B", ["P", "M", "Q"], entry_time=5)
    eng = ReactiveEngine()
    res = eng.run([A, B])
    # A 自由流: X@0 M@8 Y@16
    assert round(res["A"].sim_occ["M"]) == 8
    # B 被 A 挡在 P, 等到 A 于 26 释放 M 才进; 自由流应 13 → 次生延误 13s
    assert round(res["B"].sim_occ["M"]) == 26
    assert round(res["B"].wait_time) == 13
    assert res["A"].wait_time == 0
    assert eng.n_wait_events == 1
    assert len(eng.deadlocks) == 0


def test_no_conflict_no_wait():
    A = Train("A", ["X", "Y"], entry_time=0)
    B = Train("B", ["P", "Q"], entry_time=0)     # 无共享段
    eng = ReactiveEngine()
    res = eng.run([A, B])
    assert res["A"].wait_time == 0 and res["B"].wait_time == 0
    assert eng.n_wait_events == 0


def test_route_ready_gate_holds_until_set():
    # 下一段进路 100s 才办理 → 车在首段等到 100s
    A = Train("A", ["X", "Y"], entry_time=0, ready_at=[0, 100])
    eng = ReactiveEngine()
    res = eng.run([A])
    assert round(res["A"].sim_occ["Y"]) == 100     # 被门禁扣到进路办理时刻


def test_deadlock_broken_by_timeout():
    # 两车交叉对撞: A 占M要N, B 占N要M → 死锁, MAX_WAIT后强推
    A = Train("A", ["M", "N"], entry_time=0)
    B = Train("B", ["N", "M"], entry_time=0)
    eng = ReactiveEngine()
    eng.run([A, B])
    assert len(eng.deadlocks) >= 1                  # 死锁被检测并兜底
