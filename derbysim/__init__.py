"""derbysim —— 德比车站微观仿真器与信号员决策学习.

分层(见 docs/IMPLEMENTATION_PLAN.md):
    infra/      静态设施模型(仅标准库)
    data/       Phase 0 数据读取/对齐(pandas/pyarrow)   [M2]
    sim/        Phase 2 自研事件驱动仿真器               [M3-M4]
    validation/ 回放验证(命门)                          [M5-M6]
"""

__version__ = "0.1.0"
