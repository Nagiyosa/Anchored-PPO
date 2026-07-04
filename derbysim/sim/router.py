"""决策接口: Router 抽象 + ReplayRouter(喂日志真实进路).

这是 brief 架构图"环境⇄策略"的接缝: 现在插 ReplayRouter, 将来 RL 策略以
PolicyRouter 替换到同一 request_route 接口, 引擎零改动.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Router(ABC):
    @abstractmethod
    def request_route(self, decision) -> str | None:
        """给定一个决策点情境, 返回选定的 route_id(或 None=暂不办理)."""


class ReplayRouter(Router):
    """回放: 直接返回日志中记录的真实进路."""

    def request_route(self, decision) -> str | None:
        return decision.get("logged_route")


# PolicyRouter(Router): [Phase 4] RL 策略在此实现 request_route, 从动作掩码中选路.
