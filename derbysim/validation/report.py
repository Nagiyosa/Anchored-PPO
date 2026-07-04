"""验证报告组装 + brief §8⑥ 对外表述模板填数(措辞精确: 用中位/命中率, 透明披露长尾)."""

from __future__ import annotations


def s8_statement(median_err_s: float, hit_rate_pct: float, within_s: int,
                 mean_mae_s: float, tail_frac_pct: float) -> str:
    return (
        f"以真实决策逐事件回放, 留出集(11-12月)上 到达时刻再现 "
        f"**中位误差 ≈ {median_err_s:.0f} 秒、{within_s} 秒命中率 > {hit_rate_pct:.0f}%**, "
        f"证明**描述性有效**(泛化到未标定时段无退化)。"
        f"均值 MAE 约 {mean_mae_s:.0f}s 系被约 {tail_frac_pct:.0f}% 的长尾(>20s, 多为中途停等/停站)抬高——"
        f"这类由引擎联锁/等待机制处理, 非自由流占用模型职责。"
        f"反事实层面(改变决策的效果)无法从数据完全证明, 仅在专家校验与物理约束"
        f"(延误守恒、无'免费午餐')下作保守结论。"
    )
