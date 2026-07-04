"""全局配置: 路径、实验区域白名单、决策点、常量.

单一事实来源见 PROJECT_BRIEF.md §3.1 与 docs/IMPLEMENTATION_PLAN.md.
本模块仅用标准库, 供各层安全导入.
"""

from __future__ import annotations

from pathlib import Path

# ── 路径 ────────────────────────────────────────────────────────────────
# 仓库根 = derbysim/ 的上一级
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# 静态设施数据(小文件, 随仓库)
DATA_DERBY_INFO = REPO_ROOT / "Derby_info.csv"
DATA_ROUTE_TC = REPO_ROOT / "route_to_tc_all.csv"
DATA_PLATFORM_TC = REPO_ROOT / "platform_tc_map.csv"
DATA_PLATFORM_END_SIGNALS = REPO_ROOT / "platform_end_signals.csv"

# 大文件(gitignore, 本地存在)
DATA_TD = REPO_ROOT / "withtrainid_23-901-127.csv"          # TD 秒级事件
DATA_TRUST = REPO_ROOT / "mov_2023_9_1-2023_12_7.csv"        # TRUST 到发/延误
DATA_CIF = REPO_ROOT / "toc-full.CIF"                        # 全网 WTT

# 中间产物(gitignore)
ARTIFACTS = REPO_ROOT / "artifacts"
ARTIFACTS_SUBSET = ARTIFACTS / "subset"       # 区域过滤后的 parquet
ARTIFACTS_CALIB = ARTIFACTS / "calib"         # 标定参数
ARTIFACTS_REPORTS = ARTIFACTS / "reports"     # 验证/自检报告

# ── 实验区域 (PROJECT_BRIEF §3.1: 站心咽喉, 拓扑2跳邻域) ──────────────────
# 48 个"在范围"信号(决策点 + 站台端 + 途经). 信号号跨 TD 前缀(DC/TD/DW/DY/EC) 唯一.
IN_SCOPE_SIGNALS: frozenset[int] = frozenset({
    5040, 5044, 5045, 5046, 5049, 5053, 5054, 5055, 5056, 5061,
    5062, 5063, 5064, 5065, 5066, 5075, 5076, 5077, 5078, 5079,
    5080, 5081, 5082, 5083, 5085, 5087, 5089, 5300, 5301, 5302,
    5304, 5306, 5308, 5310, 5311, 5313, 5315, 5319, 5321, 5474,
    5475, 5484, 5486, 5487, 5488, 5489, 5491, 5493,
})

# 35 个决策点(区域内所有 >1 出口信号). 按区域分组, 便于报告/调试.
DECISION_POINTS_BY_AREA: dict[str, tuple[int, ...]] = {
    "south_throat": (
        5045, 5046, 5049, 5053, 5054, 5056, 5300, 5304,
        5306, 5308, 5310, 5311, 5313, 5315, 5319, 5321,
    ),
    "north_throat": (5076, 5063, 5064, 5065, 5066, 5078, 5080, 5082, 5085, 5087, 5089),
    "etches_park_ecs": (5475, 5484, 5486, 5487, 5488, 5489, 5491, 5493),
}
DECISION_POINTS: frozenset[int] = frozenset(
    s for group in DECISION_POINTS_BY_AREA.values() for s in group
)

# 区域 berth 集(TD 事件过滤用). 由 platform-end berth 的 2 跳 CA 邻域得 63 berth
# = 48 数字 berth(恰为 48 在范围信号) + 15 个中间/前置 X-berth(如 X302 是 5302 前置).
# 这 15 个是基础设施, 稳定, 硬编码以免每次扫 272MB TD 才能得区域定义.
REGION_INTERMEDIATE_BERTHS: frozenset[str] = frozenset({
    "EPAP", "X053", "X054", "X055", "X056", "X061", "X062", "X063",
    "X064", "X065", "X066", "X301", "X302", "X491", "X493",
})
# 完整 63-berth 区域(字符串形式, 数字 berth 与信号号同名)
REGION_BERTHS: frozenset[str] = frozenset(
    {str(s) for s in IN_SCOPE_SIGNALS} | REGION_INTERMEDIATE_BERTHS
)

# 德比站台(platform_end_signals / platform_tc_map 得 6 个)
DERBY_PLATFORMS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6})

# ── 常量 ────────────────────────────────────────────────────────────────
SEED = 20230901                 # 全局随机种子(回放应完全确定)
DST_CHANGE = "2023-10-29"       # BST→GMT 换钟(已证 TD/TRUST 跨此统一)
CALIB_SPLIT = ("2023-09-01", "2023-10-31")   # 标定集(9-10月)
HOLDOUT_SPLIT = ("2023-11-01", "2023-12-07")  # 留出验证集(11-12月)

# 自检: 常量自洽
assert len(IN_SCOPE_SIGNALS) == 48, f"in-scope 应为48, 实为 {len(IN_SCOPE_SIGNALS)}"
assert len(DECISION_POINTS) == 35, f"决策点应为35, 实为 {len(DECISION_POINTS)}"
assert DECISION_POINTS <= IN_SCOPE_SIGNALS, "决策点必须是在范围信号的子集"
assert len(REGION_BERTHS) == 63, f"区域berth应为63, 实为 {len(REGION_BERTHS)}"
