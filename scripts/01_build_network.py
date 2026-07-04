"""M1 入口: 建设施网并打印/导出自检报告.

运行:
    python scripts/01_build_network.py
产出:
    artifacts/reports/network_selfcheck.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许从仓库根直接运行(无需安装)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.infra import InfrastructureNetwork


def main() -> int:
    net = InfrastructureNetwork.load()
    report = net.self_check()

    print("=" * 60)
    print("德比实验区域 —— 设施网自检 (M1)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:32} {v}")

    # 逐决策点出口数(按区域)
    print("-" * 60)
    print("决策点出口选择度(按区域):")
    for area, sigs in config.DECISION_POINTS_BY_AREA.items():
        print(f"  [{area}]")
        for s in sigs:
            sig = net.signals.get(s)
            n = sig.n_routes if sig else 0
            exits = sorted({net.routes[r].exit for r in (sig.route_ids if sig else [])})
            print(f"    {s}: {n} 进路, 出口={exits}")

    # 导出
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_REPORTS / "network_selfcheck.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"报告已写入: {out}")

    ok = report["decision_points_match_const"] and not report["isolated_in_scope_signals"]
    print("验收:", "PASS ✅" if ok else "CHECK ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
