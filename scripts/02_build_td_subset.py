"""M2b 入口: 扫 TD → 区域子集 parquet + berth 邻接, 打印/导出报告.

运行:
    python scripts/02_build_td_subset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.data.td_loader import build_td_subset


def main() -> int:
    report = build_td_subset()
    print("=" * 60)
    print("TD 区域子集 (M2b)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:28} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_REPORTS / "td_subset.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
