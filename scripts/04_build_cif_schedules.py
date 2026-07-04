"""M2d 入口: 解析 CIF(+STP叠层) → 德比 schedule 层 parquet, 打印/导出报告."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.data.cif_parser import build_cif_schedules


def main() -> int:
    report = build_cif_schedules()
    print("=" * 60)
    print("CIF 德比 schedule 层 + STP (M2d)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:26} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_REPORTS / "cif_schedules.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
