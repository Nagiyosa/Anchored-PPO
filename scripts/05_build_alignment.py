"""M2e 入口: 三源对齐 → train_occurrences.parquet, 打印/导出报告 + 抽 3 车次核验."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.data.alignment import build_train_occurrences


def main() -> int:
    report = build_train_occurrences()
    print("=" * 60)
    print("三源对齐: 车次实例 (M2e)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:28} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_REPORTS / "train_occurrences.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
