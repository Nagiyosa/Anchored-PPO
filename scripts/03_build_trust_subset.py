"""M2c 入口: TRUST → 规范化 parquet(signed_variation), 打印/导出报告."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derbysim import config
from derbysim.data.trust_loader import build_trust_subset


def main() -> int:
    report = build_trust_subset()
    print("=" * 60)
    print("TRUST 规范化子集 (M2c)")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:26} {v}")
    config.ARTIFACTS_REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.ARTIFACTS_REPORTS / "trust_subset.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
