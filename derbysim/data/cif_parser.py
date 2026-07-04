"""CIF 固定宽解析 + STP 叠层 → 德比车次每日实际计划.

要点(H5):
- CIF 是分层数据库: BS(基本时刻表)记录带 有效期 + 运行日bitmask + STP位(P/O/N/C).
- STP 取消(C)记录**无位置体**, 靠共享 UID 取消永久车次 → 必须索引全部 BS 头(按UID).
- 德比站 TIPLOC 精确为 'DRBY'(DRBYWWJ/DRBY039 是枢纽/工务, 非站台).
- 某日某车实际计划 = 按 UID+日期 做 STP 优先级解析: C→取消; 否则 O/N 覆盖 P.

BS 定长偏移(0基): uid[3:9] dfrom[9:15] dto[15:21] days[21:28] status[29] cat[30:32]
    headcode[32:36] train_service_code[41:49] stp[79]
位置记录: TIPLOC[2:9]; 时刻 HHMM(+'H'=半分). LO:dep[10:15]; LI:arr[10:15]dep[15:20]pass[20:25]
    plat LI[33:36]; LT:arr[10:15] plat[19:22]; LO:plat[19:22].
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from derbysim import config

DERBY_TIPLOC = "DRBY"


def parse_cif_time(s: str) -> int | None:
    """'0751'->秒(自午夜); '0709H'->+30s; 空/'0000'->None."""
    s = (s or "").strip()
    if not s or s in ("0000", "00000000"):
        return None
    half = s.endswith("H")
    core = s[:-1] if half else s
    if len(core) < 4 or not core[:4].isdigit():
        return None
    hh, mm = int(core[:2]), int(core[2:4])
    return hh * 3600 + mm * 60 + (30 if half else 0)


def _yymmdd(s: str) -> dt.date | None:
    s = (s or "").strip()
    if len(s) != 6 or not s.isdigit():
        return None
    return dt.date(2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))


@dataclass
class ScheduleLayer:
    uid: str
    stp: str
    date_from: dt.date | None
    date_to: dt.date | None
    days_run: str            # 7 字符 MTWTFSS
    headcode: str
    train_service_code: str
    # 德比站计划(秒自午夜), 仅经停德比的层有值
    derby_rec: str | None = None      # LO/LI/LT
    derby_arr: int | None = None
    derby_dep: int | None = None
    derby_pass: int | None = None
    derby_platform: str | None = None

    @property
    def calls_derby(self) -> bool:
        return self.derby_rec is not None

    def runs_on(self, d: dt.date) -> bool:
        if self.date_from and d < self.date_from:
            return False
        if self.date_to and d > self.date_to:
            return False
        if len(self.days_run) == 7 and self.days_run[d.weekday()] != "1":
            return False
        return True


def _parse_bs(line: str) -> ScheduleLayer:
    return ScheduleLayer(
        uid=line[3:9].strip(),
        stp=(line[79:80] if len(line) > 79 else "").strip() or "P",
        date_from=_yymmdd(line[9:15]),
        date_to=_yymmdd(line[15:21]),
        days_run=line[21:28],
        headcode=line[32:36].strip(),
        train_service_code=line[41:49].strip(),
    )


def _fill_derby(layer: ScheduleLayer, line: str) -> None:
    rec = line[:2]
    layer.derby_rec = rec
    layer.derby_platform = line[19:22].strip() if rec in ("LO", "LT") else line[33:36].strip()
    if rec == "LO":
        layer.derby_dep = parse_cif_time(line[10:15])
    elif rec == "LT":
        layer.derby_arr = parse_cif_time(line[10:15])
    else:  # LI
        layer.derby_arr = parse_cif_time(line[10:15])
        layer.derby_dep = parse_cif_time(line[15:20])
        layer.derby_pass = parse_cif_time(line[20:25])


def iter_schedule_layers(cif_path=None):
    """流式产出所有 BS 层(经停德比者带德比时刻)."""
    cif_path = cif_path or config.DATA_CIF
    cur: ScheduleLayer | None = None
    with open(cif_path, encoding="latin-1") as f:
        for line in f:
            rt = line[:2]
            if rt == "BS":
                if cur is not None:
                    yield cur
                cur = _parse_bs(line)
            elif rt in ("LO", "LI", "LT") and cur is not None:
                if line[2:9].strip() == DERBY_TIPLOC:
                    _fill_derby(cur, line)
            elif rt in ("ZZ",):
                if cur is not None:
                    yield cur
                    cur = None
        if cur is not None:
            yield cur


def resolve(layers: list[ScheduleLayer], d: dt.date) -> ScheduleLayer | None:
    """给定同一 UID 的所有层与日期, 按 STP 优先级返回生效层(取消返回 None)."""
    applicable = [ly for ly in layers if ly.runs_on(d)]
    if not applicable:
        return None
    if any(ly.stp == "C" for ly in applicable):
        return None
    stp_layers = [ly for ly in applicable if ly.stp in ("O", "N")]
    if stp_layers:
        # 多个重叠时取日期跨度最小者(最具体)
        return min(stp_layers, key=lambda ly: _span(ly))
    perm = [ly for ly in applicable if ly.stp == "P"]
    return perm[0] if perm else applicable[0]


def _span(ly: ScheduleLayer) -> int:
    if ly.date_from and ly.date_to:
        return (ly.date_to - ly.date_from).days
    return 10_000


def build_cif_schedules(cif_path=None) -> dict:
    """解析 CIF, 保留德比相关 UID 的全部层(含C取消), 写 parquet + 报告."""
    from collections import defaultdict

    by_uid: dict[str, list[ScheduleLayer]] = defaultdict(list)
    n_bs = 0
    for ly in iter_schedule_layers(cif_path):
        n_bs += 1
        by_uid[ly.uid].append(ly)

    derby_uids = {uid for uid, lys in by_uid.items() if any(l.calls_derby for l in lys)}
    rows = []
    stp_counts: dict = {}
    for uid in derby_uids:
        for ly in by_uid[uid]:
            stp_counts[ly.stp] = stp_counts.get(ly.stp, 0) + 1
            rows.append({
                "uid": ly.uid,
                "stp": ly.stp,
                "date_from": ly.date_from,
                "date_to": ly.date_to,
                "days_run": ly.days_run,
                "headcode": ly.headcode,
                "train_service_code": ly.train_service_code,
                "calls_derby": ly.calls_derby,
                "derby_rec": ly.derby_rec,
                "derby_arr": ly.derby_arr,
                "derby_dep": ly.derby_dep,
                "derby_pass": ly.derby_pass,
                "derby_platform": ly.derby_platform,
            })
    df = pd.DataFrame(rows)
    config.ARTIFACTS_SUBSET.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_SUBSET / "cif_schedules.parquet"
    df.to_parquet(path, index=False)

    report = {
        "total_bs_records": n_bs,
        "n_derby_uids": len(derby_uids),
        "n_derby_layers": len(rows),
        "stp_counts_derby_uids": stp_counts,
        "n_calls_derby_layers": int(df["calls_derby"].sum()),
        "n_headcodes": int(df["headcode"].nunique()),
        "cif_parquet": str(path),
    }
    return report
