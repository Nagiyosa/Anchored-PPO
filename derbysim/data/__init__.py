"""数据层 (Phase 0): 读取 / 区域过滤 / 对齐 (pandas·pyarrow).

模块:
    region.py       48信号/63berth 白名单过滤 + TD id 解析
    td_loader.py    TD 秒级事件 → 区域子集 parquet + berth 邻接
    trust_loader.py TRUST 到发/延误 → 区域子集 parquet (signed_variation)   [M2c]
    cif_parser.py   CIF 固定宽 + STP 叠层 → 德比每日实际计划 parquet          [M2d]
    alignment.py    headcode↔train_id 复合键连接 + STANOX↔站台推导            [M2e]
    timeline.py     统一"事件-决策"时间线                                     [M2e]
"""
