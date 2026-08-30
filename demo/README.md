# Derby Dispatch Demo — Rail Route 式回放演示

在真实 Derby 咽喉区信号图(`derby_experiment_region.svg`)上回放各策略一整天的调度:
车头码沿轨道移动、已锁进路高亮、信号红绿灯、KPI 实时累计、双视图同步对比、周总览记分牌。
数据 = 预生成的逐事件时间线 JSON(引擎按论文评估口径重放: proactive + wide + 受控客运 cls{1,2},
锁/放事件经 `would_create_deadlock` 前瞻静音,逐日延误与 `conflict_analysis_test.json` 逐位一致)。

## 目录

- `generate_timeline.py`  跑引擎生成 `(day, policy)` 时间线 → `data/*.json`
- `app.py`                Flask 服务(只读 data/,无 torch 依赖)
- `static/`               前端(原生 JS,无构建步骤)
- `data/`                 生成的时间线 + topology.json

## 一次性准备(你的本机, PowerShell, 仓库根目录)

信号员/FCFS/lookup 已生成(2023-11-16..24)。**RL(Anchored PPO)与 BC 需要 torch,在本机跑:**

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python demo\generate_timeline.py --days 2023-11-16..2023-11-24 --policies rl
# 可选: 最强学习基线 BC 对比
.venv\Scripts\python demo\generate_timeline.py --days 2023-11-16..2023-11-24 --policies bc
```

默认 ckpt = `artifacts/reports/ppo_d73_AB_s0_best.pt`(论文基准配置)与
`artifacts/dataset/bc_fullpanel_gnn.pt`;换 ckpt 用 `--ckpt` / `--bc-ckpt`。
每天每策略约 2–10 秒。

## 启动

```powershell
.venv\Scripts\python -m pip install flask   # 若未装
$env:PYTHONPATH="."
.venv\Scripts\python demo\app.py            # -> http://127.0.0.1:5045
```

## 操作

- 空格 / ▶ 播放暂停;1×–600× 倍速;顶部时间轴拖动 seek
- 每个视图左上下拉换策略;滚轮缩放、拖拽平移(两视图同步)
- 「周总览」= 逐日 × 逐策略延误记分牌,点击任意一天进入回放
- 右栏: 双策略 KPI 对比、累计延误曲线(白线=当前时刻)、入区等待队列、主视图决策流
- 黄色/橙色车 = 受控客运(cls 1/2),灰色 = 背景车(按历史自动运行,不受策略控制)
- 演示周 2023-11-16..11-22(封存 test,RL 对 FCFS 七天全胜);11-23/24 附带(11-24 为论文失败案例日)

## 已知近似(仅影响画面,不影响数字)

- 车辆位置按"进入 TC 时刻 + 进路折线等分"插值,悬停≈停车(红框);非测绘级定位
- call-on (C) 进路复用同名 (M) 几何;区外轨道不在图上,车入区才出现
