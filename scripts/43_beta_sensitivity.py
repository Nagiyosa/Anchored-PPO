"""D82: β 敏感性图 —— 延误 vs β_r(routing anchor)/ β_t(timing anchor)。
数据 = notebooks/d82_beta_sensitivity.ipynb 的 5 个 sens_*_s0.json(服务器跑完拷回)
     + 已有点: 中心 = ppo_d73_AB_s0 前 50 轮 byR(同预算), β_r=0 端点 = abl_noroute_s0。
口径与 abl_* 消融一致: d73_AB 配置、前 50 轮预算、单种子、best-by-reward 选择。
产出: figures/pub/F15_beta_sensitivity.{svg,pdf,png}; 缺文件则跳过对应点并提示。
用法: python scripts/43_beta_sensitivity.py
"""
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
M = {"rl": "#A2665F", "fcfs": "#7C8C7D", "grid": "#E5E1DC", "dark": "#4A4A4A"}
REP = "artifacts/reports"


def by_reward(path, max_iter=None):
    """与训练一致的选择规则: 验证 reward 最高的 checkpoint(可限制迭代预算)。"""
    evs = json.load(open(path))["evals"]
    if max_iter is not None:
        evs = [e for e in evs if e["iter"] < max_iter]
    b = max(evs, key=lambda e: e["eval_episode_reward"])
    return b["eval_weighted_secondary_delay_s"] / 1000


CENTER = by_reward(f"{REP}/ppo_d73_AB_s0.json", max_iter=50)          # 123.6k @44(同预算中心)

# (β 值, 数据来源, 备注)   None 路径 = 用中心点
panels = [
    ("routing anchor", r"$\beta_r$  (timing anchor fixed at 0.01)", [
        (0.0,   f"{REP}/abl_noroute_s0.json", "0\n(no anchor)"),
        (0.01,  f"{REP}/sens_br001_s0.json",  "0.01"),
        (0.03,  None,                          "0.03\n(ours)"),
        (0.1,   f"{REP}/sens_br01_s0.json",   "0.1"),
    ]),
    ("timing anchor", r"$\beta_t$  (routing anchor fixed at 0.03)", [
        (0.0,   f"{REP}/sens_bt0_s0.json",    "0\n(no anchor)"),
        (0.005, f"{REP}/sens_bt0005_s0.json", "0.005"),
        (0.01,  None,                          "0.01\n(ours)"),
        (0.02,  f"{REP}/sens_bt002_s0.json",  "0.02"),
    ]),
]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, (title, xlabel, pts) in zip(axes, panels):
    xs, ys, labs, ours_i = [], [], [], None
    for i, (beta, path, lab) in enumerate(pts):
        if path is None:
            v = CENTER; ours_i = len(xs)
        elif os.path.exists(path):
            v = by_reward(path)
        else:
            print(f"[跳过] 缺 {path}(β={beta})"); continue
        xs.append(len(xs)); ys.append(v); labs.append(lab)
    ax.plot(xs, ys, color=M["rl"], lw=1.5, marker="o", ms=4, zorder=4)
    if ours_i is not None:
        ax.scatter([xs[ours_i]], [ys[ours_i]], s=42, color=M["rl"],
                   edgecolor=M["dark"], lw=1.2, zorder=5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}k", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=M["dark"])
    ax.axhline(181.164, color=M["fcfs"], lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, color=M["dark"])
    ax.grid(axis="y", color=M["grid"], lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("validation delay (×10³ s)\n50-iteration budget, best by reward", fontsize=8)
axes[1].text(0.98, 181.164, "FCFS", transform=axes[1].get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7, color=M["fcfs"])
fig.tight_layout()
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"figures/pub/F15_beta_sensitivity.{ext}", bbox_inches="tight", **kw)
print("saved figures/pub/F15_beta_sensitivity.*")
