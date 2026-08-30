"""D83: 其余超参敏感性图 —— lr / steps-per-iter / promote-delay-k / fail-penalty 四面板。
数据 = notebooks/d83_hparam_sensitivity.ipynb 的 sens_*_s0.json(服务器跑完拷回);
中心点 = ppo_d73_AB_s0 前 50 轮 byR(同预算, 不另跑)。β 双锚见 scripts/43(F15)。
产出: figures/pub/F16_hparam_sensitivity.{svg,pdf,png}; 缺文件自动跳过对应点。
用法: python scripts/44_hparam_sensitivity.py
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
    evs = json.load(open(path))["evals"]
    if max_iter is not None:
        evs = [e for e in evs if e["iter"] < max_iter]
    b = max(evs, key=lambda e: e["eval_episode_reward"])
    return b["eval_weighted_secondary_delay_s"] / 1000


CENTER = by_reward(f"{REP}/ppo_d73_AB_s0.json", max_iter=50)

# (面板题, x 轴标签, [(刻度标签, 文件 or None=中心)])
panels = [
    ("learning rate", "lr", [
        ("2e-5", f"{REP}/sens_lr2e5_s0.json"),
        ("5e-5\n(ours)", None),
        ("1e-4", f"{REP}/sens_lr1e4_s0.json"),
    ]),
    ("batch size", "steps per iteration (equal total env steps)", [
        ("2048", f"{REP}/sens_sp2048_s0.json"),
        ("4096", f"{REP}/sens_sp4096_s0.json"),
        ("8192\n(ours)", None),
    ]),
    ("curriculum promotion", r"promotion threshold $k$ (delay $\leq k\times$FCFS)", [
        ("1.0\n(ours)", None),
        ("1.1", f"{REP}/sens_pk11_s0.json"),
        ("1.25", f"{REP}/sens_pk125_s0.json"),
    ]),
    ("stranding penalty", r"$K_{fail}$", [
        ("25", f"{REP}/sens_fp25_s0.json"),
        ("50\n(ours)", None),
        ("100", f"{REP}/sens_fp100_s0.json"),
    ]),
]

fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.7), sharey=True)
for ax, (title, xlabel, pts) in zip(axes, panels):
    xs, ys, labs, ours_i = [], [], [], None
    for lab, path in pts:
        if path is None:
            v = CENTER; ours_i = len(xs)
        elif os.path.exists(path):
            v = by_reward(path)
        else:
            print(f"[跳过] 缺 {path}"); continue
        xs.append(len(xs)); ys.append(v); labs.append(lab)
    ax.plot(xs, ys, color=M["rl"], lw=1.5, marker="o", ms=4, zorder=4)
    if ours_i is not None and ours_i < len(xs):
        ax.scatter([xs[ours_i]], [ys[ours_i]], s=42, color=M["rl"],
                   edgecolor=M["dark"], lw=1.2, zorder=5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}k", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=M["dark"])
    ax.axhline(181.164, color=M["fcfs"], lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9.5, color=M["dark"])
    ax.grid(axis="y", color=M["grid"], lw=0.5); ax.set_axisbelow(True)
    if xs:
        ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
axes[0].set_ylabel("validation delay (×10³ s)\nequal budget, best by reward", fontsize=8)
axes[3].text(0.98, 181.164, "FCFS", transform=axes[3].get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7, color=M["fcfs"])
fig.tight_layout()
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"figures/pub/F16_hparam_sensitivity.{ext}", bbox_inches="tight", **kw)
print("saved figures/pub/F16_hparam_sensitivity.*")
