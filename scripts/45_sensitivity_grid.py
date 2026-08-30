"""D84: 六参数敏感性(lr 日程对齐版, _i100 数据) 2×3 大图(β_r / β_t / lr / batch / 课程阈值 / 搁浅罚)。
数据 = d84 notebook 的 16 个 *_i100.json + 中心(ppo_d73_AB_s0 前 50 轮 byR)
     + β_r=0 端点(abl_noroute_i100)。口径: 同预算、单种子、byR、单变量。
产出: figures/pub/F17_sensitivity_grid.{svg,pdf,png}; 缺文件自动跳过对应点。
用法: python scripts/45_sensitivity_grid.py
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


CENTER = by_reward(f"{REP}/ppo_d73_AB_s0.json")            # 123.0k, 与 headline 同数

# (面板题, x 轴标签, [(刻度标签, 文件 or None=中心)])
panels = [
    ("routing anchor", r"$\beta_r$", [
        ("0", f"{REP}/abl_noroute_i100.json"),
        ("0.01", f"{REP}/sens_br001_i100.json"),
        ("0.03\n(ours)", None),
        ("0.1", f"{REP}/sens_br01_i100.json")]),
    ("timing anchor", r"$\beta_t$", [
        ("0", f"{REP}/sens_bt0_i100.json"),
        ("0.005", f"{REP}/sens_bt0005_i100.json"),
        ("0.01\n(ours)", None),
        ("0.02", f"{REP}/sens_bt002_i100.json")]),
    ("learning rate", "lr", [
        ("2e-5", f"{REP}/sens_lr2e5_i100.json"),
        ("5e-5\n(ours)", None),
        ("1e-4", f"{REP}/sens_lr1e4_i100.json"),
        ("2e-4", f"{REP}/sens_lr2e4_i100.json")]),
    ("batch size", "steps per iteration (equal total env steps)", [
        ("2048", f"{REP}/sens_sp2048_i100.json"),
        ("4096", f"{REP}/sens_sp4096_i100.json"),
        ("8192\n(ours)", None),
        ("16384", f"{REP}/sens_sp16384_i100.json")]),
    ("curriculum promotion", r"threshold $k$ (delay $\leq k\times$FCFS)", [
        ("0.9", None),
        ("1.0\n(ours)", None),
        ("1.1", None),
        ("1.25", None)]),   # 阈值不生效(D84): 各 k 逐位同 run, 平线为可证明结果
    ("stranding penalty", r"$K_{fail}$", [
        ("25", f"{REP}/sens_fp25_i100.json"),
        ("50\n(ours)", None),
        ("100", f"{REP}/sens_fp100_i100.json"),
        ("200", f"{REP}/sens_fp200_i100.json")]),
]

fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.6), sharey=True)
for ax, (title, xlabel, pts) in zip(axes.flat, panels):
    xs, ys, labs, ours_i = [], [], [], None
    for lab, path in pts:
        if path is None:
            v = CENTER
            if "(ours)" in lab: ours_i = len(xs)
        elif os.path.exists(path):
            v = by_reward(path)
        else:
            print(f"[跳过] 缺 {path}"); continue
        xs.append(len(xs)); ys.append(v); labs.append(lab)
    ax.plot(xs, ys, color=M["rl"], lw=1.5, marker="o", ms=4, zorder=4)
    if ours_i is not None and ours_i < len(xs):
        ax.scatter([xs[ours_i]], [ys[ours_i]], s=44, color=M["rl"],
                   edgecolor=M["dark"], lw=1.2, zorder=5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}k", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.8, color=M["dark"])
    ax.axhline(181.164, color=M["fcfs"], lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_xlabel(xlabel, fontsize=7.8)
    ax.set_title(title, fontsize=9, color=M["dark"])
    ax.grid(axis="y", color=M["grid"], lw=0.5); ax.set_axisbelow(True)
    if xs:
        ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
for ax in axes[:, 0]:
    ax.set_ylabel("validation delay (×10³ s)", fontsize=8)
axes[0, 2].text(0.98, 181.164, "FCFS", transform=axes[0, 2].get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.8, color=M["fcfs"])
fig.tight_layout()
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"figures/pub/F17_sensitivity_grid.{ext}", bbox_inches="tight", **kw)
print("saved figures/pub/F17_sensitivity_grid.*")
