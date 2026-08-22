"""Method 页 1: 系统总览图(poster 级)。figures/pub/SYS_overview.{svg,pdf,png}
用法: python scripts/41_system_figure.py
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
C = {"rl": "#A2665F", "sage": "#7C8C7D", "blue": "#7D93A6", "sand": "#B8A88F",
     "band": "#F4F1EC", "ink": "#3E3E3E", "grey": "#8A8A8A", "green": "#4E7A52",
     "chipf": "#FAF8F5", "plat": "#7FA36B"}

fig, ax = plt.subplots(figsize=(12.6, 4.4))
ax.set_xlim(0, 126); ax.set_ylim(0, 44); ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.4, r=1.2, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       edgecolor=ec, facecolor=fc, lw=lw, zorder=z)
    ax.add_patch(b)
    return b


def chip(x, y, w, h, text, ec, fs=6.8):
    box(x, y, w, h, ec, C["chipf"], lw=0.9, r=0.7, z=3)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"], zorder=4)


def arrow(x1, y1, x2, y2, c=C["ink"], lw=1.6, style="-|>", cs="arc3,rad=0.0", z=5, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        color=c, lw=lw, connectionstyle=cs, zorder=z, linestyle=ls)
    ax.add_patch(a)


def stage(x, y, num, label):
    ax.add_patch(Circle((x, y + 0.05), 1.15, facecolor=C["ink"], edgecolor="none", zorder=6))
    ax.text(x, y, num, ha="center", va="center", fontsize=7.5, color="white", zorder=7,
            fontweight="bold")
    ax.text(x + 1.9, y, label, ha="left", va="center", fontsize=9.5, color=C["ink"],
            fontweight="bold", zorder=6)


# ======================= ① DATA =======================
stage(3.2, 41.5, "1", "Operational data")
box(2, 8, 17.5, 30, C["blue"], lw=1.4)
for i, (t, sub) in enumerate([("TD", "berth-step logs"), ("TRUST", "delay records"),
                              ("CIF", "timetables")]):
    y = 29.5 - i * 7.6
    box(4, y - 2.2, 13.5, 5.6, C["blue"], "#EEF2F6", lw=1.0, r=0.8, z=3)
    ax.add_patch(Rectangle((5, y - 1.4), 2.6, 3.7, facecolor=C["blue"], alpha=0.75, zorder=4))
    for k in range(3):
        ax.plot([5.35, 7.25], [y + 1.6 - k * 1.15] * 2, color="white", lw=0.7, zorder=5)
    ax.text(8.6, y + 0.85, t, fontsize=8.5, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(8.6, y - 0.95, sub, fontsize=6.6, color=C["grey"], zorder=5)
ax.text(10.75, 9.7, "provided by Network Rail", ha="center", fontsize=6.6,
        style="italic", color=C["grey"], zorder=4)

arrow(19.5, 23, 24.5, 23)

# ======================= ② SIMULATOR =======================
stage(26.2, 41.5, "2", "Microscopic simulator")
box(25, 8, 33, 30, C["sage"], lw=1.6)
# 迷你轨道图
tx0, tx1 = 27, 55.6
for j, ty in enumerate([33.5, 30.6, 27.7]):
    ax.plot([tx0, tx1], [ty, ty], color=C["ink"], lw=1.5, zorder=4)
ax.plot([33, 37.5], [33.5, 30.6], color=C["ink"], lw=1.3, zorder=4)   # 渡线
ax.plot([44, 48.5], [30.6, 27.7], color=C["ink"], lw=1.3, zorder=4)
ax.add_patch(Rectangle((39, 34.0), 7.5, 1.25, facecolor=C["plat"], edgecolor="none", zorder=4))
ax.add_patch(Rectangle((39, 28.2), 7.5, 1.25, facecolor=C["plat"], edgecolor="none", zorder=4))
for sx, sy in [(31.5, 33.5), (42.5, 30.6), (52, 27.7), (37, 33.5), (50, 30.6)]:
    ax.add_patch(Circle((sx, sy + 0.95), 0.42, facecolor="white", edgecolor=C["ink"], lw=0.9, zorder=5))
    ax.plot([sx, sx], [sy, sy + 0.55], color=C["ink"], lw=0.8, zorder=5)
ax.text(41.3, 36.6, "Derby station throat · 48 signals · 63 track circuits", ha="center",
        fontsize=6.9, color=C["grey"], zorder=5)
# 特性 chips
chip(26.8, 20.6, 14.6, 3.9, "blocking-time interlocking", C["sage"])
chip(42.4, 20.6, 14.6, 3.9, "event-driven · deterministic", C["sage"])
chip(26.8, 15.7, 14.6, 3.9, "background traffic replayed", C["sage"])
chip(42.4, 15.7, 14.6, 3.9, "60 s decision clock", C["sage"])
# 验证徽章
box(26.8, 9.6, 30.2, 4.6, C["green"], "#EFF5EF", lw=1.2, r=0.9, z=3)
ax.plot([29.2, 30.0, 31.4], [11.7, 10.9, 13.1], color=C["green"], lw=1.8,
        solid_capstyle="round", zorder=5)
ax.text(33.0, 11.9, "validation gate   signaller replay completes every day (1.0)",
        fontsize=6.9, color=C["green"], zorder=5, va="center")

# ======================= ③ LEARNING =======================
stage(64.2, 41.5, "3", "Imitation warm start")
box(63, 25.5, 32, 12.5, C["blue"], lw=1.5)
ax.text(64.6, 34.9, "Behaviour cloning warm start", fontsize=8.6, fontweight="bold", color=C["ink"], zorder=4)
ax.text(64.6, 32.6, "supervised on replayed signaller decisions", fontsize=6.8, color=C["grey"], zorder=4)
# 日志纸 → GNN 图标
box(64.8, 27.0, 5.4, 4.2, C["blue"], "white", lw=0.9, r=0.5, z=3)
for k in range(3):
    ax.plot([65.7, 69.3], [30.2 - k * 1.05] * 2, color=C["blue"], lw=0.7, zorder=4)
arrow(70.9, 29.1, 74.3, 29.1, lw=1.2)
nodes = [(77.3, 30.6), (75.9, 28.3), (78.7, 28.3), (76.6, 26.4), (78.2, 26.4)]
for (na, nb) in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]:
    ax.plot([nodes[na][0], nodes[nb][0]], [nodes[na][1], nodes[nb][1]],
            color=C["blue"], lw=0.8, zorder=4)
for nx, ny in nodes:
    ax.add_patch(Circle((nx, ny), 0.62, facecolor=C["blue"], edgecolor="white", lw=0.6, zorder=5))
ax.text(81.6, 28.6, "GNN policy\nover panel options", fontsize=6.8, color=C["ink"], zorder=4)
ax.text(81.6, 26.4, "options  hold · set route · cancel", fontsize=6.2, color=C["grey"], zorder=4)

stage(64.2, 21.2, "4", "Anchored PPO")
box(63, 3.2, 32, 15.6, C["rl"], lw=1.7)
# 交互循环 π ↔ env
ax.add_patch(Circle((68.0, 11.6), 2.4, facecolor="white", edgecolor=C["rl"], lw=1.3, zorder=4))
ax.text(68.0, 11.6, "π", ha="center", va="center", fontsize=11, color=C["rl"],
        fontweight="bold", zorder=5)
arrow(63.0, 13.6, 58.2, 13.6, c=C["rl"], lw=1.3)
arrow(58.2, 9.8, 63.0, 9.8, c=C["rl"], lw=1.3)
ax.text(60.6, 14.5, "act", fontsize=6.2, color=C["rl"], zorder=6, ha="center")
ax.text(60.5, 8.3, "observe\nreward", fontsize=5.9, color=C["rl"], zorder=6, ha="center", va="top")
# 双锚(虚线系绳 + 桩)
chip(74.4, 13.5, 19.2, 3.7, "routing anchor → timetable plan", C["rl"], fs=6.6)
chip(74.4, 8.9, 19.2, 3.7, "timing anchor → best checkpoint", C["rl"], fs=6.6)
arrow(71.1, 12.1, 74.4, 15.0, c=C["rl"], lw=0.9, ls=(0, (3, 2)), style="-")
arrow(71.1, 10.4, 74.4, 10.8, c=C["rl"], lw=0.9, ls=(0, (3, 2)), style="-")
ax.text(84.0, 7.3, "reference follows the best checkpoint", fontsize=6.2,
        style="italic", color=C["grey"], zorder=4, ha="center")
# 课程台阶
for k in range(4):
    ax.add_patch(Rectangle((64.8 + k * 1.7, 5.8), 1.45, 1.0 + k * 0.7,
                           facecolor=C["rl"], alpha=0.30 + 0.16 * k, edgecolor="none", zorder=4))
ax.text(68.3, 4.6, "curriculum  quiet → peak", fontsize=6.1, color=C["grey"], zorder=4, ha="center")

arrow(90.5, 25.5, 90.5, 19.2, c=C["ink"], lw=1.4)
ax.text(89.6, 22.3, "initialise + anchor", fontsize=6.6, color=C["ink"], zorder=6, ha="right")

# ======================= ⑤ OUTPUT =======================
stage(99.7, 41.5, "5", "Dispatch policy")
box(98.5, 12, 25, 22, C["rl"], "#FBF7F6", lw=1.7)
ax.text(111, 30.4, "completion", ha="center", fontsize=7.6, color=C["grey"], zorder=4)
ax.text(111, 27.2, "1.00", ha="center", fontsize=14, fontweight="bold", color=C["ink"], zorder=4)
ax.text(111, 22.8, "delay vs FCFS", ha="center", fontsize=7.6, color=C["grey"], zorder=4)
ax.text(111, 19.6, "−32%", ha="center", fontsize=14, fontweight="bold", color=C["rl"], zorder=4)
ax.text(111, 15.2, "wins where contention is highest", ha="center", fontsize=6.6,
        style="italic", color=C["grey"], zorder=4)

arrow(95.4, 11, 100.5, 13.8, c=C["ink"], lw=1.5, cs="arc3,rad=0.2")

fig.tight_layout()
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"figures/pub/SYS_overview.{ext}", bbox_inches="tight", **kw)
print("saved figures/pub/SYS_overview.*")
