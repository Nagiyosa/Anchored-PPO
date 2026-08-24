"""D79.2 论文图统一生成(本地跑, 无 torch)。figures/pub/ 全量重生成。

字体: Times New Roman(沙盒用公制克隆 Liberation Serif 渲染; SVG/PDF 存字体名
Times New Roman, 在 Windows/Word/PPT 打开即真 Times)。莫兰迪色系。
用法: python scripts/40_pub_figures.py
"""
import glob
import io
import json
import pickle
import zipfile

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "axes.edgecolor": "#4A4A4A", "xtick.color": "#4A4A4A", "ytick.color": "#4A4A4A",
    "text.color": "#333333", "axes.labelcolor": "#333333", "legend.frameon": False,
})
M = {"rl": "#A2665F", "fcfs": "#7C8C7D", "sig": "#8E8B93", "bc": "#7D93A6", "vppo": "#B8A88F",
     "a2c": "#C4B7A6", "iql": "#8F8CA3", "dqn": "#B99C9E", "band": "#F1EDE8", "grid": "#E5E1DC",
     "dark": "#4A4A4A", "lite": "#C9B6B0", "lookup": "#96A48F", "fsfs": "#A3AF9D", "aset": "#8D9C93"}
R = lambda f: json.load(open(f"artifacts/reports/{f}"))


def save(fig, name):
    for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
        fig.savefig(f"figures/pub/{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("saved", name)


class _Stub:
    def __init__(self, *a, **k):
        pass


class _SU(pickle.Unpickler):
    def find_class(self, m, n):
        return (lambda *a, **k: _Stub()) if m.startswith("torch") else super().find_class(m, n)

    def persistent_load(self, pid):
        return _Stub()


def loadpt(p):
    with zipfile.ZipFile(p) as z:
        n = [x for x in z.namelist() if x.endswith("data.pkl")][0]
        with z.open(n) as f:
            return _SU(io.BytesIO(f.read())).load()


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=M["grid"], lw=0.5)
    ax.set_axisbelow(True)


# ================= F1 三面板递进(各自 0 起计, 加一个机制降一台阶) =================
def f1():
    # D81: 该 run 无 .json(只有 ckpt 内嵌 evals); 从本地 artifacts 读, 不再依赖已失效的会话 uploads 目录
    d69 = loadpt("artifacts/reports/ppo_8192_anchor_s0_last.pt")["evals"]
    d72A = R("ppo_push_A_s0.json")["evals"]; d72B = R("ppo_push_AB_s0.json")["evals"]
    d73A = R("ppo_d73_A_s0.json")["evals"]
    seeds = [R("ppo_d73_AB_s0.json")["evals"], R("ppo_d74_AB_s1.json")["evals"], R("ppo_d74_AB_s2.json")["evals"]]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2), sharey=True,
                             gridspec_kw={"width_ratios": [0.8, 1.0, 1.0], "wspace": 0.06})
    import matplotlib.transforms as mtransforms
    for ax, ti, sub in zip(axes, ["Stage 1", "Stage 2", "Stage 3"],
                           ["+ timing anchor", "+ moving trust region", "+ deadlock surgery, full-peak tiers"]):
        _tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)
        ax.text(0.5, 1.10, ti, transform=_tr, ha="center", fontsize=10, fontweight="bold", color="#5A5A5A")
        ax.text(0.5, 1.03, sub, transform=_tr, ha="center", fontsize=7, color="#8A8A8A")
        for y, c in [(181.164, M["fcfs"]), (222.146, M["aset"])]:
            ax.axhline(y, color=c, lw=0.9, ls=(0, (4, 3)))
        ax.set_xlabel("iteration")
        grid(ax)
    # Stage 1: 单臂(= full method 的前身, 方法色)
    ax = axes[0]
    ax.plot([e["iter"] for e in d69], [e["eval_weighted_secondary_delay_s"] / 1000 for e in d69],
            color=M["rl"], lw=1.7)
    ax.scatter([49], [182.745], s=15, color=M["dark"], zorder=6)
    ax.annotate("182.7k", (49, 182.745), xytext=(46, 168), fontsize=8, fontweight="bold", color=M["dark"])
    ax.set_xlim(0, 78)
    # Stage 2: 双臂
    ax = axes[1]
    h_a, = ax.plot([e["iter"] for e in d72A], [e["eval_weighted_secondary_delay_s"] / 1000 for e in d72A],
                   color=M["lite"], lw=1.2)
    h_b, = ax.plot([e["iter"] for e in d72B], [e["eval_weighted_secondary_delay_s"] / 1000 for e in d72B],
                   color=M["rl"], lw=1.7)
    ax.scatter([69], [154.788], s=15, color=M["dark"], zorder=6)
    ax.annotate("154.8k", (69, 154.788), xytext=(64, 140), fontsize=8, fontweight="bold", color=M["dark"])
    ax.set_xlim(0, 102)
    # Stage 3: full = 三种子带; w/o foresight 单线
    ax = axes[2]
    ax.plot([e["iter"] for e in d73A], [e["eval_weighted_secondary_delay_s"] / 1000 for e in d73A],
            color=M["lite"], lw=1.2)
    per = [{e["iter"]: e["eval_weighted_secondary_delay_s"] / 1000 for e in evs} for evs in seeds]
    common = sorted(set(per[0]) & set(per[1]) & set(per[2]))
    mean = [np.mean([p[i] for p in per]) for i in common]
    lo = [min(p[i] for p in per) for i in common]
    hi = [max(p[i] for p in per) for i in common]
    ax.fill_between(common, lo, hi, color=M["rl"], alpha=0.20, lw=0, zorder=2)
    h_full, = ax.plot(common, mean, color=M["rl"], lw=1.7, zorder=4)
    ax.scatter([74], [123.036], s=15, color=M["dark"], zorder=6)
    ax.annotate("123.0k (−32% vs FCFS)", (74, 123.036), xytext=(18, 108),
                fontsize=8, fontweight="bold", color=M["dark"])
    ax.set_xlim(0, 102)
    for y, lab, c in [(181.164, "FCFS 181k", M["fcfs"]), (222.146, "always-set 222k", M["aset"])]:
        ax.text(100, y + 2.5, lab, color=c, fontsize=7, ha="right")
    ax.legend([h_full, h_a], ["full method (Stage-3: 3-seed mean ± min–max)", "− foresight"],
              fontsize=7.2, loc="upper right")
    axes[0].set_ylim(100, 245)
    axes[0].set_ylabel("validation delay (×10³ s)")
    fig.text(0.5, -0.03, "each stage is a separate run, re-initialised from the previous stage's best checkpoint (iterations restart at 0)",
             ha="center", fontsize=6.8, color="#8A8A8A")
    save(fig, "F1_convergence")


# ================= F2 全家福(小=好, 我们置顶加粗) =================
def f2():
    fam = [("Ours (Anchored PPO)", 123.036, M["rl"], 1.0), ("FCFS", 181.164, M["fcfs"], 1.0),
           ("Lookup", 192.874, M["lookup"], 1.0), ("FSFS", 197.154, M["fsfs"], 1.0),
           ("BC (imitation)", 199.266, M["bc"], 1.0), ("Vanilla PPO", 216.364, M["vppo"], 1.0),
           ("Always-set", 222.146, M["aset"], 1.0), ("Signaller replay", 473.698, M["sig"], 1.0),
           ("A2C", 6079.259, M["a2c"], 0.979), ("IQL (offline)", 39458.4, M["iql"], 0.825),
           ("DQN", 188395.0, M["dqn"], 0.402)]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ys = np.arange(len(fam))[::-1]                      # 第一行(我们)在最上
    for y, (n, v, c, comp) in zip(ys, fam):
        ax.barh(y, v, color=c, height=0.62, zorder=3)
        txt = f"{v:,.0f}k" if v < 1000 else f"{v/1000:,.1f}M"
        extra = "" if comp == 1.0 else f"  (completion {comp:.2f})"
        ax.text(v * 1.15, y, txt + extra, va="center", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([f[0] for f in fam], fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")      # ys[0] 对应第一条 = Ours
    ax.set_xscale("log"); ax.set_xlim(100, 900000)
    ax.set_xlabel("validation delay (×10³ s, log scale; lower is better →  best at top)")
    grid(ax, "x")
    save(fig, "F2_family")


# ================= F3 消融(小=好, full 置顶加粗, 去题注) =================
def f3():
    # 与 F1 组件命名一一对应(用户定); † = 阶段历史证据(build-up 序列, 非最终阶段同预算拆除),
    # 无 † = 受控同预算 knockout。数值均为 val 延误 best。
    abl = [("Full method",                                122.492, M["rl"],   False),
           ("− foresight features",                     128.428, "#B58A84", False),
           ("− routing anchor",                         134.188, "#BFA39E", False),
           ("− curriculum",                             136.098, "#C9B6B0", False),
           ("− deadlock surgery, full-peak tiers",    154.788, "#C9B6B0", False),
           ("− moving trust region (static anchor)",  182.745, "#D3C4BE", False),
           ("BC only (no RL)",                            199.266, M["bc"],   False),
           ("− timing anchor",                        203.956, "#D3C4BE", False),
           ("Vanilla PPO (none of the above)",            216.364, M["vppo"], False)]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ys = np.arange(len(abl))[::-1]
    for y, (n, v, c, hist) in zip(ys, abl):
        ax.barh(y, v, color=c, height=0.6, zorder=3, alpha=0.55 if hist else 1.0,
                edgecolor=c if hist else "none", lw=0.9)
        pct = "" if v == 122.492 else f"  +{(v-122.492)/122.492*100:.0f}%"
        ax.text(v + 4, y, f"{v:.0f}k{pct}", va="center", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in abl], fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 268); ax.set_xlabel("validation delay (×10³ s; lower is better)")
    grid(ax, "x")
    save(fig, "F3_ablation")


# ================= F4 日内累计(标日期; 有 busiest 文件则用) =================
def f4():
    import os
    if os.path.exists("artifacts/reports/interpret_cumgap_busiest.json"):
        cg = R("interpret_cumgap_busiest.json")
        meta = [("2023-10-23", "a  train, 2023-10-23 (busiest)"),
                ("2023-11-15", "b  validation, 2023-11-15 (busiest)"),
                ("2023-11-18", "c  test, 2023-11-18 (busiest)")]
    else:
        cg = R("interpret_cumgap_main.json")
        meta = [("2023-10-23", "a  train split — busiest day (2023-10-23)"),
                ("2023-11-13", "b  validation split — busy day (2023-11-13)"),
                ("2023-11-27", "c  test split — disruption aftermath (2023-11-27)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (date, ti) in zip(axes, meta):
        dd = cg[date]
        t0 = min(s[0][0] for k, s in dd.items() if k != "split" and s)
        for pol, c, lw in [("signaller", M["sig"], 1.2), ("fcfs", M["fcfs"], 1.3), ("rl", M["rl"], 1.6)]:
            s = dd[pol]
            ax.plot([(a - t0) / 3600 for a, _ in s], [b / 1000 for _, b in s], color=c, lw=lw)
        ends = {pol: dd[pol][-1][1] / 1000 for pol in ["signaller", "fcfs", "rl"]}
        for pol, v in ends.items():
            ax.text(22.6, v, {"rl": "RL", "fcfs": "FCFS", "signaller": "Signaller"}[pol], fontsize=6.8,
                    color={"rl": M["rl"], "fcfs": M["fcfs"], "signaller": M["sig"]}[pol],
                    va="center", fontweight="bold" if pol == "rl" else "normal")
        ax.set_title(ti, loc="left", fontsize=7.6, fontweight="bold")
        ax.set_xlabel("hours into day"); ax.set_xlim(0, 26)
        grid(ax)
    axes[0].set_ylabel("cumulative weighted delay (×10³ s)")
    save(fig, "F4_cumulative_gap")


# ================= F5 自适应耐心 =================
def f5():
    days = ["2023-10-23", "2023-11-13", "2023-11-24", "2023-11-27"]
    lbl = ["train\nbusy", "val\nbusy", "test\n(failure case)", "test\ndisruption"]
    rlw, fcw = [], []
    for d in days:
        t = R(f"interpret_trace_{d}_rl.json")["decisions"]
        rlw.append(100 * sum(1 for x in t if x["kind"] == "wait") / len(t))
        t = R(f"interpret_trace_{d}_fcfs.json")["decisions"]
        fcw.append(100 * sum(1 for x in t if x["kind"] == "wait") / len(t))
    x = np.arange(4); w = 0.34
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    b1 = ax.bar(x - w / 2, rlw, w, color=M["rl"], zorder=3, label="RL (ours)")
    b2 = ax.bar(x + w / 2, fcw, w, color=M["fcfs"], zorder=3, label="FCFS")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}",
                ha="center", fontsize=6.8, color=M["dark"])
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylabel("share of decisions that WAIT (%)"); ax.set_ylim(0, 50)
    ax.legend(fontsize=7, loc="upper left")
    grid(ax)
    save(fig, "F5_adaptive_patience")


# ================= F6 冲突归因(原版两面板: 均值条) =================
def f6():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, sp, ti in [(axes[0], "val", "a  validation, mean per train"),
                       (axes[1], "test", "b  sealed test, mean per train")]:
        ca = R(f"conflict_analysis_{sp}.json")
        bins = [b for b in ca["bins"] if b["n"] > 0]
        x = np.arange(len(bins)); w = 0.34
        ax.bar(x - w / 2, [b["rl"] / b["n"] for b in bins], w, color=M["rl"], zorder=3, label="RL (ours)")
        ax.bar(x + w / 2, [b["fcfs"] / b["n"] for b in bins], w, color=M["fcfs"], zorder=3, label="FCFS")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b['bin']}\nn={b['n']}" for b in bins], fontsize=6.8)
        ax.set_title(ti, loc="left", fontsize=7.6, fontweight="bold")
        grid(ax)
    axes[0].set_ylabel("weighted delay per train (s)")
    axes[0].legend(fontsize=6.8)
    for ax in axes:
        ax.set_xlabel("contention (co-present controlled trains)")
    save(fig, "F6_conflict_attribution")


# ================= F11 逐车 Δ 分布(小提琴+箱线, 独立图) =================
def f11():
    ca = R("conflict_analysis_test.json")
    pt = ca["per_train"]
    labels, data = [], []
    for lo, hi, lab in [(0, 4, "0-4"), (5, 9, "5-9"), (10, 14, "10-14"), (15, 24, "15-24"), (25, 10**9, "25+")]:
        v = [r["d"] for r in pt if lo <= r["contention"] <= hi and abs(r["d"]) > 1]
        if v:
            labels.append(f"{lab}\nn={len(v)}"); data.append(v)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    vp = ax.violinplot(data, showextrema=False, widths=0.85)
    for b in vp["bodies"]:
        b.set_facecolor("#E7DAD6"); b.set_edgecolor(M["rl"]); b.set_alpha(0.85); b.set_linewidth(0.8)
    bp = ax.boxplot(data, showfliers=False, widths=0.22, patch_artist=True,
                    medianprops=dict(color="white", lw=1.2),
                    boxprops=dict(facecolor=M["rl"], edgecolor=M["rl"], lw=0.8),
                    whiskerprops=dict(color=M["rl"], lw=0.9), capprops=dict(color=M["rl"], lw=0.9))
    ax.axhline(0, color="#8A8A8A", lw=0.8, ls=(0, (3, 2)))
    for i, v in enumerate(data):
        ax.scatter([i + 1], [np.mean(v)], marker="D", s=13, color=M["dark"], zorder=5)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("per-train Δ weighted delay, RL − FCFS (s)")
    ax.set_xlabel("contention (co-present controlled trains)")
    ax.set_title("Sealed test, affected trains (|Δ| > 1 s): distribution shifts down as contention grows",
                 loc="left", fontsize=7.6, fontweight="bold")
    ax.text(0.98, 0.04, "diamond = mean; white line = median; below 0 = RL saves",
            transform=ax.transAxes, fontsize=6.6, ha="right", color="#6B6B6B")
    grid(ax)
    save(fig, "F11_delta_distribution")


# ================= F7 特征置零 =================
def f7():
    pm = R("interpret_permimport_val.json")
    names = {"延误特征(dec0+panel0)": "lateness features", "细时刻fv2(dec末)": "fine time-to-plan (v2)",
             "背景前瞻fv2(cand末)": "background foresight (v2)", "占用(occ_frac+first_occ)": "occupancy",
             "候选先验(prior)": "route prior"}
    base = pm["none(基准)"]["eval_weighted_secondary_delay_s"]
    ks = list(names)
    vals = [(pm[k]["eval_weighted_secondary_delay_s"] - base) / base * 100 for k in ks]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ys = np.arange(len(ks))
    for y, i in zip(ys, order):
        v = vals[i]
        ax.barh(y, v, color=M["rl"] if v > 0 else M["bc"], height=0.58, zorder=3)
        ax.text(v + (0.08 if v >= 0 else -0.08), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([names[ks[i]] for i in order], fontsize=7.5)
    ax.axvline(0, color=M["dark"], lw=0.7)
    ax.set_xlim(-2.6, 1.2)
    ax.set_xlabel("Δ validation delay when feature group zeroed (%)")
    grid(ax, "x")
    save(fig, "F7_feature_knockout")


# ================= F8 病理时间线 =================
def f8():
    rl = R("interpret_trace_2023-11-24_rl.json"); fc = R("interpret_trace_2023-11-24_fcfs.json")
    tids = ["1D51", "1K53", "2A15"]
    t0 = min(rl["trains"][t]["entry"] for t in tids)
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    for i, tid in enumerate(tids):
        a, b = rl["trains"][tid], fc["trains"][tid]
        for rec, c, dy in [(a, M["rl"], 0.16), (b, M["fcfs"], -0.16)]:
            x0 = (rec["entry"] - t0) / 60
            x1 = (rec["entered"] - t0) / 60 if rec["entered"] and rec["entered"] > rec["entry"] else x0 + 0.4
            ax.plot([x0, x1], [i + dy] * 2, color=c, lw=7, solid_capstyle="butt", zorder=3)
            ax.scatter([x0], [i + dy], marker="|", s=44, color=M["dark"], zorder=4)
        hold = a["entered"] - a["entry"]
        ax.text((a["entered"] - t0) / 60 + 1.5, i + 0.16, f"held {hold/60:.0f} min",
                fontsize=7, va="center", color=M["rl"], fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels(tids, fontsize=7.5)
    ax.set_xlabel("minutes since first arrival at boundary")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=M["rl"], lw=5, label="RL boundary wait"),
                       Line2D([0], [0], color=M["fcfs"], lw=5, label="FCFS boundary wait")],
              fontsize=7, loc="lower right")
    ax.set_title("Failure case (test, 2023-11-24): over-patience on three services",
                 loc="left", fontsize=7.6, fontweight="bold")
    grid(ax, "x")
    save(fig, "F8_case_failure")


# ================= F9 等待尾部 =================
def f9():
    lt = R("interpret_leadtime_train.json")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for pol, c, lab in [("signaller", "#5F6E86", "Signaller replay"), ("fcfs", M["fcfs"], "FCFS"),
                        ("rl", M["rl"], "RL (ours)")]:
        w = np.array([r["entered"] - r["entry"] for day in lt.values()
                      for r in day[pol].values() if r["entered"] is not None])
        w = np.sort(w[w > 0]); cc = (1 - np.arange(len(w)) / len(w)) * 100
        ax.plot(w / 60, cc, color=c, lw=1.6, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("boundary holding time before admission (min, log)")
    ax.set_ylabel("share of held trains waiting ≥ x (%, log)")
    ax.legend(fontsize=7)
    ax.set_title("How long do held trains wait at the boundary?", loc="left", fontsize=7.6, fontweight="bold")
    ax.grid(True, which="both", color=M["grid"], lw=0.4); ax.set_axisbelow(True)
    save(fig, "F9_wait_tail")


# ================= F10 全方法收敛 =================
def f10():
    d69 = loadpt("artifacts/reports/ppo_8192_anchor_s0_last.pt")["evals"]  # D81 同上
    ours = [(e["iter"], e["eval_weighted_secondary_delay_s"]) for e in d69]
    ours += [(e["iter"] + 80, e["eval_weighted_secondary_delay_s"]) for e in R("ppo_push_AB_s0.json")["evals"]]
    ours += [(e["iter"] + 185, e["eval_weighted_secondary_delay_s"]) for e in R("ppo_d73_AB_s0.json")["evals"]]
    series = {"Ours (Anchored PPO, 3 stages)": ("rl", ours, 1.9)}
    for tag, key, lab in [("base_vppo_s0", "vppo", "Vanilla PPO (scratch)"),
                          ("base_a2c_s0", "a2c", "A2C (scratch)"),
                          ("base_dqn_s0", "dqn", "DQN (scratch)"),
                          ("base_iql_s0", "iql", "IQL (offline; x = grad-step evals)")]:
        d = R(tag + ".json")
        series[lab] = (key, [(e["iter"], e["eval_weighted_secondary_delay_s"]) for e in d["evals"]], 1.3)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for lab, (ck, s, lw) in series.items():
        ax.plot([max(p[0], 1) for p in s], [p[1] / 1000 for p in s], color=M[ck], lw=lw,
                marker="o", ms=2.2, label=lab, zorder=4 if ck == "rl" else 3)
    ax.axhline(199.266, color=M["bc"], lw=1.0, ls=(0, (4, 3)))
    ax.text(1.6, 199.266 * 1.25, "BC (no iterations)", color=M["bc"], fontsize=7)
    ax.axhline(181.164, color=M["fcfs"], lw=1.0, ls=(0, (2, 2)))
    ax.text(1.6, 181.164 * 0.58, "FCFS (best rule)", color=M["fcfs"], fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training iteration (log)"); ax.set_ylabel("validation delay (×10³ s, log)")
    ax.set_ylim(80, 500000)
    for lab, comp in [("DQN (scratch)", 0.402), ("IQL (offline; x = grad-step evals)", 0.825),
                      ("A2C (scratch)", 0.979)]:
        ck, s, _ = series[lab]
        ax.text(s[-1][0] * 1.1, s[-1][1] / 1000 * 1.15, f"completion {comp:.2f}", fontsize=6.8, color=M[ck])
    ax.annotate("final: 123.0k, completion 1.00", (285, 123.0), xytext=(20, 132), fontsize=7.5,
                fontweight="bold", color=M["dark"],
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#8A8A8A"))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", color=M["grid"], lw=0.4); ax.set_axisbelow(True)
    save(fig, "F10_all_methods_convergence")




# ================= F12 主方法三种子误差带(独立图) =================
def f12():
    seeds = [R("ppo_d73_AB_s0.json")["evals"], R("ppo_d74_AB_s1.json")["evals"], R("ppo_d74_AB_s2.json")["evals"]]
    per = [{e["iter"]: e["eval_weighted_secondary_delay_s"] / 1000 for e in evs} for evs in seeds]
    common = sorted(set(per[0]) & set(per[1]) & set(per[2]))
    mean = [np.mean([p[i] for p in per]) for i in common]
    lo = [min(p[i] for p in per) for i in common]
    hi = [max(p[i] for p in per) for i in common]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.fill_between(common, lo, hi, color=M["rl"], alpha=0.20, lw=0, zorder=2, label="seed min–max (n=3)")
    ax.plot(common, mean, color=M["rl"], lw=1.9, zorder=4, label="full method, 3-seed mean")
    for y, lab, c in [(181.164, "FCFS (best rule)", M["fcfs"]),
                      (199.266, "BC (imitation)", M["bc"]),
                      (222.146, "always-set", M["aset"])]:
        ax.axhline(y, color=c, lw=1.0, ls=(0, (4, 3)))
        ax.text(2, y + 2.5, f"{lab}  {y:.0f}k", color=c, fontsize=7.5, ha="left")
    ax.annotate("best 123.0k (seed mean 126.7k ± 5.6k)", (74, 123.0), xytext=(30, 108),
                fontsize=8, fontweight="bold", color=M["dark"],
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#8A8A8A"))
    ax.set_ylim(100, 235); ax.set_xlim(0, 100)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("validation delay (×10³ s)")
    ax.legend(fontsize=7.5, loc="upper right")
    grid(ax)
    save(fig, "F12_seed_band")




# ================= F13 课程四档 × 三种子(小多联, 档位底纹渐深) =================
def f13():
    tags = [("ppo_d73_AB_s0.json", "seed 0"), ("ppo_d74_AB_s1.json", "seed 1"), ("ppo_d74_AB_s2.json", "seed 2")]
    tier_shades = ["#F5F1EC", "#EDE6DD", "#E4DACC", "#DACDBB"]      # 档1→4 渐深
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9), sharey=True)
    for ax, (f, ti) in zip(axes, tags):
        evs = R(f)["evals"]
        iters = [e["iter"] for e in evs]
        tiers = [e.get("tier", 0) for e in evs]
        ys = [e["eval_weighted_secondary_delay_s"] / 1000 for e in evs]
        # 档位区间: 该档第一次出现的 eval iter → 下一档第一次出现
        bounds = []
        for t in range(4):
            fs = [i for i, tt in zip(iters, tiers) if tt == t]
            if fs:
                bounds.append((t, min(fs)))
        bounds.sort(key=lambda x: x[1])
        xmax = max(iters) + 4
        import matplotlib.transforms as mtransforms
        _tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        tl = ["T1", "T2", "T3", "Tier 4"]
        for j, (t, b) in enumerate(bounds):
            e = bounds[j + 1][1] if j + 1 < len(bounds) else xmax
            x0 = b - 4 if j == 0 else b
            ax.axvspan(x0, e, color=tier_shades[t], zorder=0)
            yl = 1.04 if (t == 3 or t % 2 == 0) else 1.12      # 窄档高低交错防重叠
            ax.text((x0 + e) / 2, yl, tl[t], transform=_tr,
                    ha="center", fontsize=6.8, color="#6B6B6B")
        for y, c in [(181.164, M["fcfs"]), (199.266, M["bc"]), (222.146, M["aset"])]:
            ax.axhline(y, color=c, lw=0.9, ls=(0, (4, 3)))
        ax.plot(iters, ys, color=M["rl"], lw=1.6, marker="o", ms=2.2, zorder=4)
        ax.set_title(ti, loc="left", fontsize=8, fontweight="bold", y=1.12)
        ax.set_xlabel("training iteration")
        ax.set_xlim(0, xmax)
        grid(ax)
    axes[0].set_ylabel("validation delay (×10³ s)")
    axes[0].set_ylim(100, 235)
    for y, lab, c in [(181.164, "FCFS 181k", M["fcfs"]), (199.266, "BC 199k", M["bc"]),
                      (222.146, "always-set 222k", M["aset"])]:
        axes[2].text(axes[2].get_xlim()[1] * 0.98, y + 2.5, lab, color=c, fontsize=6.8, ha="right")
    save(fig, "F13_curriculum_tiers")




# ================= F7b 特征敏感度(置换+动作翻转率; 数据到了才画) =================
def f7b():
    import os
    if not os.path.exists("artifacts/reports/interpret_sensitivity_val.json"):
        print("F7b 跳过: interpret_sensitivity_val.json 未到"); return
    d = R("interpret_sensitivity_val.json")
    names = {"lateness(dec0+panel0)": "lateness features", "fine_time_v2(dec-1)": "fine time-to-plan (v2)",
             "bg_foresight_v2(cand-1)": "background foresight (v2)", "occupancy(occ+first)": "occupancy",
             "route_prior": "route prior", "tod(sin,cos)": "time of day"}
    ks = [k for k in d if k in names]
    vals = [d[k]["flip_rate"] * 100 for k in ks]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    ys = np.arange(len(ks))
    for y, i in zip(ys, order):
        ax.barh(y, vals[i], color=M["rl"], height=0.58, zorder=3)
        ax.text(vals[i] + max(vals) * 0.02, y, f"{vals[i]:.1f}%", va="center", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([names[ks[i]] for i in order], fontsize=7.5)
    ax.set_xlabel("decisions whose argmax action flips when the feature group is permuted (%)")
    grid(ax, "x")
    save(fig, "F7_feature_sensitivity")


# ================= F14 四编码器 BC 闭环(数据到了才画) =================
def f14():
    import os
    if not os.path.exists("artifacts/reports/bc_encoders_closedloop.json"):
        print("F14 跳过: bc_encoders_closedloop.json 未到"); return
    d = R("bc_encoders_closedloop.json")
    names = {"gnn": "GNN", "deepsets": "DeepSets", "settransformer": "Set Transformer", "mlp": "MLP"}
    rows = sorted(d.items(), key=lambda kv: kv[1]["eval_weighted_secondary_delay_s"])
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    ys = np.arange(len(rows))[::-1]
    for y, (k, v) in zip(ys, rows):
        dv = v["eval_weighted_secondary_delay_s"] / 1000
        ax.barh(y, dv, color=M["bc"] if k != "gnn" else M["rl"], height=0.58, zorder=3)
        ax.text(dv + 3, y, f"{dv:.0f}k   completion {v['eval_completion_rate']:.3f}",
                va="center", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([names.get(k, k) for k, _ in rows], fontsize=7.5)
    ax.set_xlabel("closed-loop validation delay of the BC policy (×10³ s)")
    ax.set_xlim(0, max(v["eval_weighted_secondary_delay_s"] for _, v in rows) / 1000 * 1.45)
    grid(ax, "x")
    save(fig, "F14_bc_encoders")
    md = ["| Encoder | Completion | Val delay (×10³ s) | Reward |", "|---|---|---|---|"]
    for k, v in rows:
        md.append(f"| {names.get(k,k)} | {v['eval_completion_rate']:.4f} | "
                  f"{v['eval_weighted_secondary_delay_s']/1000:.1f} | {v['eval_episode_reward']:.0f} |")
    open("figures/pub/bc_encoders_table.md", "w").write("\n".join(md))
    print("saved bc_encoders_table.md")


# ================= 消融表(论文用, 取代 F3 也可) =================
def ablation_table():
    rows = [("Full method (Anchored PPO + foresight + curriculum)", "122.5", "—", ""),
            ("− foresight features", "128.4", "+4.4%", ""),
            ("− routing anchor", "134.2", "+9.1%", ""),
            ("− curriculum", "136.1", "+11.1%", ""),
            ("− deadlock surgery, full-peak tiers", "154.8", "+26.4%", "stage evidence"),
            ("− moving trust region (static anchor)", "182.7", "+49.2%", "stage evidence"),
            ("BC only (no RL)", "199.3", "+62.7%", ""),
            ("− timing anchor", "204.0", "+66.5%", "stage evidence"),
            ("Vanilla PPO (none of the above)", "216.4", "+76.6%", "")]
    md = ["| Configuration | Val delay (×10³ s) | vs full | evidence |", "|---|---|---|---|",
          *[f"| {n} | {v} | {p} | {e or 'same-budget knockout'} |" for n, v, p, e in rows],
          "",
          "note: the − timing anchor run additionally exhibits drift/instability (D70)."]
    open("figures/pub/ablation_table.md", "w").write("\n".join(md))
    print("saved ablation_table.md")


if __name__ == "__main__":
    for fn in [f1, f2, f3, f4, f5, f6, f7, f7b, f8, f9, f10, f11, f12, f13, f14, ablation_table]:
        fn()
