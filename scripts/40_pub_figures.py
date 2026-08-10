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


# ================= F1 收敛故事 =================
def f1():
    d69 = loadpt(glob.glob("/sessions/modest-focused-lamport/mnt/uploads/*ppo_8192_anchor_s0_last.pt")[0])["evals"]
    d72A = R("ppo_push_A_s0.json")["evals"]; d72B = R("ppo_push_AB_s0.json")["evals"]
    d73A = R("ppo_d73_A_s0.json")["evals"]; d73B = R("ppo_d73_AB_s0.json")["evals"]
    P1, P2, P3 = 0, 80, 185
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.axvspan(P1 - 3, P2 - 3, color=M["band"], alpha=0.55, zorder=0)
    ax.axvspan(P3 - 3, 290, color=M["band"], alpha=0.55, zorder=0)
    h_sig, = ax.plot([e["iter"] + P1 for e in d69],
                     [e["eval_weighted_secondary_delay_s"] / 1000 for e in d69], color=M["sig"], lw=1.5)
    for evs, off in [(d72A, P2), (d73A, P3)]:
        h_a, = ax.plot([e["iter"] + off for e in evs],
                       [e["eval_weighted_secondary_delay_s"] / 1000 for e in evs], color=M["lite"], lw=1.2)
    for evs, off in [(d72B, P2), (d73B, P3)]:
        h_b, = ax.plot([e["iter"] + off for e in evs],
                       [e["eval_weighted_secondary_delay_s"] / 1000 for e in evs], color=M["rl"], lw=1.7)
    for y, lab, c in [(181.164, "FCFS 181k", M["fcfs"]), (222.146, "always-set 222k", M["aset"])]:
        ax.axhline(y, color=c, lw=0.8, ls=(0, (4, 3)))
        ax.text(288, y + 3, lab, color=c, fontsize=7, ha="right")
    for x, y, s in [(49, 182.745, "182.7k"), (149, 154.788, "154.8k"), (259, 123.036, "123.0k  (−32% vs FCFS)")]:
        ax.scatter([x], [y], s=14, color=M["dark"], zorder=5)
        ax.annotate(s, (x, y), xytext=(x + 4, y - 13), fontsize=7.5, fontweight="bold", color=M["dark"])
    for x, s in [(30, "Stage 1\ntiming anchor"), (125, "Stage 2\nmoving trust region"),
                 (228, "Stage 3\ndeadlock surgery → full-peak training")]:
        ax.text(x, 236, s, fontsize=7, ha="center", color="#6B6B6B")
    ax.legend([h_b, h_a, h_sig], ["full method (with foresight)", "w/o foresight (ablation arm)", "stage-1 run"],
              fontsize=7, loc="center left", bbox_to_anchor=(0.315, 0.70))
    ax.set_ylim(100, 245); ax.set_xlim(-4, 290)
    ax.set_xlabel("training iteration (stage-sequential)")
    ax.set_ylabel("validation delay (×10³ s)")
    grid(ax)
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
    abl = [("Full method", 122.492, M["rl"]), ("− foresight features", 128.428, "#B58A84"),
           ("− routing anchor", 134.188, "#BFA39E"), ("− curriculum", 136.098, "#C9B6B0"),
           ("BC only (no RL)", 199.266, M["bc"]), ("Vanilla PPO (none)", 216.364, M["vppo"])]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ys = np.arange(len(abl))[::-1]
    for y, (n, v, c) in zip(ys, abl):
        ax.barh(y, v, color=c, height=0.6, zorder=3)
        pct = "" if v == 122.492 else f"  +{(v-122.492)/122.492*100:.1f}%"
        ax.text(v + 4, y, f"{v:.0f}k{pct}", va="center", fontsize=7, color=M["dark"])
    ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in abl], fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 265); ax.set_xlabel("validation delay (×10³ s; lower is better)")
    grid(ax, "x")
    save(fig, "F3_ablation")


# ================= F4 日内累计(标日期; 有 busiest 文件则用) =================
def f4():
    import os
    if os.path.exists("artifacts/reports/interpret_cumgap_busiest.json"):
        cg = R("interpret_cumgap_busiest.json")
        meta = [("2023-10-23", "a  train split — busiest day (2023-10-23)"),
                ("2023-11-15", "b  validation split — busiest day (2023-11-15)"),
                ("2023-11-18", "c  test split — busiest day (2023-11-18)")]
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


# ================= F6 冲突归因: 均值条 + 受影响车 Δ 箱线 =================
def f6():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7),
                             gridspec_kw={"width_ratios": [1.6, 2.4, 2.2]})
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
    # c: test 受影响车(|Δ|>1s)逐箱 Δ 箱线
    ca = R("conflict_analysis_test.json")
    pt = ca["per_train"]
    labels, data = [], []
    for lo, hi, lab in [(0, 4, "0-4"), (5, 9, "5-9"), (10, 14, "10-14"), (15, 24, "15-24"), (25, 10**9, "25+")]:
        v = [r["d"] for r in pt if lo <= r["contention"] <= hi and abs(r["d"]) > 1]
        if v:
            labels.append(f"{lab}\nn={len(v)}"); data.append(v)
    ax = axes[2]
    bp = ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True,
                    medianprops=dict(color=M["dark"], lw=1.1),
                    boxprops=dict(facecolor="#E7DAD6", edgecolor=M["rl"], lw=0.9),
                    whiskerprops=dict(color=M["rl"], lw=0.9), capprops=dict(color=M["rl"], lw=0.9))
    ax.axhline(0, color="#8A8A8A", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_title("c  test, per-train Δ(RL−FCFS), affected trains", loc="left", fontsize=7.6, fontweight="bold")
    ax.set_ylabel("Δ weighted delay (s)")
    ax.set_xlabel("contention (co-present controlled trains)")
    grid(ax)
    save(fig, "F6_conflict_attribution")


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
    d69 = loadpt(glob.glob("/sessions/modest-focused-lamport/mnt/uploads/*ppo_8192_anchor_s0_last.pt")[0])["evals"]
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


# ================= 消融表(论文用, 取代 F3 也可) =================
def ablation_table():
    rows = [("Full method (Anchored PPO + foresight + curriculum)", 122.492, "—"),
            ("w/o foresight features", 128.428, "+4.4%"),
            ("w/o routing anchor", 134.188, "+9.1%"),
            ("w/o curriculum", 136.098, "+11.1%"),
            ("w/o timing anchor", "drift → plateau collapse", "(D70)"),
            ("BC only (no RL)", 199.266, "+62.7%"),
            ("Vanilla PPO (no warm start, no anchors, no curriculum)", 216.364, "+76.6%")]
    md = ["| Configuration | Val delay (×10³ s) | vs full |", "|---|---|---|"]
    for n, v, p in rows:
        md.append(f"| {n} | {v if isinstance(v,str) else f'{v:.1f}'} | {p} |")
    open("figures/pub/ablation_table.md", "w").write("\n".join(md))
    print("saved ablation_table.md")


if __name__ == "__main__":
    for fn in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, ablation_table]:
        fn()
