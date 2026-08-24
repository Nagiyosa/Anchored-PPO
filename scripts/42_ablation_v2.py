"""D81.1: 消融图/表 v2(与 v1 并存, 不覆盖 F3/ablation_table.md)。
两处口径统一(用户 2026-08-23 拍板"单独另做一个,都保留"):
  1) 全部行统一 **best-by-reward** 选择(= 实际部署的模型选择规则, `_best.pt` 同源);
     Full = 123,035.5(v1 的 122,492.5 是曲线 min-delay, 事后 oracle 选择);
     − moving trust region = 183,700.5(v1 的 182,745 同为 min-delay)。
  2) "− deadlock surgery, full-peak tiers" 改用 **AB 臂** ppo_push_AB_s0(157,245.0):
     A 臂(v1 的 154,787.5)是 fv1、连前瞻特征也没有, 该差值混入了 foresight 的贡献;
     AB 臂带前瞻, 与 full 的唯一差别 = deadlock 手术 + full-peak tiers, 单变量语义干净。
数值全部从结果文件现读(非硬编码)。产出: figures/pub/F3_ablation_v2.{svg,pdf,png} + ablation_table_v2.md
用法: python scripts/42_ablation_v2.py
"""
import io
import json
import pickle
import zipfile

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
M = {"rl": "#A2665F", "bc": "#7D93A6", "vppo": "#B8A88F", "grid": "#E5E1DC", "dark": "#4A4A4A"}
R = lambda f: json.load(open(f"artifacts/reports/{f}"))


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


def by_reward(evals):
    """统一选择规则: 验证 reward 最高的 checkpoint(与训练时 _best.pt 的选择一致)。"""
    e = max(evals, key=lambda x: x.get("eval_episode_reward", -9e9))
    return e["iter"], e["eval_weighted_secondary_delay_s"]


rows = []                                   # (名称, 迭代, 延误秒, 来源, 颜色)
it, v = by_reward(R("ppo_d73_AB_s0.json")["evals"])
rows.append(("Full method", it, v, "ppo_d73_AB_s0", M["rl"]))
it, v = by_reward(R("ppo_d73_A_s0.json")["evals"])
rows.append(("− foresight features", it, v, "ppo_d73_A_s0", "#B58A84"))
it, v = by_reward(R("abl_noroute_s0.json")["evals"])
rows.append(("− routing anchor", it, v, "abl_noroute_s0", "#BFA39E"))
it, v = by_reward(R("abl_nocur_s0.json")["evals"])
rows.append(("− curriculum", it, v, "abl_nocur_s0", "#C9B6B0"))
it, v = by_reward(R("ppo_push_AB_s0.json")["evals"])                      # D81.1: AB 臂(v1 用 A 臂)
rows.append(("− deadlock surgery, full-peak tiers", it, v, "ppo_push_AB_s0", "#C9B6B0"))
it, v = by_reward(loadpt("artifacts/reports/ppo_8192_anchor_s0_last.pt")["evals"])
rows.append(("− moving trust region (static anchor)", it, v, "ppo_8192_anchor_s0_last.pt", "#D3C4BE"))
v = R("base_bc_eval.json")["eval_weighted_secondary_delay_s"]
rows.append(("BC only (no RL)", None, v, "base_bc_eval", M["bc"]))
it, v = by_reward(R("ppo_8192_s0.json")["evals"])
rows.append(("− timing anchor", it, v, "ppo_8192_s0", "#D3C4BE"))
it, v = by_reward(R("base_vppo_s0.json")["evals"])
rows.append(("Vanilla PPO (none of the above)", it, v, "base_vppo_s0", M["vppo"]))

full = rows[0][2]
rows.sort(key=lambda r: r[2])                                             # 小=好, full 自然置顶

# ---- 图 ----
fig, ax = plt.subplots(figsize=(5.6, 3.1))
ys = np.arange(len(rows))[::-1]
for y, (n, it, v, src, c) in zip(ys, rows):
    ax.barh(y, v / 1000, color=c, height=0.6, zorder=3)
    pct = "" if v == full else f"  +{(v - full) / full * 100:.0f}%"
    ax.text(v / 1000 + 4, y, f"{v / 1000:.0f}k{pct}", va="center", fontsize=7, color=M["dark"])
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(0, 268); ax.set_xlabel("validation delay (×10³ s; lower is better)")
ax.grid(axis="x", color=M["grid"], lw=0.5); ax.set_axisbelow(True)
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"figures/pub/F3_ablation_v2.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("saved figures/pub/F3_ablation_v2.*")

# ---- 表 ----
lines = ["| Configuration | Val delay (×10³ s) | vs full | best iter | source run |",
         "|---|---|---|---|---|"]
for n, it, v, src, c in rows:
    pct = "—" if v == full else f"+{(v - full) / full * 100:.1f}%"
    lines.append(f"| {n} | {v / 1000:.1f} | {pct} | {it if it is not None else '—'} | {src} |")
lines.append("")
lines.append("note: every row is selected **by validation reward** (the deployment selection rule, "
             "matching each run's `_best.pt`); v1 mixed selection rules and used the fv1 arm for the "
             "deadlock-surgery row. The − timing anchor run additionally exhibits drift/instability (D70).")
open("figures/pub/ablation_table_v2.md", "w").write("\n".join(lines) + "\n")
print("saved figures/pub/ablation_table_v2.md")
for n, it, v, src, c in rows:
    print(f"  {n:<40s} {v:>10.1f}  it={it}")
