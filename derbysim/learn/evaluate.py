"""M11d 决策匹配率(headline, D12/D14/D16, 纯 python 可测): 按信号宏平均 + 长尾单列 + 优先级分层 + 基线.

只在**真多选**决策(n_legal>1)上算 headline(单选平凡 100% 另计)。
recs 每条: {signal, service_class, chosen_route, pred_route, n_legal, legal_route_ids}。
"""
from __future__ import annotations
from collections import Counter

PRIOS = {"1", "2", "3", "5", "6"}


def _acc(rs):
    return (sum(r["chosen_route"] == r["pred_route"] for r in rs) / len(rs)) if rs else float("nan")


def match_metrics(recs, route_counts=None, tail_le=5) -> dict:
    multi = [r for r in recs if r["n_legal"] > 1]
    sig = {}
    for r in multi:
        sig.setdefault(r["signal"], []).append(r)
    per_sig = {s: _acc(rs) for s, rs in sig.items()}
    macro = sum(per_sig.values()) / len(per_sig) if per_sig else float("nan")
    prio = {}
    for r in multi:
        c = str(r["service_class"]); c = c if c in PRIOS else "other"
        prio.setdefault(c, []).append(r)
    out = {
        "n_multi": len(multi),
        "overall_acc": _acc(multi),                       # 微平均(会被高频信号主导)
        "macro_signal_acc": macro,                        # **headline**: 按信号宏平均
        "n_signals": len(sig),
        "per_priority_acc": {c: _acc(rs) for c, rs in prio.items()},
        "per_signal_acc": per_sig,
    }
    if route_counts is not None:
        tail = [r for r in multi if route_counts.get(r["chosen_route"], 0) <= tail_le]
        out["longtail_acc"] = _acc(tail)
        out["n_longtail"] = len(tail)
    return out


# ---- 基线(D10, 仿真内/数据内实现) ----
def fit_most_frequent(train_recs) -> dict:
    c = {}
    for r in train_recs:
        if r["n_legal"] > 1:
            c.setdefault(r["signal"], Counter())[r["chosen_route"]] += 1
    return {s: cnt.most_common(1)[0][0] for s, cnt in c.items()}


def predict_most_frequent(fitted, rec):
    legal = list(rec["legal_route_ids"])
    r = fitted.get(rec["signal"])
    return r if r in legal else legal[0]


def predict_random_legal(rec, rng):
    legal = list(rec["legal_route_ids"])
    return legal[int(rng.integers(0, len(legal)))]


def expected_random_acc(recs) -> float:
    """随机合法基线的期望准确率 = mean(1/n_legal) over multi(闭式, 免采样方差)。"""
    multi = [r for r in recs if r["n_legal"] > 1]
    return sum(1.0 / r["n_legal"] for r in multi) / len(multi) if multi else float("nan")
