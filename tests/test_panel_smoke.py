"""M12a panel-option pointer + critic smoke(torch). 沙盒无 torch → **自动 skip**; 本地 GPU 跑。

验: 四编码器前向/形状 · padding 屏蔽 · **BC 热启动整体迁移**(载入 M11 no-prior 权重后, 面板的选路
分支 logits 与独立 BC 在同一输入上**逐元素相等**) · 选路条件分布(KL-to-BC 用) · 确定性 · 可学 ·
设备无关。**安全掩码不在此层**(选项集由环境给, D6)。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")               # 无 torch 时跳过(沙盒)

from derbysim.learn.models import ENCODERS, PointerBC
from derbysim.learn.panel import KIND_CODE, OptionSlot, collate_panels
from derbysim.learn.panel_models import NEG, PanelOptionPolicy, pick_device, to_tensors

DIMS = dict(dec_dim=18, conf_dim=8, cand_dim=20, cancel_dim=14 + 20,
            train_dim=14, global_dim=7)


def _synth_panel(rng, T=3, Mc=3, Ma=4, C=2, N=6):
    """造一个"已编码"的面板(形状与 panel.encode() 一致), 选项 = WAIT + set(T×Ma) + cancel(C)."""
    d = DIMS
    x_cand = rng.standard_normal((T, Ma, d["cand_dim"])).astype("float32")
    cand_mask = np.zeros((T, Ma), "float32")
    slots = [OptionSlot("wait")]
    for t in range(T):
        na = int(rng.integers(2, Ma + 1))
        cand_mask[t, :na] = 1.0
        for j in range(na):
            slots.append(OptionSlot("set", f"T{t}", f"R{t}_{j}", t, j))
    for c in range(C):
        slots.append(OptionSlot("cancel", f"X{c}", None, c, -1))
    x_conf = rng.standard_normal((T, Mc, d["conf_dim"])).astype("float32")
    conf_mask = (rng.random((T, Mc)) > 0.3).astype("float32")
    return {"slots": slots, "options": slots, "train_rows": [f"T{t}" for t in range(T)],
            "cancel_rows": [f"X{c}" for c in range(C)],
            "x_dec": rng.standard_normal((T, d["dec_dim"])).astype("float32"),
            "x_conf": x_conf, "conf_mask": conf_mask,
            "x_cand": x_cand, "cand_mask": cand_mask,
            "x_cancel": rng.standard_normal((C, d["cancel_dim"])).astype("float32"),
            "cancel_mask": np.ones((C,), "float32"),
            "x_panel": rng.standard_normal((N, d["train_dim"])).astype("float32"),
            "panel_mask": np.ones((N,), "float32"),
            "x_global": rng.standard_normal((d["global_dim"],)).astype("float32"),
            "n_options": len(slots)}


def _batch(B=4, seed=0, device=None):
    rng = np.random.default_rng(seed)
    enc = [_synth_panel(rng, T=int(rng.integers(1, 4))) for _ in range(B)]
    return to_tensors(collate_panels(enc), device or torch.device("cpu")), enc


def test_all_encoders_forward_shapes_and_padding_mask():
    b, enc = _batch()
    for name in ENCODERS:
        torch.manual_seed(0)
        m = PanelOptionPolicy(name, **DIMS, hidden=16)
        logits, value, aux = m(b)
        B, O = b["opt_kind"].shape
        assert logits.shape == (B, O) and value.shape == (B,)
        pad = b["opt_mask"] < 0.5
        assert (logits[pad] <= NEG).all()                     # padding 选项被屏蔽
        assert torch.isfinite(logits[~pad]).all()             # 环境给的选项全部有限可选
        assert torch.isfinite(value).all()


def test_bc_warm_start_transfers_whole_routing_branch():
    """§10-A 的关键: 载入 M11 no-prior BC 权重后, 面板选路 logits == 独立 BC 的 logits.

    若只迁移了候选特征而没迁编码器, 这条会失败。
    """
    b, _enc = _batch(seed=3)
    for name in ENCODERS:
        torch.manual_seed(7)
        bc = PointerBC(name, DIMS["dec_dim"], DIMS["conf_dim"], DIMS["cand_dim"], hidden=16)
        bc.eval()
        torch.manual_seed(99)                                  # 面板策略用**不同**初始化
        pol = PanelOptionPolicy(name, **DIMS, hidden=16)
        pol.load_bc_state_dict(bc.state_dict())
        pol.eval()
        B, T = b["x_dec"].shape[0], b["x_dec"].shape[1]
        with torch.no_grad():
            ref = bc(b["x_dec"].reshape(B * T, -1),
                     b["x_conf"].reshape(B * T, b["x_conf"].shape[2], -1),
                     b["conf_mask"].reshape(B * T, -1),
                     b["x_cand"].reshape(B * T, b["x_cand"].shape[2], -1),
                     b["cand_mask"].reshape(B * T, -1))
            got = pol.set_logits(b["x_dec"], b["x_conf"], b["conf_mask"],
                                 b["x_cand"], b["cand_mask"]).reshape(B * T, -1)
        fin = torch.isfinite(ref)
        assert fin.any()
        assert torch.allclose(got[fin], ref[fin], atol=1e-6), f"{name}: routing branch diverged"


def test_wait_and_cancel_heads_are_new_not_from_bc():
    """§10-D: WAIT/cancel 是 BC 里不存在的子决策, 必须是新参数(从头学), 不能伪造时机先验。"""
    bc = PointerBC("mlp", DIMS["dec_dim"], DIMS["conf_dim"], DIMS["cand_dim"], hidden=16)
    pol = PanelOptionPolicy("mlp", **DIMS, hidden=16)
    bc_keys = set(bc.state_dict())
    new_keys = {k for k in pol.state_dict() if not k.startswith("pointer.")}
    assert not (new_keys & {f"pointer.{k}" for k in bc_keys})
    assert any(k.startswith("wait_head") for k in new_keys)
    assert any(k.startswith("cancel_head") for k in new_keys)
    assert any(k.startswith("critic") for k in new_keys)


def test_routing_conditional_is_a_distribution_over_legal_routes():
    """KL-to-BC 只作用在这个条件分布上(§10-B)。"""
    b, _enc = _batch(seed=5)
    torch.manual_seed(0)
    m = PanelOptionPolicy("deepsets", **DIMS, hidden=16)
    lp = m.routing_log_probs(b["x_dec"], b["x_conf"], b["conf_mask"],
                             b["x_cand"], b["cand_mask"])
    p = lp.exp()
    rows = b["cand_mask"].sum(-1) > 0
    assert torch.allclose(p.sum(-1)[rows], torch.ones_like(p.sum(-1)[rows]), atol=1e-5)
    assert (p * (b["cand_mask"] < 0.5)).max() < 1e-6           # 非候选位概率 ~0


def test_determinism_same_seed():
    b, _enc = _batch(seed=11)

    def run():
        torch.manual_seed(123)
        m = PanelOptionPolicy("settransformer", **DIMS, hidden=16)
        m.eval()
        with torch.no_grad():
            lg, v, _ = m(b)
        return lg, v
    l1, v1 = run()
    l2, v2 = run()
    assert torch.equal(l1, l2) and torch.equal(v1, v2)


def test_policy_and_critic_learn():
    import torch.nn.functional as F
    b, _enc = _batch(B=16, seed=2)
    torch.manual_seed(0)
    m = PanelOptionPolicy("mlp", **DIMS, hidden=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    target = (b["opt_mask"].cumsum(1) == 1).float().argmax(1)   # 第一个有效选项
    ret = torch.zeros(b["opt_mask"].shape[0])

    def step():
        lg, v, _ = m(b)
        loss = F.cross_entropy(lg, target) + F.mse_loss(v, ret)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()
    l0 = step()
    for _ in range(60):
        lN = step()
    assert lN < l0


def test_device_agnostic():
    dev = pick_device()
    b, _enc = _batch(seed=1, device=dev)
    m = PanelOptionPolicy("gnn", **DIMS, hidden=16).to(dev)
    lg, v, _ = m(b)
    assert lg.device.type == dev.type and v.device.type == dev.type
