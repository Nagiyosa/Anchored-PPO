"""M11c BC smoke(torch). 沙盒无 torch → **自动 skip**; 本地 GPU 运行验证四编码器前向/掩码/训练/确定性."""
from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")               # 无 torch 时跳过(沙盒)
from derbysim.learn.models import PointerBC, ENCODERS
from derbysim.learn.collate import collate


def _synth(B=8, Mc=3, Ma=4, Dd=18, Dc=8, Da=20, seed=0):
    rng = np.random.default_rng(seed); s = []
    for _ in range(B):
        nc = int(rng.integers(0, Mc + 1)); na = int(rng.integers(2, Ma + 1))
        s.append({"x_dec": rng.standard_normal(Dd).astype("float32"),
                  "x_conf": rng.standard_normal((nc, Dc)).astype("float32"),
                  "x_cand": rng.standard_normal((na, Da)).astype("float32"),
                  "y": int(rng.integers(0, na)), "n_legal": na})
    return s, (Dd, Dc, Da)


def _tensors(samples, dims):
    return {k: torch.as_tensor(v) for k, v in collate(samples, *dims).items()}


def test_all_encoders_forward_and_mask():
    samples, dims = _synth(); t = _tensors(samples, dims)
    pad = t["cand_mask"] < 0.5
    for enc in ENCODERS:
        torch.manual_seed(0)
        m = PointerBC(enc, *dims, hidden=16)
        logits = m(t["x_dec"], t["x_conf"], t["conf_mask"], t["x_cand"], t["cand_mask"])
        assert logits.shape == (len(samples), t["x_cand"].shape[1])
        assert torch.isinf(logits[pad]).all()               # 非法候选屏蔽
        assert torch.isfinite(logits[~pad]).all()


def test_train_step_reduces_loss():
    import torch.nn.functional as F
    samples, dims = _synth(B=64); t = _tensors(samples, dims)
    torch.manual_seed(0)
    m = PointerBC("deepsets", *dims, hidden=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    def step():
        lg = m(t["x_dec"], t["x_conf"], t["conf_mask"], t["x_cand"], t["cand_mask"])
        loss = F.cross_entropy(lg, t["y"]); opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()
    l0 = step()
    for _ in range(40):
        lN = step()
    assert lN < l0                                           # 能学(损失下降)


def test_determinism_same_seed():
    samples, dims = _synth(); t = _tensors(samples, dims)

    def run():
        torch.manual_seed(123)
        m = PointerBC("mlp", *dims, hidden=16)
        return m(t["x_dec"], t["x_conf"], t["conf_mask"], t["x_cand"], t["cand_mask"])
    a, c = run(), run()
    assert torch.allclose(a[torch.isfinite(a)], c[torch.isfinite(c)])
