"""M12b PPO smoke(torch). 沙盒无 torch → 自动 skip; 本地 GPU 跑。

验: KL-to-BC 的**正确性**(策略==BC 时 KL 必须为 0) · 只对齐选路(WAIT/cancel 不进 KL) ·
PPO 更新能跑且参数真的动 · 确定性 · driver 只回环境给的选项下标 · 端到端(真实窗口跑一小局再更新一次)。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter, collate_panels
from derbysim.learn.panel_models import PanelOptionPolicy, pick_device, to_tensors
from derbysim.learn.ppo import (PPOConfig, TorchPanelDriver, frozen_bc, masked_categorical,
                                ppo_update, prepare_batch, routing_kl)
from derbysim.learn.rollout import make_random_policy, run_episode
from derbysim.learn.scenarios import build_episode_trains, congested_windows, load_assets
from derbysim.sim.reward import RewardedDynamicEnv

ENC = "deepsets"
HID = 16


def _setup(n_dec=24):
    net, params, dwell, ptc = load_assets()
    ad = PanelAdapter(net, Featurizer(net, ptc))
    w = congested_windows("train", top_k=1)[0]
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    res = run_episode(env, ad, make_random_policy(np.random.default_rng(0)),
                      build_episode_trains(w, seed=0), max_decisions=n_dec)
    return ad, res, (net, params, ptc, dwell, w)


def _policy(ad, seed=0):
    torch.manual_seed(seed)
    return PanelOptionPolicy(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim,
                             ad.cancel_dim, ad.train_dim, ad.global_dim, hidden=HID)


def test_kl_to_bc_is_zero_when_policy_equals_bc():
    """最关键的一条: 载入同一份 BC 权重后, KL 必须**恰好为 0**。非 0 = 两边输入/布局对不上。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID,
                   pol.pointer.state_dict(), dev)
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)
    kl = routing_kl(pol, bc, batch)
    assert float(kl.item()) == pytest.approx(0.0, abs=1e-6)


def test_kl_to_bc_is_positive_when_policy_differs():
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad, seed=0).to(dev).eval()
    other = _policy(ad, seed=999)
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID,
                   other.pointer.state_dict(), dev)
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)
    assert float(routing_kl(pol, bc, batch).item()) > 1e-4


def test_kl_ignores_wait_and_cancel_heads():
    """§10-B: WAIT/cancel 完全放开 —— 改动这两个头不应改变 KL(它只看选路分布)。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, None, dev)
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)
    before = float(routing_kl(pol, bc, batch).item())
    with torch.no_grad():
        for head in (pol.wait_head, pol.cancel_head):
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.add_(torch.randn_like(m.weight))
    assert float(routing_kl(pol, bc, batch).item()) == pytest.approx(before, abs=1e-6)


def test_driver_only_returns_environment_offered_options():
    ad, res, (net, params, ptc, dwell, w) = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev)
    drv = TorchPanelDriver(pol, dev, generator=torch.Generator(device=dev).manual_seed(0))
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    out = run_episode(env, ad, drv, build_episode_trains(w, seed=0), max_decisions=30)
    assert out.transitions
    for t in out.transitions:
        assert 0 <= t.option_idx < t.enc["n_options"]
        assert np.isfinite(t.logp) and np.isfinite(t.value)
    assert out.metrics["n_collisions"] == 0


def test_ppo_update_runs_and_moves_parameters():
    ad, res, _ = _setup(n_dec=40)
    dev = pick_device()
    pol = _policy(ad).to(dev)
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, pol.pointer.state_dict(), dev)
    cfg = PPOConfig(minibatch=16, update_epochs=2, kl_beta=0.1)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in pol.parameters()]
    stats = ppo_update(pol, bc, opt, prepare_batch(res.transitions, cfg), cfg, dev)
    assert stats["n_minibatch"] > 0
    for k in ("policy_loss", "value_loss", "entropy", "kl_to_bc", "clip_frac"):
        assert np.isfinite(stats[k]), k
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters()))


def test_ppo_update_is_deterministic():
    ad, res, _ = _setup(n_dec=32)
    dev = pick_device()
    cfg = PPOConfig(minibatch=16, update_epochs=2, kl_beta=0.1)
    data = prepare_batch(res.transitions, cfg)

    def once():
        pol = _policy(ad, seed=5).to(dev)
        bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, pol.pointer.state_dict(), dev)
        opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
        return ppo_update(pol, bc, opt, data, cfg, dev)
    a, b = once(), once()
    assert a == b


def test_zero_beta_disables_the_kl_term():
    ad, res, _ = _setup(n_dec=24)
    dev = pick_device()
    pol = _policy(ad).to(dev)
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, None, dev)
    cfg = PPOConfig(minibatch=16, update_epochs=1, kl_beta=0.0)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    stats = ppo_update(pol, bc, opt, prepare_batch(res.transitions, cfg), cfg, dev)
    assert stats["kl_to_bc"] == 0.0


def test_masked_categorical_never_samples_padding():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    dist = masked_categorical(logits, mask)
    assert float(dist.probs[0, 2:].sum().item()) == pytest.approx(0.0, abs=1e-9)
    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(dist.probs.expand(200, -1), 1, generator=g)
    assert int(draws.max().item()) <= 1
