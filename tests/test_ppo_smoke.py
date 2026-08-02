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
from derbysim.learn.ppo import (PPOConfig, RunningMoments, TorchPanelDriver, frozen_bc,
                                greedy_option, masked_categorical, ppo_update, prepare_batch,
                                routing_kl)
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
    with torch.no_grad():                       # 两个小初始化的随机 pointer 分布几乎一样(KL~1e-5),
        for prm in other.pointer.parameters():  # 那测的是"随机初始化很接近", 不是 KL 会不会动。
            prm.mul_(8.0)                       # 放大权重 → 分布真正不同, 才是这条测试要验的事。
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID,
                   other.pointer.state_dict(), dev)
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)
    assert float(routing_kl(pol, bc, batch).item()) > 1e-4


def test_kl_ignores_kind_and_cancel_heads():
    """§10-B: WAIT/cancel 完全放开 —— 改动这两个头不应改变 KL(它只看选路分布)。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, None, dev)
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)
    before = float(routing_kl(pol, bc, batch).item())
    with torch.no_grad():
        for head in (pol.kind_head, pol.cancel_head):
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


def test_driver_denormalises_the_critic_output():
    """critic 预测的是**归一化值**, driver 必须还原成原始单位再交给 GAE。

    否则 δ = r + γV' − V 里 r 是原始单位(运营日整局 −3 万量级)而 V 是 ~N(0,1),
    两边量纲差 4 个数量级, 优势直接是垃圾 —— 而且不会报错, 只会安静地训不出东西。
    """
    ad, res, (net, params, ptc, dwell, w) = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev)
    rn = RunningMoments(mean=-3000.0, var=800.0 ** 2, count=1e6)
    plain = TorchPanelDriver(pol, dev, greedy=True)
    normed = TorchPanelDriver(pol, dev, greedy=True, ret_norm=rn)
    env = RewardedDynamicEnv(net, params, ptc, dwell)
    obs, _ = env.reset(build_episode_trains(w, seed=0))
    dec = env.decision()
    enc = ad.encode(obs, dec)
    i0, lp0, v_hat = plain(enc)
    i1, lp1, v_raw = normed(enc)
    assert (i0, lp0) == (i1, lp1)                     # 动作与 logp 不受影响
    assert v_raw == pytest.approx(v_hat * rn.std + rn.mean, rel=1e-5)
    assert v_raw < -1000.0                            # 确实回到了原始量级, 不是 ~N(0,1)


def test_return_normalisation_keeps_value_loss_in_a_sane_range():
    """写死的 reward_scale 在运营日上会让值损失 ~10^4–10^5, 把策略梯度整个压掉。"""
    ad, res, _ = _setup(n_dec=40)
    dev = pick_device()
    big = [type(t)(t.enc, t.option_idx, t.logp, t.value, t.reward * 500.0, t.dt, t.done)
           for t in res.transitions]                  # 把奖励放大到运营日量级
    cfg_off = PPOConfig(minibatch=16, update_epochs=1, kl_beta=0.0,
                        normalise_returns=False, reward_scale=1.0)
    cfg_on = PPOConfig(minibatch=16, update_epochs=1, kl_beta=0.0)

    def value_loss(cfg, rn):
        pol = _policy(ad, seed=5).to(dev)
        bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, None, dev)
        opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
        return ppo_update(pol, bc, opt, prepare_batch(big, cfg, ret_norm=rn), cfg, dev)["value_loss"]
    off, on = value_loss(cfg_off, None), value_loss(cfg_on, RunningMoments())
    assert on < off / 100.0, f"归一化后值损失 {on:.3g} 相对 {off:.3g} 没有明显下降"
    assert on < 100.0


def test_greedy_picks_the_most_likely_kind_not_the_singleton_wait():
    """扁平 argmax 会系统性地选 WAIT —— 因子化之后必须**按类别贪心**(见 `greedy_option`)。

    实测抓到: 初始体检 WAIT 100% / completion 0.028(与"永远WAIT"同级)。原因不是策略偏好, 而是
    WAIT 类只有 1 个选项独占 p=0.15, 而 set 类的 0.80 摊到 N 条候选上 —— N>5.3 时每条都低于 WAIT。
    这里构造一个"set 类概率最高但候选很多"的批, 断言贪心必须落在 set 上。
    """
    B, O, N = 3, 9, 8
    kind = torch.zeros(B, O, dtype=torch.long); kind[:, 1:1 + N] = 1        # 0=WAIT, 其余=set
    mask = torch.zeros(B, O); mask[:, :1 + N] = 1.0
    k_lp = torch.log(torch.tensor([[0.15, 0.80, 0.05]]).expand(B, 3).contiguous())
    logits = torch.full((B, O), float("-inf"))
    logits[:, 0] = k_lp[:, 0]                                              # WAIT: log 0.15
    logits[:, 1:1 + N] = k_lp[:, 1:2] - torch.log(torch.tensor(float(N)))  # 每条 set: log(0.8/8)
    batch = {"opt_kind": kind, "opt_mask": mask}
    flat = logits.argmax(dim=-1)
    hier = greedy_option(logits, batch, {"kind_log_probs": k_lp})
    assert (flat == 0).all(), "前提: 扁平 argmax 确实会选 WAIT(这正是要修的偏差)"
    assert (hier >= 1).all() and (hier <= N).all(), f"按类别贪心必须落在 set 上, 得到 {hier.tolist()}"


def test_importance_ratio_is_one_when_nothing_changed():
    """**dropout 必须在 PPO 更新期关闭**, 否则重要性比比较的是两个不同的网络。

    `old_logp` 是 rollout 期采的(driver 里 `policy.eval()`, dropout 关)。若更新期开 `train()`,
    `ratio = exp(logp - old_lp)` 会带 dropout 噪声 —— 与参数有没有更新无关。实测后果: 1M 步那轮
    `approx_kl` 在任何学习之前就有 0.023-0.081 > `target_kl=0.02`, 240 迭代触发 **895 次**早停,
    每轮只做约 4 个 minibatch(本该 256), **实际只执行了 1.5% 的更新**, 曲线全程平坦。

    钉死方式: 用**同一份策略、eval 模式**重算**同一批被记录动作**的 logp 当作 `old_logp`, 再以
    `lr=0` 跑一次更新 —— 参数不动、动作不变, `approx_kl` 必须 ≈ 0。
    (第一版这条测试是错的: 我用 driver 重采 logp, 而 driver 会**重新采样动作**, 于是 old_logp 对应
     的动作与 `data["actions"]` 不是同一个, 自然对不上 —— 那测的是"两个不同动作的 logp 不同"。)
    """
    ad, res, _ = _setup(n_dec=40)
    dev = pick_device()
    pol = _policy(ad).to(dev)
    bc = frozen_bc(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim, HID, None, dev)
    data = prepare_batch(res.transitions, PPOConfig())
    pol.eval()
    old = []
    for t in res.transitions:                       # **被记录动作**在 eval 模式下的 logp
        b = to_tensors(collate_panels([t.enc]), dev)
        with torch.no_grad():
            logits, _v, _aux = pol(b)
            dist = masked_categorical(logits, b["opt_mask"])
            old.append(float(dist.log_prob(
                torch.tensor([t.option_idx], device=dev)).item()))
    data["old_logp"] = np.asarray(old, dtype=np.float32)
    cfg = PPOConfig(minibatch=64, update_epochs=1, kl_beta=0.0, target_kl=None, lr=0.0)
    opt = torch.optim.Adam(pol.parameters(), lr=0.0)
    stats = ppo_update(pol, bc, opt, data, cfg, dev)
    assert stats["n_minibatch"] > 0
    assert abs(stats["approx_kl"]) < 5e-3, f"approx_kl={stats['approx_kl']:.4f} —— dropout 没关?"
