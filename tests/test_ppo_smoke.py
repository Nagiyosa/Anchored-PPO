"""M12b PPO smoke(torch). 沙盒无 torch → 自动 skip; 本地 GPU / 服务器跑。

D59 后**选路 KL 锚到规则参照**(schedule 计划路 / lookup 最频路), 不再是冻结神经 BC。故 KL 的
正确性判据改为: 参照指向策略**最可能**的进路时 KL 小、指向**最不可能**时 KL 大(= 确实在把选路
拉向参照); 只对齐选路(WAIT/cancel 不进 KL); 无参照 → KL=0。其余(PPO 更新能跑且参数真的动 ·
确定性 · driver 只回环境给的选项下标 · 归一化 · 贪心 · dropout 关) 不变。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from derbysim.learn.features import Featurizer
from derbysim.learn.panel import PanelAdapter, collate_panels
from derbysim.learn.panel_models import PanelOptionPolicy, pick_device, to_tensors
from derbysim.learn.ppo import (PPOConfig, RunningMoments, TorchPanelDriver,
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
    env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
    res = run_episode(env, ad, make_random_policy(np.random.default_rng(0)),
                      build_episode_trains(w, seed=0), max_decisions=n_dec)
    return ad, res, (net, params, ptc, dwell, w)


def _policy(ad, seed=0):
    torch.manual_seed(seed)
    return PanelOptionPolicy(ENC, ad.dec_dim, ad.conf_dim, ad.cand_dim,
                             ad.cancel_dim, ad.train_dim, ad.global_dim, hidden=HID)


def _attach_extreme_ref(pol, enc, dev, which: str):
    """给单个 enc 挂参照: 每个有效选路行, 参照列 = 策略当前**最可能**('best')或**最不可能**('worst')
    的候选进路。返回带 ref_col/ref_valid 的 enc 副本。"""
    b = to_tensors(collate_panels([enc]), dev)
    lp = pol.routing_log_probs(b["x_dec"], b["x_conf"], b["conf_mask"],
                              b["x_cand"], b["cand_mask"])[0].detach()      # [T,Ma]
    cm = b["cand_mask"][0]                                                  # [T,Ma]
    T, _ = lp.shape
    rc = np.zeros(T, np.int64)
    rv = np.zeros(T, np.float32)
    for t in range(T):
        valid = cm[t] > 0.5
        if bool(valid.any()):
            fill = float("-inf") if which == "best" else float("inf")
            masked = lp[t].masked_fill(~valid, fill)
            rc[t] = int(masked.argmax().item() if which == "best" else masked.argmin().item())
            rv[t] = 1.0
    e = dict(enc)
    e["ref_col"], e["ref_valid"] = rc, rv
    return e


def _kl_over(pol, encs, dev, which):
    batch = to_tensors(collate_panels([_attach_extreme_ref(pol, e, dev, which) for e in encs]), dev)
    return float(routing_kl(pol, batch).item())


def test_routing_kl_pulls_toward_reference():
    """最关键的一条: 参照指向策略最可能进路时 KL 应**明显小于**指向最不可能进路时。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    encs = [t.enc for t in res.transitions]
    kl_best = _kl_over(pol, encs, dev, "best")
    kl_worst = _kl_over(pol, encs, dev, "worst")
    assert kl_best < kl_worst, f"best {kl_best:.4f} 应 < worst {kl_worst:.4f}"
    assert kl_best >= 0.0 and kl_worst > 1e-3


def test_routing_kl_is_zero_without_reference():
    """无参照(batch 不含 ref_col / ref_valid 全 0)→ KL 恒为 0(等价关掉锚)。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    batch = to_tensors(collate_panels([t.enc for t in res.transitions]), dev)   # 未挂 ref
    assert float(routing_kl(pol, batch).item()) == 0.0


def test_kl_ignores_kind_and_cancel_heads():
    """§10-B: WAIT/cancel 完全放开 —— 改动这两个头不应改变选路 KL(它只看 routing_log_probs)。"""
    ad, res, _ = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev).eval()
    encs = [_attach_extreme_ref(pol, t.enc, dev, "best") for t in res.transitions]
    batch = to_tensors(collate_panels(encs), dev)
    before = float(routing_kl(pol, batch).item())
    with torch.no_grad():
        for head in (pol.kind_head, pol.cancel_head):
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.add_(torch.randn_like(m.weight))
    assert float(routing_kl(pol, batch).item()) == pytest.approx(before, abs=1e-6)


def test_driver_only_returns_environment_offered_options():
    ad, res, (net, params, ptc, dwell, w) = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev)
    drv = TorchPanelDriver(pol, dev, generator=torch.Generator(device=dev).manual_seed(0))
    env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
    out = run_episode(env, ad, drv, build_episode_trains(w, seed=0), max_decisions=30)
    assert out.transitions
    for t in out.transitions:
        assert 0 <= t.option_idx < t.enc["n_options"]
        assert np.isfinite(t.logp) and np.isfinite(t.value)
    assert out.metrics["n_collisions"] == 0


def _rollout_with_ref(ad, res, dev, pol):
    """把 best 参照挂到每个转移的 enc 上(模拟 collect_rollout 的 ref_builder), 供更新用。"""
    for t in res.transitions:
        e = _attach_extreme_ref(pol, t.enc, dev, "best")
        t.enc["ref_col"], t.enc["ref_valid"] = e["ref_col"], e["ref_valid"]
    return res


def test_ppo_update_runs_and_moves_parameters():
    ad, res, _ = _setup(n_dec=40)
    dev = pick_device()
    pol = _policy(ad).to(dev)
    _rollout_with_ref(ad, res, dev, pol)
    cfg = PPOConfig(minibatch=16, update_epochs=2, kl_beta=0.1)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in pol.parameters()]
    stats = ppo_update(pol, opt, prepare_batch(res.transitions, cfg), cfg, dev)
    assert stats["n_minibatch"] > 0
    for k in ("policy_loss", "value_loss", "entropy", "routing_kl", "clip_frac"):
        assert np.isfinite(stats[k]), k
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters()))


def test_ppo_update_is_deterministic():
    ad, res, _ = _setup(n_dec=32)
    dev = pick_device()
    cfg = PPOConfig(minibatch=16, update_epochs=2, kl_beta=0.1)

    def once():
        pol = _policy(ad, seed=5).to(dev)
        _rollout_with_ref(ad, res, dev, pol)
        data = prepare_batch(res.transitions, cfg)
        opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
        return ppo_update(pol, opt, data, cfg, dev)
    a, b = once(), once()
    assert a == b


def test_zero_beta_disables_the_kl_term():
    ad, res, _ = _setup(n_dec=24)
    dev = pick_device()
    pol = _policy(ad).to(dev)
    _rollout_with_ref(ad, res, dev, pol)
    cfg = PPOConfig(minibatch=16, update_epochs=1, kl_beta=0.0)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    stats = ppo_update(pol, opt, prepare_batch(res.transitions, cfg), cfg, dev)
    assert stats["routing_kl"] == 0.0


def test_masked_categorical_never_samples_padding():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    dist = masked_categorical(logits, mask)
    assert float(dist.probs[0, 2:].sum().item()) == pytest.approx(0.0, abs=1e-9)
    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(dist.probs.expand(200, -1), 1, generator=g)
    assert int(draws.max().item()) <= 1


def test_driver_denormalises_the_critic_output():
    """critic 预测的是**归一化值**, driver 必须还原成原始单位再交给 GAE。"""
    ad, res, (net, params, ptc, dwell, w) = _setup()
    dev = pick_device()
    pol = _policy(ad).to(dev)
    rn = RunningMoments(mean=-3000.0, var=800.0 ** 2, count=1e6)
    plain = TorchPanelDriver(pol, dev, greedy=True)
    normed = TorchPanelDriver(pol, dev, greedy=True, ret_norm=rn)
    env = RewardedDynamicEnv(net, params, ptc, dwell, proactive=True, reserve_window="wide")
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
        opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
        return ppo_update(pol, opt, prepare_batch(big, cfg, ret_norm=rn), cfg, dev)["value_loss"]
    off, on = value_loss(cfg_off, None), value_loss(cfg_on, RunningMoments())
    assert on < off / 100.0, f"归一化后值损失 {on:.3g} 相对 {off:.3g} 没有明显下降"
    assert on < 100.0


def test_greedy_picks_the_most_likely_kind_not_the_singleton_wait():
    """扁平 argmax 会系统性地选 WAIT —— 因子化之后必须**按类别贪心**(见 `greedy_option`)。"""
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
    """**dropout 必须在 PPO 更新期关闭**, 否则重要性比比较的是两个不同的网络。"""
    ad, res, _ = _setup(n_dec=40)
    dev = pick_device()
    pol = _policy(ad).to(dev)
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
    stats = ppo_update(pol, opt, data, cfg, dev)
    assert stats["n_minibatch"] > 0
    assert abs(stats["approx_kl"]) < 5e-3, f"approx_kl={stats['approx_kl']:.4f} —— dropout 没关?"
