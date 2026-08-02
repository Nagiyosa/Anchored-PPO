# Changelog

Microscopic blocking-time simulator + signaller route-decision learning for the Derby station throat.
Milestone-based, newest first. Authoritative sources: `PROJECT_BRIEF.md` (decision log D1–D25),
`docs/IMPLEMENTATION_PLAN.md`, `docs/M9_DESIGN.md`–`docs/M12_DESIGN.md`.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [In progress] Phase 4 (M12): online RL — closed-loop dispatching

Design in `docs/M12_DESIGN.md`. Core = maskable PPO (KL-to-BC + no-prior BC warm start) on the M9
counterfactual environment with M10 rewards; discrete actions + event-driven SMDP (D24); the value
anchor is closed-loop quality vs the signaller, not the BC match rate (D23).

### Decisions and doc alignment (2026-07-30)
- **D25 — the MILP/alt-graph optimality-bound baseline is cancelled** (supersedes that item in D10).
  A microscopic MILP would be model-mismatched against the validated simulator, so both beating it and
  losing to it are uninterpretable, and the marking weight of a CS/RL thesis sits on method, ablations
  and learning-based baselines rather than an OR bound. Strong baselines therefore rest on offline RL
  (IQL/CQL/AWAC) + human replay + strong heuristics (ARS proxy / FSFS / FCFS); an optional light in-sim
  beam search may serve as an optimality reference, otherwise the optimality gap is stated as future
  work. Claims of the form "approaches the bound" are banned. MILP removed as a promised baseline from
  the brief (D10 row, architecture diagram, roadmap table), HANDOFF, the implementation plan and
  `M12_DESIGN`; only the third-party literature citation remains. Brief bumped to v3.0.
- **M12a design points recorded before coding**: the state adapter rebuilds the M8-style context per
  awaiting train (not per option, matching M11); occupancy features stay physical occupancy derived
  from `trains[].cur_tc` and never from `reservations` (lock state) — swapping the semantics of that
  slot would silently degrade the BC warm start; the warm start is pinned to `route_prior={}`.
- `overlap_depth` unified to "passed through in M12a, robustness ablation in M13".

### M12a — framework: state adapter + panel-option pointer + critic (2026-07-30)
- **State adapter** (`derbysim/learn/panel.py`, numpy, sandbox-testable): rebuilds the M8-style
  `deciding` / `conflict_trains` / `candidate_tc_occupancy` / `decision_signal` per **awaiting train**
  from the M9 panel obs, so the whole M11 stack (featuriser, encoder, scoring head) transfers, not
  just the candidate features. Occupancy is derived from each train's `cur_tc` only — never from
  `reservations`; on the M8 side a train holds exactly one TC at a time (`env.py` sets it on enter and
  deletes it on leave), so this is an equivalence, not an approximation. A live `route_prior` is
  rejected at construction.
- **Panel option featurisation**: `set(t,r)` options are grouped into rows by train, which is exactly
  the M11 batch layout; `cancel(t)` = train features ++ its pending route's candidate features; `WAIT`
  is scored from the panel summary. Emitted `OptionSlot`s align position-by-position with
  `PanelDecision.options`, so an argmax index maps straight back to an environment-offered action.
  `collate_panels()` pads several decisions into a batch for M12b PPO.
- **Policy** (`derbysim/learn/panel_models.py`, device-agnostic torch): the set branch *is* the M11
  `PointerBC`, so no-prior BC weights load wholesale; `wait_head` / `cancel_head` / `critic` are new
  and learn from scratch (no fabricated timing prior). `routing_log_probs()` exposes the routing
  conditional, which is the only place KL-to-BC applies.
- **Environment**: `overlap_depth` now passes through `DynamicDerbyEnv` and `RewardedDynamicEnv`
  (default 1, behaviour unchanged); the panel obs gained `rem_path`, `dec_signal` and `pending_route`,
  all of which are consequences of the agent's own past actions or plain physical position, and so are
  observable under Path A online control (M9 §3), unlike the M8 replay snapshot.
- **Tests**: 108 passed, 2 skipped. Sandbox checks cover hand-constructed context truth, the
  occupancy-vs-lock-state discipline, the no-prior guard, option alignment and masks on a real window,
  bit-identical determinism, the flattened gather index arithmetic, and the `overlap_depth`
  passthrough. **Honest limitation**: torch is not installable in the sandbox, so `panel_models.py`
  has not been run — `tests/test_panel_smoke.py` (four encoders, whole-routing-branch warm-start
  transfer, conditional normalisation, determinism, learning, device agnosticism) awaits the local
  GPU. We do not claim the adapter reproduces M8's runtime output: under M9's free timing the decision
  instants differ by construction, so a moment-by-moment comparison is not the right test.

### M12b — PPO: SMDP rollouts, GAE and KL-to-BC (2026-07-30)
- **Decisions (design §10-F/G/H)**: KL-to-BC is a **fixed coefficient swept manually** (β is the paper's
  independent variable, so "how far can we loosen the leash before it pays / before it breaks" reads
  straight off a curve). **The SMDP discount defaults to γ=1** — discounting would make deferring pain
  cheap (150s twenty minutes out is worth ~9.4s at a 5-min half-life), which is the very "leave the hard
  train standing" failure M10's stranding penalty exists to block, and it would break the identity
  "sum of per-step differences = total weighted secondary delay" that ties training to the M13 metric.
  The time-based machinery `γ(Δt)=exp(−βΔt)` is kept (per-step discounting would be wrong under
  time-extended WAIT) and swept as a sensitivity in M12e. Training scenarios are real congested windows
  only, since injecting delay into quiet windows measurably does nothing (Δ=0).
- **Code**: `smdp.py` (time discount + SMDP GAE), `scenarios.py` (congested windows, chain cleaning,
  primary-delay injection, train/val/test split), `rollout.py` (policy injected as a callable, so the
  bookkeeping is testable without torch; records Δt and closed-loop metrics), `ppo.py` (maskable PPO,
  KL-to-BC restricted to the routing conditional and masked to valid routing rows), `scripts/13`.
- **Three real problems found by running, not by inspection**:
  1. **The delay measure had a blind spot** (see M10_DESIGN F7): only "route requested but queued" time
     was counted, so *never dispatching a train cost nothing*. Measured on one window: a random policy
     (18/30 completed, 0s delay, reward −233) scored **better** than dispatching promptly (26/30,
     45758s, −527) — PPO would have learned to stall. Fixed by accounting for time a train spends held
     awaiting dispatch; the prompt policy's numbers are unchanged, so the new term only bites stalling.
  2. **Holdout contamination**: the train split excluded val but not test, leaking 11-27/11-18 windows
     into training. Fixed and pinned by a test.
  3. **Wrong ranking statistic**: ranking windows by total late minutes is dominated by a few extreme
     outliers and picked a 10-train, 8-decision midnight window with zero CIF coverage; ranking by the
     number of late trains gives 30–43 trains and 522–1558 decisions per window.
- **Tests**: 118 passed, 3 skipped. `ppo.py` and `panel_models.py` remain unrun in the sandbox (no torch);
  the discount/GAE maths is unit-tested in numpy, and `tests/test_ppo_smoke.py` awaits the local GPU —
  its key assertion is that KL-to-BC is *exactly* zero when the policy carries the BC weights.

### M12b — smoke-run post-mortem and reward calibration (2026-07-30)
The 40-episode smoke run showed the pipeline works (warm start effective, zero deadlocks, no crashes)
but was not learning — and could not have shown learning under the metrics as they stood. Three real
problems, each found by running rather than by inspection:
1. **`p_incomplete` was charged to trains that cannot possibly finish** (empty `goal.exit_group`: their
   in-region chain is a single fragment, so `_at_goal_exit` is always False). 1–7 such trains per
   window, up to 21%. **Truth reference**: replaying the signaller's own logged decisions over 8 real
   congested windows reaches *every* train that has an exit goal (0 misses, 8/8), and the misses are
   exactly this set — so the human cannot "complete" them either. Gated by `reward.can_complete`, and
   the completion-rate denominator now counts completable trains only (M10_DESIGN F8).
2. **The training curve was dominated by which window got sampled**, not by policy quality: episode
   length swung 180–4586 steps. Rollouts are now collected against a **step budget**, and progress is
   measured on a **fixed evaluation set** (same windows, same injection seeds) alongside baseline
   reference lines.
3. **`DynTrain` is stateful**, so reusing train objects across policies made three different baselines
   report identical numbers (completion 1.0, delay 620, reward 0) — all fake. Evaluation sets now pass
   train *factories*; pinned by a test.

**Reward calibration closed (M10 §9 open item 4): `c_delay` is left unchanged**, contrary to the
hypothesis we started from. With the false penalty removed, the four reference lines are strictly
monotone and the human scores *positive*:

| policy | completion | weighted secondary delay | reward |
|---|---|---|---|
| signaller replay | 1.000 | 930 | +4.2 |
| dispatch-when-possible | 0.860 | 54,463 | −612.6 |
| random legal | 0.577 | 177,654 | −2009.3 |
| always WAIT | 0.009 | 551,943 | −6079.2 |

The earlier "delay is 88% of the reward" reading was an artefact of the false penalty plus a bad
policy: 92–99% of the delay came from trains that never finished. Near human level the delay term is
~9 reward units, comparable to the on-time bonuses, while a poor policy gets a strong "stop stranding
trains" gradient — a sensible progression, so rescaling would not be justified. Tests: 121 passed,
3 skipped.

### M12d — the bottleneck is compounding trajectory divergence, not route choice or ordering (2026-08-02)
Two diagnostic experiments over four validation operating days.
- **Overlap depth is a real but secondary factor.** Dispatch-when-possible scores 0.396 / 0.353 /
  0.288 at overlap depths 0 / 1 / 2 while the signaller stays at 1.000 throughout. Region capacity is
  genuinely tightened by the overlap model, but nothing here explains a gap between 1.000 and 0.35.
  It belongs in the limitations and in the M13 robustness ablation, not in the causal account.
- **The timing-axis experiment was mis-designed, and finding out why was the real result.** Five
  ordering rules produced bit-identical outcomes. The reason is that the panel offers at most one
  train at a time: of 1,017 signaller decision points, 914 offer exactly one train and **none offer
  two or more**; the same holds under dispatch-when-possible. Choosing which train to serve first
  does not exist in the action space — the engine serialises awaiting trains. The measurement was of
  a degree of freedom that is not there. The agent's only timing lever is whether to act on the train
  in front of it or wait, and waiting is itself the reordering mechanism.
- **What is actually wrong.** Under dispatch-when-possible, **72% of decision points offer no set
  option at all**, against 10% for the signaller. It is not declining to act; there is nothing to act
  on, because trains are stuck and the reachability and deadlock masks have removed every candidate.
  Meanwhile the signaller's own action mix is 11% wait and 89% set — it is *also* dispatching
  whenever it can. The only difference is which route, and the lookup reproduces the human's route
  choice 90.6% of the time while doing *worse* in closed loop. The one explanation consistent with
  all of this is covariate shift: the logged route is the choice made in a world where the whole day
  worked, and one step off that trajectory the "most frequent" route is no longer the right one.
  Error compounds decision by decision and the environment offers no recovery. This is the textbook
  compounding-error problem of imitation learning, and it accounts at once for BC improving from
  0.627 to 0.933 with no closed-loop movement, for the lookup's high match rate and poor closed-loop
  result, and for 860k PPO steps producing nothing. It is also precisely the problem online RL exists
  to solve, so the premise of M12 stands.
- Next, in order: a single-deviation recovery test (force one alternative legal route at step k of the
  signaller replay, sweep k, and see whether the day recovers); a redesigned timing axis that varies
  *when not to dispatch* rather than which train to serve; and an audit of why 72% of decision points
  offer nothing, checking whether any of it is a modelling defect such as queued requests never being
  retried.

### M12d — the bottleneck is *when* to dispatch, not *which* route (2026-08-02)
- **The kind anchor works but does not unlock learning.** A KL from the action-kind distribution to
  the signaller's measured mix (`kind_anchor_beta = 0.05`) held cancellations at 0% for all 200
  iterations, against the 50–58% drift of the previous run. Completion nonetheless wandered between
  0.187 and 0.296 with no trend over 860k steps, peaking at 0.289 — still short of
  dispatch-when-possible at 0.297.
- **The diagnosis came from splitting completion by entry time.** Over six equal bins across one
  operating day: the signaller holds 1.00 in every bin; dispatch-when-possible starts at 0.83 and
  decays monotonically to 0.24; the zero-learning lookup starts at 0.52 and decays to 0.29. Every
  automated policy shows the same shape — a decent start, monotone decay, no recovery. The region
  silts up over the day and never clears. This is not a single bad decision; it is cumulative.
- **Routing is not the bottleneck.** The lookup reproduces the human's route choice 90.6% of the
  time and yet does *worse* in closed loop than picking the first available option (0.389 against
  0.461). That single comparison explains everything that had been puzzling: BC improving from 0.627
  to 0.933 overall while closed-loop performance did not move at all (0.284 to 0.289), and 860k PPO
  steps producing nothing. We had been optimising a quantity that is not binding.
- **Framing correction.** Describing this as "ordering versus routing" was wrong. D24 and §14.5 item
  2 define Path A as the agent having full control of *when to dispatch, whether to dispatch, and
  which route*, and M11 §0 states plainly that BC learns which route and not when. Deciding which
  train to serve and when is already part of "when". So the correct statement is that the bottleneck
  falls exactly on the half that BC by definition does not cover and only M12's RL can learn — which
  sharpens the M12 claim rather than changing the story: routing is already near its ceiling
  (0.906–0.933 against a candidate-feature ceiling of 0.928), so all remaining headroom is in timing.
- Added `scripts/17_diagnose_bottleneck.py` with two decisive experiments: overlap depth 0/1/2 to
  test whether region capacity is modelled too tightly, and a timing-axis by routing-axis crossing to
  read off which axis explains the variance in closed-loop quality. Also moved training completion to
  the front of the per-iteration log — it was computed every iteration all along and simply never
  printed, so a flat run could not be spotted early.

### M12d — two more silent failures found by the warm-start check (2026-08-01)
The pre-training action-mix check added alongside the factorisation immediately paid for itself.
- **The `target_kl` guard sat before the optimiser step.** If the first minibatch exceeded the
  threshold, the whole update was skipped with `n_minibatch = 0` and training silently did nothing.
  It never fired before because the policy was near-uniform and matched the behaviour data; once the
  warm start actually took effect the policy became peaked and it triggered immediately. Moved after
  the step, matching the usual PPO implementations.
- **Greedy evaluation cannot use a flat argmax over a factorised distribution.** The check reported
  WAIT 100% and completion 0.028 — the level of the always-wait red-team policy. With
  log p(option) = log p(kind) + log p(option | kind), WAIT is a single option holding all of p=0.15
  while the set branch spreads 0.80 across N candidates, so once N exceeds about 5.3 every individual
  route is less probable than WAIT and the argmax must pick WAIT; waiting then lets trains
  accumulate, which raises N further. Sampling is unaffected, so training was fine and only greedy
  evaluation was distorted. `greedy_option()` now takes the most likely kind first and then the best
  option within it, with a test that pins the failure mode.

All three problems in this round — uncalibrated logit scales diluting the warm start, the guard
placed before the step, and the flat argmax — share a property: no single metric reports them. They
were only visible by loading the weights and running one real day end to end and looking at the
action mix.

### M12d — the warm start was never actually taking effect (2026-08-01)
Smoke run 6 passed every hygiene check — explained variance 0.431 to 0.974, return std steady, zero
collisions, zero deadlocks — and still learned nothing: completion frozen at 0.208 across four
evaluations, with the last two identical to the digit, against 0.385 for dispatch-when-possible and
0.109 for random. KL-to-BC sat at 0.0006 throughout, which we had been reading as "the routing branch
is faithfully tracking BC". The action mix told the real story: WAIT 0%, set 42–45%, cancel 55–58%,
while the environment only offers a cancel option at 11.7% of decision points under the signaller's
trajectory. The policy was taking cancel almost whenever it was available, and each cancellation
queued more requests which offered more cancel options — a self-amplifying loop. Entropy of 0.84–0.93
nats over roughly 2.4 effective options confirms the policy was close to uniform, which is why
completion landed between random and dispatch-when-possible.

The cause is structural. `PanelOptionPolicy` put WAIT, the BC pointer's route scores, and a randomly
initialised cancel head into a single softmax. The pointer's logit scale is only meaningful between
routes, because it was trained under a softmax over routes alone; placed alongside untrained heads
the joint distribution is uncalibrated, and the routing knowledge from the warm start is diluted to
near-uniform. KL-to-BC constrains only the routing conditional by design, so nothing pulled it back.
This explains the long-standing puzzle of why a warm-started PPO could not beat a zero-learning
lookup: it never started from BC's level at all.

Fixed structurally rather than by tuning: the action distribution is now factorised as
log p(option) = log p(kind) + log p(option | kind). The set conditional is exactly the pointer, so the
warm start transfers in full and now coincides with the object KL-to-BC acts on. A three-way
`kind_head` replaces `wait_head`, masked by which kinds are available, with its final layer zeroed and
its bias set to log of a calibrated mix (0.15 wait / 0.80 set / 0.05 cancel) drawn from the
signaller's measured 10.2/89.8/0.0 over 2,857 steps and the measured 10.44% cancellation rate. Three
guards were added, since nothing else reports this class of failure: an initial-mix test, a test that
the set conditional is exactly the pointer distribution, and a pre-training action-mix check in
`scripts/13` that warns loudly if cancel exceeds 20%.

### M11e — with the leak closed, the no-prior BC loses to a zero-learning lookup on its own features (2026-08-01)
The `--no-prior` DeepSets run under the corrected split gives holdout macro 0.499, overall 0.627,
long-tail 0.269, against a validation macro of 0.609 — a generalisation gap of 0.11 that the old
holdout-based early stopping had hidden entirely. To separate "the task is hard" from "the model is
bad", two zero-learning predictors were evaluated on exactly the same features and the same split:

| predictor (D31, holdout, multi-choice only) | macro | overall | long-tail |
|---|---|---|---|
| feature-signature lookup (no route prior, no signal id) | 0.632 | 0.885 | 0.058 |
| per-signal most frequent | 0.650 | 0.906 | 0.000 |
| `route_prior` feature alone, argmax | 0.650 | 0.906 | 0.000 |
| no-prior neural BC (clean split) | 0.499 | 0.627 | 0.269 |
| with-prior neural BC (old, leaky split) | 0.714–0.757 | 0.899–0.907 | 0.19–0.31 |

Three things follow. The `route_prior` feature *is* the most-frequent baseline — the three metrics
agree to the digit — so the with-prior model is a network with a lookup table bolted on as a feature,
and the two must never be presented as independent. On head decisions that model does not beat the
table it is handed: its overall accuracy, 0.899–0.907, is the table's 0.906. Its one real
contribution is the long tail, 0.19–0.31 against 0.00–0.06. And the no-prior model falls well short
of what its own features support, 0.499 against 0.632 macro and 0.627 against 0.885 overall, so the
root cause identified in M11 §11 — the model and its training, not the features — was never actually
fixed; M11d's remedy was to feed the lookup in as a feature and route around it.

This bears directly on where M12 starts: decision A warm-starts PPO from the no-prior model, so the
policy begins below a zero-learning table on the same inputs and KL then holds it there. That is
consistent with the earlier closed-loop finding where a lookup scored 0.883 against PPO's 0.791 after
336k steps. Three candidate remedies are recorded in M11_DESIGN §15 — fix the no-prior model so
scoring is candidate-dominant with the state only modulating it (decision features are 117-dim, 36 of
them signal one-hot, against 20-dim candidate features, so the state drowns the candidates); abandon
decision A and warm-start from the with-prior model; or distil the lookup into the pointer network.
The choice must be made on closed-loop quality via `scripts/16`, not on match rate (D23).

### M11e — four encoders on the D31 dataset, and early stopping was reading the holdout (2026-08-01)
- **Re-ran on the rebuilt dataset** (with prior): MLP 0.757 macro / 0.907 overall, GNN 0.753 / 0.906,
  DeepSets 0.749 / 0.899 (longtail 0.308, best of the four), Set-Transformer 0.714 / 0.905; the
  per-signal most-frequent baseline is 0.650 and random 0.306. Against the old dataset's 0.85–0.86
  this looks like a regression, but the comparison is invalid because the task changed: the baseline
  fell from 0.801 to 0.650. D31 put back the trains the old filter was selectively discarding —
  empty-stock multi-choice decisions are up 84% — and empty stock does not have one dominant route
  per signal, so a per-signal lookup collapses on it. **BC's margin over the baseline went from
  +0.05 to +0.099, roughly doubling.** Putting the hard traffic back makes a learned policy more
  valuable relative to a table, not less, which strengthens the D9 and D23 story rather than
  weakening it. The four encoders still tie (spread 0.043, within the epoch-to-epoch noise of the
  validation curve), so D9's answer is unchanged: the architecture does not matter.
- **Found and fixed: early stopping was selecting on the holdout set.** `scripts/12` passed
  `val_samples=ho` to `train_bc`, so the per-epoch macro was computed on the holdout, the best epoch
  restored from it, and then the same set reported as the held-out result — the logged
  `best_val_macro` and the reported `holdout macro` are identical to the digit. With the per-epoch
  value swinging between 0.63 and 0.76, taking the maximum over eleven epochs is itself a selection,
  so the reported figures are optimistically biased by an estimated 0.03 to 0.08. This has been
  present since early stopping was introduced in M11d. The training段 is now split by date, the last
  15% (9 days, 10-23 to 10-31) serving as validation for early stopping and checkpoint selection,
  with the holdout untouched until the final report; the route prior is fitted on the reduced
  training set so the validation prior features are out of sample. Reports now also carry the
  validation figures so the size of the bias is visible. **The four numbers above must be re-run
  before they count.**
- Also fixed: `scripts/12` overwrote `bc_<enc>.json` and `.pt` in place, and this run destroyed the
  previous with-prior reports. Timestamped copies are now written alongside.

### M12c — BC dataset rebuilt under D31, and the oscillator question settled (2026-08-01)
- **BC dataset regenerated** into `bc_dataset_all_d31.parquet` (the previous version is kept whole in
  `artifacts/dataset_pre_D31_backup/`; `scripts/11` now takes `BC_FULL_DIR` so a change of definition
  produces a new directory rather than overwriting). Two definitions changed at once: broken chains
  are split rather than dropped (D31), and the CIF lookup switched to per-run nearest matching, which
  the RL path had already adopted. Decisions go from 48,021 to 52,294 (+8.9%) and genuine multi-choice
  decisions from 21,876 to 24,430 (+11.7%). The interesting column is traffic type: empty stock
  multi-choice samples go from 1,745 to 3,208 (**+84%**), light engines +36%, class 3 +66%, freight
  +19%, against +4% for passenger — exactly the trains the old filter was selectively discarding.
  Independent corroboration of the CIF fix: the share of `|delay_min| > 120 min` drops from 1.1% to
  **0.0%** and the delay quantiles narrow from p1 = −62.6 / p99 = 57.3 to −19.7 / 26.0, so those
  "extreme outliers" were mostly runs holding another run's planned time rather than real lateness.
  The ±120 clip decided in M11 §8 is now inert. Still to do on GPU: re-run `scripts/12` and
  re-adjudicate every M11 matching-rate number and decision A.
- **The oscillator is not an exploit.** On three validation days it appeared to beat
  dispatch-when-possible (0.575 vs 0.561), which touches red-team invariant 2. Re-checked on all
  eleven validation operating days: dispatch-when-possible wins, 0.297 against 0.294, with 1.77% less
  delay, and the oscillator is ahead on only 2 of 11 days. Per-day reward differences run from
  −14,662 to +12,186 — the day-to-day variance is far larger than the gap between the two policies,
  so three days cannot separate them. Small-sample inference again. Two things follow: the pytest
  red-team audit over two 3-hour windows is a smoke gate, not evidence, and any claim that a policy
  beats a reference must be re-checked on all validation operating days before it enters a conclusion
  or triggers a change to the environment; and invariant 2 is reworded to judge by design intent,
  since the oscillator stopped being degenerate once the cancel time gate closed — what it now does
  is retry a stuck request, which is crude but legitimate.
- **Reference lines, all eleven validation operating days, real history**: signaller 1.0000
  (2,533/2,533) / 94,849 s / +181; dispatch-when-possible 0.297; oscillate 0.294. Zero collisions,
  zero deadlocks, and the signaller is at 1.000 on every single day.

### M12c — three scope decisions frozen: D30 / D31 / D32 (2026-08-01)
- **D30 — the encoder ablation crosses only the primary method.** PPO gets all four encoders × 3
  seeds; DQN, IQL, CQL, AWAC and AWAC-online each use DeepSets × 3 seeds. D9 asks whether the
  architecture matters, and M11 already answered that on BC (the four encoders tie at macro
  0.85–0.86); confirming it once on PPO answers it for the online setting, whereas crossing all six
  methods repeats the same finding six times. Every method and every other ablation axis is still
  done — only this one axis stops being a Cartesian product. Runs needing environment rollouts drop
  from 117 to about 57.
- **D31 — broken chains are split into separate movements instead of the whole train being dropped.**
  The old filter discarded 303 of 2,056 trains over ten days, but selectively: 5.5% of express and
  5.6% of stopping passenger services against 45.8% of empty stock, 32.3% of light engines and 17.5%
  of freight. 159 of 169 empty-stock breaks look like `TEDJ → TEDG` — a train entering Etches Park
  depot, sitting in track we do not model, and coming back out. Those are two independent movements,
  not one broken chain, so `dataset.split_broken_chains` now emits one train per continuous segment,
  keyed `<id>@<n>`, with entry taken from the observed occupancy of the segment's first circuit,
  `planned_ref` given only to the first segment, and stops assigned to whichever segment contains the
  platform. Both the BC and RL construction paths use it. On 2023-11-05 the day goes from 176 to 192
  trains; the signaller still completes 1.000 with zero collisions. **The BC dataset must be
  regenerated and all M11 numbers re-run** — not yet done.
- **D32 — throughput: policy on CPU during rollout, then run the matrix concurrently.** Of the 5.6 ms
  per rollout step, 4.06 ms is the kernel-launch round trip for a batch-of-one forward on a
  hidden-64 network, while pure CPU simulation and encoding take 1.5 ms; GPU compute is 11.5% of wall
  time, so a faster card buys nothing. `--rollout-device` now defaults to cpu. The useful side effect
  is that a run becomes a pure single-core CPU job, so the ~57 runs can go 32-wide without contending
  for the GPU: 319 hours serial becomes roughly 4. Prebuilding trains with deepcopy and thinning
  evaluation come next; vectorised parallel environments are deferred until there is a bit-exact
  equivalence test, and rewriting the simulator core is ruled out because it would touch the D6
  safety guarantee. Also noted: the 2M-step budget was assumed, not measured — the first thing after
  the definitions settle is a full learning curve.

### M12c — no synthetic perturbation, terminating trains get a goal (2026-08-01)
Two definitional decisions, both frozen; they change what the headline number means.
- **D28: synthetic delay injection is removed from training, validation and testing.** The recorded
  entry times already contain the real delay of that day, so the counterfactual we actually want to
  pose — "given the same trains arriving at the region boundary at the same times, would a different
  set of routing decisions have produced less delay inside the region?" — is well posed without any
  injection, and injection only muddies it. The primary-delay share α = 0.36 comes from Palmqvist's
  Swedish case study and has never been validated on Derby data; what we had validated was only the
  marginal distribution of the injected magnitudes, never that the resulting congestion pattern
  resembles reality. Those are not the same claim. Measured over three validation operating days,
  injection raises the signaller's weighted secondary delay from 6,544 s to 16,487 s — 2.5× — while
  the replayed human never gets to re-decide, so our own human reference was being systematically
  depressed. Robustness under disruption will instead be reported by stratifying the real days by
  observed disruption. The `inject` default flips from True to False so that forgetting the flag
  cannot silently perturb a run, and `--eval-seeds` is gone. Limitation to state in the paper: we
  cannot test conditions worse than anything observed. Note the environment is fully deterministic
  (`_adv` and dwell are point estimates), so 65 training days replayed ~90 times each is a real
  overfitting risk — but a detectable one, and the grounded fix if it appears is sampling the
  M5-calibrated per-TC traversal residuals, not α.
- **D29: trains that terminate at Derby now have a goal instead of being excluded.** 1,651 of 18,650
  trains across all 98 days have an empty exit group. The previous note attributed this to window
  boundary fragments; diagnosis shows otherwise — only 2 of 45 in a three-day sample sit near the
  boundary, while 82% of a twenty-day sample end on a platform track circuit with headcodes 1C, 2A,
  1K, 1F, 1V, 1D. They are terminating passenger services whose headcode dies at the platform. They
  do have a goal: reach the platform and stop. Excluding them meant an agent could leave a
  terminating train standing in the throat at no cost while it blocked a platform. Completion is now
  judged by `has_reached_goal`, which checks the exit group for through trains and `served_platform`
  for terminating ones. The completable denominator goes from 477 to 513 over three validation days,
  the excluded set from 45 to 9, and the signaller still scores 1.000 (513/513) — which is itself
  evidence the goal definition is right.
- Reference lines on three validation operating days, real history: signaller 1.000 / 6,544 s /
  +166; oscillate-set-cancel 0.575; dispatch-when-possible 0.561; loop-the-train 0.183; random
  0.113; always-wait 0.014; zero collisions and zero deadlocks throughout. Open: oscillate now edges
  out dispatch-when-possible, which touches red-team invariant 2; the red-team audit runs on 3-hour
  windows and does not see it, so this needs checking on operating days.
- **Open finding — the chain-continuity filter is selective.** It drops 303 of 2,056 trains over ten
  days (14.7%), but unevenly: 5.5% of express and 5.6% of stopping passenger services against 45.8%
  of empty stock moves, 32.3% of light engines and 17.5% of freight. The cause is Etches Park depot:
  159 of 169 empty-stock breaks look like `TEDJ → TEDG`, a train entering the depot, sitting in
  track we do not model, and coming back out. That is a modelling boundary, not dirty data, and
  automatic repair is not viable (of 303 breaks, 1 has a unique connecting path, 216 have several,
  86 have none). The consequence is that our scenarios contain roughly half the empty-stock movement
  that really occurs, and empty stock is a major source of throat conflict at a terminus — so
  conflict intensity is systematically understated, in the BC dataset as well as in the RL
  scenarios. Options recorded in M12_DESIGN §15; splitting a broken chain into separate movements is
  the most faithful but would require rebuilding the BC dataset and re-running all of M11.
- Tests: **143 passed, 3 skipped**.

### M12c — full-data recheck: three open findings (2026-08-01)
Not fixed, recorded for decision (`docs/M12_DESIGN.md` §14).
- **8.85% of trains have no exit target, and it is not the day boundary.** Across all 98 days,
  1,651 of 18,650 trains have an empty `goal.exit_group` and are therefore excluded from the
  completion denominator. Both D27 and M10_DESIGN F8 state that these are the trains cut by the
  window boundary; that is wrong — in a three-day sample only 2 of 45 sit near the boundary, and
  80% of them have a route chain of length 1 against a typical length of 4. Cause not yet
  established: chain cleaning truncating at a discontinuity, gaps in the TD feed, or services that
  genuinely terminate inside the region. This is a denominator definition, so it moves the headline.
- **The fixed evaluation set contradicts its own design.** M10_DESIGN §4.4 is titled "fixed
  evaluation (real historical windows, no injection)", but `scripts/13` builds it with injection
  seeds 0 and 1. Measured over three validation operating days, injection raises the signaller's
  weighted secondary delay from 6,544 s to 16,487 s — 2.5× — while the replayed human never gets to
  re-decide, so the human reference is systematically depressed and we are comparing against an
  unfairly weak anchor. Proposal: headline evaluation returns to real history, injection becomes a
  robustness ablation reported separately.
- **The experiment matrix is larger than stated.** The "47 runs" figure was my own arithmetic and is
  too low. §14.2/§14.4 commit six learning methods crossed with four encoders and multiple seeds,
  plus KL, half-life, warm-start and mask ablations — about 111 training runs, roughly 26 days
  serial at the current throughput. Needs a decision on whether the encoder ablation really crosses
  all six methods or only the primary one.
- **Correction to a stale claim**: M12_DESIGN §3/§8 said "the environment is fast (13.5k decision
  steps/s), so it is not the bottleneck". That number is from the M7/M8 static engine. Measured now:
  1,157 steps/s for the bare dynamic engine, 667 with reward and the state adapter, 180 in actual
  training — 20× off. That sentence is why nobody looked at performance for months.

### M12c — evaluation pooling, return normalisation, and a second collision (2026-08-01)
- **Evaluation now pools across episodes instead of weighting each day equally.** With whole
  operating days the per-day train count spans 5 to 241. `M11_DESIGN` §2 had already identified and
  recorded the three thin days (09-01 = 5 decisions, 09-30 = 17, 12-07 = 37, against a median of
  568) and decided to keep them — correct for BC, where they are 0.01% of 48,021 samples, but in RL
  a day is a whole episode: one thin day was 9% of the validation score, and 12-07 sits in the test
  split and would have gone straight into the M13 headline. Completion is now Σreached / Σcompletable;
  delay, reward, steps and collisions are reported as totals; action fractions are weighted by steps;
  the old per-episode mean is kept as `..._macro`. Training needed no change — step-budgeted
  collection already weights days by their length, which is the "weight by day volume" that M11 left
  as optional. We did **not** drop the thin days: a "drop days not fully covered by the data range"
  rule would miss 09-30, which sits mid-period.
- **`reward_scale` replaced by running return normalisation** (`smdp.RunningMoments`). The constant
  0.01 was calibrated on 3-hour windows; on operating days within a single episode |r| spans five
  orders of magnitude (median 0.00, p99 826, max 8,968, with the terminal alone accounting for 29% of
  total |r|) and episode returns run from +40 for the signaller to −31,966 for dispatch-when-possible.
  The critic now predicts normalised values and the driver un-normalises before GAE — get that wrong
  and the advantage is off by four orders of magnitude without any error being raised, so it is
  pinned by a test. No PopArt weight compensation, so the critic re-adapts when (μ,σ) shift; drift is
  slow because the statistics are running rather than per-batch, and `--no-return-norm` is kept for
  the ablation.
- **A second collision, and the reporting change is what found it.** The random policy scored one
  collision on 2023-11-07. Root cause: route chains are chosen online by the policy, so a policy can
  drive a train in a circle — `RDC5061A→RDC5075A→RDC5083A` traverses `TDAB→TDAC` three times — and
  `_release_behind(idx-2)` assumed a path never revisits a track circuit. The train released the lock
  under its own wheels, another train took it, and the first train drove back in. This is not a data
  defect: across all 98 days and 18,650 trains, zero planned chains contain a repeated track circuit.
  Fixed by refusing to release any circuit still present in `path[pos:]`. Added `loop_the_train` to
  the standing red team and a direct invariant test. Note the old evaluation code averaged collisions
  over episodes, so one collision in three days printed as 0 — the defect had been reachable and
  invisible.
- Tests: **143 passed, 3 skipped**.

### M12c — scenarios reworked into continuous operating days (2026-07-31)
The user pushed back twice, and both times the design was wrong rather than the tuning.
- **An episode is now a whole operating day, 03:00 to 03:00, over all 98 days** (train 65 / val 11 /
  test 22, split purely by date, no disruption filter; `scenarios.operating_days`, `build_day_trains`,
  `day_horizon`, with the split fingerprints pinned in tests). This replaces "the 6 most disrupted
  3-hour windows", which was 0.77% of the data while BC trained on all 98 days — an asymmetry we had
  never stated. The environment is a continuous process in time; slicing it and keeping only the top
  7.9% of windows meant the agent only ever saw states that were already bad, never the build-up or
  the recovery, and always started from an empty region. The original justification — "injecting
  delay into a quiet window changes nothing" — was measured on one window; stratified over the full
  range, median windows carry as much learning signal as the top decile (23–32 trains, 35–70 genuine
  multi-choice decisions, dispatch-when-possible completing only 0.633–0.950). Injection is indeed
  inert in quiet windows, but "injection is inert" is not "the window has no conflict".
- **The 03:00 cut is chosen from data**, on the right criterion: not "which hour has fewest requests"
  but "where does cutting slice fewest trains in half". Re-deriving trips from raw TD by the
  90-minute gap rule over 98 days: 03:00 cuts 0.80 trips/day, 02:00 0.90, midnight 2.49, 06:00 4.11.
  The obvious measurement was wrong — `train_occurrences` is built per calendar date, so
  midnight-spanning trips are already split there and the table cannot detect midnight cuts at all.
- `congested_windows` is kept for stratified reporting (normal vs disrupted, §14.0) and for a
  "train on disruption only vs train on everything" ablation.
- **A latent CIF defect surfaced and was fixed.** `build_cif_map` was keyed by headcode alone, so a
  headcode running twice in a day had its earlier run overwritten by the later one: 3.0% of
  (headcode, date) pairs, 6.0% of trips, 814 of them carrying another run's planned time (1M00's
  09:31 run was given the 22:43 schedule, 13.2 hours out). The 3-hour-window path had this all along.
  `build_cif_index` now matches per run by nearest first event and honours the existing
  `cif_reliable` flag — where alignment is unreliable no planned time is given, because a schedule
  22.8 hours out is worse than none. Real coverage is 79.4%; the 88% we used to see was partly wrong
  values.
- Tests: **137 passed, 3 skipped**. Smoke over three val operating days (143–194 trains each):
  signaller 1.000 completion / +11.4 reward, dispatch-when-possible 0.550, oscillate-set-cancel
  0.545, random 0.118, always-wait 0.006 — strictly ordered, zero collisions, zero deadlocks.

### M12b — three training runs, three real causes found (2026-07-30)
None of the three runs learned, but each one was diagnosed to a real cause rather than written off as
"needs more steps". Recorded here because the diagnostic method is the transferable part.
1. **Run 1 (93k steps) — reward hacking.** Evaluation reward approached the human's while completion
   was 0.217; the arithmetic did not add up. Cause: `cancel` could undo a route in the *same instant*,
   freezing the simulation clock so standing time cost nothing, and truncated episodes skipped the
   terminal settlement. A do-nothing oscillator scored +3.0 where competent dispatching scored −1660.
2. **Red-team audit** (user: "RL finding holes is normal — the point is to rule them out
   systematically, not to muddle through"). Three more holes surfaced: deliberately deadlocking
   escaped the incomplete penalties entirely (−3050 vs −3500 for the same stranding); the episode end
   time depended on the policy, so doing less truncated your own bill; and trains queued at the entry
   route accrued nothing at all. Fixed by two structural rules — terminal settlement is independent of
   how the episode ends, and no unbounded actions in zero simulated time — plus a fixed horizon and
   entry-queue timing. A standing battery of eight deliberately exploitative policies now pins the
   invariants (`derbysim/learn/redteam.py`, `tests/test_redteam.py`).
3. **Run 2 (87k steps) — no learning at all.** KL pinned at 0.004, two evaluations bit-identical.
   Measured cause: GAE λ=0.95 has an effective horizon of 20 steps against episodes of 143–642 steps
   with reward concentrated at the end, so the first decision saw **0.2%** of the episode return.
   λ default moved to 1.0, rewards scaled for the critic.
4. **Run 3 (87k steps) — policy moved, in the wrong direction.** Completion fell 0.567 → 0.388, worse
   than random. Measured cause: with an untrained critic the advantage correlates **0.77–0.96** with
   step index — it encodes *position in the episode*, not action quality, so early actions are pushed
   down and late ones up regardless of merit. Added a critic warm-up phase (value head only, policy
   frozen) and made **explained variance** a per-iteration diagnostic; EV≈0 is the direct detector for
   this failure and should have been printed from the start.

**Reading order for the next run, fixed**: explained variance first, then KL/entropy, then whether
completion and delay improve *together*. If EV stays near zero, the problem is the critic — not the
learning rate and not the step budget.

## [In progress] Phase 4 (M11): BC warm-start + decision-match rate

Behaviour cloning of the signaller's route choice (headline = decision-match rate, D21), as a warm-start
prior for M12 RL. `derbysim/learn/` (features / collate / models / train_bc / evaluate), `scripts/11`
(full 98-day dataset), `scripts/12` (training CLI). Design in `docs/M11_DESIGN.md`.

### M11a-c
- **Data**: full 98-day decision dataset (48021 decisions / 21876 multi-choice, 0 mask-misses), temporal
  split 9-10 train / 11-12 holdout; delay clip ±120 (|dm|>120 only 1.1%).
- **Model**: pluggable pointer over legal routes with reachability-based candidate features; 4 encoders
  (MLP / DeepSets / Set-Transformer / GNN), device-agnostic, no torch-version pin. Non-torch parts
  (features / collate / evaluate) unit-tested; torch smoke skips without torch (run on GPU).

### M11d — results + honest analysis (decision A)
- Encoders **tied** (macro 0.850-0.857, overall 0.941-0.945; spread <1%) → D9 ablation answer:
  architecture doesn't matter for this small structured state.
- Root-cause dig (lookup on identical features hits 0.93 overall / 0.72 macro): first BC underperformed a
  lookup → **model overfit, not features/data**. Fixed via train-fit per-(signal,route) choice prior
  (=baseline as a feature) + signal one-hot + dropout + weight-decay + val-macro early-stopping → BC
  beats baseline (0.86 macro / 0.94 overall).
- **But** the prior turns BC into a routine follower: long-tail (0.8% of decisions, the deviations we
  care about) collapsed below random. Long-tail weakness is characterised, not a bug: data-limited
  (3 months; §3 always said the tail is for synthetic/RL, not more real data), information-limited
  (deviation reasons often off-state), and inherently RL's domain (D23).
- **Decision A**: report both sides honestly; warm-start uses the no-prior (+reg) BC (`--no-prior`, keeps
  deviation signal, long-tail > random) so KL-to-BC doesn't bias RL away from deviations; the prior
  version is the routine ceiling. BC is a descriptive result + safe prior, **not the headline** (D23).

---

## Phase 4 (M10): reward + perturbation

Attach an environment reward to the counterfactual env (`DynamicDerbyEnv`) and inject data-grounded
perturbations so online RL (M12) has a learning signal. Reward is a *small* term (imitation stays the
headline via BC+KL, M11/M12); safety stays structural. Design in `docs/M10_DESIGN.md`. Non-invasive:
reward lives in `derbysim/sim/reward.py` (`RewardedDynamicEnv` subclass), verified `env_dynamic.py` untouched.

### M10a — dense secondary-delay reward (2026-07-23)
- Secondary delay is a **small real objective delivered as per-step differences** (ΣλΔwait), TOPS-priority
  weighted (D16). **Not** claimed as policy-invariant potential shaping — that is self-contradictory
  (invariant = no incentive to cut delay); corrected in `PROJECT_BRIEF.md §7`. Robustness to the M9 ~4×
  inflation comes from relative reporting (common-mode cancels) + the M13 overlap check, not invariance.
- Realized delay counts in-progress blocking (`now − blocked_since`), else the signal is lumpy.
- Deadlock terminal penalty. Reward = pure function of engine snapshot before→after a step.

### M10b — platform 3-tier + on-time + stranded terminal (2026-07-23)
- **Platform service, 3-tier soft penalty**: booked 0 / re-platform −ε / passenger-no-stop −P_nostop.
  Grounded in UK practice (WTT/docker booked = default; re-platform is a deliberate manager decision; ARS
  routes only to booked). Served platform derived post-hoc from the path; engine untouched.
- **on-time** bonus (coarse: single arr/dep/pass reference; tolerance tunable).
- **Non-completion / stranded** = terminal loss (−P_incomplete/train); under natural termination a
  not-through train was starved by the policy (F4). Per-completion credit assignment.

### M10c — data-grounded perturbation injection (2026-07-23)
- Real Derby TRUST distribution: 50% LATE; late-only median 4 min, p90 19 min, heavy tail to ~1 h.
  Non-parametric bootstrap from the empirical lateness pool (not a Gaussian / fixed value).
- Inject **primary** delay only: observed delay = primary + secondary, so injecting the full observed
  delay would double-count the secondary the simulator regenerates. Scale by **α = 0.36** (Palmqvist et
  al. Swedish macro-sim ~36% primary / 64% secondary; literature 20–50%) — a *sourced* sensitivity
  parameter, not a guess, not a headline. `p_late = 0.50` = real Derby LATE fraction. Clip 120 min
  (drop headcode-reuse artifacts). Route unchanged → reachability preserved. Deterministic (seeded).
- Add-trains = resample real services (clone route_chain) → topologically valid by construction.
- Fixed evaluation = real historical high-disruption windows (no injection); reproducible.

### M10 review + fixes (2026-07-23)
Full review on real data surfaced 4 real issues (M10_DESIGN §12):
- **R1 (fixed)** — `realized_weighted_delay` counted in-progress blocking for *done* trains that the M9
  engine leaves with a stale `blocked_since` → phantom delay growing with the clock (inflated ~1.5k→460k
  under perturbation). Fix: in-progress only for non-done trains.
- **R2/R3 (perturbation redirected)** — entry-delay injection is inert in uncongested windows (Δ=0, no
  overlaps to perturb); synthetic add-trains (`add_resampled_trains`) broke the no-collision invariant
  (reservations keyed by `route_id`; clones reuse `route_id`s). Both removed/kept-honest. New direction:
  **real high-disruption windows** (`rank_high_disruption_windows`) built with chain-continuity filtering
  (`build_clean_window_dyn_trains`) + delay injection as congestion augmentation. Busy-window raw builds
  carry H6-dirty trains that collide on replay; the same dataset filter → 0 collisions.
- **R4 (deferred)** — platform −P_nostop mis-fired on window-boundary fragments (real stoppers like 1O06
  whose stop falls outside the captured window). Gated off (`enable_nostop=False`); keep booked 0 /
  re-platform −ε; the no-stop term waits for M13 boundary machinery.

### Tests / status
- test_reward.py 9 + test_perturb.py 6 → **88 passed**. Determinism enforced throughout.
- Pending (M13): overlap_depth passthrough for the robustness sweep; boundary machinery for −P_nostop.

---

## Phase 3 (M9): dynamic routing + full physical locking + robust deadlock handling

Counterfactual environment — the policy actually "drives the panel" (Path A: set route / cancel / wait).
From here it deliberately diverges from the log; validation switches to physical consistency.
Design in `docs/M9_DESIGN.md`. Decided: Path A + explicit cancel, free routing (platform is a consequence
of route choice), deterministic, geometric mask, overlap included, no artificial cap on how early a route
may be set, deadlock = terminal failure, obs includes current lock state.

### M9 review round 2 (examiner) — fixes (2026-07-16)
- **platform-service = soft goal, not a hard action filter**: it was pruning the signaller's real
  route on 22.4% of decisions (routes toward an exit but away from platforms). Hard reachability now
  = exit-direction only; platform service moves to the M10 reward. (goal.py)
- **H6 per-trip split**: build_env_trains splits a headcode into runs on >90-min gaps (data.alignment
  rule), so reused headcodes with far-apart trips aren't glued; BC dataset regenerated clean.
- **decision signal by physical continuation**: offer routes whose first TC continues the train's
  current TC (successor), unioned with signal/paired-end — not signal-name matching. Full-day
  false-negative 21.8% -> 0 on continuous chains. Broken chains (H6 out-of-order, ~15%) filtered from
  the BC dataset (dataset.chain_is_continuous). (env_dynamic.py, dataset.py)
- **conflict-dynamics validation (honest)**: log-timing-honored M8/ReactiveEngine = M6-validated
  ~observed (112s/3 conflicts); M9 free-timing overlap-1 = 431s (~4x, inflated by free timing +
  overlap). overlap highly sensitive (depth 0/1/2 = 224/431/3698s) -> reward must be potential-based
  (relative). Real delay conservation (monotone / attributable) replaces the trivial wait>=0 claim.
- overlap confirmed data-grounded (topology), only its length is approximated; configurable depth.
- cancel exercised in a full episode (deterministic); multi-window robustness confirmed (path
  consistency / determinism / safety across peak/off-peak/calibration/holdout/full-day).
- corrected over-claims in docs (§4/§10/§11); +tests, 73 pass.

### M9d — DynamicDerbyEnv: full Path A counterfactual env (2026-07-16)
- `sim/env_dynamic.py`: reservation-driven engine (advance freely inside a locked route; block
  only when a request queues; sectional + on-completion release) with the full Path A action set
  `{WAIT} ∪ {set(train, route)} ∪ {cancel(train)}` (PanelDecision/PanelAction), deadlock-lookahead
  masking + terminal-failure on detected deadlock, goal-group termination, and a panel `obs`
  (all in-region trains + current reservation lock state + time). `DynamicDerbyEnv` gym wrapper.
- consistency regression vs M8 (fed the logged decisions): **paths bit-identical for all 34/34
  trains**; free-flow occupation matches (median 0s, 61% ≤2s = the M6 ~2s validation); the timing
  tail (mean 98s) is the intended divergence from dropping M8's log-time gate (agent controls
  timing in Path A). Physical sanity: 0 collisions, no deadlock, delay conservation.
- Path A smoke: env is drivable by a non-replay policy, safely (0 collisions, terminates).
- fixes: reachability counts "route ends at the goal exit" as reached; platform-reversal decision
  signal (arrive one end / depart the other) via paired platform ends.
- known limitation: goal-reachability filter is conservative (occasional false negatives) — harmless
  in replay, may under-offer routes under aggressive free-routing; to be hardened in M11. +8 tests. 68 pass.

### M9c — Deadlock detection + lookahead (2026-07-16)
- `sim/deadlock.py`: wait-for graph from reservation state (a train with a QUEUED request waits on the
  train holding the conflicting reservation); DFS cycle detection = deadlock.
- Deadlock → episode terminal failure (per M9). `would_create_deadlock`: one-step lookahead
  (snapshot → hypothetically request → check → rollback) for action-mask pruning.
- Under full physical locking, blocking only occurs at the queued-request layer, so deadlocks surface as
  cycles among queued requests. M8's `MAX_WAIT` timeout retires in M9d. +4 tests. 61 pass.

### M9b — Goal + reachability (2026-07-16)
- `sim/goal.py`: exit direction groups derived from topology (exits sharing approach signals are unioned;
  8 groups, e.g. 5093/5095/5097 northbound, 5323/5329/5331 southbound).
- Goal = reach any exit in the train's onward-direction group (not the single logged exit);
  Reachability = in-region BFS terminating at boundary exits.
- Hard action constraint (`route_keeps_goal`): prune routes that would strand a train (goal no longer
  reachable) — dead-ends are invalid actions, not strategy, so filtered rather than learned.
- Passenger service: stopping trains must route through a platform before exiting. +5 tests. 57 pass.

### M9a — Route reservation engine (2026-07-16)
- `sim/reservation.py`: full physical locking — setting a route reserves its TCs + overlap; a conflicting
  request is QUEUED (not rejected) and auto-locks once the blocking reservation is released sectionally.
- Realises the D22 request/lock split: request layer = geometric mask, lock layer = this module
  (matching "a request may be over an occupied TC; the lock waits for it to clear" — 2.2% in data).
- overlap = the TC immediately beyond a route's end signal (available in data); it raises cross-signal
  route-pair conflicts 16.6% → 21.2% (+28%), tightening conflict realism. +6 tests. 52 pass.

---

## Phase 3 (M7–M8): gym environment

### M8 — Leak-free state + physical trigger (D18/D22) (2026-07-16)
- Physical trigger: platform-exit signals fire on dwell entry; approach signals fire N[S] berths upstream
  (N calibrated from 55249 decisions, look-back spans route segments). 91% of real multi-choice decisions
  fire early.
- Leak-free state (`_build_state`): a causally-prior snapshot at the decision instant — excludes this
  decision's consequences (lock / signal-clear / committed motion / the answer), keeps observables
  (occupancy / approaching trains' positions / delay / CIF target / time-of-day). State = deciding train +
  all conflict trains. 0 leaks.
- Action mask = geometric legal exits (D22, request layer: 2.2% of real requests are over an occupied TC;
  occupancy is a state feature, learned by the policy).
- Decoupling keeps M7 bit-equivalence (ask at the physical trigger, advance on log req_time).
- Lessons: no-leak comes from state construction, not early snapshots; decision signal = the requested
  route's start signal (fixes platform pairing / ECS reversal).
- Phase-1 BC decision dataset as env side-product (`sim/dataset.py` + `scripts/10`; 2023-11-06:
  597 decisions / 301 real multi-choice). +6 tests.

### M7 — Pausable gym-style DerbyEnv (2026-07-11)
- `sim/env.py`: generator-based reactive engine (yield at decision point → send route → materialize route
  TCs incrementally) + `DerbyEnv` (reset/step).
- Engine-equivalence regression: replay mode == `ReactiveEngine` bit-for-bit (path/ready_at/dwell/
  wait_time/sim_occ). Entry route treated as non-decision. 40 tests.

---

## Phase 2 (M3–M6+): microscopic simulator + validation

### M6+ — Reactive blocking-time engine (keystone) (2026-07-04)
- `sim/reactive.py`: trains advance TC-by-TC, stopping when the next section is occupied — secondary delay
  emerges causally (train-waits-for-train).
- Honest argument: full-chain second accuracy is the wrong target for a differential objective (speed
  variance is common-mode); characterise holds (98% exogenous / 2% TC-conflict).

### M6 — Holdout replay validation (keystone gate) (2026-07-04)
- `validation/{metrics,replay,report}.py`: forward replay on the Nov–Dec holdout, event-by-event.
  Occupation median error 1.0s (= calibration set, generalises), 10s hit-rate 91%, zero bias.
  Descriptive validity established.

### M5 — Event-driven calibration (2026-07-04)
- `validation/calibration.py`: pair occupations via Panel_Request-lock → Route-release events, fit per-TC
  params + platform dwell (Sept–Oct). Residual median 2s, zero bias.

### M4 — Discrete-event engine + interlocking + replay (2026-07-04)
- `sim/{events,engine,interlocking,router}.py`: heapq event queue + TC-mutex mask + `ReplayRouter`.
  1h replay locks all decisions with 0 interlocking violations via wait-and-retry deferral.

### M3 — Blocking-time occupation model (2026-07-04)
- `sim/occupation.py`: given (train, route, entry time) compute per-TC occupation intervals; per-TC params
  estimated from TD. Validated MAE 6.4s over real traversals.

---

## Phase 0 (M0–M2): scaffold + data alignment

### M2 — Phase-0 data alignment (2026-07-04)
- `derbysim/data/*`: region filter + TD/TRUST/CIF loaders (incl. full CIF STP layering) + 3-source
  train-occurrence alignment (25k occurrences, 88% tri-matched, arrival chain 90.9% ≤1min). Per-run
  splitting (headcode reuse), cif_reliable flag.

### M0 + M1 — Scaffold + infrastructure network (2026-07-04)
- `derbysim/` package skeleton + `config.py` (48-signal whitelist / 35 decision points) + `infra/`: build
  `InfrastructureNetwork` (signals / berth adjacency / routes / TC sequences / platforms) with self-check
  and unit tests.
