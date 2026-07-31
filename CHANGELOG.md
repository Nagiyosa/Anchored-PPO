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
