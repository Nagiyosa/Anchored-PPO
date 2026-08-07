# Changelog

Microscopic blocking-time simulator + signaller route-decision learning for the Derby station throat.
Milestone-based, newest first. Authoritative sources: `PROJECT_BRIEF.md` (decision log D1–D48),
`docs/IMPLEMENTATION_PLAN.md`, `docs/M9_DESIGN.md`–`docs/M12_DESIGN.md`.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [In progress] Phase 4 (M12): online RL — closed-loop dispatching

Design in `docs/M12_DESIGN.md`. Core = maskable PPO (KL-to-BC + no-prior BC warm start) on the M9
counterfactual environment with M10 rewards; discrete actions + event-driven SMDP (D24); the value
anchor is closed-loop quality vs the signaller, not the BC match rate (D23).

### D58 — Evaluation metric suite + reward redesigned to mirror Rail Route (per-train terminal); branch point resolved (2026-08-07, user)
- **Branch point resolved**: the TD-occupation validation (`scripts/28`, `occupation_vs_td_20260807-184618.json`)
  showed all three modes drift 40–78s median from TD (reactive most centered at bias 0 — but that is a
  tautology: reactive just replays the per-TC calibration, meaningless). The user's key correction: **we should
  NOT chase matching TD/the signaller** — the signaller is a *reference baseline, not a target* (an optimal
  policy reacting immediately is naturally faster than the imperfect signaller); we compare policies on
  delay-type dispatching metrics. So the "wide + req_time faithful anchor" premise is **dropped as unnecessary**.
  Anchor = **narrow** (relative trigger, no stalls, a clean approximation — approximate is fine for a
  reference); **wide = the RL freedom / ablation** (RR-style, lets RL reserve earlier to beat the signaller) —
  i.e. back to the original D57 split (narrow reference/BC, wide RL ablation).
- **Evaluation metric suite (⑧)** — floor = safety (completion, 0 collisions, 0 deadlocks). Headline =
  **weighted secondary delay (λ-priority, D16/D24) + completion + punctuality (PPM-style: % exiting within ±T
  of the CIF plan) + number of unscheduled stops**. Strong support = infrastructure occupation ratio (UIC-406
  flavour, the blocking-time-native selling point) + delay recovery (exit − entry signed) + delay tail
  (p90/p95/max, D42). Support = platform adherence / throughput / total held time. All stratified by λ class
  (passenger / ECS / freight / light-engine, D44).
- **Reward redesigned to mirror Rail Route's scoring** (RR: +10 per train arrived, −2/min late, −1 wrong
  platform, braking penalty, time bonus): switch from the per-step global Δ-delay (D24) to a **per-train
  terminal reward** = `base(completion) − w_d·(weighted secondary delay) − w_s·(unscheduled stops)`;
  gridlock / incomplete use the D46 large penalty + charge-to-horizon; the time bonus is skipped (entry times
  are fixed, already captured by delay). **Platform is NOT in the reward** (goal-mask forces it to 100% for
  terminating trains → meaningless; a D44 second-order effect for through trains; another knob / variance —
  eval only). The per-train terminal structure also **addresses D45** (the per-step global Δ was dominated by
  exogenous day-variance with muddy credit assignment → the critic couldn't fit it; per-train settlement gives
  clean credit and aligns reward with the eval metrics). Other metrics (PPM / occupation / recovery / tail /
  platform) are eval-only, not reward terms — as in RR (only the dominant, smooth terms are scored).
- **Implemented + verified**: `RewardConfig.reward_mode="per_train"` (default; "per_step" kept for ablation),
  `_reward_per_train` = `base_completion − w_delay·secondary_delay − w_stops·n_stops` per arrival, deadlock /
  incomplete via D46; new `DynTrain.n_stops` (in-region forced stops; entry-boundary wait excluded);
  platform/on-time removed from the reward. 2-val-day check: completion 1.0, 0 invariants; per_train and
  per_step give identical action SHA (replay ignores reward → the ⑥ gate is unaffected); stops 928 (reactive)
  → 72 (narrow) — proactive stops trains 12.9× less, now reflected directly in the reward.
- **Eval suite (`scripts/29`) + BC full-panel dataset gen (`scripts/30`) implemented + smoked**. Eval suite
  (λ-stratified: weighted secondary delay / completion / PPM / stops headline + tail / platform / throughput):
  val signaller comp 1.0, PPM≤300s 99.9%, stops 137; **greedy gridlocks at 0.34 completion** (no timing ⇒
  gridlock — a further quantitative confirmation of the timing thesis). **Occupation ratio (UIC-406) decided
  NOT to build** (throat sequencing ≠ line-capacity study; trains fixed ⇒ ratio ~constant across policies;
  the capacity limit is already captured by completion/gridlock; not worth the per-TC interval instrumentation).
  BC full-panel dataset (`scripts/30`): narrow-proactive signaller replay → per-epoch `(obs, options,
  chosen_idx)` samples cloning the full panel {WAIT/SET/CANCEL}, D34 split + fingerprint + per-class report;
  3-val-day smoke ≈ 90% SET / 10% WAIT / **0 CANCEL** (the replay doesn't reproduce logged cancels — a known
  limitation; BC won't learn cancel, RL explores it). Remaining ⑦/⑧ (mostly server): full 96-day dataset, train
  `PanelOptionPolicy` on `chosen_idx` (= full-panel BC), warm-start RL (per_train reward), redo baselines + Step 9.
- **BC full-panel training script (`scripts/31`) implemented**: clones `chosen_idx` (the full panel action)
  by NLL on `PanelOptionPolicy.forward` option log-probs (the set branch = the M11 PointerBC, so the whole
  thing warm-starts RL). The data path (reconstruct dec → `PanelAdapter.encode` → `collate_panels` → dims →
  label validity) is verified without torch (dims = dec117/conf8/cand21/cancel40/train19/global7; all
  `chosen_idx` land within the panel `opt_mask`); the torch forward/training runs on the server (torch can't be
  installed in the sandbox — expected, per the local-smoke / server-training discipline). Server commands are in
  HANDOFF (full dataset `gen all` + `combine`; BC `scripts/31 deepsets 30`; eval `scripts/29 val`).
- **Baseline ladder (`scripts/32`) rebuilt on the new proactive env — a decisive result**: every zero-learning
  baseline gridlocks (lookup / fcfs / fsfs **0.49**, greedy 0.35, random 0.32 completion; weighted delay
  explodes to 38–51M as ~half the trains strand and charge to horizon), while the **signaller's timing alone
  reaches 1.000** (delay 9,165s). Crucially **lookup collapses to 0.49** — in the old reactive/D41-contaminated
  env it was the "strongest safe baseline ≈ signaller +8.7%" (D42); this **confirms D47** that the old figure
  was a free-drop artifact. lookup=fcfs=fsfs coincide because none does strategic WAIT (no timing ⇒ same
  gridlock; routing/order can't save it). Timing is the lever from ~0.3 to 1.0 — BC/RL approaching the
  signaller = solving the problem. (`baseline_ladder_val_20260807-220658.json`.)
- **⑥ gate re-verified after all reward/eval/BC changes**: reactive (100,273.5s) and proactive-narrow
  (54,652.5s) both still gate_pass True, deterministic, 0 chain-mismatch / 0 invariants — the many changes
  since sealing (per-train reward, n_stops, req_times threading, wide-gated reserve_due, leak fix, entry-as-
  decision, window switch) did not disturb the sealed foundation.
- **Code audit since the new engine (2026-08-07, user-requested) + fixes**: (1) featurizer (`panel.py`/
  `features.py`) audited — it never touches `decision.logged`/`logged_time` (no answer-leak), obs carries no
  `req_time`/logged route, `rem_path` in M9 is locked-only (D54-compliant), and it encodes entry decisions
  (cur_tc=None) without error (200/200). (2) Fixed `scripts/30`: added the decision `signal` column (in-region
  SETs populated, entry/WAIT null — correct) needed for route_prior/lookup. (3) Fixed `scripts/31`: fit
  route_prior from the train split (was a constant 0.5 = dead feature; now 112 (sig,route) keys on a smoke).
  (4) **PPM corrected to the D58 spec** — it was implemented as secondary-delay-relative (making the signaller
  a trivial 1.0, i.e. signaller-as-standard, wrong); D58 already specified PPM = on-time vs the CIF **plan**.
  Now: delay = actual arrival/pass (`max(sim_occ)`) − `planned_ref` (already coalesced arr→dep→pass); PPM =
  on-time / **all scheduled trains** (stranded = not on-time). `scripts/29`+`32` consistent; signaller
  PPM≤300s 1.0 → **0.57** (limited by exogenous primary delays), lookup/fcfs/fsfs 0.245, greedy 0.19,
  random 0.075. (5) D45 voided (ran on the D41-contaminated engine; retest on the new engine+reward).

### ⚠️ KEY BRANCH POINT — wide+req_time faithful-anchor attempt (superseded by D58 above) (2026-08-07)
- **What was tried (D57.1)**: after the user's critique that the calibrated 141s trigger is a median/detour, we
  attempted a Rail-Route-faithful anchor: wide window (reserve from frontier-known) + a req_time-aware replay
  (`make_signaller_replay_policy` defers a logged SET until `now >= its logged req_time`) + a `reserve_due`
  event scheduled at each route's `req_time` (opens a panel + clears the de-dup signature). `req_times` is now
  threaded EnvTrain → DynTrain (`to_dyn_trains`); all of this is **gated to `reserve_window="wide"`**.
- **Result — not faithful**: proactive-wide replay completes but trains **stall at frontiers**: day-0
  `held_time` 23,674s vs narrow 698s (34×), delay 42,525 vs 1,674; over 4 days a train strands (completion
  0.9988, not all done). **Root cause**: ⑤ removed the req_time advancement pacing, so under free advancement
  the train races ahead on its current route, reaches the frontier early, then waits for the **absolute**
  req_time to lock the next route. Free advancement (⑤) and absolute-req_time locking fight each other.
  Narrow's **relative** trigger (N berths back from the frontier) is robust to pace — that is a feature, not a
  detour.
- **Open question (to be arbitrated by data)**: is wide's extra 23k held_time the **real** secondary delay
  (signallers' trains do wait at signals; narrow/reactive lock ahead and may **under-count** it) or **artificial**
  (free advancement racing ahead)? **The only arbiter is validating sim occupation times against raw TD.**
- **State at this branch point**: reactive (100,273.5s, sha 398b5dbd5844) and proactive-narrow (54,652.5s, sha
  8319330458ad) are **bit-identical and ⑥-sealed**; the wide+req_time code is flag-isolated and does not touch
  them. The anchor stays **narrow** until the TD-occupation validation decides. If a later problem traces here,
  this is the checkpoint to return to.

### Proactive Stage 1 — advance-trigger implemented behind a flag + signaller-replay gate passed (2026-08-06)
- `DynamicEngine(proactive=True)` restores M8's advance-triggering (reuses `decision_advance.json`): candidates
  are generated at the frontier window (M8 `_schedule_next_decision` ported: dwell start / N berths upstream)
  rather than only when a train runs out of locked track. Flag defaults to **False** (reactive unchanged);
  propagated through `DynamicDerbyEnv` / `RewardedDynamicEnv`. New DynTrain fields `next_trigger_idx/kind`;
  `_actionable` proactive branch = deadline (awaiting) ∪ trains past their advance trigger with no queued
  request. `_append_locked` reschedules the next trigger after each commit.
- **Gate (full 11 val days, signaller replay)**: reactive default **bit-identical** (completion 1.000,
  weighted delay **100,273.5s**, 6042 sets — matches D37/D39). Proactive: completion **1.000**, 2809/2809 done,
  **0 collisions / 0 deadlocks / 0 truncations / 0 D36 invariant violations / 0 ghost locks**, deterministic
  (double-pass identical action SHA), same **6042** routes committed.
- **Finding**: proactive replay cuts weighted secondary delay **100,273.5 → 54,374s (−46%)** — routes locked
  ahead ⇒ fewer forced stops. First quantitative sign that the timing lever reduces delay for the signaller
  anchor (Stage-1 figure, not final).
- **Stage 2 (de-dup, D50) done + verified**: a proactive SET candidate is re-offered only when its context
  signature `(trigger_idx, holders of its legal continuations' footprints)` changes (mirrors the `cancel_sig`
  discipline); stored in `_time_slice`. Full 11 val gate: completion **1.000**, 6042 sets, 0 col/dead/trunc,
  0 D36 violations, all done, deterministic; **steps 10,327 → 6,866** (vs reactive 6,159 — the ~700 extra are
  genuine advance decision points, not spam); weighted delay 54,652.5s (still −46% vs reactive). The action
  set `{WAIT}∪{SET}∪{CANCEL}` and string-stacking (consecutive SETs in a cut) were already present.
- **Stage 3a (leak audit + fix, D54/D22) done + verified**: `_build_obs` computed `delay_min =
  (entry_time − planned_ref)/60` for **all** non-done trains, but `entry_time` is the **actual** first-TC
  occupation — for a not-yet-entered (approaching) train that lies in the future, so exposing its realized
  arrival delay leaks "the realized future" (D54's guardrail). Fix: `_delay_min` returns None until the train
  has entered (approaching-train timing may come only from the CIF plan, never actual entry_time). Shared bug
  on both reactive/proactive paths. Verified: reactive + proactive replay **bit-identical** (action SHA
  398b5dbd / 35713231 unchanged — replay ignores features), and approaching-train realized-delay leaks 183 → 0.
- **Stage 3b (entry-as-decision (a) + reserve_window switch, D57) done + verified**: proactive `_on_arrive` no
  longer auto-commits the entry route; the train becomes an `awaiting_entry` candidate whose only SET option is
  the logged entry route (routing fixed, timing decided; WAIT holds it at the boundary and accrues delay per
  D56). Reactive keeps D14.8 (auto-entry) untouched. `reserve_window` param: **narrow** (calibrated trigger =
  141s, faithful human timing — replay/BC use this) vs **wide** (frontier-known ⇒ reserve immediately, = RR's
  free set-ahead — RL default, narrow as ablation). Full 11 val gate: reactive **bit-identical** (100,273.5s /
  6042 sets); proactive **narrow** completion **1.000**, all done, 0 col/dead/D36, deterministic, delay
  **54,652.5s** (unchanged — replay admits at arrival), sets 8851 (= 6042 + 2809 entry decisions). **Wide
  sanity confirms the by-role split**: wide replay completes 1.000 but delay **balloons to 316,694.8s (+480%)**
  because it reserves the whole chain up-front and holds all overlaps from entry — i.e. wide would corrupt the
  human anchor, exactly why replay/BC must stay narrow.
- **Proactive engine now functionally complete + replay-verified**: advance-trigger (①), string + de-dup (②),
  reserve-ahead + train-keyed locking (④ core), policy-reservation advancement gate (⑤), leak-safe delay (③
  fix), entry-as-decision (a), narrow/wide window (D57). Remaining are RL/BC-facing and belong with ⑦: richer ③
  featurizer wiring, ④ cancel_locked, then BC full-panel (⑦) + baselines/eval/Step9 redo (⑧).
- **⑥ sealed — full Step-7 gate on the proactive engine (committed script + timestamped artifact)**:
  `scripts/20` gains `--proactive` / `--reserve-window`; in proactive mode it deliberately does **not** compare
  against `HISTORICAL_REPORT` (that is old reactive/D48-voided data — comparing would point at wrong content),
  only self-certifying invariants. Proactive-narrow gate (`d33_step7_proactive_20260807-130744.json`):
  **gate_pass True**, full protocol (11 days × 2 passes), deterministic, completion 1.000 every day, 2809/2809
  done, **0 collisions / 0 deadlocks / 0 truncations / 0 D36 invariant violations / 0 reference-chain
  mismatches**, delay 54,652.5s. Reactive gate still passes unchanged (100,273.5s, chain-miss 0) — the script
  edit didn't disturb the reactive path. The proactive environment is now a sealed, reproducible foundation for
  ⑦/⑧.

### D56 — Block ⑤ WAIT/delay/gridlock settlement settled + overlap sensitivity measured (take depth-1 consistent + report sensitivity) (2026-08-06, user)
- **Delay/gridlock accounting reused unchanged**: `held_time` weighted by service-class λ, incomplete trains
  charged to the common horizon (D46 gridlock = game-over), deadlock −50 + terminal settlement (D6). The shape
  matches RR (on-time / gridlock = failure) and reality; weighted-secondary-delay + per-operating-day episodes
  are our declared evaluation conventions (D23/D24).
- **Advancement gate: log req_time → the policy's locked reservation**: a train advances into the next segment
  iff (a) the policy has locked that segment and (b) the next TC is physically free; late reservation = the
  train stops = delay accrues — the **mechanical seat of the timing lever**. This gate matches RR/reality
  (wait at red until the route is set + clear); the old log-req_time gate was a replay artifact. Signaller
  replay reproduces the logged timing → still completion 1.0 (the ⑥ anchor).
- **WAIT is free** (no hold_start; the train keeps running); **entry-wait accrues delay** (with entry now a
  decision, a late admission makes the train wait at the boundary and accrue planned-vs-actual delay;
  generalises the `env_dynamic:209` fix).
- **Overlap settled as (a)**: **depth-1 (~180m colour-light standard) as the default, always consistent
  between the signaller anchor and the RL policy, with the 0/1/2 sensitivity reported as a first-class
  robustness result.** Measured now (`scripts/27_overlap_sensitivity_replay.py`, val 11 days, signaller replay,
  artifact `reports/overlap_sensitivity_replay_val_20260807-005042.json`): **completion is 1.000 / 0 collisions
  / 0 deadlocks at depth 0/1/2 (primary metric D23 robust); but weighted delay is first-order sensitive —
  depth-0 −14.3%, depth-2 +548.8%.** So overlap moves delay but not completion; hence depth-1 (grounded),
  consistent comparison (the D23/D24 relative anchor stays fair), sensitivity reported. **Correction**: the
  earlier "overlap is probably second-order" guess was overturned by the data (the user's push to measure now
  was right). **Swinging overlap (c) is declared a limitation / future work** (a fixed depth-1 over-charges
  absolute delay vs real swinging overlaps, but the relative comparison is unaffected; (c) adds engine dynamics
  + lesson-#1 risk, not built now).

### D55 — Block ④ reservation/overlap semantics settled: reuse train-keyed reservation (RR queue) + per-train reservation string + cancel_locked guard + keep deadlock mask (stricter than RR) (2026-08-06, user)
- **Reservation layer (`reservation.py`) reused as-is**: train-keyed footprint locking (route TCs + overlap),
  FIFO queueing on conflict, `_pump_queue` auto-locks when track clears = RR's queue-until-clear, native.
- **Engine-layer core change**: `env_dynamic.py`'s single `pending_request` per train → a **per-train string
  of reservations (locked run + queued tail)**; `_wake`/`_pump_queue` promote queued→locked as track clears.
- **cancel_locked (the ② deferred semantics)**: release that route's footprint (`release_route` exists) +
  roll back the `actual_chain` tail + restore matchability; **guard = the train has not yet entered that
  segment (commit_idx not yet entered)**; once occupied it can't be cancelled (interlocking invariant).
  Consistent with RR (unlock any not-yet-reached segment).
- **overlap**: depth-1 declared approximation; **spatially = the first TC of the next hop, but semantically
  ≠ the route body** (overlap = an uncommitted safety margin; route = the committed travel path). Same-train
  consecutive locks don't self-conflict; `release_overlap` releases sectionally past the exit signal; the C4
  sensitivity scan stays in M13.
- **Deadlock mask kept (option i, stricter than RR)**: `would_create_deadlock` masks only a one-step provable
  never-clearing cycle; rationale = the D23/D41 safety-by-construction pillar + lower RL trial-and-error cost.
  **Dynamic gridlock still exists and is game-over (D46)** and is unaffected by the one-step mask. RR itself
  does NOT prevent deadlock; being stricter here is deliberate.
- **Cancel-rate measurement**: a rigorous reproducible classifier (region-edge + run-split + cancel-vs-
  consume-vs-reroute) is built alongside the implementation to measure the locked-not-occupied vs queued split.

### D54 — Block ③ state settled: full CIF timetable (Rail-Route-style) + conflicts from physical/own-reservations/schedule + proactive features (2026-08-06, user)
- **Conflict/approach derivation**: from **current physical occupancy + the policy's own committed/locked
  reservations + the CIF timetable** — **never from other trains' logged-future paths** (M8 `_build_state`
  `env.py:213` uses `tr.path[tr.pos:]`, fine for offline replay but must change online: those are other
  trains' answers and are unavailable online). Occupancy exposes **physical** position only (rebuilt from
  cur_tc, D22 §11), never reservation intent.
- **Schedule exposure = the full CIF plan (like Rail Route shows the whole timetable)**: the plan is known
  ahead of the day and is not a consequence of any decision ⇒ **leak-free and RR-consistent**. **Guardrail**:
  expose the **plan (CIF) + realized state up to `now`** only; **never the realized future** (actual future
  occupancy / actual arrival / actual route taken = outcomes = leakage). Approaching trains carry planned
  arrival + class + goal + signed delay-so-far; actual arrival/delay becomes visible only once it happens
  (≤ now). How far the encoder actually attends is an M12 modelling choice, not a semantics/fidelity one — no
  hard window.
- **New proactive features**: time-to-deadline (how long until the train runs out of locked track),
  reserved-frontier depth (how many routes already stacked), approaching-train ETAs.
- **Entry (D52)**: the entry train's before-state = CIF planned arrival + class + goal + current physical
  freeness of candidate entry platforms + approaching conflicts; the logged entry route is never exposed.
- **Anti-leakage audit (D22 ⑥)**: re-run the blind-policy / ablation audit on the new state; every new
  feature must pass "can't reconstruct the answer from it."

### D53 — Block ② action space settled: single-route SET (string via consecutive SETs) / WAIT / CANCEL (2026-08-06, user)
- **SET**: one route at a time, appended from the **reservation frontier**; "pre-stack a string" = **consecutive
  SETs within one time-cut** (reuse the D39 cut loop), depth 0..5 decided by how many SETs the policy issues
  before a WAIT — **no compound "k-at-once" action, no fixed depth** (D51 anchor: median 0 / p95 2).
- **WAIT**: the global epoch-terminator (per D39, ends the cut and advances to > now); the train keeps running
  on locked track; "don't reserve this one yet" = just don't SET it and WAIT — it stays a candidate and
  resurfaces per the de-dup rule.
- **CANCEL**: scope = any reserved route the train has **not yet occupied** (queued-not-locked **or**
  locked-but-not-yet-passed); **once occupied it cannot be cancelled = an interlocking invariant, enforced by
  construction** (cancel is only offered for not-yet-occupied reservations); mirrors Rail Route (CTRL-unlock,
  hard once the train passes). M9 `_cancellable` (`env_dynamic.py:228`) today cancels only queued-not-locked;
  the **unlock + footprint-release + chain-tail rollback semantics for cancelling a locked route are deferred
  to ④**. A rigorous reproducible cancel-rate/state measurement (region-edge + run-split + cancel-vs-consume-
  vs-reroute) is deferred to ④; the existing 10.44% has no committed source and quick proxies (67–74%) are
  region-edge artifacts, so not cited.
- **Action set** = `{WAIT} ∪ {SET(train, route)} ∪ {CANCEL(train)}`, reusing the `_build_options` (`:259`) shape
  with candidates now drawn from the ① frontier window. **Mask** = geometry ∩ reachability ∩ in-scope ∩
  not-true-deadlock; **occupancy does not mask (request layer)**, a true deadlock does. Multi-train joint panel
  inherited from D39.

### D50–D52 — Proactive refactor pinned block by block: candidate-①, depth, entry, order (2026-08-06, user)
- **D50 (candidate generation ① settled).** A train is a decision candidate at its **reservation frontier**
  (end of its locked route chain), not its physical position. The reservable **window** = [earliest: the
  previous route is committed and the next decision signal S is known → latest: the train runs out of locked
  track = the old reactive point, now demoted to a deadline / the "lead = 0" degenerate case]. M8's calibrated
  trigger (`env.py:145 _schedule_next_decision`, 141s / N berths upstream) becomes the **signaller reference /
  BC target**, no longer a hard trigger. **De-dup**: an unchanged candidate is not re-asked every event (reuse
  the `env_dynamic.py:228 _cancellable` blocker_sig/cancel_sig discipline), else we reproduce "asked 614×
  in 3h" and 6× longer episodes that break GAE. **Accurate framing**: restore M8's advance trigger + request-layer
  semantics (`env.py:176`) into `DynamicEngine` (which today only sets awaiting at `:484`), then extend from
  "fixed calibrated lead, one route" to Rail-Route-style free choice (when + how many). Don't reinvent.
- **D51 (concurrent reservation depth measured reproducibly — replaces the void D49 figure).**
  `scripts/26_measure_reservation_depth.py` (reusing `build_env_trains_for_window` + `split_broken_chains`,
  same D31/D35 train set), full 98 days / 25,812 trains / **79,562 Panel_Requests**; artifact
  `artifacts/reports/reservation_depth_full_20260806-222715.json`. Definition: at each request for route k,
  the count of the train's routes with `req_time ≤ t_k and first-TC occ > t_k` (booked-not-yet-occupied).
  **Result (excl-self): median 0, p75 1, p90 2, p95 2, p99 3, max 5**; ≥1 40.5%, **≥2 16.2%, ≥3 only 1.1%**;
  strict variant near-identical (edge truncation negligible). **By class: freight (6) deepest (median 1, ≥2 32%),
  ECS (2) 22%, passenger (1) 16%, light-engine (0) 11%** — same populations as the route-deviation cluster (D44).
  **Conclusion: signallers do pre-stack a string, but shallow.** The void "median 2 / max 74" (D48/D49) is
  disproved (max 74 is structurally impossible: per-train chain p99=4/max=6, D34). **Action design: allow a
  string, depth 0..5 chosen by the policy, no fixed window/depth.**
- **D52 (entry becomes a decision, reverting D14.8; reserve-onto-occupied via a Rail Route queue).**
  Decision 1: D14.8 (§14.8) excluded the entry route as "decided out-of-region." Data disproves this — over
  10 days / 2,316 entry segments the first route's start signal is **100% in-scope (48-whitelist), 99% a
  multi-option decision point (of 35), 99% with >1 geometric exit** ⇒ it is our panel's real Panel_Request
  (which platform to admit to). So entry becomes a decision (like Rail Route accept-to-platform); **landing
  depends on ③** (the train isn't in-region yet, so ③ must expose approaching trains / schedule to give entry
  a leak-free "before" state). Decision 2: pre-booking onto still-occupied-but-will-clear track is a real
  action (D22: 2.2% of requests; `env.py:176` request-layer mask is already occupancy-agnostic) — adopt the
  Rail Route **queue** semantics: REQUEST clears on geometry+reachability, LOCK is deferred until the section
  clears (auto-locks), and the deadlock look-ahead blocks only a **true deadlock (a never-clearing cycle)**,
  not "occupied now, will clear." Detailed reservation/overlap semantics deferred to ④.
- **Order**: discuss/implement ② (action space) → ③ (state, incl. entry) → ④ (reservation semantics). Hard
  dependency ②→③ (D22's leak-free "strip this decision's consequences" needs the action shape first);
  soft dependency ③→② (state makes "when to reserve" learnable but doesn't change the action shape).
- **Process discipline (lesson #15 correction, applied)**: the depth number is now a committed reproducible
  script + timestamped artifact, not a sandbox inline computation, and it reused existing pipeline machinery.

### D48 — The environment's decision model is fundamentally wrong: reality is proactive, the env is reactive (2026-08-06, verified against data)
- Data (TD, 92,534 route settings): the signaller issues each Panel_Request a median of 147s before the
  train occupies the route's first TC (p10 67 / p90 254); 99.9% are set >2s ahead, 91.7% >60s ahead, only
  0.1% at-arrival. Setting routes ~2.5 min in advance is the norm.
- But the current environment only lets you set a route for a train that is ALREADY awaiting (out of
  locked track, stopped at its signal) — there is no advance-reservation action. So the environment never
  exposes the core timing lever (when to reserve each train's next route). This explains almost everything:
  heuristics gridlock (they can only react at arrival, can't sequence ahead), and the RL can't learn timing
  (it's a missing mechanism, not a warm-start or reward problem). WAIT's true meaning: don't reserve yet;
  the train keeps running on its locked track and only stops if the next route comes too late.
- Analogy (user): this is essentially Rail Route — you see the schedule and freely decide when to set which
  route. The environment should be a mini Rail Route; the RL is an AI script maximising the score.
- Redesign direction: the decision is "when to reserve each in-region / approaching train's next route"
  (the lead time is chosen by the policy from train type / signal state / conflicts — NOT a fixed window);
  the state exposes approaching trains; signaller replay must still reach 1.0 under the new model. This is
  the deepest rework, touching env/data/BC/RL/eval end to end; discuss the design piece by piece first.

### D46 — Revert D41's strand-termination: a stranded train is a real gridlock (2026-08-06, user)
- Engine-level root error: D41 terminated + released the locks of a train that reaches an in-scope
  boundary with no continuation. But a real stranded train sits on the track holding its locks and
  blocks everything behind it (gridlock). D41 made "dump a hard train into a dead-end" consequence-free
  and also let it escape the delay charge (`realized_weighted_delay` only charges not-done trains to the
  horizon; a terminated stranded train is done and escapes). The PPO bring-up exploited exactly this:
  complete the easy trains, strand ~10% of the hard ones cheaply.
- Fix: remove the left_region termination. A stranded train stays awaiting, keeps holding its locks, and
  accrues delay to the horizon — bad routing self-penalises. Signaller replay is bit-identical (it never
  strands, completion 1.0); greedy completion drops realistically to 0.55–0.70 with delay in the millions
  (locks block others), 0 collisions/deadlocks/invariants, no hang; 40+ regression tests pass.
  "Don't drop trains" is now enforced by the environment's physics.
- Residual of the same class: route_cap (needs a policy to actively loop a train 12 times to trigger —
  not a cheap dump); flagged, watch.

### D45 — PPO bring-up fails to learn: credit-assignment / reward-variance problem (2026-08-06)
- β0.1 warm-started from the fixed no-prior BC, 100 iters (early-stopped at 24): held-out eval is stuck
  at the BC start (0.90 / 232k) from iter 1, no convergence, ~2× worse than lookup (109k). Root cause:
  EV is negative/wildly swinging throughout — the critic cannot fit the returns, so advantages are noise
  and PPO has no useful gradient (lesson 11 realized). Why: delay is a whole-day global quantity and, with
  γ=1, the return is dominated by exogenous which-day variance (train delay swings 16× across iters), which
  the state cannot distinguish. No reward hacking / no cancel drift. More steps won't help (not undertrained).
- "RL can't beat lookup" is rejected as a conclusion: lookup is imperfect (most-frequent route only, never
  waits strategically) and beatable; the earlier "timing headroom ≈ 0" inference is void (ARS-proxy only
  equalled the signaller because it peeks at the logged route).
- Fix direction: (1) free probe `--half-life 900` (localise returns → lower variance → does EV recover?);
  (2) if confirmed, an objective-preserving per-day difference reward (reward = how much better than a
  reference policy on the same day) to cancel the exogenous day variance while keeping γ=1's objective.

### D42–D44 — Step 9 done, routing-prior closed-loop adjudicated, Step 10 fixed as B (2026-08-05)
- Step 9 full scan (train+val): (1) single-point route deviation is heavy-tailed — median Δdelay 0 but
  26% catastrophic cascades, and catastrophe is almost entirely from being forced onto atypical routes
  (common-route 3% vs rare-route 57%); (2) a single forced WAIT is smooth (97.75% small delay rise, 0.2%
  completion drop). So routing is high-stakes/heavy-tailed, timing is a smooth delay dial.
- M11 re-run correctly on D35 per §18 (`--no-route-weight --epochs 30`): no-prior BC macro 0.68–0.76 >
  lookup 0.65 (the earlier `--epochs 15` without `--no-route-weight`, macro 0.53, is void).
- Routing-prior closed-loop on the corrected engine (held-out val): signaller 1.0000/100,274; lookup
  0.9972/+8.7% (strongest safe baseline); greedy +39%; BC 0.90/+101–131%. Decision A adjudicated: match
  rate is disconnected from closed-loop — BC beats the lookup on match yet loses badly closed-loop
  ("the more it imitates, the worse"), because its deviations land on rare-route landmines (Step 9 ①).
- Methodology fix (corrects D40): fitted models (lookup/BC/RL) must be evaluated held-out, not on
  train+val; lookup's honest baseline is the held-out val +8.7% (trainval +2.9% is in-sample).
- Step 10 fixed as option B: keep the full action space (routing freedom), warm-start + KL-anchor to the
  fixed no-prior BC (bc_deepsets_noprior_nw_d35.pt), sweep β from tight to loose (B sits tight), RL mainly
  learns timing plus when a BC route deviation pays off; must beat the held-out lookup; safe passage is
  guaranteed structurally (mask + reachability + in-scope). Data basis: 46.7% of decisions are multi-choice,
  where the signaller deviates from the most-frequent route 9.21% of the time — concentrated in ECS 27% /
  freight 20% / light-engine 45% (passenger only 1–5%), 67% to the 2nd-most-frequent — context-dependent
  deviations that lookup cannot express but BC can.

### D41 — decision space confined to in-scope routes; terminate at region boundary (2026-08-05)
- Step 9 ③ found greedy had 54.8% awaiting rows with "no option". The cause was not funnel
  over-filtering (goal/deadlock filters = 0) but the engine offering **out-of-scope routes** at the
  region edge, stranding policies at boundary dead-ends (all no-geometry was one out-of-scope TC, TDBN).
- Clarified: `in_scope` is a per-route start-signal property; long routes carry out-of-scope track, so a
  route-level cut isn't clean. Reachability (`_reach`) already expands in-region only and treats the edge
  as an exit, and real trains use 0/2809 out-of-scope routes — so the decision problem is closed within
  in-scope routes + boundary termination.
- Fix (reachability untouched): (1) `_legal` offers only in-scope routes; (2) a train that reaches its
  path end with no in-scope continuation terminates — success if at its goal exit, else `left_region`
  (incomplete, locks released) instead of stranding and holding locks. Audit geometry helpers synced.
- Verified on full data: signaller **bit-identical** (completion 1.000, n_decisions 371/557/514, delay
  100,273.5s, n_done=n_trains, all safety/invariant/mismatch 0, deterministic); greedy
  `awaiting_no_geometry` 4987→0, no-set 33,920→0, 0 collisions/deadlocks/invariants; Step 7/8 gates pass;
  regression green.
- Reframing finding: on the corrected in-scope engine completion is near-saturated — trivial greedy hits
  0.9916 — because `route_keeps_goal` guarantees completability. The differentiator is delay: greedy
  139,796s vs signaller 100,274s (+39%), reward −705 vs +101. Completion is a safety floor; the learning
  signal is reducing secondary delay via timing (D23/D24). Step 9 redirected: ③ closed, ① measures delay
  cascade, ② (timing/WAIT) becomes central.

### D33 step 8 — timing-contract gate passed; events batched per common time-cut (2026-08-04, D39)
- Reworked the event loop so each timestamp forms exactly one panel: `DynamicEngine.run` +
  new `_drain_slice`/`_time_slice` (replacing `_panel_phase`) drain every event at `t` (including
  same-time events a SET schedules) before opening one joint panel; the agent may SET consecutively
  within the cut, and a WAIT ends the cut so the loop advances to strictly `> t` (D24). Physical
  lock/release/occupancy logic is unchanged.
- Step 7 re-run (`d33_step7_20260804-215608.json`): gate_pass, deterministic, completion 1.000,
  0 collision/deadlock/truncation/invariant/mismatch. Weighted secondary delay is **100,273.5s,
  bit-identical to pre-fix**; the only change is `wait` 167→117 and `n_steps` 6209→6159 (`set`
  unchanged) — 50 zero-duration WAITs removed, no physical timing moved, nothing to attribute.
- Step 8 re-run (`d33_step8_20260804-215748.json`): all four contract points pass. Signaller
  `next_event_same_time` 307→0, `zero_dt_wait` 28→0, `wait_while_same_time_pending` 126→0,
  `max_set_trains` 1→2; WAIT always Δt>0 (min 4s / median 38s); `awaiting_no_geometry` 0.
- Gate tests `tests/test_timing_contract.py` go red→green across all val days; regression subset
  (`test_env_dynamic/test_reward/test_redteam/test_smdp_rollout/test_panel`) is 73 passed.
- Honest boundaries (see M12_DESIGN §18.13): greedy `awaiting_no_geometry=4987` is greedy's own bad
  routing (completion 0.62), not a contract bug; the signaller's clean 1.9% WAIT is still a replay-policy
  residual, and whether it can serve as `kind_anchor` is a Step 9 question. Environment fix complete;
  PPO / Step 9–10 unblocked. M11-on-D35 rerun remains a separate GPU task.

### D33 step 8 — timing-contract gate failed (2026-08-04)
- The 11-day audit preserves all safety/state guarantees, but finds 28 zero-duration WAITs and 307
  panels opened while another event at the same timestamp is still pending. WAIT therefore does not yet
  satisfy D24's time-extended macro-action contract.
- Greedy replay has 4,179 multi-awaiting panels but never exposes set options for more than one train,
  with 5,028 awaiting rows lacking a geometry option. Same-time events are being serialised into panels
  before the timestamp is fully drained, so the current interface does not represent the intended
  "when/which train" decision.
- Step 8 is recorded as failed. Both the historical 10.2% and current 2.69% WAIT shares are invalid as
  human timing anchors; PPO remains paused pending an agreed same-time event batching contract and a
  repeat of the Step 7/8 gates.

### D33 step 7 — full validation replay and determinism gate (2026-08-04)
- Two fresh passes over all 11 validation days are exactly deterministic by per-day action, physical
  path/timing and core-metric fingerprints. All 2,809 trains terminate; all 2,487 completable trains
  complete, with zero collision, deadlock, truncation, invariant violation or reference-chain mismatch.
- Current signaller delay is 100,273.5 seconds, reward 101.265 and 6,209 steps. Against the preserved
  pre-D33 report, completion is unchanged while delay is +5,424.5 seconds, reward −79.38 and steps −402;
  lifecycle-correct delay is allowed to change and is not forced to reproduce the defective timing.
- Added the Step 8 read-only audit for the geometry→goal→deadlock-safe→set funnel, simultaneous panel
  exposure and WAIT duration contract. The new replay WAIT share is 2.69%, but it is not treated as a
  human timing target because position replay does not reconstruct original request timestamps.

### D33 step 6 — counterfactual terminal-state invariants (2026-08-04)
- D35 is now materialised as 53,750 decisions (train 33,280 / holdout 20,470), including 25,077
  multi-choice decisions. Its combined SHA-256 is
  `0b86c129a06bf9bee3762559edc001ae0251d7063178c3816eb22b550f9f23e9`.
- Every episode now audits that done trains own no lock/request state, every reservation belongs to a
  known unfinished train with an exact reverse index, and `actual_chain` reconstructs `path`, route IDs
  and commit indices exactly. Violations are reported, never silently repaired, and enter episode and
  fixed-evaluation metrics.
- Injected residual-lock, orphan-owner and divergent-path states are detected; all reference and red-team
  policies report zero violations. The directly affected suites pass: dynamic environment 22, red team 7,
  rollout 25 (**54 passed**). Training remains disabled pending the full 11-day step-7 baseline.
- Added a timestamped Step 7 runner for two fresh passes over all 11 validation days. It fingerprints
  actions, actual paths/timings and metrics per day, enforces the complete safety/state/replay gate, and
  reports attributable deltas against the preserved pre-D33 baseline without requiring delay equality.

### D33 step 5 — position-ordered replay and D35 dataset contract (2026-08-04)
- A full-data audit proved that `Panel_Request` order is not always physical traversal order: signallers
  can pre-request a downstream route before its feeder. Position reordering is accepted only when route
  IDs are unique, every route has an exclusive observed Track timestamp, and discontinuities strictly
  decrease; ambiguous shunts remain split at the modelling boundary.
- Across 98 BC days, 773 of 22,023 trains are reliably reordered and 1,614 decisions are recovered;
  across 96 complete RL days, the corresponding figures are 751 of 21,963 and 1,562. Online replay now
  matches an unconsumed historical route among the current physical options and consumes it only after
  the route is fully locked; queued/cancelled requests do not advance the reference.
- The D31 52,294-row dataset, fingerprint, checkpoints and match rates are superseded. D35 has been built
  in `full_d35`; BC training accepts an explicit dataset, versions D35 outputs, and archives every
  checkpoint. Reordering merges false fragments to 16,631 train vehicles while leaving p99 route length
  at 4, so the cap of 12 remains supported.
- Dataset generation now defaults explicitly to `full_d35` and announces its per-day and combined targets.
  This closes a discovered CLI footgun where omitting `BC_FULL_DIR` silently reused the legacy `full/`
  shards and reproduced the obsolete 48,021-row dataset.
- Relevant data, simulation, rollout, reward, reservation, compatibility and red-team regressions pass:
  **118 passed**. A three-day signaller smoke remains 1.000 completion with zero collision/deadlock; this
  is not the formal 11-day baseline. Training stays disabled until steps 6–7 pass.

### D33 step 4 — goal termination, loop guard and queued cancel (2026-08-04)
- Completion now follows actual state: through trains must physically reach their exit group, while
  terminating trains must complete dwell at the specific platform inferred as their observed target.
  Platform tiers and booked-service metrics now use actual served platforms, not planned paths.
- The train-only total actual-route cap of 12 is a failure guard only and goal success takes precedence.
  Cancel removes only a request that is still queued and unlocked; queue promotion cannot be rewound.
- Red-team looping exposed a second-order lock bug: a partial holder from an earlier occurrence of the
  same route could masquerade as a complete new lock. Lock confirmation now requires full-footprint
  ownership, and overlap release retains TCs still needed by the future physical path. All seven
  exploit-policy regressions are collision-free; 101 relevant regressions pass.
- `1M00` proves logged request order can be downstream-before-upstream and therefore differ from
  traversal order. This is retained as a hard input to step 5's position-matched signaller replay;
  the 11-day baseline has not yet been rerun and training remains disabled.

### D33 step 3 — actual-route reservation lifecycle (2026-08-04)
- Same-train route extension now transfers each shared boundary TC from the old route holder to the
  new holder atomically. `reserved` and `holder_fp` remain exact reverse indices, empty holders are
  removed, and updates touch only affected TCs rather than scanning the full reservation table.
- Added path-aligned actual-route instance indices. Entering the next locked actual route releases the
  previous route's remaining overlap; route bodies remain sectionally released behind the train and
  final reservations remain until region exit. This is a GB-mainline-aligned approximation, not an
  exact Derby SSI reconstruction.
- New ownership and lifecycle guards plus dynamic-engine, deadlock, reward, rollout, panel and red-team
  regressions pass (88 relevant tests). Terminal/goal/cap/cancel semantics remain step 4; training stays off.

### D33 step 2 — separate reference, request, committed and physical state (2026-08-04)
- Removed `DynTrain.route_ptr` instead of renaming it: one cursor can no longer stand for both logged
  and counterfactual progress. The logged `route_chain` is converted to an immutable tuple;
  `pending_request` exists only while a request is queued; `actual_chain` is appended only after the
  route is physically locked; and `path`, parallel `path_route_ids`, plus `pos` represent movement.
- Next-signal derivation, collision attribution, goal-exit checks and terminal release now read the
  actual committed chain. The external observation key stays `pending_route` for M12a compatibility,
  but its sole source is the independent queued-request state.
- Added guards proving that an alternate lock cannot modify the reference chain, a queued route is not
  committed before lock, cancel cannot rewind actual progress, and the physical path follows the
  chosen route. Two non-overlapping regression batches cover the dynamic environment, rewards,
  rollouts, red-team policies, datasets and M7/M8 compatibility: 110 passed.
- Stop/continue is deliberately still a temporary historical-length rule, logged replay remains
  depth-indexed and release timing is unchanged; these belong to D33 steps 3–5, so training stays off.

### D33 steps 0–1 — pre-fix audit and complete operating days (2026-08-04)
- The original D31 train-only audit covered 64 complete operating days and 17,583 fragments. D35 later
  merged false request-order fragments to 16,631 trains. Logged chains including
  entry have p99=4 and max=6, giving a total-actual-route safety cap of 12; the cap prevents infinite
  cycling and never defines success. The 52,294-row BC fingerprint is retained only as superseded history.
- D34 excludes boundary-partial 2023-09-01 and 2023-12-07 from RL episodes while preserving raw TD and
  all 98 BC days. RL now uses 96 complete days (train64/val11/test21); 2023-09-30 remains and evaluation
  stays micro-averaged. Repeated route-body TCs were rejected as loop evidence after audit validation.

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
  do have a goal: complete dwell at their observed target platform. Excluding them meant an agent
  could leave a terminating train standing in the throat at no cost while it blocked a platform.
  D33 step 4 tightened `has_reached_goal`: through trains require actual exit, while terminating
  trains require `reached_terminal` after target-platform dwell; visiting an arbitrary platform or a
  planned path is insufficient. The old three-day denominator change (477 to 513, excluded 45 to 9)
  and signaller score 1.000 are retained as pre-D33 evidence, not the final step-7 baseline.
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
