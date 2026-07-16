# Changelog

Microscopic blocking-time simulator + signaller route-decision learning for the Derby station throat.
Milestone-based, newest first. Authoritative sources: `PROJECT_BRIEF.md` (decision log D1–D22),
`docs/IMPLEMENTATION_PLAN.md`, `docs/M9_DESIGN.md`.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [In progress] Phase 3 (M9): dynamic routing + full physical locking + robust deadlock handling

Counterfactual environment — the policy actually "drives the panel" (Path A: set route / cancel / wait).
From here it deliberately diverges from the log; validation switches to physical consistency.
Design in `docs/M9_DESIGN.md`. Decided: Path A + explicit cancel, free routing (platform is a consequence
of route choice), deterministic, geometric mask, overlap included, no artificial cap on how early a route
may be set, deadlock = terminal failure, obs includes current lock state.

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
