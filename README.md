# Simulation-to-Policy: Microscopic Railway Junction Dispatching with Reinforcement Learning

MSc project (University of Birmingham, School of Computer Science, with BCRRE). The system builds a
microscopic blocking-time simulator of the **Derby station throat** from real operational railway data,
then learns a dispatching policy on top of it — behaviour cloning of the human signaller as a warm
start, followed by **Anchored PPO**, a PPO variant regularised towards the timetable and its own best
checkpoint. The learned policy completes every controlled service and reduces class-weighted secondary
delay by **32% against the strongest rule-based baseline (FCFS)** on held-out validation days, and wins
18 of 21 days on a sealed test set it never touched during development.

## Pipeline

1. **Operational data** — TD berth-step logs, TRUST delay records and CIF timetables provided by
   Network Rail, covering 96 operating days at Derby (48 signals, 63 track circuits, 35 decision
   points). Split 64 train / 11 validation / 21 sealed-test days.
2. **Microscopic simulator** (`derbysim/sim/`) — event-driven, deterministic, blocking-time
   interlocking with proactive route reservation. Only passenger services (classes 1–2, ~190/day) are
   controlled; all other traffic is replayed from history as background constraints. Validation gate:
   replaying the recorded signaller decisions completes every day (completion 1.0).
3. **Imitation warm start** (`derbysim/learn/`) — a GNN policy over panel options (hold / set route /
   cancel), behaviour-cloned from the replayed signaller decisions.
4. **Anchored PPO** (`derbysim/learn/ppo.py`, `scripts/13_train_ppo.py`) — PPO with
   - a **routing anchor**: KL towards the timetabled route (β_r = 0.03),
   - a **timing anchor**: per-state KL over action kinds towards a frozen reference (β_t = 0.02 → 0.01),
   - a **moving trust region**: the anchor reference follows the best checkpoint, with automatic
     rollback and learning-rate rescue on degradation,
   - a **curriculum** from quiet to full-peak days, promoting only at completion 1.0 and delay ≤ FCFS.
   Training proceeds in three stages, each re-initialised from the previous stage's best checkpoint.
5. **Evaluation** — closed-loop and outcome-based: completion rate as a feasibility floor, then
   TOPS-class-weighted secondary delay (seconds; lower is better). No per-decision accuracy metrics.

## Headline results

| Policy (validation, 11 days) | Completion | Weighted delay (s) |
|---|---|---|
| **Anchored PPO (ours)** | **1.000** | **123,036** |
| FCFS (strongest rule) | 1.000 | 181,164 |
| Lookup / FSFS / Always-set | 1.000 | 192,874 / 197,155 / 222,146 |
| BC only (no RL) | 1.000 | 199,267 |
| Vanilla PPO | 1.000 | 216,365 |

Three seeds: mean −30.1% vs FCFS (worst seed −26.5%), completion 1.000 across seeds. On the sealed
test set (21 days, touched once with the deployed checkpoint) the policy matches or beats FCFS on all
aggregate metrics, wins 18 days with 1 tie, and its advantage grows monotonically with contention —
up to −2,932 s/train in the most congested bins. Full ablations (9 configurations) and a documented
failure case are in `figures/pub/` and the changelogs.

## Repository layout

```
derbysim/            core library
  sim/               simulator: engine, interlocking, blocking-time env, rewards
  learn/             features, encoders, BC, PPO/A2C, IQL, rollout, baselines
scripts/             numbered pipeline steps (01 data build → 42 figures), one per stage
notebooks/           server-side training/eval launchers (72-core parallel runs; d84_sens_rerun.ipynb
                     is the hyperparameter-sensitivity sweep, resumable via --resume)
artifacts/           processed datasets, calibration, checkpoints, result JSONs
figures/pub/         publication figures F1–F14 + system overview (SVG/PDF/PNG)
demo/                Rail Route-style playback web demo (Flask + SVG board)
tests/               unit tests
docs/                design documents (historical; the changelogs are authoritative)
```

`CHANGELOG.md` and `chinese_change_log.md` record every decision (D-numbered) and are the
authoritative project history.

## Getting started

```bash
pip install -e .                      # Python 3.10+, torch required for training only
python -m pytest tests/               # unit tests (no torch needed for most)
python scripts/13_train_ppo.py --help # training entry point (all flags documented)
python scripts/40_pub_figures.py      # regenerate all publication figures
cd demo && python app.py              # interactive playback demo (localhost)
```

Training was run on a 72-core CPU server; evaluation and figure generation run locally.

## Data statement

Raw TD/TRUST/CIF data were provided by Network Rail through BCRRE for research use and are not
redistributed here; `artifacts/` contains only derived, processed forms used by the pipeline.

## Acknowledgements

Supported by the Birmingham Centre for Railway Research and Education (BCRRE), with operational data
provided by Network Rail.
