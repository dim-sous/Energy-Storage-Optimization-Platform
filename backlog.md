# Backlog — Active Issues & Future Work

> **Frozen versions (v1–v4)** live in [archive/](archive/). They are not part
> of active development and should not be modified.
>
> **Historical gate reports** for v1–v5 are in [archive/gate_reports.md](archive/gate_reports.md).

---

## Current state (2026-04-15, post trust-reset)

**HEAD:** see `git log --oneline | head -10`. Working tree is the only
authoritative source of truth in this project. Anything not currently in
the source code, the git history, or the empirical results listed below
should be re-derived rather than recalled.

### Trust reset (2026-04-15)

This session purged the backlog, the design docs, and the suspect
empirical results because they were generated under heavy cognitive load
and at least one audit subagent had a documented factual error
(claiming LP and EMS lacked capacity revenue in their objectives, which
was false on inspection).

**What was deleted:**
- All "Bug A-E", "Concern F-I", "F1-F6", "EMS-A/B/C" audit-derived
  findings that were logged in this file
- `docs/realism_fix_1_design.md` (RF1 implementation is now the truth)
- `docs/wow_factor_1_design.md` (speculative pre-investigation thinking)
- `D1-D5` future-work catalog (speculative)
- `Wow Factor 1` plan (speculative)
- The two session-recap memory files in
  `.claude/projects/-home-user-bess/memory/`
- `results/v5_big_experiment.json` and `results/v5_big_traces/` (moved
  to `results/_quarantine_pre_2026_04_15/` — generated on a configuration
  that no longer exists, do not use without re-running)

**What survived** (only things verifiable on the current HEAD):
- The git commit history
- The current source code
- The 5-strategy ladder (verified in
  [comparison/run_v5_comparison.py](comparison/run_v5_comparison.py))
- One reproduced empirical data point: the cleanup sanity test on day 0
  produced LP $53.99, ems_clamps $54.78, economic_mpc $54.02
- The user's restated proposition (below)

### Restated proposition (user-approved, 2026-04-15)

> EMS + MPC strategies are strictly ≥ ALL other strategies on
> every metric we care about — profit, delivery score, constraint
> satisfaction, SOH preservation, robustness under disturbance — on the
> honest RF1 baseline, at the current ledger calibration, without any
> new architectural machinery. Activation tracking lives in the plant.
> There is no strategy-layer PI.

The proposition has not been verified on the current HEAD with a
multi-day experiment. Verifying it (or empirically rejecting it) is
the next concrete deliverable.

### Strategy ladder

| # | Strategy | Role |
|---|---|---|
| 1 | `rule_based` | Naive baseline |
| 2 | `deterministic_lp` | Commercial baseline (LP, mean-substitution) |
| 3 | `ems_clamps` | Canonical "EMS alone" (stochastic two-stage program) |
| 4 | `tracking_mpc` | Sanity control (kept as broken baseline) |
| 5 | `economic_mpc` | EMS + MPC. Production v5. |

**Canonical pitch comparison:** `economic_mpc` − `ems_clamps`.

### Architectural facts (verifiable in source)

- Activation tracking is performed inside `BatteryPlant.step()` — see
  [core/physics/plant.py](core/physics/plant.py). The strategy layer
  outputs `[setpoint_pnet, p_reg_committed]` only.
- There is no strategy-layer PI controller. `core/pi/` does not exist.
- The `Plan` dataclass at [core/planners/plan.py](core/planners/plan.py)
  carries both probability-weighted averages and per-scenario
  trajectories from `EconomicEMS`. The MPC currently consumes only the
  averages (mean-substitution).
- The plant clips dispatch in two passes (base setpoint, then setpoint
  + activation), so `p_delivered` is correctly attributed to the FCR
  portion only — see [core/physics/plant.py](core/physics/plant.py).

### Pending work

1. **Re-establish empirical ground truth.** Run a fresh single-day
   sanity on the current HEAD (5 strategies) and a fresh 5-day big
   experiment (3 subsets × 5 strategies). The output of these runs
   becomes the only trusted empirical baseline going forward. Cost:
   ~75 minutes wall.
2. **Investigate the value leak** (after step 1 produces a baseline).
   Walk one specific day hour-by-hour, comparing `economic_mpc`
   against `ems_clamps`, and find where the MPC strategy loses money
   relative to the EMS-alone strategy. The investigation is a *reading*,
   not an experiment — output is a precise diagnosis with code
   citations, not a fix proposal.
3. **Decide on next direction** based on the diagnosis.

### How to resume after a session break

1. Read this section of [backlog.md](backlog.md) (the current state).
2. Run `git log --oneline | head -10` to confirm HEAD.
3. Read [comparison/run_v5_comparison.py](comparison/run_v5_comparison.py)
   `STRATEGY_FACTORIES` to confirm the 5-strategy ladder.
4. Look at `results/` for the latest empirical baseline; if it doesn't
   exist or is outdated, re-run it before trusting any conclusion that
   depends on it.
5. Do not consult any pre-2026-04-15 audit findings, design docs, or
   memory entries. They were purged because they were not trustworthy.

---

## Future work — upcoming versions (intent, not commitments)

These are *roadmap intent*, not derived from any audit. They describe
what each subsequent version *would* add, in rough order of value.

- **v6** — Unscented Kalman Filter (replace EKF)
- **v7** — Joint state and parameter estimation (online R_internal,
  capacity, efficiency)
- **v8** — ACADOS NMPC (replace CasADi/IPOPT, RTI, control blocking)
- **v9** — Degradation-aware MPC (SOH in MPC state, profit-vs-degradation
  tradeoff)
- **v10** — Disturbance forecast uncertainty (scenario-based MPC, chance
  constraints)
- **v11** — Measurement and communication delays
- **v12** — Multi-battery system with central EMS coordinator
- **v13** — Grid-connected inverter model (id, iq, Vdc dynamics)
- **v14** — Market bidding optimization (day-ahead, reserve, intraday)
