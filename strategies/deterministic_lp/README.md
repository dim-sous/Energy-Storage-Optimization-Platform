# deterministic_lp — Linear Program (commercial baseline)

**Pitch-visible:** yes
**Composition:** `DeterministicLP` planner (no MPC)
**Solver:** `scipy.optimize.linprog` with the HiGHS backend

## What it does (plain words)

Collapses the forecast scenarios to a single deterministic mean and
solves a 24-hour linear program for the hourly arbitrage and FCR
commitment plan, accounting for power-budget headroom, FCR endurance,
degradation cost, and a soft terminal SOC anchor. Mean substitution =
no scenario hedging. The honest commercial baseline: every commercial
BESS EMS today ships some form of this.

## Optimal Control Problem

### Notation

| Symbol            | Meaning                                              | Units                 |
|-------------------|------------------------------------------------------|-----------------------|
| `N`               | planning horizon                                     | hours (= 24)          |
| `Δt_h`            | step size in hours                                   | h (= 1)               |
| `Δt_s`            | step size in seconds                                 | s (= 3600)            |
| `k`               | time index, k = 0..N−1                               | —                     |
| `p_e[k]`          | forecast-mean energy price                           | \$/kWh                |
| `p_r[k]`          | forecast-mean reg-capacity price                     | \$/(kW·h)             |
| `E_nom`           | nominal pack capacity                                | kWh                   |
| `η_c`, `η_d`      | charge / discharge efficiency                        | —                     |
| `P_max`           | max power                                            | kW                    |
| `SOC_min`,`SOC_max` | SOC bounds                                         | —                     |
| `SOC_0`           | initial SOC (from EKF)                               | —                     |
| `SOC_term`        | terminal SOC target                                  | —                     |
| `α_deg`           | arbitrage-throughput degradation rate                | 1/(kW·s)              |
| `α_reg`           | reg-cycling degradation rate                         | 1/(kW·s)              |
| `c_deg`           | degradation cost                                     | \$/SOH lost           |
| `T_end`           | endurance horizon                                    | h (= 0.5)             |
| `W_term`          | terminal-anchor L1 weight                            | \$/SOC-unit (= 500)   |

### Decision variables

```
P_chg[k] ∈ [0, P_max]      for k = 0..N−1
P_dis[k] ∈ [0, P_max]      for k = 0..N−1
P_reg[k] ∈ [0, P_max]      for k = 0..N−1
z⁺,  z⁻  ≥ 0               (L1 slacks for the terminal anchor)
```

The state `SOC[k]` is **not** a free variable; it is reconstructed from
the inputs via the linear recursion below.

### State recursion (linear, embedded in constraints)

```
SOC[k] = SOC_0 + (Δt_h / E_nom) · Σ over j=0..k−1 of [ η_c · P_chg[j] − P_dis[j] / η_d ]
```

### Objective

The LP minimises the negation of expected profit:

```
minimise   Σ over k=0..N−1 of [
                p_e[k] · P_chg[k] · Δt_h                          ← charge cost
              − p_e[k] · P_dis[k] · Δt_h                          ← discharge revenue
              − p_r[k] · P_reg[k] · Δt_h                          ← capacity revenue
              + c_deg · Δt_s · ( α_deg · (P_chg[k] + P_dis[k])
                               + α_reg · P_reg[k] )               ← degradation cost
            ]
            + W_term · ( z⁺ + z⁻ )                                ← L1 terminal anchor
```

All terms are **linear** in the decision variables — that is what makes
this an LP and not an NLP. Note in particular that the SOC dynamics are
linear in `(P_chg, P_dis)` (no nonlinear OCV-coupled current solve), and
the terminal anchor uses an L1 rather than L2 penalty so it can be
encoded with two non-negative slack variables.

### Constraints

**Power budget** (committed reg power must leave headroom for the
planned chg/dis dispatch):

```
P_chg[k] + P_reg[k] ≤ P_max          for k = 0..N−1
P_dis[k] + P_reg[k] ≤ P_max          for k = 0..N−1
```

**SOC bounds with FCR endurance margin** (the most-recently-committed
reg power `P_reg[k−1]` must be sustainable for `T_end` hours in either
direction without leaving the SOC envelope):

```
SOC[k] + (T_end · η_c / E_nom)        · P_reg[k−1] ≤ SOC_max     for k = 1..N
SOC[k] − (T_end / (E_nom · η_d))      · P_reg[k−1] ≥ SOC_min     for k = 1..N
```

**Terminal SOC anchor** (encoded as a linear equality with non-negative
slacks; the objective penalty `W_term · (z⁺ + z⁻)` then implements the
L1 penalty `W_term · |SOC[N] − SOC_term|`):

```
SOC[N] − SOC_term  =  z⁺ − z⁻
```

### What this LP does NOT model

- **Thermal dynamics.** No temperature state and no thermal constraint.
  The plant runs hot whenever the LP plans aggressive dispatch; the
  ledger does not bill thermal violations.
- **OCV nonlinearity.** SOC dynamics are linear in `(P_chg, P_dis)`;
  no voltage-coupled current solve.
- **V_rc transient dynamics.** No charge-transfer or diffusion modes.
- **Multi-cell pack effects.** Plans against pack-level energy.
- **Stochastic forecast uncertainty.** Scenarios are collapsed to the
  mean before solving — the LP ignores variance.
- **Closed-loop state feedback within the hour.** The LP solves once
  per hour against the most recent EKF estimate; within the hour the
  held setpoint is dispatched open-loop.
