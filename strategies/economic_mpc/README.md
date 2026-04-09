# economic_mpc — Economic NLP MPC (per minute)

**Pitch-visible:** yes (the production strategy)
**Composition:** `EconomicEMS` planner (hourly, see `strategies/ems_clamps`) + `EconomicMPC` (per-minute closed-loop)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Re-solves a 60-minute deterministic NLP every minute against the live
EKF state estimate. Plans charge/discharge under a soft anchor on the
EMS strategic SOC plan plus an economic term on energy arbitrage. The
committed FCR power is **exogenous** — set hourly by the EMS, treated
as a parameter (not a decision variable) inside this layer. Activation
tracking lives in the plant; the MPC does not see the FCR signal
directly.

## Optimal Control Problem

This is a **deterministic NLP solved at 60 s cadence** with a 1-hour
prediction horizon and control-horizon blocking after `Nc = 20` steps.

### Notation

| Symbol             | Meaning                                                | Units            |
|--------------------|--------------------------------------------------------|------------------|
| `N`                | prediction horizon                                     | (= 60 MPC steps = 60 min) |
| `Nc`               | control horizon (decisions are blocked beyond Nc)      | (= 20 MPC steps = 20 min) |
| `Δt_h`             | step size in hours                                     | h (= 1/60)       |
| `Δt_s`             | step size in seconds                                   | s (= 60)         |
| `k`                | prediction step index, k = 0..N−1                      | —                |
| `j(k)`             | control-blocking index, `j(k) = min(k, Nc − 1)`        | —                |
| `soc_0`,`temp_0`   | initial state from EKF (set per solve)                 | —                |
| `soh̄`              | EKF SOH estimate, **frozen** as a parameter            | —                |
| `p̂_e[k]`           | forecast-mean energy price for step k                  | \$/kWh           |
| `P̄_reg[k]`         | committed FCR power, ZOH-expanded from the EMS plan    | kW               |
| `soc_ref[k]`       | EMS SOC reference (current implementation: end-of-hour value, repeated) | — |
| `u_prev`           | last applied controls `(P_chg_prev, P_dis_prev)`       | kW               |

| Symbol             | Meaning                                                | Default          |
|--------------------|--------------------------------------------------------|------------------|
| `w_e`              | energy-arbitrage weight                                | 1                |
| `w_deg`            | degradation-cost weight                                | 1                |
| `Q_soc_anchor`     | per-step SOC anchor                                    | 10               |
| `Q_term_econ`      | terminal SOC anchor                                    | 1e3              |
| `R_delta_econ`     | rate-of-change penalty                                 | 1e−2             |
| `λ_soc`            | SOC slack penalty                                      | 1e6              |
| `λ_temp`           | temperature slack penalty                              | 1e7              |

### Decision variables

```
P_chg[j], P_dis[j] ∈ [0, P_max]      for j = 0..Nc−1
SOC[k], T[k] ∈ ℝ                      for k = 0..N
ε[k]      ≥ 0                         for k = 0..N           ← SOC slack
ε_temp[k] ≥ 0                         for k = 0..N           ← temperature slack
```

For `k ≥ Nc`, the control is **blocked**: the same `P_chg[Nc−1]` and
`P_dis[Nc−1]` are reused. SOH is frozen at `soh̄`, not a state.
**`P_reg` is not a decision variable** — it is the parameter `P̄_reg[k]`.

### Prediction model (2-state, frozen SOH)

```
state:  x = (SOC, T)
inputs: P_chg[j(k)], P_dis[j(k)], P̄_reg[k]


  dSOC                  η_c · P_chg − P_dis / η_d
  ────  =  ──────────────────────────────────────────
   dt                       soh̄ · E_nom · 3600


  dT          Q_joule(P_net, T)  −  h_cool · (T − T_amb)
  ────  =   ─────────────────────────────────────────────
   dt                          C_thermal
```

with `P_net = P_dis − P_chg`. The thermal Joule term receives the
**committed reg power** in addition to the planned chg/dis, so the
predicted heating accounts for it. `soh̄` is a parameter (not a state)
and stays constant for the duration of one solve.

**Discrete-time:**

```
( SOC[k+1], T[k+1] )  =  F(  ( SOC[k], T[k] ),
                             ( P_chg[j(k)], P_dis[j(k)], P̄_reg[k] );
                             soh̄ )
```

with `F` one explicit RK4 step at `Δt = 60 s`.

### Objective

```
minimise   Σ over k=0..N−1 of [
              − w_e · p̂_e[k] · ( P_dis[j(k)] − P_chg[j(k)] ) · Δt_h     ← energy revenue (negated)
              + w_deg · c_deg · α_deg · ( P_chg[j(k)] + P_dis[j(k)] ) · Δt_s   ← arbitrage degradation
              + Q_soc_anchor · ( SOC[k] − soc_ref[k] )²                  ← soft EMS SOC anchor
            ]
            + R_delta_econ · ( ( P_chg[0]  − P_chg_prev )²              ← rate-of-change at k=0
                             + ( P_dis[0]  − P_dis_prev )² )
            + R_delta_econ · Σ over k=1..Nc−1 of [
                  ( P_chg[k] − P_chg[k−1] )²                            ← rate-of-change inside Nc
                + ( P_dis[k] − P_dis[k−1] )²
              ]
            + Q_term_econ · ( SOC[N] − soc_ref[N] )²                    ← terminal SOC anchor
            + λ_soc  · Σ over k=0..N of ε[k]²                           ← SOC slack penalty
            + λ_temp · Σ over k=0..N of ε_temp[k]²                      ← temperature slack penalty
```

The reg-cycling degradation term `c_deg · α_reg · P̄_reg[k]` is
**constant** with respect to the decision variables (since `P̄_reg[k]`
is exogenous), so it has zero gradient and is omitted from the
objective. The simulator's ledger bills it post-hoc.

### Constraints

**Initial conditions:**

```
SOC[0] = soc_0,    T[0] = temp_0
```

**Dynamics:**

```
( SOC[k+1], T[k+1] )  =  F( ( SOC[k], T[k] ), ( P_chg[j(k)], P_dis[j(k)], P̄_reg[k] ); soh̄ )
```

for k = 0..N−1.

**SOC bounds (soft):**

```
SOC_min − ε[k]  ≤  SOC[k]  ≤  SOC_max + ε[k]            for k = 0..N
```

**Temperature bounds (soft):**

```
T_min − ε_temp[k]  ≤  T[k]  ≤  T_max + ε_temp[k]        for k = 0..N
```

**Power budget (full prediction horizon, post F18 fix):**

```
P_chg[j(k)] + P̄_reg[k]  ≤  P_max          for k = 0..N−1
P_dis[j(k)] + P̄_reg[k]  ≤  P_max          for k = 0..N−1
```

This constraint is enforced over the **full** horizon `k ∈ [0, N)`,
not just the unblocked control window `j ∈ [0, Nc)`, so the held
control values cannot propagate a physically infeasible plan into the
blocked region of the prediction.

### What this MPC does NOT have

- **No `P_reg` decision variable.** Reg power is exogenous (parameter from the EMS).
- **No endurance constraint.** Currently differs from `tracking_mpc` in this respect.
- **No stochasticity.** Single deterministic price horizon (forecast mean), single 2-state trajectory.
- **No SOH state.** Frozen as a parameter; the slow SOH dynamics are modelled by the EMS, not the MPC.
- **No V_rc transient states.**
- **No multi-cell pack model.** Plans against pack-mean SOC.
- **Per-step SOC anchor uses the end-of-hour reference at all `k`** in
  the current implementation — see audit finding F33 for the
  structural issue this creates inside an hour.

### Empirical status (post-audit)

The post-audit big experiment (3 subsets × 5 days × 5 strategies × 2
plant configurations) shows that on the current data pipeline (hourly
day-ahead prices, no intraday or real-time signals), the per-minute
economic re-optimization layer **does not produce a positive return**
relative to the EMS-only baseline (`ems_clamps`). Specifically:

- `economic_mpc` loses to `ems_clamps` in 28 of 30 days, by
  \$0.06–\$0.85/day.
- The loss is largest in the **volatile** subset, not the stressed
  subset, suggesting the failure mode is **horizon myopia**: the 60-min
  MPC cannot see the full-day arbitrage shape that the EMS captures.
- The MPC's economic term has **zero intra-hour gradient** because
  `p̂_e[k]` is constant within an hour (day-ahead resolution).
- Per-minute decisions are therefore driven entirely by the SOC anchor
  — a tracking signal that does no economic work.

This is a **data-pipeline gap**, not (necessarily) an MPC formulation
bug. The MPC layer would only earn its compute cost if it received
intraday/real-time prices the EMS does not see. See the OCP write-ups
for `ems_clamps` and `deterministic_lp` and the audit
[backlog](../../backlog.md) for the full diagnosis.
