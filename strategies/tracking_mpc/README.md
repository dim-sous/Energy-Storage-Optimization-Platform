# tracking_mpc — Tracking NLP MPC (controlled-experiment baseline)

**Pitch-visible:** no (controlled-experiment baseline for `economic_mpc`)
**Composition:** `EconomicEMS` planner (hourly, see `strategies/ems_clamps`) + `TrackingMPC` (per-minute closed-loop)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Same prediction model and exogenous-`P_reg` handling as `economic_mpc`.
Differs in **exactly one place**: the cost function **tracks** the EMS
plan in (SOC, P_chg, P_dis) space instead of optimizing arbitrage.
Also includes a short-horizon FCR endurance constraint that
`economic_mpc` does not have.

## Optimal Control Problem

This is a **deterministic NLP solved at 60 s cadence** with a 1-hour
prediction horizon and control-horizon blocking after `Nc` steps —
structurally identical to `economic_mpc`, with a tracking objective
substituted for the economic one and a soft endurance constraint added.

### Notation

Same notation as `economic_mpc/README.md`, plus:

| Symbol             | Meaning                                                | Default          |
|--------------------|--------------------------------------------------------|------------------|
| `Q_soc`            | per-step SOC tracking weight                           | 1e4              |
| `R_power`          | power-reference tracking weight                        | 1                |
| `R_delta`          | rate-of-change penalty                                 | 10               |
| `Q_terminal`       | terminal SOC anchor                                    | 1e5              |
| `T_end_mpc`        | MPC short-horizon endurance                            | 5/60 h ≈ 0.083 h |
| `λ_soc`,`λ_temp`   | slack penalties                                        | 1e6, 1e7         |
| `p_chg_ref[k]`,`p_dis_ref[k]` | EMS power references at MPC cadence         | kW               |
| `P̄_reg[k]`         | committed FCR power, ZOH from EMS plan                 | kW               |

### Decision variables

```
P_chg[j], P_dis[j] ∈ [0, P_max]      for j = 0..Nc−1
SOC[k], T[k] ∈ ℝ                      for k = 0..N
ε[k]      ≥ 0                         for k = 0..N           ← SOC slack
ε_temp[k] ≥ 0                         for k = 0..N           ← temperature slack
ε_end[k]  ≥ 0                         for k = 0..N           ← endurance slack
```

For `k ≥ Nc`, the control is blocked: same `P_chg[Nc−1]` and
`P_dis[Nc−1]` are reused. **`P_reg` is not a decision variable** — it
is the parameter `P̄_reg[k]` from the EMS plan.

### Prediction model

Identical to `economic_mpc`: 2-state RK4 with frozen SOH and the
EMS-committed `P̄_reg[k]` entering the thermal Joule heating term:

```
( SOC[k+1], T[k+1] )  =  F(  ( SOC[k], T[k] ),
                             ( P_chg[j(k)], P_dis[j(k)], P̄_reg[k] );
                             soh̄ )
```

See `strategies/economic_mpc/README.md` for the explicit ODE.

### Objective

```
minimise   Σ over k=0..N−1 of [
              Q_soc   · ( SOC[k] − soc_ref[k] )²                            ← SOC tracking
            + R_power · (   ( P_chg[j(k)] − p_chg_ref[k] )²
                          + ( P_dis[j(k)] − p_dis_ref[k] )² )               ← power-reference tracking
            ]
            + R_delta · (   ( P_chg[0]  − P_chg_prev )²                     ← rate-of-change at k=0
                          + ( P_dis[0]  − P_dis_prev )² )
            + R_delta · Σ over k=1..Nc−1 of [
                  ( P_chg[k] − P_chg[k−1] )²                                ← rate-of-change inside Nc
                + ( P_dis[k] − P_dis[k−1] )²
              ]
            + Q_terminal · ( SOC[N] − soc_ref[N] )²                         ← terminal SOC anchor
            + λ_soc  · Σ over k=0..N of ε[k]²
            + λ_temp · Σ over k=0..N of ε_temp[k]²
            + λ_soc  · Σ over k=0..N of ε_end[k]²
```

There is **no economic term** (no `−price · (P_dis − P_chg) · Δt_h`)
and **no degradation term** in the cost. Both are subsumed into the
tracking of the EMS plan, which itself optimised against energy prices
and degradation cost at hourly cadence.

### Constraints

**Initial conditions, dynamics, SOC bounds, temperature bounds, and
power-budget headroom over the full horizon** are identical to
`economic_mpc`. See `strategies/economic_mpc/README.md` for the full
expressions. The one addition is the endurance constraint:

**Endurance (soft, at every predicted step):**

```
SOC[k] + (T_end_mpc · η_c       / E_nom) · P̄_reg[k_p]  ≤  SOC_max + ε_end[k]
SOC[k] − (T_end_mpc / (E_nom · η_d))     · P̄_reg[k_p]  ≥  SOC_min − ε_end[k]
```

for `k = 0..N`, with `k_p = min(k, N − 1)` (the reg-power index, since
`P̄_reg` has length `N`).

The horizon `T_end_mpc = 5 minutes` is deliberately shorter than the
EMS's own `T_end = 30 minutes` — the EMS already enforces 30-minute
strategic headroom; the MPC adds a tactical 5-minute cushion on top,
scaled to typical OU activation persistence.

### Difference vs `economic_mpc` (single-knob comparison)

| Element                       | `tracking_mpc`              | `economic_mpc`                     |
|-------------------------------|-----------------------------|------------------------------------|
| SOC tracking weight           | `Q_soc = 1e4` (dominant)    | `Q_soc_anchor = 10` (soft)         |
| Power-reference tracking      | `R_power · ‖u − u_ref‖²`    | (none)                             |
| Energy-arbitrage term         | (none)                      | `−w_e · p̂_e · (P_dis − P_chg) · Δt_h` |
| Degradation in objective      | (none)                      | `w_deg · c_deg · α_deg · (P_chg + P_dis) · Δt_s` |
| Endurance constraint          | `T_end_mpc = 5 min` (soft)  | (none)                             |
| Terminal anchor               | `Q_terminal = 1e5`          | `Q_term_econ = 1e3`                |
| Rate-of-change penalty        | `R_delta = 10`              | `R_delta_econ = 1e−2`              |

The two strategies share the **same prediction model**, the **same
exogenous P_reg handling**, the **same fallback path**, and the **same
data inputs**. The only product-relevant degree of freedom is the cost
function. This makes the two-strategy comparison a controlled experiment.

### What this MPC does NOT have

- **No `P_reg` decision variable.** Reg power is exogenous (parameter from the EMS).
- **No stochasticity.**
- **No SOH state.** Frozen as a parameter.
- **No V_rc transient states.**
- **No multi-cell pack model.**
- **Per-step SOC anchor uses the end-of-hour reference at all `k`** in
  the current implementation — same audit finding F33 as `economic_mpc`.

## History

Pre-2026-04-15 this file described a "`Q_soc = 1e4` dominated old v5
stack" with a fictitious `P_reg` decision variable that the adapter
discarded — a strawman baseline that made the `economic_mpc` vs
`tracking_mpc` comparison meaningless. The audit redesigned
`TrackingMPC` to drop the fictitious decision variable, add the
endurance constraint, and use the same exogenous `P_reg` handling as
`economic_mpc`. After the redesign, the comparison is honest.
