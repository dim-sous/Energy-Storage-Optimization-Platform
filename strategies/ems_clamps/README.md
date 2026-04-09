# ems_clamps — Stochastic NLP (EMS)

**Pitch-visible:** yes (canonical "EMS alone" pitch baseline)
**Composition:** `EconomicEMS` (stochastic NLP planner) + open-loop dispatch (no MPC)
**Solver:** CasADi `Opti` + IPOPT (MUMPS linear solver)

## What it does (plain words)

Solves a scenario-based two-stage stochastic NLP every hour over `S`
forecast price scenarios with non-anticipativity. Uses a 3-state
nonlinear ODE (SOC, SOH, T) with Arrhenius-coupled degradation and a
linearised current/thermal coupling. Plans hourly arbitrage and FCR
commitment hedged across price uncertainty. Holds the planner's hourly
setpoint constant for the next hour and dispatches it open-loop —
there is no MPC layer.

## Optimal Control Problem

### Notation

| Symbol             | Meaning                                                | Units            |
|--------------------|--------------------------------------------------------|------------------|
| `N`                | planning horizon                                       | hours (= 24)     |
| `S`                | number of price scenarios                              | (= 5)            |
| `π[s]`             | probability of scenario s, with Σ π[s] = 1             | —                |
| `Δt_h`, `Δt_s`     | step size in hours, in seconds                         | h (= 1), s (= 3600) |
| `k`                | time index, k = 0..N−1                                 | —                |
| `s`                | scenario index, s = 1..S                               | —                |
| `p_e[s,k]`,`p_r[s,k]` | per-scenario forecast prices                        | \$/kWh, \$/(kW·h) |
| `E_nom`            | nominal pack capacity                                  | kWh              |
| `n_m`              | modules in pack (the `n_modules` parameter)            | (= 4)            |
| `η_c`, `η_d`       | charge / discharge efficiency                          | —                |
| `P_max`            | max power                                              | kW               |
| `SOC_min`,`SOC_max`,`T_min`,`T_max` | state bounds                          | —                |
| `SOC_0`,`SOH_0`,`T_0` | initial state (from EKF)                            | —                |
| `SOC_term`         | terminal SOC target                                    | —                |
| `α_deg`,`α_reg`    | degradation rates                                      | 1/(kW·s)         |
| `c_deg`            | degradation cost                                       | \$/SOH lost      |
| `ā`                | E[ \|activation\| ]                                    | (= 0.04)         |
| `T_end`            | endurance horizon                                      | h (= 0.5)        |
| `κ(T)`             | Arrhenius factor (defined below)                       | —                |
| `Q_soc_term`,`Q_soh_term` | terminal SOC and SOH penalties                  | (each = 1e4)     |
| `λ_soc`,`λ_end`    | soft-constraint slack penalties                        | (each = 1e5)     |

### Decision variables (per scenario s)

```
P_chg[s,k], P_dis[s,k], P_reg[s,k] ∈ [0, P_max]      for k = 0..N−1
SOC[s,k] ∈ ℝ                                          for k = 0..N
SOH[s,k] ∈ [0.5, 1.001]                               for k = 0..N
T[s,k]   ∈ [T_min, T_max]                             for k = 0..N
ε_soc[s,k] ≥ 0                                         for k = 0..N
ε_end[s,k] ≥ 0                                         for k = 0..N−1
```

### Continuous-time dynamics (3-state ODE)

```
state:  x = (SOC, SOH, T)
input:  u = (P_chg, P_dis, P_reg)


  dSOC                  η_c · P_chg − P_dis / η_d
  ────  =  ──────────────────────────────────────────
   dt                   SOH · E_nom · 3600

                          (η_c − 1/η_d) · ā · P_reg              ← expected
              +  ─────────────────────────────────────              FCR
                          SOH · E_nom · 3600                        efficiency drain


  dSOH         κ(T)
  ────  =  −  ──────  · ( α_deg · (P_chg + P_dis) + α_reg · |P_reg| )
   dt          n_m


  dT          Q_joule(P_net, T)  −  h_cool · (T − T_amb)
  ────  =   ─────────────────────────────────────────────
   dt                          C_thermal
```

with `P_net = P_dis − P_chg`. The Arrhenius acceleration factor is

```
κ(T) = exp[ (E_a / R_gas) · ( 1 / T_ref^K  −  1 / T^K ) ]
```

with `T^K = T + 273.15`. The Joule heating uses the linearised current
`I = |P_net| · 1000 / V_oc` through the total DC resistance
`R_tot = R_0 + R_1 + R_2`:

```
Q_joule(P_net, T) = I² · R_tot
```

**Discrete-time:**

```
x[s, k+1] = F( x[s, k],  u[s, k] )       k = 0..N−1
```

where `F` is one explicit RK4 step at `Δt = 3600 s`.

### Objective (expected cost across scenarios)

```
minimise   Σ over s=1..S of   π[s] · J[s]
```

with the per-scenario cost

```
J[s] =   − Σ over k=0..N−1 of [
              p_e[s,k] · (P_dis[s,k] − P_chg[s,k]) · Δt_h        ← energy revenue
            + p_r[s,k] ·  P_reg[s,k]               · Δt_h        ← capacity revenue
            − c_deg · Δt_s · ( α_deg · (P_chg[s,k] + P_dis[s,k])
                             + α_reg ·  P_reg[s,k] )             ← degradation cost
          ]
       +   Q_soc_term · ( SOC[s, N] − SOC_term )²                ← terminal SOC
       +   Q_soh_term · ( SOH[s, N] − SOH_0    )²                ← terminal SOH
       +   λ_soc · Σ over k=0..N   of  ε_soc[s,k]²
       +   λ_end · Σ over k=0..N−1 of  ε_end[s,k]²
```

### Constraints (per scenario s)

**Initial conditions:**

```
SOC[s, 0] = SOC_0,    SOH[s, 0] = SOH_0,    T[s, 0] = T_0
```

**Dynamics:**

```
x[s, k+1]  =  F( x[s, k],  u[s, k] )      for k = 0..N−1
```

**SOC bounds (soft):**

```
SOC_min − ε_soc[s,k]  ≤  SOC[s,k]  ≤  SOC_max + ε_soc[s,k]      for k = 0..N
```

**Power budget:**

```
P_chg[s,k] + P_reg[s,k]  ≤  P_max
P_dis[s,k] + P_reg[s,k]  ≤  P_max
```

**Endurance (soft):**

```
SOC[s, k+1] + (T_end · η_c       / E_nom) · P_reg[s,k]  ≤  SOC_max + ε_end[s,k]
SOC[s, k+1] − (T_end / (E_nom · η_d))     · P_reg[s,k]  ≥  SOC_min − ε_end[s,k]
```

### Non-anticipativity (cross-scenario coupling)

The first-stage decisions — i.e., the action that gets executed
**now**, *before* the realised scenario is known — must agree across
all scenarios:

```
P_chg[s, 0] = P_chg[1, 0]
P_dis[s, 0] = P_dis[1, 0]                for all s = 2..S
P_reg[s, 0] = P_reg[1, 0]
```

This is what makes it a true two-stage stochastic program rather than
`S` independent deterministic problems averaged after the fact. Without
non-anticipativity, the planner would "see" the realised scenario from
`k = 0` and the comparison vs `deterministic_lp` would be unfair.

### What this NLP does NOT model

- **V_rc transient dynamics** (charge-transfer + diffusion modes). Time
  constants ≤ 400 s ≪ Δt_ems = 3600 s, so they decay within one EMS
  step.
- **Multi-cell pack effects.** Plans against pack-mean SOC; cell-level
  imbalance is left to the plant's balancer.
- **Sub-hour activation realisation.** Uses only the expected magnitude
  `ā = 0.04` in the SOC drain term — does not condition on the actual
  activation sample.
- **Closed-loop feedback inside the hour.** The hourly setpoint is held
  open-loop until the next EMS solve.
