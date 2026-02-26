# ems/ — Stochastic Energy Management System

The EMS solves a **scenario-based stochastic nonlinear program (NLP)** every hour over a 24-hour rolling horizon. It determines the economically optimal charge, discharge, and regulation schedule considering uncertainty in energy and regulation prices.

---

## Problem Formulation

### Decision Variables

For each scenario `s ∈ {1, ..., S}` and each hour `k ∈ {0, ..., N-1}`:

```
P_chg[s,k] ≥ 0     Charging power [kW]
P_dis[s,k] ≥ 0     Discharging power [kW]
P_reg[s,k] ≥ 0     Regulation capacity [kW]
SOC[s,k]            State of charge trajectory [-]
SOH[s,k]            State of health trajectory [-]
ε_soc[s,k] ≥ 0     Soft SOC constraint slack [-]
```

Total decision variables: `S × (3N + 2(N+1) + (N+1)) = S × (6N + 3)`. With S=5, N=24: **723 variables**.

### Objective

Maximize expected profit across all scenarios:

```
max  Σ_s  π_s · [ Σ_k ( revenue_energy[s,k] + revenue_reg[s,k] − cost_deg[s,k] )
                   − penalty_terminal_soc[s] − penalty_terminal_soh[s]
                   − penalty_soft_soc[s] ]
```

Where `π_s` is the probability of scenario `s`, and:

**Energy arbitrage revenue** (per hour):
```
revenue_energy[s,k] = price_energy[s,k] · (P_dis[s,k] − P_chg[s,k]) · (dt_ems / 3600)
```

The battery earns revenue when discharging (positive net power) and pays when charging (negative net power). The factor `dt_ems / 3600` converts the hourly time step to hours for the $/kWh price unit.

**Regulation capacity revenue** (per hour):
```
revenue_reg[s,k] = price_reg[s,k] · P_reg[s,k] · (dt_ems / 3600)
```

The battery earns a capacity payment for making P_reg kW available, regardless of actual dispatch.

**Degradation cost** (per hour):
```
cost_deg[s,k] = C_deg · α_deg · (P_chg[s,k] + P_dis[s,k] + P_reg[s,k]) · dt_ems
```

Where `C_deg` is the monetary cost per unit of SOH lost [$/unit SOH] and `α_deg` is the degradation rate [1/(kW·s)].

**Terminal penalties:**
```
penalty_terminal_soc[s] = w_soc · (SOC[s,N] − SOC_terminal)²
penalty_terminal_soh[s] = w_soh · (SOH[s,N] − SOH_init)²
```

These prevent the optimizer from depleting the battery (or its health) by end of horizon.

**Soft SOC penalty:**
```
penalty_soft_soc[s] = Σ_k  1e5 · ε_soc[s,k]²
```

### Constraints

**Dynamics** (per scenario, per hour):
```
x_{k+1} = F_rk4(x_k, u_k)     (RK4 integrator at dt = 3600 s)

where x = [SOC, SOH],  u = [P_chg, P_dis, P_reg]
```

**Initial conditions:**
```
SOC[s,0] = SOC_init     (from current EKF estimate)
SOH[s,0] = SOH_init     (from current EKF estimate)
```

**SOC bounds** (soft):
```
SOC_min − ε_soc[s,k]  ≤  SOC[s,k]  ≤  SOC_max + ε_soc[s,k]
ε_soc[s,k] ≥ 0
```

**SOH bounds** (hard):
```
0.5 ≤ SOH[s,k] ≤ 1.001
```

The upper bound is relaxed to 1.001 to prevent numerical infeasibility when the initial SOH estimate is very close to 1.0.

**Power bounds:**
```
0 ≤ P_chg[s,k] ≤ P_max
0 ≤ P_dis[s,k] ≤ P_max
0 ≤ P_reg[s,k] ≤ P_max · regulation_fraction
```

**Power budget** (the battery cannot simultaneously use full power for dispatch and regulation):
```
P_chg[s,k] + P_reg[s,k] ≤ P_max
P_dis[s,k] + P_reg[s,k] ≤ P_max
```

### Non-Anticipativity Constraints

The first-stage (current hour) decisions must agree across all scenarios:

```
P_chg[s,0] = P_chg[0,0]     ∀ s ∈ {1, ..., S}
P_dis[s,0] = P_dis[0,0]     ∀ s ∈ {1, ..., S}
P_reg[s,0] = P_reg[0,0]     ∀ s ∈ {1, ..., S}
```

This enforces the fundamental principle of stochastic programming: you cannot make different decisions for the current hour based on future prices you haven't observed yet. Only future hours (k ≥ 1) are allowed to differ across scenarios.

### Output

The EMS returns **probability-weighted average** references:

```
P_chg_ref[k] = Σ_s  π_s · P_chg*[s,k]
P_dis_ref[k] = Σ_s  π_s · P_dis*[s,k]
P_reg_ref[k] = Σ_s  π_s · P_reg*[s,k]
SOC_ref[k]   = Σ_s  π_s · SOC*[s,k]
SOH_ref[k]   = Σ_s  π_s · SOH*[s,k]
```

These hourly references are then interpolated to the MPC time resolution:
- **Power**: zero-order hold (constant within each hour)
- **State**: linear interpolation between hourly knot points

---

## Solver Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Solver | IPOPT | Interior-point method for large-scale NLP |
| Linear solver | MUMPS | Open-source sparse symmetric solver |
| Max iterations | 3,000 | Large budget for difficult scenarios |
| Tolerance | 1e-6 | Primary optimality tolerance |
| Acceptable tol | 1e-4 | Fallback if primary tolerance not reached |
| Warm start | yes | Reuses barrier parameter from previous solve |

---

## Fallback Behavior

If the solver fails (infeasibility or max iterations), the EMS returns **zero-power references** with constant SOC/SOH trajectories. The MPC will then hold the battery at its current state until the next EMS solve succeeds. This ensures the system always has valid references.
