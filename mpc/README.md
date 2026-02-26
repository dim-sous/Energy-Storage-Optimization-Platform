# mpc/ — Nonlinear Tracking MPC with Control Horizon Blocking

The MPC tracks the hourly EMS references in real time, running every 60 seconds. It uses a 60-step (1 hour) prediction horizon with a 10-step (10 minute) control horizon to produce smooth, constraint-respecting power commands.

---

## Problem Formulation

The NLP is **prebuilt once** at initialization and re-solved every 60 seconds by updating the parameters (current state and reference trajectories). This avoids the overhead of reconstructing the problem graph at each time step.

### Decision Variables

```
P_chg[j],  j = 0, ..., Nc-1     Free charging power [kW]        (10 variables)
P_dis[j],  j = 0, ..., Nc-1     Free discharging power [kW]     (10 variables)
P_reg[j],  j = 0, ..., Nc-1     Free regulation power [kW]      (10 variables)
SOC[k],    k = 0, ..., N         SOC trajectory [-]              (61 variables)
SOH[k],    k = 0, ..., N         SOH trajectory [-]              (61 variables)
ε[k],      k = 0, ..., N         Soft-constraint slack [-]       (61 variables)
```

Total: **213 decision variables** (vs 243 without control horizon blocking).

### Control Horizon Blocking

For prediction steps `k ≥ Nc`, the control input is held constant at the last free value:

```
u_k = u_{min(k, Nc-1)}
```

Concretely, with Nc = 10 and N = 60:
- Steps 0-9: independent control variables (MPC can choose freely)
- Steps 10-59: control held at u₉ (the last free value)

This reduces the number of power decision variables from 180 (3 × 60) to 30 (3 × 10) while maintaining the full 60-step prediction horizon for stability and constraint satisfaction.

### Parameters (set each solve)

```
soc_0          Current SOC estimate (from EKF)
soh_0          Current SOH estimate (from EKF)
soc_ref[k]     SOC reference trajectory (from EMS, interpolated)
soh_ref[k]     SOH reference trajectory (from EMS, interpolated)
p_chg_ref[k]   Charge power reference (from EMS)
p_dis_ref[k]   Discharge power reference (from EMS)
p_reg_ref[k]   Regulation power reference (from EMS)
```

### Objective

```
min  Σ_{k=0}^{N-1} [ Q_soc · (SOC_k − soc_ref_k)²
                     + Q_soh · (SOH_k − soh_ref_k)²
                     + R_power · ( (P_chg_j − p_chg_ref_k)²
                                 + (P_dis_j − p_dis_ref_k)²
                                 + (P_reg_j − p_reg_ref_k)² ) ]
   + Σ_{k=1}^{Nc-1} R_delta · ( (P_chg_k − P_chg_{k-1})²
                                + (P_dis_k − P_dis_{k-1})²
                                + (P_reg_k − P_reg_{k-1})² )
   + Q_terminal · (SOC_N − soc_ref_N)²
   + Σ_{k=0}^{N}  slack_penalty · ε_k²
```

Where `j = min(k, Nc-1)` maps prediction step to control variable index.

**Cost term breakdown:**

| Term | Weight | Purpose |
|------|--------|---------|
| SOC tracking | Q_soc = 1e4 | Follow the EMS SOC plan |
| SOH tracking | Q_soh = 1e2 | Gentle SOH tracking (slow dynamics) |
| Power tracking | R_power = 1.0 | Follow the EMS power plan |
| Rate-of-change | R_delta = 10.0 | Smooth power transitions |
| Terminal | Q_terminal = 1e5 | Drive SOC to reference at end of horizon |
| Slack penalty | slack_penalty = 1e6 | Penalize SOC bound violations |

### Constraints

**Dynamics** (per step):
```
[SOC_{k+1}, SOH_{k+1}]ᵀ = F_rk4([SOC_k, SOH_k]ᵀ, [P_chg_j, P_dis_j, P_reg_j]ᵀ)
```

**Initial conditions** (equality, from EKF estimate):
```
SOC_0 = soc_0
SOH_0 = soh_0
```

**SOC bounds** (soft):
```
SOC_min − ε_k ≤ SOC_k ≤ SOC_max + ε_k,    ε_k ≥ 0
```

**SOH bounds** (hard):
```
0.5 ≤ SOH_k ≤ 1.001
```

**Power bounds:**
```
0 ≤ P_chg_j ≤ P_max
0 ≤ P_dis_j ≤ P_max
0 ≤ P_reg_j ≤ 0.3 · P_max
```

**Power budget:**
```
P_chg_j + P_reg_j ≤ P_max
P_dis_j + P_reg_j ≤ P_max
```

---

## Warm-Starting

After each successful solve, the MPC caches the optimal trajectories and **shifts them** by one step for the next solve:

```
P_chg_init[0:Nc-1] ← P_chg_opt[1:Nc]
P_chg_init[Nc-1]   ← P_chg_opt[Nc-1]     (repeat last value)
```

The same shift is applied to P_dis, P_reg, SOC, and SOH. This provides an excellent initial guess that is typically very close to the next solution, reducing IPOPT iterations from ~50 (cold start) to ~5 (warm start).

---

## Feasibility Guarantee

The soft SOC constraints ensure the NLP is **always feasible**, even when:
- The EKF provides an estimate outside the SOC limits (due to estimation error)
- The EMS reference is infeasible at the current state
- Model mismatch causes the predicted trajectory to violate bounds

The slack penalty (1e6) is large enough that the solver only uses slack as a last resort, but its presence guarantees a solution exists.

---

## Output

The MPC returns the **first control action** only:

```
u_cmd = [P_chg[0], P_dis[0], P_reg[0]]
```

This is applied to the plant for the next 60 seconds, after which the MPC re-solves with updated state feedback (receding horizon principle).

---

## Solver Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Solver | IPOPT | Interior-point NLP solver |
| Max iterations | 500 | Sufficient for warm-started solves |
| Tolerance | 1e-6 | Primary optimality tolerance |
| Acceptable tol | 1e-4 | Fallback tolerance |
| Warm start | yes | Exploits previous solution |
| Initial barrier (mu_init) | 1e-3 | Warm-start compatible barrier |

If the solver fails, the MPC falls back to the EMS reference for the current step: `u_cmd = [p_chg_ref[0], p_dis_ref[0], p_reg_ref[0]]`.
