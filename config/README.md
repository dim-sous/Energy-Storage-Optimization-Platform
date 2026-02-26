# config/ — Parameter Definitions

All tunable parameters for the platform are centralized in `parameters.py` as **frozen dataclasses**. Freezing ensures parameters are immutable after construction, preventing accidental mutation during a simulation run.

---

## Dataclasses

### `BatteryParams`

Physical and electrochemical properties of the BESS.

| Parameter | Symbol | Default | Unit | Description |
|-----------|--------|---------|------|-------------|
| `E_nom_kwh` | E_nom | 200 | kWh | Nameplate energy capacity |
| `P_max_kw` | P_max | 100 | kW | Maximum charge/discharge power |
| `SOC_min` | SOC_min | 0.10 | — | Minimum allowable state of charge |
| `SOC_max` | SOC_max | 0.90 | — | Maximum allowable state of charge |
| `SOC_init` | SOC_0 | 0.50 | — | Initial state of charge |
| `SOH_init` | SOH_0 | 1.00 | — | Initial state of health |
| `SOC_terminal` | SOC_T | 0.50 | — | Terminal SOC target for EMS |
| `eta_charge` | η_c | 0.95 | — | Charging efficiency |
| `eta_discharge` | η_d | 0.95 | — | Discharging efficiency |
| `alpha_deg` | α_deg | 2.78e-9 | 1/(kW·s) | Degradation rate coefficient |

**Degradation rate calibration:**

The degradation model is:

```
dSOH/dt = -α_deg · (P_chg + P_dis + |P_reg|)
```

For one full 200 kWh charge-discharge cycle at 100 kW:
- Cycle duration: 2 × (200 kWh / 100 kW) × 3600 s/h = 14,400 s
- Average power throughput: 100 kW
- SOH loss per cycle: α_deg × 100 × 14,400 = 2.78e-9 × 100 × 14,400 ≈ 0.004 (0.4%)
- Cycles to 80% SOH: 0.20 / 0.004 = 50 full cycles

This is intentionally accelerated for demonstration. Real lithium-ion cells typically achieve 3,000-5,000 cycles to 80% SOH. To model a real system, set `alpha_deg ≈ 2.78e-11`.

### `TimeParams`

Multi-rate time discretization. All values in **seconds** except `sim_hours`.

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `dt_ems` | 3,600 | s | EMS re-solve period |
| `dt_mpc` | 60 | s | MPC and estimator period |
| `dt_estimator` | 60 | s | EKF/MHE sampling period (= dt_mpc) |
| `dt_sim` | 1 | s | Plant integration step |
| `sim_hours` | 24 | h | Total simulation duration |

The ratio between time scales determines the simulation structure:
- `dt_ems / dt_mpc = 60`: 60 MPC solves per EMS solve
- `dt_mpc / dt_sim = 60`: 60 plant steps per MPC solve
- Total plant steps: `sim_hours × 3600 / dt_sim = 86,400`

### `EMSParams`

Stochastic energy management system optimizer tuning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ems` | 24 | Planning horizon (hours / steps) |
| `n_scenarios` | 5 | Number of stochastic price scenarios |
| `regulation_fraction` | 0.3 | Maximum regulation power as fraction of P_max |
| `degradation_cost` | 50.0 | Penalty for SOH loss [$/unit SOH] |
| `terminal_soc_weight` | 1e4 | Quadratic penalty on SOC deviation from SOC_terminal at end of horizon |
| `terminal_soh_weight` | 1e4 | Quadratic penalty on SOH loss at end of horizon |

### `MPCParams`

Nonlinear tracking MPC tuning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_mpc` | 60 | Prediction horizon [steps at dt_mpc = 60s → 60 min] |
| `Nc_mpc` | 10 | Control horizon [steps at dt_mpc = 60s → 10 min] |
| `Q_soc` | 1e4 | SOC tracking weight |
| `Q_soh` | 1e2 | SOH tracking weight |
| `R_power` | 1.0 | Power reference tracking weight |
| `R_delta` | 10.0 | Control rate-of-change penalty |
| `Q_terminal` | 1e5 | Terminal SOC penalty |
| `slack_penalty` | 1e6 | Soft SOC constraint violation penalty |

**Weight rationale:**

- `Q_soc >> Q_soh`: SOC changes rapidly and must be tracked tightly; SOH drifts slowly and over-reacting to SOH estimation noise would cause jerky control.
- `Q_terminal >> Q_soc`: Strong terminal penalty ensures the MPC drives SOC toward the reference at the end of its horizon, preventing myopic behavior.
- `R_delta > R_power`: Penalizing control changes more than control magnitude produces smoother power commands, reducing mechanical stress and improving grid-friendliness.
- `slack_penalty >> Q_soc`: Violating SOC bounds is strongly penalized but allowed as a last resort, guaranteeing feasibility.

### `EKFParams`

Extended Kalman Filter tuning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_soc` | 1e-6 | Process noise variance for SOC |
| `q_soh` | 1e-12 | Process noise variance for SOH |
| `r_soc_meas` | 1e-4 | Measurement noise variance (σ ≈ 0.01) |
| `p0_soc` | 1e-3 | Initial SOC estimation uncertainty |
| `p0_soh` | 1e-2 | Initial SOH estimation uncertainty |

**Tuning rationale:**

- `q_soh << q_soc`: SOH changes ~10,000x slower than SOC. Setting very low process noise for SOH tells the EKF to trust the model prediction (SOH barely changes) rather than trying to estimate rapid SOH fluctuations from noisy SOC-only measurements.
- `p0_soh >> p0_soc`: Initial SOH uncertainty is larger because SOH is never directly measured, so the filter starts with less confidence about SOH.

### `MHEParams`

Moving Horizon Estimation tuning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_mhe` | 30 | Estimation window [steps → 30 min] |
| `arrival_soc` | 1e3 | Weight on SOC arrival cost |
| `arrival_soh` | 1e4 | Weight on SOH arrival cost |
| `w_soc_meas` | 1e4 | Measurement residual weight |
| `w_process_soc` | 1e2 | Process noise weight (SOC) |
| `w_process_soh` | 1e8 | Process noise weight (SOH) |

**Tuning rationale:**

- `w_process_soh >> w_process_soc`: Very high SOH process noise penalty prevents the MHE from absorbing model errors into SOH corrections. Since SOH is unobserved, without this constraint the optimizer would use SOH as a "free variable" to explain any SOC mismatch.
- `arrival_soh > arrival_soc`: Strong arrival cost for SOH anchors the estimate to the prior, preventing drift.
