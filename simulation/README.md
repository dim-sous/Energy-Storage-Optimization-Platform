# simulation/ — Multi-Rate Simulation Coordinator

The `MultiRateSimulator` orchestrates the entire closed-loop simulation, coordinating the plant, EMS, MPC, EKF, and MHE at their respective time scales.

---

## Multi-Rate Loop Structure

The simulation runs a single loop at the plant time step (dt_sim = 1 s) with conditional execution of slower components:

```
for sim_step in range(86,400):              # 24 hours at 1 s
    t = sim_step · dt_sim

    if t mod dt_ems == 0:                    # every 3,600 s (24 times)
        EMS.solve(x_est, price_scenarios)
        interpolate references to MPC resolution

    if t mod dt_mpc == 0:                    # every 60 s (1,440 times)
        y = plant.get_measurement()
        x_ekf = EKF.step(u, y)
        x_mhe = MHE.step(u, y)
        u = MPC.solve(x_ekf, references)

    plant.step(u)                            # every 1 s (86,400 times)
```

### Timing Summary

| Event | Period | Count (24 h) | What happens |
|-------|--------|-------------|--------------|
| Plant step | 1 s | 86,400 | Apply current u to battery, record true state |
| Measurement | 60 s | 1,440 | Read noisy SOC from plant |
| EKF update | 60 s | 1,440 | Predict + update with new measurement |
| MHE update | 60 s | 1,440 | Append to window, solve NLP |
| MPC solve | 60 s | 1,440 | Compute new control action |
| EMS solve | 3,600 s | 24 | Re-optimize economic schedule |

---

## Reference Interpolation

The EMS produces **hourly** references. The MPC consumes **minute-by-minute** references. The `interpolate_ems_to_mpc()` function bridges this gap:

**Power references — zero-order hold:**
```
P_ref_mpc[k] = P_ref_ems[⌊k / ratio⌋]     ratio = dt_ems / dt_mpc = 60
```

Each hourly power value is repeated 60 times. This is appropriate because the EMS power decision is a commitment for the full hour.

**State references — linear interpolation:**
```
SOC_ref_mpc[k] = interp(k / ratio, [0, 1, ..., N], SOC_ref_ems)
```

SOC and SOH references are linearly interpolated between hourly knot points. This produces smooth reference trajectories for the MPC to track, avoiding discontinuous jumps at hour boundaries.

---

## Profit Calculation

Revenue is computed at each MPC step (60 s resolution) using the **actually applied** power and the first scenario's prices:

```
energy_profit[k] = price_energy[h] · (P_dis_applied − P_chg_applied) · (dt_mpc / 3600)
reg_profit[k]    = price_reg[h] · P_reg_applied · (dt_mpc / 3600)
deg_cost[k]      = C_deg · α_deg · (P_chg + P_dis + P_reg) · dt_mpc
net_profit[k]    = energy_profit[k] + reg_profit[k] − deg_cost[k]
```

Where `h = ⌊sim_step / steps_per_ems⌋` maps the current simulation step to the corresponding hour for price lookup.

The cumulative profit is tracked for visualization and final reporting.

---

## Data Flow

```
EMS output (hourly):
  P_chg_ref[24], P_dis_ref[24], P_reg_ref[24], SOC_ref[25], SOH_ref[25]
        │
        ▼ interpolate_ems_to_mpc
MPC references (per minute):
  P_chg_ref_mpc[1440], SOC_ref_mpc[1441], ...
        │
        ▼ _extract_ref (sliding window)
MPC input (60-step window):
  soc_ref[61], p_chg_ref[60], ...
        │
        ▼ MPC.solve
Control action:
  u = [P_chg, P_dis, P_reg]
        │
        ▼ plant.step (60 times at 1 s)
True state:
  [SOC_true, SOH_true]
```

The `_extract_ref` helper extracts a window of the appropriate length from the interpolated references, padding with the last value if the window extends beyond the available data. This handles end-of-horizon gracefully.

---

## Output Dictionary

The simulator returns a comprehensive dictionary with data at multiple time resolutions:

| Key | Shape | Resolution | Description |
|-----|-------|-----------|-------------|
| `time_sim` | (86401,) | 1 s | Plant time axis |
| `soc_true` | (86401,) | 1 s | True SOC trajectory |
| `soh_true` | (86401,) | 1 s | True SOH trajectory |
| `time_mpc` | (1440,) | 60 s | MPC/estimator time axis |
| `soc_ekf` | (1440,) | 60 s | EKF SOC estimates |
| `soh_ekf` | (1440,) | 60 s | EKF SOH estimates |
| `soc_mhe` | (1440,) | 60 s | MHE SOC estimates |
| `soh_mhe` | (1440,) | 60 s | MHE SOH estimates |
| `power_applied` | (1440, 3) | 60 s | [P_chg, P_dis, P_reg] applied |
| `cumulative_profit` | (1440,) | 60 s | Running total profit [$] |
| `energy_profit` | (1440,) | 60 s | Per-step energy revenue [$] |
| `reg_profit` | (1440,) | 60 s | Per-step regulation revenue [$] |
| `deg_cost` | (1440,) | 60 s | Per-step degradation cost [$] |
| `total_profit` | scalar | — | Final cumulative profit [$] |
| `soh_degradation` | scalar | — | Total SOH loss [-] |
| `ems_p_chg_refs` | list of arrays | hourly | EMS charge references (each solve) |
| `ems_soc_refs` | list of arrays | hourly | EMS SOC references (each solve) |
| `prices_energy` | (48,) | hourly | Energy prices (scenario 1) |
| `prices_reg` | (48,) | hourly | Regulation prices (scenario 1) |
