# v3_pack_model — Multi-Cell Battery Pack with Active Cell Balancing

Extends the v2 thermal model from a single lumped battery to a **multi-cell pack** of N cells in series, each with unique physical parameters drawn from manufacturing tolerances. An active proportional cell balancing controller equalises SOC across cells. EMS, MPC, EKF, and MHE remain pack-level (3-state) — they see aggregated pack states while the BMS handles cell balancing transparently, creating realistic model mismatch.

## What Changed from v2

| Component | v2_thermal_model | v3_pack_model |
|-----------|-----------------|---------------|
| **Plant** | Single `BatteryPlant` | `BatteryPack` wrapping N `BatteryPlant` instances |
| **Parameters** | Homogeneous | Per-cell variation (capacity, resistance, degradation, initial SOC) |
| **Balancing** | None | Active proportional controller per cell |
| **Optimizer model** | Matches plant exactly | Pack-level 3-state (model mismatch vs multi-cell plant) |
| **Visualization** | 6-panel (3x2) | 8-panel (4x2) with cell-level SOC, temperature, and balancing plots |
| **Metrics** | 16 standard metrics | 16 standard + 6 cell-level metrics |

**Unchanged**: EMS, MPC, EKF, MHE, price generation — all remain pack-level 3-state. Optimizer dimensions and solve times are identical to v2.

## Multi-Cell Pack Architecture

```
Pack-Level Optimizer (EMS/MPC/EKF/MHE)
  sees: x_pack = [SOC_mean, SOH_min, T_max]
        u_pack = [P_chg, P_dis, P_reg]
                    │
                    ▼
┌──────────────────────────────────┐
│         BatteryPack (BMS)        │
│                                  │
│  Power split: P_cell_i = P/N    │
│  + Balancing:  P_bal_i           │
│                                  │
│  ┌────────┐ ┌────────┐          │
│  │ Cell 1 │ │ Cell 2 │  ...     │
│  │E=51.2kW│ │E=48.8kW│          │
│  │R=2.3mΩ │ │R=2.7mΩ │          │
│  └────────┘ └────────┘          │
│                                  │
│  Aggregation:                    │
│    SOC = mean(cells)             │
│    SOH = min(cells)              │
│    T   = max(cells)              │
└──────────────────────────────────┘
```

## Per-Cell Parameter Scaling

For N cells in series, pack parameters are divided equally then perturbed:

| Parameter | Per-cell formula | Variation |
|-----------|-----------------|-----------|
| `E_nom_cell` | `E_nom_pack / N * (1 +/- spread)` | +/-3% capacity |
| `P_max_cell` | `P_max_pack / N` | No variation |
| `R_cell` | `R_pack / N * (1 +/- spread)` | +/-8% resistance |
| `C_th_cell` | `C_th_pack / N` | No variation |
| `h_cool_cell` | `h_cool_pack / N` | No variation |
| `V_cell` | `V_pack / N` | No variation |
| `alpha_cell` | `alpha_pack * (1 +/- spread)` | +/-5% degradation |
| `SOC_init_cell` | `SOC_init +/- spread` | +/-2% initial SOC |

**Physical consistency**: Series cells carry identical current. `I_cell = (P_pack/N * 1000) / (V_pack/N) = P_pack * 1000 / V_pack = I_pack`. All dynamics remain self-consistent.

## Active Cell Balancing

A proportional controller redistributes power between cells to equalise SOC:

```
P_bal_i = gain * (SOC_avg - SOC_i)

  clipped to [-max_balancing_power, +max_balancing_power] per cell
  then zero-sum enforced: bal -= mean(bal)  (energy conservation)

Application:
  P_bal_i > 0  →  added to cell's charge power
  P_bal_i < 0  →  added to cell's discharge power
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `balancing_gain` | 50.0 | Proportional gain [kW / unit SOC error] |
| `max_balancing_power` | 1.0 kW | Max per-cell balancing power |
| `balancing_enabled` | True | Enable/disable balancing |

With 4 cells and default gain, the controller reduces initial 2.4% SOC spread to ~0.2% in steady state.

## Pack-Level Aggregation

The `BatteryPack` presents the same interface as `BatteryPlant` using these aggregation rules:

| Pack state | Formula | Rationale |
|------------|---------|-----------|
| `SOC_pack` | `mean(cell SOCs)` | Most representative for pack energy content |
| `SOH_pack` | `min(cell SOHs)` | Industry-standard weakest-link; pack is only as healthy as its worst cell |
| `T_pack` | `max(cell temps)` | Thermal safety; hottest cell determines thermal limit |

## Model Mismatch

This version intentionally introduces **realistic model mismatch**:

- The optimizer (EMS/MPC) uses a single-cell 3-state model with nominal pack parameters
- The plant is a multi-cell pack with per-cell variation and balancing dynamics
- The estimators (EKF/MHE) see pack-aggregated measurements, not individual cells

This creates the kind of mismatch seen in real BESS installations where the control system uses a simplified pack model while each cell behaves slightly differently.

## Module Structure

```
v3_pack_model/
├── main.py                   # Entry point: VERSION_TAG="v3_pack_model"
├── config/
│   └── parameters.py         # All v2 params + PackParams (n_cells, spreads, balancing)
├── models/
│   └── battery_model.py      # v2 dynamics + BatteryPack class wrapping N BatteryPlants
├── ems/
│   └── economic_ems.py       # Unchanged from v2 (pack-level 3-state)
├── mpc/
│   └── tracking_mpc.py       # Unchanged from v2 (pack-level 3-state)
├── estimation/
│   ├── ekf.py                # Unchanged from v2 (pack-level 3-state)
│   └── mhe.py                # Unchanged from v2 (pack-level 3-state)
├── simulation/
│   └── simulator.py          # Accepts PackParams; logs cell-level arrays + balancing power
├── visualization/
│   └── plot_results.py       # 4x2 layout: pack + cell SOCs, cell temps, balancing, profit
├── data/
│   └── price_generator.py    # Unchanged from v2
└── stress_test.py            # 10-test stress suite with pack-specific tests
```

## Running

```bash
# From repository root
uv run python v3_pack_model/main.py

# Run stress tests
uv run python v3_pack_model/stress_test.py

# Compare with v1 and v2
uv run python -m comparison.process_results
uv run python -m comparison.compare_versions
```

## Stress Tests

10 tests covering extreme conditions + pack-specific scenarios (all PASS):

| # | Test | Key Finding |
|---|------|-------------|
| 1 | Max power cycling (100 kW, 4h) | T_max=28.3°C |
| 2 | High ambient (40°C) | Arrhenius ratio 1.47x |
| 3 | SOC boundary saturation | Clamps correctly |
| 4 | Rapid power reversals | T_max=27.4°C |
| 5 | Thermal decay to ambient | Matches analytical |
| 6 | EKF convergence from bad init | Error 0.0158 → 0.0010 |
| 7 | MPC temperature constraint | Safe fallback at T_max |
| 8 | Cell imbalance recovery (±10%) | Spread reduced 72% in 2h |
| 9 | Balancing saturation | Stable under extreme variation |
| 10 | Weakest-cell degradation | Pack SOH = min cell SOH correctly |

Results plotted to `results/v3_pack_model_stress_tests.png`.

## Results vs v1/v2

| Metric | v1_baseline | v2_thermal_model | v3_pack_model |
|--------|------------|-----------------|---------------|
| Total profit | $35.33 | $35.23 | $35.20 |
| SOH degradation | 0.972% | 1.001% | 0.261% |
| Max temperature | -- | 27.8 degC | 28.0 degC |
| Avg MPC solve | 61 ms | 114 ms | 116 ms |
| Avg estimator solve | 46 ms | 79 ms | 80 ms |

### Cell-Level Metrics (v3 only)

| Metric | Value |
|--------|-------|
| Max SOC imbalance | 2.44% (initial, before balancing settles) |
| Avg SOC imbalance | 0.18% (steady-state with balancing active) |
| SOH spread (final) | 0.017% across 4 cells |
| Max temp spread | 0.29 degC between cells |
| Balancing energy | 2.88 kWh over 24 hours |

Profit is comparable to v2 — slight reduction from balancing energy losses. Solve times are identical since optimizer dimensions are unchanged. Pack-level SOH uses weakest-link (min) aggregation, so the reported degradation differs from the lumped single-cell model in v2. Estimator RMSE increases slightly, reflecting realistic model mismatch between the pack-level optimizer model and the multi-cell plant.
