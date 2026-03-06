# Hierarchical Nonlinear BESS Control Platform

A complete Python platform for **optimal scheduling, real-time dispatch, and state estimation** of a grid-connected Battery Energy Storage System (BESS). It simultaneously participates in **energy arbitrage** and **frequency regulation** markets while tracking battery degradation — all in closed-loop, running at three distinct time scales.

This is the same control architecture deployed in commercial utility-scale BESS installations, implemented end-to-end using CasADi and IPOPT. The platform is evolving through **incremental, versioned upgrades** toward an industry-grade digital twin — each version is independently runnable with quantitative metrics and version-to-version comparison.

---

## What This Platform Does

The platform answers the core question every BESS operator faces: **"Given uncertain market prices, how should I charge, discharge, and bid regulation capacity over the next 24 hours — and how do I execute that plan in real time while protecting the battery?"**

It does this through a two-layer control hierarchy:

| Layer | Runs Every | Role |
|-------|-----------|------|
| **Energy Management System (EMS)** | 1 hour | Decides *what to do*: the economically optimal charge, discharge, and regulation schedule over a 24-hour rolling horizon, considering 5 stochastic price scenarios |
| **Model Predictive Controller (MPC)** | 1 minute | Decides *how to do it*: tracks the EMS plan in real time, compensating for model mismatch, measurement noise, and constraint violations |

Two state estimators run alongside the MPC to reconstruct the battery's internal state from noisy sensor readings:

| Estimator | Method | Purpose |
|-----------|--------|---------|
| **EKF** (Extended Kalman Filter) | Recursive Bayesian | Fast, lightweight SOC/SOH estimation used as feedback for MPC |
| **MHE** (Moving Horizon Estimation) | Optimization-based | Higher-accuracy SOC/SOH estimation over a 30-minute sliding window |

The physical battery plant is simulated at **1-second resolution** with a nonlinear model. The base version uses a 2-state model (SOC + SOH); the thermal upgrade extends this to 3 states (SOC + SOH + Temperature) with Arrhenius-coupled degradation. The pack model further extends the plant to a multi-cell battery pack with per-cell parameter variation and active cell balancing.

---

## Architecture

```
                  Stochastic Price Scenarios
                  (5 scenarios x 48 hours)
                          │
                          ▼
    ┌──────────────────────────────────────────────┐
    │         Economic EMS  (every 3 600 s)        │
    │                                              │
    │  • 24-hour rolling horizon                   │
    │  • Scenario-based stochastic NLP             │
    │  • Energy arbitrage + regulation revenue     │
    │  • Battery degradation penalty               │
    │  • Non-anticipativity constraints             │
    │                                              │
    │  Output: P_chg_ref, P_dis_ref, P_reg_ref,   │
    │          SOC_ref, SOH_ref  (hourly)          │
    └───────────────────┬──────────────────────────┘
                        │  zero-order hold (power)
                        │  linear interpolation (state)
                        ▼
    ┌──────────────────────────────────────────────┐
    │       Tracking MPC  (every 60 s)             │
    │                                              │
    │  • 60-step prediction horizon (1 hour)       │
    │  • 10-step control horizon (blocking)        │
    │  • Tracks EMS references in closed loop      │
    │  • Soft SOC constraints for feasibility      │
    │  • Warm-started IPOPT solves (~5 iterations) │
    │                                              │
    │  Output: P_chg, P_dis, P_reg  (applied now)  │
    └───────┬───────────────────────┬──────────────┘
            │                       │
            ▼                       ▼
    ┌───────────────┐     ┌─────────────────────┐
    │   EKF         │     │   MHE               │
    │  (every 60 s) │     │  (every 60 s)       │
    │               │     │                     │
    │  SOC + SOH    │     │  SOC + SOH          │
    │  estimate     │     │  estimate           │
    │  (recursive)  │     │  (30-min window)    │
    └───────┬───────┘     └─────────┬───────────┘
            │  state feedback       │  comparison
            ▼                       ▼
    ┌──────────────────────────────────────────────┐
    │       Battery Plant  (every 1 s)             │
    │                                              │
    │  • 2-state nonlinear model (SOC + SOH)       │
    │  • RK4 integration at 1-second resolution    │
    │  • SOC-limited power saturation              │
    │  • Noisy SOC measurement (SOH unobserved)    │
    └──────────────────────────────────────────────┘
```

### Why Two Layers?

A single optimizer cannot efficiently handle both economic planning and real-time execution:

- The **EMS** needs a long horizon (24 hours) and considers multiple price scenarios to hedge against forecast uncertainty. This is a large nonlinear program that takes seconds to solve — too slow for real-time control, but it only needs to run once per hour.

- The **MPC** needs fast execution (sub-second) and high-fidelity state feedback to reject disturbances and track the economic plan precisely. It runs every minute with a shorter horizon, using the EMS references as targets.

This separation is standard in the BESS industry. The EMS is the "trader" that decides the dispatch schedule. The MPC is the "operator" that executes it safely.

### Why Two Estimators?

SOH (State of Health) is never directly measured — it must be inferred from the slow drift in SOC dynamics over time. This makes it extremely difficult to estimate:

- The **EKF** is fast and recursive but can struggle with the very different time scales of SOC (changes in minutes) and SOH (changes over months). It provides the primary feedback signal for the MPC.

- The **MHE** solves an optimization problem over a sliding window of past data, which is more robust to nonlinearity and can enforce physical bounds directly. It serves as a cross-check and provides higher-quality estimates for offline analysis.

Both estimators are initialized identically and receive the same measurements, so their estimates can be compared to assess estimation quality.

---

## Revenue Streams

The platform optimizes across two simultaneous revenue streams:

### Energy Arbitrage
Buy low, sell high. The battery charges when energy prices are low (typically overnight and midday) and discharges when prices peak (morning and evening demand peaks). Revenue depends on the price spread and round-trip efficiency losses.

### Frequency Regulation
Reserve capacity for the grid operator to call upon for frequency regulation. The battery earns a capacity payment ($/kW/h) for making power available, regardless of whether it is actually dispatched. Regulation commitments reduce the power available for arbitrage, creating a trade-off that the EMS optimizes.

### Degradation Cost
Every kWh cycled through the battery degrades its cells. The optimizer includes a degradation penalty that represents the replacement cost of lost battery life. This prevents the optimizer from over-cycling the battery for marginal revenue.

---

## Multi-Rate Timing

The simulation coordinates four different time scales in a single loop:

| Component | Time Step | Steps in 24 h | Purpose |
|-----------|-----------|---------------|---------|
| Plant | 1 s | 86,400 | High-fidelity battery physics |
| EKF | 60 s | 1,440 | State estimation (recursive) |
| MHE | 60 s | 1,440 | State estimation (optimization) |
| MPC | 60 s | 1,440 | Real-time dispatch optimization |
| EMS | 3,600 s | 24 | Economic scheduling |

The inner loop (1 s) applies the most recent MPC command to the plant model. Every 60 seconds, the estimators process a new SOC measurement and the MPC recomputes its dispatch. Every hour, the EMS re-solves the economic schedule with updated state estimates and fresh price scenarios.

---

## Project Structure

```
battery_optimization_platform/
├── pyproject.toml                 # Dependencies: casadi, numpy, matplotlib, scipy
│
├── v1_baseline/                   # Version 1: frozen baseline (2-state, timing instrumented)
│   ├── main.py                    #   Independent entry point
│   ├── config/parameters.py       #   All tunable parameters (frozen dataclasses)
│   ├── models/battery_model.py    #   2-state nonlinear model (CasADi + numpy plant)
│   ├── ems/economic_ems.py        #   Stochastic energy management system
│   ├── mpc/tracking_mpc.py        #   Nonlinear tracking MPC with control blocking
│   ├── estimation/{ekf,mhe}.py    #   EKF + MHE state estimators
│   ├── simulation/simulator.py    #   Multi-rate simulation coordinator
│   ├── visualization/plot_results.py
│   └── data/price_generator.py    #   Stochastic price scenario generator
│
├── v2_thermal_model/              # Version 2: 3-state thermal model upgrade
│   ├── main.py                    #   Adds temperature state, Arrhenius degradation
│   ├── stress_test.py             #   8-test stress suite with plots
│   └── ...                        #   Same module structure as v1
│
├── v3_pack_model/                 # Version 3: multi-cell pack with active balancing
│   ├── main.py                    #   N-cell BatteryPack wrapping BatteryPlant instances
│   ├── stress_test.py             #   10-test stress suite with pack-specific tests
│   └── ...                        #   Same module structure as v2 + PackParams
│
├── comparison/                    # Cross-version comparison infrastructure
│   ├── metrics.py                 #   Metric computation from simulation results
│   ├── process_results.py         #   Load .npz files and produce metrics JSON
│   └── compare_versions.py        #   Side-by-side table, CSV, and bar chart
│
├── backlog.md                     # Gate review reports (validation, evaluation, stress tests)
│
└── results/                       # Simulation outputs (auto-generated)
    ├── v*_results.npz             #   Raw time series per version
    ├── v*_metrics.json            #   Computed metrics per version
    ├── v*_stress_tests.png        #   Stress test visualizations
    ├── version_comparison.csv     #   All versions side-by-side
    └── version_comparison.png     #   Comparison bar charts
```

Each version is fully self-contained with its own `README.md` documenting the mathematical formulations and changes from its predecessor.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
cd battery_optimization_platform
uv sync
```

Each version is independently runnable from the repository root:

```bash
# Run the frozen baseline
uv run python v1_baseline/main.py

# Run the thermal model upgrade
uv run python v2_thermal_model/main.py

# Run the pack model (multi-cell with balancing)
uv run python v3_pack_model/main.py
```

Results (`.npz` time series and `.png` plots) are saved to `results/`.

### Comparing Versions

After running one or more versions, generate a side-by-side comparison:

```bash
# Compute metrics from saved .npz files
uv run python -m comparison.process_results

# Print comparison table, save CSV and bar chart
uv run python -m comparison.compare_versions
```

### Expected Output

Each simulation takes approximately 5-10 minutes (86,400 plant steps with 1,440 optimization solves). Console output shows:

```
==============================================================
  BESS HIERARCHICAL CONTROL PLATFORM  [v2_thermal_model]
==============================================================
  Battery:    200 kWh / 100 kW
  SOC range:  [0.10, 0.90]
  ...
==============================================================
  RESULTS SUMMARY  [v2_thermal_model]
==============================================================
  Total profit:     $35.23
  SOH degradation:  1.0009%
  Final SOC:        0.1361
  Final Temp:       25.42 degC
  Avg MPC solve:    114.3 ms
==============================================================
```

---

## Output Visualization

The platform generates a 6-panel figure showing:

| Panel | What It Shows | Why It Matters |
|-------|--------------|----------------|
| **SOC Estimation** | True SOC vs EKF and MHE estimates | Validates that estimators track the real battery state through noisy measurements |
| **SOH Estimation** | True SOH vs EKF and MHE estimates | Shows how well the system tracks degradation — the hardest estimation problem since SOH is never directly measured |
| **Power Dispatch** | MPC applied power overlaid on EMS reference | Lets you visually verify that the MPC faithfully executes the economic plan; thick transparent lines are the EMS targets, thin solid lines are what the MPC actually applied |
| **Market Prices** | Energy and regulation prices over 24 hours | Context for understanding dispatch decisions — you should see charging during low prices and discharging during peaks |
| **Revenue Breakdown** | Cumulative energy revenue, regulation revenue, degradation cost, and net profit | Shows how value is created and where costs come from |
| **Battery Degradation** | SOH loss over 24 hours | Quantifies the physical cost of the dispatch strategy |

---

## Configuration

All parameters are organized into frozen dataclasses in each version's `config/parameters.py`. Key settings:

### Battery

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E_nom_kwh` | 200 | Nameplate energy capacity [kWh] |
| `P_max_kw` | 100 | Maximum charge/discharge power [kW] |
| `SOC_min` / `SOC_max` | 0.10 / 0.90 | Operational SOC window |
| `eta_charge` / `eta_discharge` | 0.95 / 0.95 | One-way efficiencies |
| `alpha_deg` | 2.78e-9 | Degradation rate [1/(kW·s)] |

### Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ems` | 24 | EMS planning horizon [hours] |
| `n_scenarios` | 5 | Number of stochastic price scenarios |
| `N_mpc` | 60 | MPC prediction horizon [minutes] |
| `Nc_mpc` | 10 | MPC control horizon [minutes] |
| `N_mhe` | 30 | MHE estimation window [minutes] |

### Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt_ems` | 3,600 s | EMS re-solve period |
| `dt_mpc` | 60 s | MPC/estimator period |
| `dt_sim` | 1 s | Plant simulation step |
| `sim_hours` | 24 h | Total simulation duration |

---

## Solver Technology

All optimization problems (EMS, MPC, MHE) are formulated using **CasADi's Opti stack** and solved with the **IPOPT** interior-point method (with the MUMPS linear algebra backend).

Key solver features used:
- **Automatic differentiation**: CasADi computes exact gradients and Hessians of the battery dynamics, avoiding finite differences entirely
- **Warm-starting**: The MPC caches and shifts the previous solution to initialize the next solve, reducing iterations from ~50 (cold) to ~5 (warm) in steady state
- **Soft constraints**: SOC bounds use slack variables with heavy quadratic penalties, guaranteeing that the MPC always finds a feasible solution even under model mismatch
- **RK4 integration**: All dynamics use 4th-order Runge-Kutta integration, shared as a single CasADi function across all modules to ensure consistency

---

## Design Decisions

### Split Charge/Discharge Variables
Charge and discharge power are separate non-negative decision variables rather than a single signed power variable. This avoids piecewise efficiency modeling (charge and discharge have different efficiencies) and eliminates the need for binary variables or big-M constraints. Simultaneous charge and discharge is never optimal and is automatically avoided by the solver.

### Control Horizon Blocking
The MPC uses 10 free control steps followed by 50 steps where the control is held constant at the last free value. This dramatically reduces the number of decision variables (from 180 to 30 power variables) while maintaining a long prediction horizon for stability. The first control action is applied, and the problem is re-solved at the next time step.

### Non-Anticipativity Constraints
The EMS enforces that the first-stage decisions (current hour) must be identical across all price scenarios. This is the standard stochastic programming approach — you cannot make different decisions for different futures before observing which future materializes.

### Degradation as a Soft Penalty
Battery degradation is included as a cost term rather than a hard constraint on SOH. This lets the optimizer make principled trade-offs: cycle the battery more aggressively when revenue is high, and conserve it when margins are thin.

---

## Version Upgrades

The platform evolves through incremental, independently runnable upgrades. Each version adds one major capability while preserving everything from previous versions.

| Version | Name | Key Addition | Difficulty |
|---------|------|-------------|------------|
| **v1** | `v1_baseline` | Frozen copy of base platform with timing instrumentation | — |
| **v2** | `v2_thermal_model` | 3rd state (temperature), lumped-parameter thermal model, Arrhenius degradation coupling, temperature constraints in MPC/EMS | LOW |
| **v3** | `v3_pack_model` | Multi-cell battery pack (N cells in series), per-cell parameter variation, active proportional cell balancing, 8-panel visualization with cell-level plots | LOW |
| v4 | `v4_electrical_rc_model` | 2RC equivalent circuit model, voltage states (V_rc1, V_rc2), terminal voltage measurement, OCV(SOC) lookup | LOW-MEDIUM |
| v5 | `v5_ukf_estimator` | Unscented Kalman Filter replacing EKF, sigma points, unscented transform | MEDIUM |
| v6 | `v6_parameter_estimation` | Joint state + parameter estimation via MHE, online R_internal/capacity/efficiency identification | MEDIUM |
| v7 | `v7_acados_nmpc` | Real-time NMPC using ACADOS, multiple shooting, Real-Time Iteration (RTI) | HIGH |
| v8 | `v8_degradation_aware_mpc` | SOH in MPC state, degradation cost in MPC objective, SOH floor constraint | HIGH |
| v9 | `v9_disturbance_forecast_uncertainty` | Stochastic price forecasts in MPC, scenario-based MPC, robustness comparison | HIGH |
| v10 | `v10_measurement_delay` | Measurement delay, actuator delay, random communication latency | HIGH |
| v11 | `v11_multi_battery_system` | Multiple batteries with local MPC and central EMS coordination | VERY HIGH |
| v12 | `v12_grid_inverter_model` | Grid-connected inverter dynamics (id, iq, Vdc), power converter model, reactive power | VERY HIGH |
| v13 | `v13_market_bidding` | Day-ahead bidding, reserve bidding, market participation optimization | VERY HIGH |

### Version Comparison (v1 → v2 → v3)

| Metric | v1_baseline | v2_thermal_model | v3_pack_model |
|--------|------------|-----------------|---------------|
| Total profit | $35.33 | $35.23 | $35.20 |
| SOH degradation | 0.972% | 1.001% | 0.261% |
| Max temperature | — | 27.8 °C | 28.0 °C |
| Avg MPC solve time | 61 ms | 114 ms | 116 ms |
| Avg estimator solve time | 46 ms | 79 ms | 80 ms |
| Max SOC imbalance | — | — | 2.44% |
| Avg SOC imbalance | — | — | 0.18% |
| SOH spread (final) | — | — | 0.017% |
| Balancing energy | — | — | 2.88 kWh |

**v1 → v2**: The thermal model introduces ~3% more degradation via Arrhenius coupling at elevated temperatures, with a proportional increase in solve times from the additional state dimension.

**v2 → v3**: The pack model wraps 4 cells in series with manufacturing variation (capacity ±3%, resistance ±8%, degradation ±5%). EMS/MPC/EKF/MHE remain pack-level (3-state) — they see aggregated pack states while the BMS handles cell balancing transparently. Active proportional balancing reduces initial 2.4% SOC spread to ~0.2% steady-state. Solve times are unchanged since optimizer dimensions are identical to v2. Pack-level SOH reports the weakest cell (min), reflecting industry-standard weakest-link aggregation.

---

## Typical Results

For a 200 kWh / 100 kW battery over 24 hours with synthetic price data:

- **Net profit**: $30-40 (energy arbitrage + regulation revenue - degradation cost)
- **SOH degradation**: ~0.5-1.0% (accelerated for demo visibility; real batteries degrade ~0.01% per day)
- **EKF accuracy**: SOC error < 1%, SOH error < 0.1%
- **MHE accuracy**: SOC error < 1%, SOH converges within ~2 hours
- **MPC tracking**: Power commands closely follow EMS references within 1 minute
- **Solver reliability**: Zero solver failures across all 1,440 MPC solves and 24 EMS solves

---

## Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Optimization modeling | CasADi | >= 3.6.0 |
| NLP solver | IPOPT (bundled with CasADi) | 3.14+ |
| Linear algebra backend | MUMPS (bundled with IPOPT) | — |
| Numerics | NumPy | >= 1.24.0 |
| Estimator covariance | SciPy | >= 1.10.0 |
| Visualization | Matplotlib | >= 3.7.0 |

---

## License

MIT
