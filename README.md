# Hierarchical Nonlinear BESS Control Platform

A complete Python platform for **optimal scheduling, real-time dispatch, and state estimation** of a grid-connected Battery Energy Storage System (BESS). It simultaneously participates in **energy arbitrage** and **frequency regulation** markets while tracking battery degradation — all in closed-loop, running at three distinct time scales.

This is the same control architecture deployed in commercial utility-scale BESS installations, implemented end-to-end in ~1,500 lines of Python using CasADi and IPOPT.

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

The physical battery plant is simulated at **1-second resolution** with a nonlinear 2-state model that captures both energy throughput and calendar/cycling degradation.

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
├── main.py                      # Entry point — runs the full pipeline
├── pyproject.toml               # Dependencies: casadi, numpy, matplotlib, scipy
├── config/
│   └── parameters.py            # All tunable parameters (6 frozen dataclasses)
├── models/
│   └── battery_model.py         # 2-state nonlinear model (CasADi + numpy plant)
├── ems/
│   └── economic_ems.py          # Stochastic energy management system
├── mpc/
│   └── tracking_mpc.py          # Nonlinear tracking MPC with control blocking
├── estimation/
│   ├── ekf.py                   # Extended Kalman Filter
│   └── mhe.py                   # Moving Horizon Estimation
├── simulation/
│   └── simulator.py             # Multi-rate simulation coordinator
├── data/
│   ├── price_generator.py       # Stochastic price scenario generator
│   └── prices.csv               # Historical price data (backward compatible)
└── visualization/
    └── plot_results.py          # 6-panel result figure
```

Each subfolder contains its own `README.md` with full mathematical formulations.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
cd battery_optimization_platform
uv sync
uv run python main.py
```

Or with pip:

```bash
pip install casadi numpy matplotlib scipy
python main.py
```

### Expected Output

The simulation takes approximately 5-10 minutes (86,400 plant steps with 1,440 optimization solves). Console output shows:

```
==============================================================
  BESS HIERARCHICAL CONTROL PLATFORM
==============================================================
  Battery:    200 kWh / 100 kW
  SOC range:  [0.10, 0.90]
  ...
EMS solve at t=0 s (hour 0), SOC=0.500, SOH=1.000000
MPC step 0: u=[...] SOC_ekf=0.500 SOH_ekf=1.000000
...
==============================================================
  RESULTS SUMMARY
==============================================================
  Total profit:     $XX.XX
  SOH degradation:  X.XXXX%
  Final SOC:        0.XXXX
==============================================================
```

A `results.png` figure is saved to the project root (see below).

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

All parameters are organized into frozen dataclasses in `config/parameters.py`. Key settings:

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
