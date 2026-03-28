# BESS Optimization Platform

A production-grade Python platform for **optimal scheduling, real-time dispatch, and state estimation** of grid-connected battery energy storage systems (BESS). Optimizes **energy arbitrage** and **frequency regulation** (capacity bidding + real-time delivery) while actively managing battery degradation.

Built with the same hierarchical control architecture deployed in commercial utility-scale installations. Evolves through **incremental, gated upgrades** from baseline to industry-grade digital twin.

## What It Solves

Every BESS operator faces the same question: *Given uncertain market prices and stochastic grid signals, how should I charge, discharge, and bid regulation capacity over the next 24 hours — and how do I execute that plan in real time while protecting the battery?*

This platform answers it end-to-end:

- **Economic scheduling** — stochastic 24-hour optimization across multiple price scenarios, jointly optimizing arbitrage revenue, regulation capacity payments, and degradation costs
- **Real-time dispatch** — nonlinear model predictive control (2-state SOC/T prediction with SOH frozen) tracking the economic schedule while enforcing SOC, thermal, and voltage constraints
- **Regulation delivery** — feedforward controller tracking stochastic grid activation signals at 4s resolution, with SOC safety clamping and recovery bias
- **State estimation** — EKF reconstructing battery internals (SOC, SOH, temperature, RC voltages) from noisy sensors
- **Multi-cell pack modeling** — per-cell parameter variation with active balancing, weakest-link SOH tracking
- **High-fidelity plant** — 2RC equivalent circuit with NMC OCV polynomial, Arrhenius degradation, and thermal dynamics

## Architecture

```
 Stochastic Prices ──► EMS (3600s)  ──► MPC (60s) ──► Reg Ctrl (4s) ──► Plant (4s)
                       24h horizon      2-state        feedforward       5-state
                       N scenarios      SOC + T        SOC clamp         2RC pack
                                        ◄──── EKF (60s) ◄────────────   noisy meas
```

**Market model:** PJM RegD-style centrally dispatched activation signals. Energy and regulation prices are scenario-based with 5 stochastic scenarios.

## Versioned Upgrades

Each version adds one major capability, passes a **4-stage gate** (validation, evaluation, comparison, stress testing), and is frozen before the next begins. See each version's `README.md` for details and `backlog.md` for gate reports.

| Version | What It Adds | Key Metrics | Status |
|---------|-------------|-------------|--------|
| **v1** Baseline | EMS + MPC + EKF, energy arbitrage + regulation capacity | $21.36 profit, 65ms MPC | Frozen |
| **v2** Thermal | Temperature state, Arrhenius degradation coupling | $21.21 profit, 113ms MPC | Frozen |
| **v3** Pack | 4-cell pack, active cell balancing | $21.25 profit, 146ms MPC | Frozen |
| **v4** Electrical RC | 2RC circuit, NMC OCV, voltage measurement | $21.73 profit, 188ms MPC | Frozen |
| **v5** Regulation | Activation delivery, feedforward controller, MPC simplification | $22.66 profit, 95.3% delivery, 154ms MPC | In development |

## Quick Start

```bash
uv sync                                       # install dependencies
uv run python v5_regulation_activation/main.py # run latest version (v5)
uv run python v5_regulation_activation/stress_test.py  # 20 stress tests
```

Each version is independently runnable. Results go to `results/`.

## Technical Stack

CasADi + IPOPT for nonlinear optimization, NumPy for numerics, Matplotlib for visualization. All optimization models use automatic differentiation and warm-started interior-point solving.

## Roadmap

| Upcoming | Description |
|----------|------------|
| v6 | Unscented Kalman Filter (UKF) |
| v7 | Online parameter estimation |
| v8 | Real-time NMPC with ACADOS |
| v9 | Degradation-aware MPC |
| v10-v14 | Uncertainty, delays, multi-battery, inverter, market bidding |

---

*Each version contains its own `README.md` with implementation details. See `backlog.md` for gate review reports.*
