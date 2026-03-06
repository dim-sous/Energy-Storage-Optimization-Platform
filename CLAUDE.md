## Current State
- Completed: v1, v2, v3 (4-stage gate passed: validation, evaluation, comparison, stress testing)
- Next: v4
- Frozen (do not modify): v1, v2, v3
- Gate reports: see backlog.md

## If context is unclear Re-read this file top to bottom. Ask me to confirm the current version.

You are assisting in the development of an industry-grade battery digital twin, control, and optimization platform in Python.
Version 1 (baseline) is already implemented. All further development follows an incremental engineering process toward a robust, production-grade system.
────────────────────────────────────────
BEHAVIORAL PRINCIPLES
────────────────────────────────────────
1. PROPOSE BEFORE IMPLEMENTING
Before writing any code for a new upgrade, output a short proposal:
What does this upgrade add and why does it matter?
What are the expected benefits and risks?
Any dependencies on prior upgrades or known implementation pitfalls?
For non-trivial changes, wait for confirmation. If you see a better approach or a meaningful tradeoff, surface it first.
2. ONE UPGRADE AT A TIME
Each upgrade creates a new versioned folder. Never merge multiple upgrades into a single step. Each version must be independently runnable and its changes reversible.
3. THREE-STAGE GATE — mandatory before moving to the next upgrade
A. Validation — confirm the implementation is physically and mathematically consistent
B. Evaluation — compute and log the standard metrics (see below)
C. Comparison — generate plots comparing this version to the previous one and to v1
4. PRODUCTION-GRADE CODE
Use modular architecture, type hints, docstrings, configuration files, logging, and exception handling throughout. Use explicit physical units everywhere. Prefer clarity and maintainability over cleverness.
5. AVOID OVER-ENGINEERING
Prefer the smallest implementation that meaningfully advances the system. If a proposed approach significantly increases complexity or compute time, explain the tradeoff and suggest a lighter alternative.
6. COMPUTATIONAL AWARENESS
For each upgrade, note the effect on simulation and solver time. Ensure the system remains tractable. Flag any upgrade that risks making real-time operation infeasible.
────────────────────────────────────────
VERSIONING STRUCTURE
────────────────────────────────────────
battery_optimization_platform/
├── v1_baseline/
├── v2_thermal_model/
├── v3_pack_model/
├── v4_electrical_rc_model/
├── v5_ukf_estimator/
├── v6_parameter_estimation/
├── v7_acados_nmpc/
├── v8_degradation_aware_mpc/
├── v9_disturbance_forecast_uncertainty/
├── v10_measurement_delay/
├── v11_multi_battery_system/
├── v12_grid_inverter_model/
├── v13_market_bidding/
└── results/version_comparison.csv
────────────────────────────────────────
STANDARD METRICS — compute and store after every version
────────────────────────────────────────
Control: RMSE_SOC_tracking, RMSE_power_tracking
Estimation: RMSE_SOC_estimation, RMSE_SOH_estimation
Economic: total_profit, total_degradation_cost
Computational: avg_mpc_solve_time, max_mpc_solve_time, estimator_solve_time
Store in results/version_comparison.csv. Generate comparison plots for: SOC, SOH, temperature, voltage, power, profit, solver time.
────────────────────────────────────────
UPGRADE BACKLOG — implement in priority order
────────────────────────────────────────
v2 — Thermal Model (Low difficulty / Very High importance)
Add temperature state to x = [SOC, SOH, T]. Thermal dynamics: dT/dt = (I²·R_int − k_cool·(T − T_amb)) / C_th. Add temperature constraints to MPC. Couple thermal dynamics to degradation. Update EKF and MHE.
v3 — Multi-Cell Pack Model (Low-Medium / High)
Multi-cell battery pack with active cell balancing. Model cell-to-cell variation, balancing logic, and pack-level constraints.
v4 — 2RC Electrical Model (Low-Medium / Very High)
Replace simple model with 2RC equivalent circuit. States: V_rc1, V_rc2. Terminal voltage: V = OCV(SOC) − V_rc1 − V_rc2 − I·R_int. Add voltage measurement. Update EKF, MHE, MPC constraints.
v5 — Unscented Kalman Filter (Medium / Very High)
Replace EKF with UKF. Implement sigma points, unscented transform, prediction, correction, covariance update. Compare UKF vs EKF estimation accuracy.
v6 — Joint State and Parameter Estimation (Medium / Very High)
Estimate R_internal, capacity, and efficiency online. Augment state: x_aug = [x, R_int, Capacity]. Implement via MHE.
v7 — ACADOS NMPC (High / Extremely High)
Replace CasADi/IPOPT MPC with ACADOS. Implement multiple shooting, Real-Time Iteration (RTI), control blocking. Measure and compare solve time and control performance.
v8 — Degradation-Aware MPC (High / Extremely High)
Add SOH to MPC state. Add degradation cost to objective. Enforce SOH ≥ SOH_min constraint. Characterize profit vs degradation tradeoff.
v9 — Disturbance Forecast Uncertainty (High / High)
Add stochastic price forecasts. Implement scenario-based MPC. Compare profit and robustness against v8.
v10 — Measurement and Communication Delays (High / High)
Add measurement delay, actuator delay, random latency. Update estimator and MPC to compensate.
v11 — Multi-Battery System (Very High / High)
Simulate multiple batteries, each with local MPC, coordinated by a central EMS.
v12 — Grid-Connected Inverter Model (Very High / High)
Add inverter dynamics: states id, iq, Vdc. Implement power converter model, power limits, reactive power control.
v13 — Market Bidding Optimization (Very High / Medium)
Add day-ahead bidding, reserve bidding, and market participation optimization.
────────────────────────────────────────
The goal is a well-tested, physically realistic, and maintainable platform that improves measurably at each step — not one that is maximally complex.
────────────────────────────────────────