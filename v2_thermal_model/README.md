# v2_thermal_model — 3-State Thermal Model Upgrade

Extends the v1 baseline from 2 states (SOC, SOH) to **3 states (SOC, SOH, Temperature)** by adding lumped-parameter thermal dynamics with Arrhenius-coupled degradation. Temperature now directly accelerates battery aging, and all optimizers (EMS, MPC, EKF, MHE) are upgraded to handle the third state dimension.

## What Changed from v1

| Component | v1_baseline | v2_thermal_model |
|-----------|------------|-----------------|
| **State vector** | `x = [SOC, SOH]` (2-state) | `x = [SOC, SOH, T]` (3-state) |
| **Degradation** | `dSOH/dt = -alpha * P_total` | `dSOH/dt = -alpha * kappa(T) * P_total` |
| **Thermal dynamics** | None | `dT/dt = (I^2*R - h*(T-T_amb)) / C_th` |
| **Measurement** | `y = [SOC_meas]` (1 output) | `y = [SOC_meas, T_meas]` (2 outputs) |
| **MPC constraints** | SOC bounds only | SOC bounds + temperature ceiling (45 degC) |
| **EMS planning** | SOC + degradation cost | SOC + degradation cost + thermal awareness |
| **EKF/MHE** | 2-state estimation | 3-state estimation with temperature tuning |

## Thermal Dynamics

The battery temperature evolves according to a lumped-parameter energy balance:

```
dT/dt = (Q_joule - Q_cool) / C_thermal   [degC/s]

where:
  Q_joule = I^2 * R_internal              [W]     (Joule heating)
  Q_cool  = h_cool * (T - T_ambient)      [W]     (convective cooling)
  I       = P_total * 1000 / V_nominal     [A]     (pack current from total power)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `R_internal` | 0.010 Ohm | Internal resistance |
| `C_thermal` | 150,000 J/K | Thermal mass |
| `h_cool` | 50 W/K | Cooling coefficient |
| `T_ambient` | 25.0 degC | Ambient temperature |
| `V_nominal` | 800 V | Nominal pack voltage |

**Thermal time constant**: `tau = C_thermal / h_cool = 3,000 s = 50 min`

At full power (100 kW): `I = 125 A`, `Q_joule = 156 W`, steady-state `dT = 3.1 degC`.

## Arrhenius Degradation Coupling

Temperature accelerates degradation through an Arrhenius factor:

```
kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))

dSOH/dt = -alpha_deg * kappa(T) * (P_chg + P_dis + |P_reg|)
```

| Temperature | kappa | Degradation effect |
|-------------|-------|--------------------|
| 25 degC (ref) | 1.00 | Baseline rate |
| 35 degC | 1.30 | 30% faster |
| 45 degC | 1.66 | 66% faster |

With `E_a = 20,000 J/mol`, the coupling is moderate — enough to be meaningful without dominating the economics.

## Complete ODE System

```
dSOC/dt = (eta_c * P_chg - P_dis / eta_d) / (SOH * E_nom * 3600)     [1/s]
dSOH/dt = -alpha_deg * kappa(T) * (P_chg + P_dis + |P_reg|)           [1/s]
dT/dt   = (I^2 * R_internal - h_cool * (T - T_ambient)) / C_thermal   [degC/s]
```

All three equations are implemented as a single CasADi symbolic function shared by EMS, MPC, EKF, and MHE, with RK4 integration at the respective time steps.

## Module Structure

```
v2_thermal_model/
├── main.py                   # Entry point: VERSION_TAG="v2_thermal_model"
├── config/
│   └── parameters.py         # Adds ThermalParams; extends EKF/MHE/MPC params
├── models/
│   └── battery_model.py      # 3-state CasADi dynamics + BatteryPlant with thermal ODE
├── ems/
│   └── economic_ems.py       # 3-state stochastic EMS with temperature references
├── mpc/
│   └── tracking_mpc.py       # 3-state tracking MPC with temperature ceiling constraint
├── estimation/
│   ├── ekf.py                # 3-state EKF (3x3 covariance, 2 measurements)
│   └── mhe.py                # 3-state MHE with temperature arrival/stage costs
├── simulation/
│   └── simulator.py          # Multi-rate coordinator logging temperature traces
├── visualization/
│   └── plot_results.py       # 6-panel layout including temperature panel
├── data/
│   └── price_generator.py    # Unchanged from v1
└── stress_test.py            # 8-test stress suite with plots
```

## Running

```bash
# From repository root
uv run python v2_thermal_model/main.py

# Run stress tests
uv run python v2_thermal_model/stress_test.py

# Compare with v1
uv run python -m comparison.process_results
uv run python -m comparison.compare_versions
```

## Stress Tests

8 tests covering extreme conditions (all PASS):

| # | Test | Key Finding |
|---|------|-------------|
| 1 | Max power cycling (100 kW, 4h) | T_max=28.1°C |
| 2 | High ambient (40°C) | Arrhenius ratio 1.47x |
| 3 | Low ambient (0°C) | Temperature stable |
| 4 | SOC boundary saturation | Clamps correctly |
| 5 | Rapid power reversals | T_max=27.2°C |
| 6 | Thermal decay to ambient | Matches analytical |
| 7 | EKF convergence from bad init | Error 0.0095 → 0.0012 |
| 8 | MPC temperature constraint | Safe fallback at T_max |

Results plotted to `results/v2_thermal_model_stress_tests.png`.

## Results vs v1

| Metric | v1_baseline | v2_thermal_model |
|--------|------------|-----------------|
| Total profit | $35.33 | $35.23 |
| SOH degradation | 0.972% | 1.001% |
| Max temperature | -- | 27.8 degC |
| Avg MPC solve | 61 ms | 114 ms |
| Avg estimator solve | 46 ms | 79 ms |

The thermal model introduces ~3% more degradation via Arrhenius coupling. Solve times roughly double due to the additional state dimension (3x3 instead of 2x2 in Jacobians/Hessians). Profit is marginally lower as the optimizer now accounts for thermal degradation costs.
