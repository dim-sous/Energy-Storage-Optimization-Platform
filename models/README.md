# models/ — Nonlinear 2-State Battery Model

This module defines the battery dynamics shared by every component in the platform: the EMS, MPC, EKF, and MHE all use the same underlying model, ensuring consistency across the hierarchy.

---

## State Space Formulation

**State vector:**

```
x = [SOC, SOH]ᵀ ∈ ℝ²
```

- **SOC** (State of Charge): fraction of available energy, SOC ∈ [0, 1]
- **SOH** (State of Health): fraction of remaining capacity, SOH ∈ [0, 1]

**Input vector:**

```
u = [P_chg, P_dis, P_reg]ᵀ ∈ ℝ³,  all ≥ 0
```

- **P_chg**: charging power [kW]
- **P_dis**: discharging power [kW]
- **P_reg**: frequency regulation capacity [kW]

All inputs are non-negative. Charge and discharge are modeled as separate variables to handle asymmetric efficiencies without introducing binary variables or piecewise functions.

**Measurement:**

```
y = SOC + v,    v ~ N(0, σ²),   σ = 0.01
```

Only SOC is measured (via coulomb counting or voltage-based estimation in practice). SOH is a **hidden state** that must be inferred from the slow drift in SOC dynamics.

---

## Continuous-Time Dynamics

```
dSOC/dt = (η_c · P_chg − P_dis / η_d) / (SOH · E_nom · 3600)

dSOH/dt = −α_deg · (P_chg + P_dis + |P_reg|)
```

### SOC Equation

The SOC derivative represents the net energy flow into the battery, normalized by the effective capacity:

- `η_c · P_chg`: energy actually stored (after charging losses)
- `P_dis / η_d`: energy extracted from the battery (before discharging losses)
- `SOH · E_nom · 3600`: effective capacity in kW·s. The factor 3600 converts E_nom from kWh to kW·s. The SOH factor models capacity fade — as the battery degrades, the same kW of power causes a larger SOC change.

### SOH Equation

The SOH derivative is a simple throughput-based degradation model:

- The degradation rate is proportional to the total power throughput: charge + discharge + regulation
- `α_deg` has units [1/(kW·s)] — it represents SOH lost per unit of energy throughput
- Regulation power uses absolute value since regulation can be called in either direction

This is a simplified Ah-throughput model. More sophisticated models (e.g., Arrhenius temperature dependence, stress factors for depth-of-discharge) can be substituted by modifying the `_ode` method without changing the rest of the platform.

---

## Discrete-Time Integration (RK4)

The continuous-time ODE is integrated using the **4th-order Runge-Kutta method**:

```
k₁ = f(xₖ, uₖ)
k₂ = f(xₖ + (dt/2)·k₁, uₖ)
k₃ = f(xₖ + (dt/2)·k₂, uₖ)
k₄ = f(xₖ + dt·k₃, uₖ)

x_{k+1} = xₖ + (dt/6)·(k₁ + 2k₂ + 2k₃ + k₄)
```

This is implemented **once** as a CasADi `Function` (`build_casadi_rk4_integrator`) and called with different time steps:

| Consumer | dt | Purpose |
|----------|-----|---------|
| EMS | 3,600 s | Hourly state prediction over 24-hour horizon |
| MPC | 60 s | Minute-by-minute state prediction over 1-hour horizon |
| EKF | 60 s | State prediction for Kalman filter |
| MHE | 60 s | Dynamics constraint in estimation NLP |
| Plant | 1 s | High-fidelity simulation (numpy, not CasADi) |

RK4 provides O(dt⁴) accuracy — significantly better than Euler (O(dt)) at minimal computational cost, especially important for the 3,600 s EMS time step where Euler integration would accumulate substantial error.

---

## CasADi Symbolic Functions

Two CasADi functions are exported:

### `build_casadi_dynamics(bp) → ca.Function`

Returns the continuous-time ODE `f(x, u) → ẋ` as a CasADi `Function`. This is used internally by the RK4 integrator and can be used directly for analysis (e.g., computing equilibria).

### `build_casadi_rk4_integrator(bp, dt) → ca.Function`

Returns the discrete-time map `F(x, u) → x_next` as a CasADi `Function`. This is the primary interface used by all optimizers and estimators.

Because these are CasADi symbolic functions, they support:
- **Automatic differentiation**: exact Jacobians and Hessians for the EKF and IPOPT
- **Code generation**: can be compiled to C for production deployment
- **Embedding in NLPs**: used directly as constraints in the EMS, MPC, and MHE optimization problems

---

## BatteryPlant Class

The `BatteryPlant` class is a **numpy-based** implementation of the same dynamics, used as the "ground truth" plant in simulation. It adds:

- **SOC-limited power saturation**: if a control command would push SOC outside [SOC_min, SOC_max], the state is clamped (back-calculation). This models the real behavior of battery management systems that cut off charge/discharge at limits.
- **Noisy measurement**: `get_measurement()` returns SOC + Gaussian noise (σ = 0.01). SOH is never included in the measurement — estimators must infer it.
- **Input clamping**: all power inputs are clipped to [0, P_max_kw] before integration.

The plant integrates at `dt_sim = 1 s` resolution — 60x finer than the control/estimation loop — providing a high-fidelity reference against which the estimators and controllers are evaluated.
