# estimation/ — State Estimation (EKF + MHE)

This module implements two complementary estimators for **joint SOC/SOH estimation** from noisy SOC-only measurements. Both run every 60 seconds alongside the MPC.

The core challenge: **SOH is never directly measured**. It must be inferred from the slow drift in SOC dynamics — as the battery degrades, the effective capacity decreases, causing the same power input to produce a slightly larger SOC change. Detecting this tiny signal through noisy SOC measurements is an extremely difficult observability problem.

---

## Extended Kalman Filter (`ekf.py`)

### State-Space Model

**Prediction step** (nonlinear):
```
x̂⁻_{k+1} = F(x̂_k, u_k)
```

Where `F` is the RK4 integrator at dt = 60 s (same model used by the MPC).

**Measurement model** (linear):
```
y_k = H · x_k + v_k,    H = [1  0],    v_k ~ N(0, R)
```

Only SOC is measured. The measurement matrix `H = [1, 0]` extracts SOC from the state vector.

### Algorithm

**Predict:**
```
x̂⁻ = F(x̂, u)                           State prediction (nonlinear)
A = ∂F/∂x |_{x̂,u}                       Jacobian (CasADi autodiff)
P⁻ = A · P · Aᵀ + Q                     Covariance prediction
```

The Jacobian `A` is computed using **CasADi automatic differentiation** of the RK4 integrator — not finite differences. This gives exact derivatives and is a key advantage of the CasADi framework.

**Update:**
```
ỹ = y − H · x̂⁻                          Innovation
S = H · P⁻ · Hᵀ + R                     Innovation covariance
K = P⁻ · Hᵀ · S⁻¹                       Kalman gain
x̂ = x̂⁻ + K · ỹ                          State correction
P = (I − K·H) · P⁻ · (I − K·H)ᵀ + K · R · Kᵀ    Joseph form
```

The **Joseph form** for the covariance update is used instead of the standard `P = (I - KH)P` because it guarantees positive semi-definiteness even with numerical rounding.

### State Projection

After both predict and update steps, the state estimate is projected onto the feasible region:
```
SOC_est = clip(SOC_est, 0, 1)
SOH_est = clip(SOH_est, 0.5, 1.0)
```

This prevents the EKF from producing physically impossible estimates that would cause downstream MPC infeasibility.

### Noise Matrices

```
Q = diag(q_soc, q_soh) = diag(1e-6, 1e-12)     Process noise
R = [r_soc_meas] = [1e-4]                        Measurement noise
P₀ = diag(p0_soc, p0_soh) = diag(1e-3, 1e-2)   Initial covariance
```

The extreme ratio `q_soc / q_soh = 1e6` reflects the fact that SOC changes ~1e6 times faster than SOH. With `q_soh = 1e-12`, the filter barely adjusts SOH through measurement updates, instead trusting the degradation model almost entirely.

---

## Moving Horizon Estimation (`mhe.py`)

### Formulation

MHE solves an optimization problem over a sliding window of the most recent `M` measurements (up to N_mhe = 30 steps = 30 minutes):

```
min   Σ_{arrival} + Σ_{stage}
```

**Decision variables:**
```
X ∈ ℝ^{2 × (M+1)}     State trajectory [SOC; SOH] over the window
W ∈ ℝ^{2 × M}         Process noise (additive disturbance) sequence
```

Total variables: `2(M+1) + 2M = 4M + 2`. At full window (M=30): **122 variables**.

**Arrival cost** (anchors the initial state to a prior):
```
J_arrival = w_arr_soc · (SOC_0 − SOC_arrival)² + w_arr_soh · (SOH_0 − SOH_arrival)²
```

Where `x_arrival` is the estimate at the beginning of the window from the previous MHE solve (propagated forward as the window slides).

**Stage cost** (per step):
```
J_stage = Σ_{k=0}^{M-1} [ w_meas · (SOC_{k+1} − y_k)²
                          + w_proc_soc · w_soc_k²
                          + w_proc_soh · w_soh_k² ]
```

**Dynamics constraints** (with additive process noise):
```
x_{k+1} = F(x_k, u_k) + w_k
```

Where `F` is the RK4 integrator and `w_k = [w_soc_k, w_soh_k]ᵀ` is the process noise that the optimizer can choose to explain measurement residuals.

**State bounds:**
```
SOC_min − 0.05 ≤ SOC_k ≤ SOC_max + 0.05     (relaxed for feasibility)
0.5 ≤ SOH_k ≤ 1.0
```

### Sliding Window Operation

At each time step:
1. Append the new `(u, y)` pair to circular buffers
2. If the buffer exceeds N_mhe, trim the oldest entry and update the arrival cost reference to the previously estimated state at that time
3. Build and solve the NLP for the current window
4. Return the estimate at the end of the window: `x̂ = X[:, -1]`

### Warm-Starting

The previous optimal state trajectory is cached and reused:
- If the window size hasn't changed (steady state), the previous `X_opt` is used directly as the initial guess
- Otherwise, the estimate is initialized from the current `x̂` at all time steps
- Process noise `W` is initialized to zero

### Comparison with EKF

| Property | EKF | MHE |
|----------|-----|-----|
| Computational cost | O(n²) per step | NLP solve (~300 IPOPT iterations) |
| Handles nonlinearity | First-order linearization | Exact nonlinear model |
| Handles constraints | Post-hoc projection | Native bound constraints |
| Memory | Current state + covariance | 30-step window of data |
| Initialization | Sensitive to P₀ | Robust (arrival cost) |
| SOH estimation | Good with proper tuning | Better (optimization-based) |

Both estimators are included for comparison and cross-validation. In practice, the EKF provides the primary feedback to the MPC due to its simplicity and reliability.

---

## Solver Configuration (MHE)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Solver | IPOPT | Interior-point NLP solver |
| Max iterations | 300 | Small problem, converges quickly |
| Tolerance | 1e-6 | High accuracy for estimation |
| Warm start | yes | Exploits previous window solution |

The NLP is **rebuilt each step** (unlike the MPC which prebuilds once). This is acceptable because the MHE problem is small (~122 variables at full window) and CasADi's problem construction is fast.

If the solver fails, the MHE propagates the previous estimate through the dynamics model: `x̂ = F(x̂_prev, u_last)`. This ensures continuity even during solver failures.
