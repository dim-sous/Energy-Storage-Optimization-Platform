"""Nonlinear 2-state battery energy storage system model.

Provides:
  1. CasADi symbolic dynamics (shared by EMS, MPC, EKF, MHE).
  2. A numpy-based high-fidelity plant for closed-loop simulation.

State vector   x = [SOC, SOH]
Input vector   u = [P_charge, P_discharge, P_reg]   (all >= 0, kW)
Measurement    y = SOC_measured   (SOH is NOT measured)

Continuous-time dynamics
------------------------
  dSOC/dt = (eta_c * P_chg  -  P_dis / eta_d) / (SOH * E_nom * 3600)
  dSOH/dt = -alpha_deg * (P_chg + P_dis + |P_reg|)

The factor 3600 converts E_nom from kWh to kW*s so that the derivative
has units [1/s] when power is in kW.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, TimeParams


# ---------------------------------------------------------------------------
#  CasADi symbolic dynamics  (reused by EMS, MPC, EKF, MHE)
# ---------------------------------------------------------------------------

def build_casadi_dynamics(bp: BatteryParams) -> ca.Function:
    """Return a CasADi Function  f(x, u) -> x_dot  (continuous-time ODE).

    Parameters
    ----------
    bp : BatteryParams

    Returns
    -------
    ca.Function  with signature  (x[2], u[3]) -> x_dot[2]
    """
    x = ca.MX.sym("x", 2)     # [SOC, SOH]
    u = ca.MX.sym("u", 3)     # [P_chg, P_dis, P_reg]

    SOC, SOH = x[0], x[1]
    P_chg, P_dis, P_reg = u[0], u[1], u[2]

    # Effective capacity in kW*s
    E_eff_kws = SOH * bp.E_nom_kwh * 3600.0

    dSOC_dt = (bp.eta_charge * P_chg - P_dis / bp.eta_discharge) / E_eff_kws

    # Regulation power contributes to degradation (absolute value)
    dSOH_dt = -bp.alpha_deg * (P_chg + P_dis + ca.fabs(P_reg))

    x_dot = ca.vertcat(dSOC_dt, dSOH_dt)
    return ca.Function("battery_ode", [x, u], [x_dot], ["x", "u"], ["x_dot"])


def build_casadi_rk4_integrator(bp: BatteryParams, dt: float) -> ca.Function:
    """Return a single-step RK4 integrator  F(x, u) -> x_next.

    Parameters
    ----------
    bp : BatteryParams
    dt : float
        Integration time step [s].

    Returns
    -------
    ca.Function  with signature  (x[2], u[3]) -> x_next[2]
    """
    f = build_casadi_dynamics(bp)

    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 3)

    k1 = f(x, u)
    k2 = f(x + dt / 2.0 * k1, u)
    k3 = f(x + dt / 2.0 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step", [x, u], [x_next], ["x", "u"], ["x_next"])


# ---------------------------------------------------------------------------
#  Numpy plant model for simulation
# ---------------------------------------------------------------------------

class BatteryPlant:
    """High-fidelity 2-state BESS plant integrated at ``dt_sim`` resolution.

    The plant uses RK4 internally and generates noisy SOC measurements.
    SOH is a hidden state — it is never directly measured.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    seed : int
        Random seed for measurement noise.
    """

    def __init__(self, bp: BatteryParams, tp: TimeParams, seed: int = 42) -> None:
        self.bp = bp
        self.tp = tp
        self._rng = np.random.default_rng(seed)

        # True state  [SOC, SOH]
        self._x = np.array([bp.SOC_init, bp.SOH_init], dtype=np.float64)

        # Measurement noise standard deviation (sqrt of r_soc_meas ~ 0.01)
        self._meas_std = 0.01

    # ---- continuous-time ODE (numpy) ----
    def _ode(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        SOC, SOH = x[0], x[1]
        P_chg, P_dis, P_reg = u[0], u[1], u[2]

        E_eff = SOH * self.bp.E_nom_kwh * 3600.0
        dSOC = (self.bp.eta_charge * P_chg - P_dis / self.bp.eta_discharge) / E_eff
        dSOH = -self.bp.alpha_deg * (P_chg + P_dis + abs(P_reg))
        return np.array([dSOC, dSOH])

    # ---- single RK4 step (numpy) ----
    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._ode(x, u)
        k2 = self._ode(x + dt / 2.0 * k1, u)
        k3 = self._ode(x + dt / 2.0 * k2, u)
        k4 = self._ode(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- public interface ----
    def step(self, u: np.ndarray) -> tuple[np.ndarray, float]:
        """Integrate one ``dt_sim`` step and return (x_new, y_meas).

        Parameters
        ----------
        u : ndarray, shape (3,)
            [P_charge, P_discharge, P_reg]  (kW, all >= 0).

        Returns
        -------
        x_new : ndarray, shape (2,)
            Updated true state [SOC, SOH].
        y_meas : float
            Noisy SOC measurement.
        """
        bp = self.bp

        # Clamp inputs to physical bounds
        u_clamped = np.array([
            np.clip(u[0], 0.0, bp.P_max_kw),
            np.clip(u[1], 0.0, bp.P_max_kw),
            np.clip(u[2], 0.0, bp.P_max_kw),
        ])

        x_new = self._rk4_step(self._x, u_clamped, self.tp.dt_sim)

        # --- SOC saturation with back-calculation ---
        if x_new[0] < bp.SOC_min:
            x_new[0] = bp.SOC_min
        elif x_new[0] > bp.SOC_max:
            x_new[0] = bp.SOC_max

        # SOH can only decrease; clamp to valid range
        x_new[1] = np.clip(x_new[1], 0.5, 1.0)

        self._x = x_new.copy()
        y_meas = self.get_measurement()
        return self._x.copy(), y_meas

    def get_measurement(self) -> float:
        """Return a noisy SOC measurement (SOH is NOT measured)."""
        noise = self._rng.normal(0.0, self._meas_std)
        return float(np.clip(self._x[0] + noise, 0.0, 1.0))

    def get_state(self) -> np.ndarray:
        """Return the true state [SOC, SOH] (for logging only)."""
        return self._x.copy()

    def reset(
        self,
        soc: float | None = None,
        soh: float | None = None,
    ) -> None:
        """Reset plant state."""
        self._x[0] = soc if soc is not None else self.bp.SOC_init
        self._x[1] = soh if soh is not None else self.bp.SOH_init
