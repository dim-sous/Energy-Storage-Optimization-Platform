"""Extended Kalman Filter for joint SOC / SOH estimation.

State:        x = [SOC, SOH]
Measurement:  y = SOC_measured   (scalar — SOH is NOT measured)

The state transition Jacobian is computed analytically via CasADi
automatic differentiation of the shared RK4 integrator.

Runs every ``dt_estimator`` = 60 s.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, EKFParams, TimeParams
from models.battery_model import build_casadi_rk4_integrator


class ExtendedKalmanFilter:
    """EKF for joint SOC / SOH estimation.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    ep : EKFParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EKFParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep

        # State estimate
        self.x_hat = np.array([bp.SOC_init, bp.SOH_init], dtype=np.float64)

        # Error covariance
        self.P = np.diag([ep.p0_soc, ep.p0_soh]).astype(np.float64)

        # Process noise covariance
        self.Q = np.diag([ep.q_soc, ep.q_soh]).astype(np.float64)

        # Measurement noise covariance (scalar -> 1x1)
        self.R = np.array([[ep.r_soc_meas]], dtype=np.float64)

        # Measurement matrix:  y = H @ x  =>  y = SOC
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Build CasADi Jacobian
        self._build_jacobian()

    # ------------------------------------------------------------------
    #  CasADi setup
    # ------------------------------------------------------------------

    def _build_jacobian(self) -> None:
        """Compute the Jacobian function  A(x, u) = df_discrete / dx."""
        F_step = build_casadi_rk4_integrator(self.bp, self.tp.dt_estimator)

        x_sym = ca.MX.sym("x", 2)
        u_sym = ca.MX.sym("u", 3)
        x_next = F_step(x_sym, u_sym)

        # Jacobian of the discrete-time map w.r.t. the state
        A_sym = ca.jacobian(x_next, x_sym)

        self._A_func = ca.Function("A_ekf", [x_sym, u_sym], [A_sym])
        self._F_step = F_step

    def _f_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the discrete-time state transition (numpy)."""
        return np.array(self._F_step(x, u)).flatten()

    def _A_eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the state transition Jacobian (numpy, 2x2)."""
        return np.array(self._A_func(x, u))

    # ------------------------------------------------------------------
    #  EKF steps
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray) -> None:
        """Prediction (time-update) step.

        Parameters
        ----------
        u : ndarray, shape (3,)
            Control input [P_chg, P_dis, P_reg] applied during this interval.
        """
        # Jacobian at *prior* estimate (before prediction update)
        A = self._A_eval(self.x_hat, u)

        # State prediction
        x_pred = self._f_eval(self.x_hat, u)
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)
        x_pred[1] = np.clip(x_pred[1], 0.5, 1.0)

        # Covariance prediction
        self.P = A @ self.P @ A.T + self.Q
        self.x_hat = x_pred

    def update(self, y_meas: float) -> np.ndarray:
        """Correction (measurement-update) step.

        Uses the Joseph form for numerical stability of the covariance
        update:  P = (I - K H) P (I - K H)^T  +  K R K^T

        Parameters
        ----------
        y_meas : float
            Noisy SOC measurement.

        Returns
        -------
        x_hat : ndarray, shape (2,)
            Updated estimate [SOC_est, SOH_est].
        """
        # Innovation
        y_pred = self.H @ self.x_hat                      # (1,)
        innov = np.array([y_meas]) - y_pred                # (1,)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R           # (1, 1)

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)          # (2, 1)

        # State update
        self.x_hat = self.x_hat + (K @ innov).flatten()

        # Project state onto feasible region
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)     # SOC in [0, 1]
        self.x_hat[1] = np.clip(self.x_hat[1], 0.5, 1.0)     # SOH in [0.5, 1]

        # Covariance update — Joseph form
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x_hat.copy()

    def step(self, u: np.ndarray, y_meas: float) -> np.ndarray:
        """Combined predict + update.

        Parameters
        ----------
        u : ndarray, shape (3,)
            Control input applied during the preceding interval.
        y_meas : float
            SOC measurement at end of the interval.

        Returns
        -------
        x_hat : ndarray, shape (2,)
            [SOC_est, SOH_est]
        """
        self.predict(u)
        return self.update(y_meas)

    def get_estimate(self) -> np.ndarray:
        """Return current state estimate [SOC_est, SOH_est]."""
        return self.x_hat.copy()
