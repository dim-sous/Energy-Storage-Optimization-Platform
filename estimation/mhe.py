"""Moving Horizon Estimation for joint SOC / SOH estimation.

Formulation
-----------
  min   arrival_cost(x_0 - x_bar)
      + sum_k  w_soc_meas  * (SOC_k  - y_k)^2
      + sum_k  w_process_soc * w_soc_k^2  +  w_process_soh * w_soh_k^2

  s.t.  x_{k+1} = f(x_k, u_k) + w_k
        SOC_min - margin  <=  SOC_k  <=  SOC_max + margin
        0.5  <=  SOH_k  <=  1.0

Estimation window:  N_mhe = 30 steps  (30 minutes at dt_estimator = 60 s).
Solved with CasADi Opti / IPOPT.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, MHEParams, TimeParams
from models.battery_model import build_casadi_rk4_integrator

logger = logging.getLogger(__name__)


class MovingHorizonEstimator:
    """MHE for joint SOC / SOH estimation.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    mp : MHEParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MHEParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp

        # Circular buffers for the estimation window
        self._u_buf: list[np.ndarray] = []
        self._y_buf: list[float] = []

        # Arrival cost reference state
        self._x_arrival = np.array([bp.SOC_init, bp.SOH_init], dtype=np.float64)

        # Current estimate
        self.x_hat = np.array([bp.SOC_init, bp.SOH_init], dtype=np.float64)

        # CasADi integrator (discrete-time, dt_estimator)
        self._F_step = build_casadi_rk4_integrator(bp, tp.dt_estimator)

        # Warm-start cache for the state trajectory
        self._prev_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def step(self, u: np.ndarray, y_meas: float) -> np.ndarray:
        """Process a new (input, measurement) pair and return the estimate.

        Parameters
        ----------
        u : ndarray, shape (3,)
            [P_chg, P_dis, P_reg] applied during the preceding interval.
        y_meas : float
            Noisy SOC measurement at end of interval.

        Returns
        -------
        x_hat : ndarray, shape (2,)
            [SOC_est, SOH_est]
        """
        N = self.mp.N_mhe

        self._u_buf.append(u.copy())
        self._y_buf.append(y_meas)

        # Trim to window
        if len(self._u_buf) > N:
            # Update arrival cost from the discarded portion
            if self._prev_X is not None and self._prev_X.shape[1] > 1:
                self._x_arrival = self._prev_X[:, 1].copy()
            self._u_buf = self._u_buf[-N:]
            self._y_buf = self._y_buf[-N:]

        M = len(self._u_buf)
        return self._solve_mhe(M)

    def get_estimate(self) -> np.ndarray:
        """Return current state estimate [SOC_est, SOH_est]."""
        return self.x_hat.copy()

    # ------------------------------------------------------------------
    #  NLP construction and solve
    # ------------------------------------------------------------------

    def _solve_mhe(self, M: int) -> np.ndarray:
        """Build and solve the MHE NLP for the current window of size *M*."""
        bp = self.bp
        mp = self.mp

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(2, M + 1)       # state trajectory [SOC; SOH]
        W = opti.variable(2, M)           # process noise sequence

        # ----- Objective -----
        cost = 0.0

        # Arrival cost
        cost += mp.arrival_soc * (X[0, 0] - self._x_arrival[0]) ** 2
        cost += mp.arrival_soh * (X[1, 0] - self._x_arrival[1]) ** 2

        # Stage costs
        for k in range(M):
            # Dynamics with additive process noise
            x_k = X[:, k]
            u_k = ca.DM(self._u_buf[k])
            x_next_nom = self._F_step(x_k, u_k)

            opti.subject_to(X[0, k + 1] == x_next_nom[0] + W[0, k])
            opti.subject_to(X[1, k + 1] == x_next_nom[1] + W[1, k])

            # Measurement cost (only SOC observed — associate with post-step state)
            y_k = self._y_buf[k]
            cost += mp.w_soc_meas * (X[0, k + 1] - y_k) ** 2

            # Process noise penalty
            cost += mp.w_process_soc * W[0, k] ** 2
            cost += mp.w_process_soh * W[1, k] ** 2

        opti.minimize(cost)

        # ----- State bounds -----
        soc_margin = 0.05
        for k in range(M + 1):
            opti.subject_to(
                opti.bounded(bp.SOC_min - soc_margin, X[0, k], bp.SOC_max + soc_margin)
            )
            opti.subject_to(opti.bounded(0.5, X[1, k], 1.0))

        # ----- Initial guess (warm-start) -----
        if self._prev_X is not None and self._prev_X.shape[1] == M + 1:
            opti.set_initial(X, self._prev_X)
        else:
            for k in range(M + 1):
                opti.set_initial(X[0, k], self.x_hat[0])
                opti.set_initial(X[1, k], self.x_hat[1])
        opti.set_initial(W, 0.0)

        # ----- Solver -----
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            X_opt = np.array(sol.value(X))   # (2, M+1)
            self._prev_X = X_opt
            self.x_hat = X_opt[:, -1].copy()
        except RuntimeError:
            # Solver failed — keep previous estimate, propagate with dynamics
            logger.warning("MHE solver failed; keeping previous estimate.")
            if len(self._u_buf) > 0:
                x_pred = np.array(
                    self._F_step(self.x_hat, self._u_buf[-1])
                ).flatten()
                self.x_hat = x_pred

        return self.x_hat.copy()
