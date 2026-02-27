"""Nonlinear tracking MPC with control horizon blocking.

Runs every ``dt_mpc`` = 60 s.

Prediction horizon:  N_mpc  = 60 steps  (60 minutes)
Control horizon:     Nc_mpc = 10 steps  (10 minutes)
After Nc_mpc the control input is held constant at the last free value.

States:  x = [SOC, SOH]
Inputs:  u = [P_chg, P_dis, P_reg]   (all >= 0, kW)

Objective
---------
  min  sum_k  Q_soc (SOC_k - ref)^2  +  Q_soh (SOH_k - ref)^2
     + sum_k  R_power ( ||u_k - u_ref_k||^2 )
     + sum_{k=1..Nc}  R_delta ( ||u_k - u_{k-1}||^2 )
     + Q_terminal (SOC_N - ref)^2
     + slack_penalty * sum eps_k^2

Soft constraints on SOC; hard bounds on SOH and power.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, MPCParams, TimeParams
from models.battery_model import build_casadi_rk4_integrator

logger = logging.getLogger(__name__)


class TrackingMPC:
    """Nonlinear tracking MPC with control horizon blocking.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    mp : MPCParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MPCParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp

        # Warm-start caches
        self._prev_P_chg: np.ndarray | None = None
        self._prev_P_dis: np.ndarray | None = None
        self._prev_P_reg: np.ndarray | None = None
        self._prev_SOC: np.ndarray | None = None
        self._prev_SOH: np.ndarray | None = None

        self._build_problem()

    # ------------------------------------------------------------------
    #  NLP construction (called once)
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self.mp.N_mpc      # 60
        Nc = self.mp.Nc_mpc    # 10
        bp = self.bp
        mp = self.mp

        F_mpc = build_casadi_rk4_integrator(bp, self.tp.dt_mpc)

        opti = ca.Opti()

        # ---- Decision variables ----
        P_chg = opti.variable(Nc)          # Free charge power  [kW]
        P_dis = opti.variable(Nc)          # Free discharge power  [kW]
        P_reg = opti.variable(Nc)          # Free regulation power  [kW]
        SOC = opti.variable(N + 1)         # SOC trajectory  [-]
        SOH = opti.variable(N + 1)         # SOH trajectory  [-]
        eps = opti.variable(N + 1)         # Soft-constraint slack  [-]

        # ---- Parameters (set each solve) ----
        soc_0 = opti.parameter()
        soh_0 = opti.parameter()
        soc_ref_p = opti.parameter(N + 1)
        soh_ref_p = opti.parameter(N + 1)
        p_chg_ref_p = opti.parameter(N)
        p_dis_ref_p = opti.parameter(N)
        p_reg_ref_p = opti.parameter(N)
        # Previously applied control (for first-move smoothness)
        u_prev_p = opti.parameter(3)

        # ---- Initial conditions ----
        opti.subject_to(SOC[0] == soc_0)
        opti.subject_to(SOH[0] == soh_0)

        # ---- Build cost ----
        cost = 0.0

        for k in range(N):
            j = min(k, Nc - 1)   # control horizon blocking

            # Dynamics
            x_k = ca.vertcat(SOC[k], SOH[k])
            u_k = ca.vertcat(P_chg[j], P_dis[j], P_reg[j])
            x_next = F_mpc(x_k, u_k)

            opti.subject_to(SOC[k + 1] == x_next[0])
            opti.subject_to(SOH[k + 1] == x_next[1])

            # SOC tracking
            cost += mp.Q_soc * (SOC[k] - soc_ref_p[k]) ** 2
            # SOH tracking
            cost += mp.Q_soh * (SOH[k] - soh_ref_p[k]) ** 2
            # Power tracking (all three inputs)
            cost += mp.R_power * (
                (P_chg[j] - p_chg_ref_p[k]) ** 2
                + (P_dis[j] - p_dis_ref_p[k]) ** 2
                + (P_reg[j] - p_reg_ref_p[k]) ** 2
            )

            # Rate-of-change penalty
            if k == 0:
                # First move: penalise jump from previously applied control
                cost += mp.R_delta * (
                    (P_chg[0] - u_prev_p[0]) ** 2
                    + (P_dis[0] - u_prev_p[1]) ** 2
                    + (P_reg[0] - u_prev_p[2]) ** 2
                )
            elif k < Nc:
                # Subsequent free control steps
                cost += mp.R_delta * (
                    (P_chg[k] - P_chg[k - 1]) ** 2
                    + (P_dis[k] - P_dis[k - 1]) ** 2
                    + (P_reg[k] - P_reg[k - 1]) ** 2
                )

        # Terminal cost
        cost += mp.Q_terminal * (SOC[N] - soc_ref_p[N]) ** 2

        # Soft-constraint penalty
        for k in range(N + 1):
            cost += mp.slack_penalty * eps[k] ** 2

        opti.minimize(cost)

        # ---- Constraints ----
        opti.subject_to(opti.bounded(0.0, P_chg, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_dis, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_reg, bp.P_max_kw * 0.3))
        opti.subject_to(eps >= 0)

        for k in range(N + 1):
            opti.subject_to(SOC[k] >= bp.SOC_min - eps[k])
            opti.subject_to(SOC[k] <= bp.SOC_max + eps[k])
        opti.subject_to(opti.bounded(0.5, SOH, 1.001))

        # Power budget
        for j in range(Nc):
            opti.subject_to(P_chg[j] + P_reg[j] <= bp.P_max_kw)
            opti.subject_to(P_dis[j] + P_reg[j] <= bp.P_max_kw)

        # ---- Solver options ----
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_init": 1e-3,
        }
        opti.solver("ipopt", opts)

        # ---- Store handles ----
        self._opti = opti
        self._P_chg = P_chg
        self._P_dis = P_dis
        self._P_reg = P_reg
        self._SOC = SOC
        self._SOH = SOH
        self._eps = eps
        self._soc_0 = soc_0
        self._soh_0 = soh_0
        self._soc_ref_p = soc_ref_p
        self._soh_ref_p = soh_ref_p
        self._p_chg_ref_p = p_chg_ref_p
        self._p_dis_ref_p = p_dis_ref_p
        self._p_reg_ref_p = p_reg_ref_p
        self._u_prev_p = u_prev_p

    # ------------------------------------------------------------------
    #  Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x_est: np.ndarray,
        soc_ref: np.ndarray,
        soh_ref: np.ndarray,
        p_chg_ref: np.ndarray,
        p_dis_ref: np.ndarray,
        p_reg_ref: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the MPC control action.

        Parameters
        ----------
        x_est     : ndarray (2,)   [SOC_est, SOH_est]
        soc_ref   : ndarray (N+1,)
        soh_ref   : ndarray (N+1,)
        p_chg_ref : ndarray (N,)
        p_dis_ref : ndarray (N,)
        p_reg_ref : ndarray (N,)
        u_prev    : ndarray (3,) or None
            Previously applied control [P_chg, P_dis, P_reg].
            Used for first-move rate-of-change penalty.
            If None, defaults to the current power references.

        Returns
        -------
        u_cmd : ndarray (3,)   [P_chg, P_dis, P_reg]
        """
        N = self.mp.N_mpc
        Nc = self.mp.Nc_mpc
        opti = self._opti

        # Pad references if shorter than required
        soc_ref = self._pad(soc_ref, N + 1)
        soh_ref = self._pad(soh_ref, N + 1)
        p_chg_ref = self._pad(p_chg_ref, N)
        p_dis_ref = self._pad(p_dis_ref, N)
        p_reg_ref = self._pad(p_reg_ref, N)

        # Clip state estimate to feasible region
        soc_0_val = float(np.clip(x_est[0], self.bp.SOC_min, self.bp.SOC_max))
        soh_0_val = float(np.clip(x_est[1], 0.5, 1.0))

        # Default u_prev to current references if not provided
        if u_prev is None:
            u_prev = np.array([p_chg_ref[0], p_dis_ref[0], p_reg_ref[0]])

        # Set parameters
        opti.set_value(self._soc_0, soc_0_val)
        opti.set_value(self._soh_0, soh_0_val)
        opti.set_value(self._soc_ref_p, soc_ref)
        opti.set_value(self._soh_ref_p, soh_ref)
        opti.set_value(self._p_chg_ref_p, p_chg_ref)
        opti.set_value(self._p_dis_ref_p, p_dis_ref)
        opti.set_value(self._p_reg_ref_p, p_reg_ref)
        opti.set_value(self._u_prev_p, u_prev)

        # Warm-start from previous solution
        if self._prev_P_chg is not None:
            opti.set_initial(self._P_chg, self._prev_P_chg)
            opti.set_initial(self._P_dis, self._prev_P_dis)
            opti.set_initial(self._P_reg, self._prev_P_reg)
            opti.set_initial(self._SOC, self._prev_SOC)
            opti.set_initial(self._SOH, self._prev_SOH)
        else:
            opti.set_initial(self._P_chg, p_chg_ref[:Nc])
            opti.set_initial(self._P_dis, p_dis_ref[:Nc])
            opti.set_initial(self._P_reg, p_reg_ref[:Nc])
            opti.set_initial(self._SOC, soc_ref)
            opti.set_initial(self._SOH, soh_ref)
        opti.set_initial(self._eps, 0.0)

        # Solve
        try:
            sol = opti.solve()

            pc = np.array(sol.value(self._P_chg)).flatten()
            pd = np.array(sol.value(self._P_dis)).flatten()
            pr = np.array(sol.value(self._P_reg)).flatten()
            soc_opt = np.array(sol.value(self._SOC)).flatten()
            soh_opt = np.array(sol.value(self._SOH)).flatten()

            u_cmd = np.array([pc[0], pd[0], pr[0]])

            # Cache shifted solution for warm-start
            self._prev_P_chg = np.append(pc[1:], pc[-1])
            self._prev_P_dis = np.append(pd[1:], pd[-1])
            self._prev_P_reg = np.append(pr[1:], pr[-1])
            self._prev_SOC = np.append(soc_opt[1:], soc_opt[-1])
            self._prev_SOH = np.append(soh_opt[1:], soh_opt[-1])

        except RuntimeError as exc:
            logger.warning("MPC solver failed: %s", str(exc)[:200])
            u_cmd = np.array([p_chg_ref[0], p_dis_ref[0], p_reg_ref[0]])

        return u_cmd

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad (or truncate) an array to *target_len* by repeating the last value."""
        if len(arr) >= target_len:
            return arr[:target_len]
        pad_val = arr[-1] if len(arr) > 0 else 0.0
        return np.concatenate([arr, np.full(target_len - len(arr), pad_val)])
