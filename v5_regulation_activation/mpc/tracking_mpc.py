"""Nonlinear tracking MPC with temperature constraints and regulation headroom.

v5_regulation_activation: uses 2-state prediction model (SOC, T) with SOH
frozen as a parameter.  SOH changes ~0.001 over the 1-hour MPC horizon —
negligible for control.  The EMS handles SOH economics at the hourly level.

Hierarchical estimation-control separation:
  - EKF/MHE: 5-state (SOC, SOH, T, V_rc1, V_rc2) — full observability model
  - MPC:     2-state (SOC, T) + SOH parameter      — fast, tractable prediction
  - EMS:     3-state (SOC, SOH, T)                  — coarse hourly planning

The MPC objective combines SOC tracking (closed-loop state feedback) with
power reference tracking (economic timing from EMS) and rate-of-change
smoothness.  Temperature is predicted for constraint enforcement only
(no tracking term — the soft constraint handles thermal safety).

Runs every ``dt_mpc`` = 60 s.

States:  x = [SOC, T]  (SOH frozen as parameter)
Inputs:  u = [P_chg, P_dis, P_reg]   (all >= 0, kW)
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from config.parameters import (
    BatteryParams,
    ElectricalParams,
    MPCParams,
    ThermalParams,
    TimeParams,
)
from models.battery_model import build_casadi_dynamics_3state

logger = logging.getLogger(__name__)


def _build_2state_integrator(
    bp: BatteryParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    dt: float,
    soh_param: ca.MX,
) -> ca.Function:
    """Build a 2-state (SOC, T) RK4 integrator with SOH as an external parameter.

    Internally uses the 3-state ODE but fixes SOH to *soh_param* and
    discards the dSOH output.

    Returns
    -------
    ca.Function  (x2[2], u[3], soh[1]) -> x2_next[2]
    """
    f3 = build_casadi_dynamics_3state(bp, thp, elp)

    x2 = ca.MX.sym("x2", 2)   # [SOC, T]
    u = ca.MX.sym("u", 3)
    soh = ca.MX.sym("soh", 1)

    # Reconstruct 3-state vector with frozen SOH
    x3 = ca.vertcat(x2[0], soh, x2[1])  # [SOC, SOH, T]

    def rk4_f(x2_in: ca.MX) -> ca.MX:
        x3_in = ca.vertcat(x2_in[0], soh, x2_in[1])
        xdot3 = f3(x3_in, u)
        return ca.vertcat(xdot3[0], xdot3[2])  # [dSOC, dT], skip dSOH

    k1 = rk4_f(x2)
    k2 = rk4_f(x2 + dt / 2.0 * k1)
    k3 = rk4_f(x2 + dt / 2.0 * k2)
    k4 = rk4_f(x2 + dt * k3)
    x2_next = x2 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_2s", [x2, u, soh], [x2_next],
                        ["x2", "u", "soh"], ["x2_next"])


class TrackingMPC:
    """Nonlinear tracking MPC with simplified 2-state prediction model.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    mp  : MPCParams
    thp : ThermalParams
    elp : ElectricalParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MPCParams,
        thp: ThermalParams,
        elp: ElectricalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp
        self.thp = thp
        self.elp = elp

        # Solver failure tracking
        self.last_solve_failed = False

        # Warm-start caches
        self._prev_P_chg: np.ndarray | None = None
        self._prev_P_dis: np.ndarray | None = None
        self._prev_P_reg: np.ndarray | None = None
        self._prev_SOC: np.ndarray | None = None
        self._prev_TEMP: np.ndarray | None = None

        self._build_problem()

    # ------------------------------------------------------------------
    #  NLP construction (called once)
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self.mp.N_mpc      # 60
        Nc = self.mp.Nc_mpc    # 20
        bp = self.bp
        mp = self.mp
        thp = self.thp

        opti = ca.Opti()

        # ---- Decision variables ----
        P_chg = opti.variable(Nc)
        P_dis = opti.variable(Nc)
        P_reg = opti.variable(Nc)
        SOC = opti.variable(N + 1)
        TEMP = opti.variable(N + 1)
        eps = opti.variable(N + 1)         # Soft SOC slack
        eps_temp = opti.variable(N + 1)    # Soft temperature slack

        # ---- Parameters (set each solve) ----
        soc_0 = opti.parameter()
        soh_p = opti.parameter()           # SOH frozen as parameter
        temp_0 = opti.parameter()
        soc_ref_p = opti.parameter(N + 1)
        p_chg_ref_p = opti.parameter(N)
        p_dis_ref_p = opti.parameter(N)
        p_reg_ref_p = opti.parameter(N)
        u_prev_p = opti.parameter(3)

        # ---- 2-state integrator with SOH parameter ----
        F_mpc = _build_2state_integrator(
            bp, thp, self.elp, self.tp.dt_mpc, soh_p,
        )

        # ---- Initial conditions ----
        opti.subject_to(SOC[0] == soc_0)
        opti.subject_to(TEMP[0] == temp_0)

        # ---- Build cost ----
        cost = 0.0

        for k in range(N):
            j = min(k, Nc - 1)   # control horizon blocking

            # Dynamics (2-state)
            x2_k = ca.vertcat(SOC[k], TEMP[k])
            u_k = ca.vertcat(P_chg[j], P_dis[j], P_reg[j])
            x2_next = F_mpc(x2_k, u_k, soh_p)

            opti.subject_to(SOC[k + 1] == x2_next[0])
            opti.subject_to(TEMP[k + 1] == x2_next[1])

            # SOC tracking — closed-loop state feedback
            cost += mp.Q_soc * (SOC[k] - soc_ref_p[k]) ** 2

            # Power tracking — economic timing signal from EMS
            cost += mp.R_power * (
                (P_chg[j] - p_chg_ref_p[k]) ** 2
                + (P_dis[j] - p_dis_ref_p[k]) ** 2
                + (P_reg[j] - p_reg_ref_p[k]) ** 2
            )

            # Rate-of-change penalty
            if k == 0:
                cost += mp.R_delta * (
                    (P_chg[0] - u_prev_p[0]) ** 2
                    + (P_dis[0] - u_prev_p[1]) ** 2
                    + (P_reg[0] - u_prev_p[2]) ** 2
                )
            elif k < Nc:
                cost += mp.R_delta * (
                    (P_chg[k] - P_chg[k - 1]) ** 2
                    + (P_dis[k] - P_dis[k - 1]) ** 2
                    + (P_reg[k] - P_reg[k - 1]) ** 2
                )

        # Terminal cost
        cost += mp.Q_terminal * (SOC[N] - soc_ref_p[N]) ** 2

        # Soft-constraint penalties
        for k in range(N + 1):
            cost += mp.slack_penalty * eps[k] ** 2
            cost += mp.slack_penalty_temp * eps_temp[k] ** 2

        opti.minimize(cost)

        # ---- Constraints ----
        opti.subject_to(opti.bounded(0.0, P_chg, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_dis, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_reg, bp.P_max_kw * 0.3))
        opti.subject_to(eps >= 0)
        opti.subject_to(eps_temp >= 0)

        for k in range(N + 1):
            opti.subject_to(SOC[k] >= bp.SOC_min - eps[k])
            opti.subject_to(SOC[k] <= bp.SOC_max + eps[k])
            opti.subject_to(TEMP[k] <= thp.T_max + eps_temp[k])
            opti.subject_to(TEMP[k] >= thp.T_min - eps_temp[k])

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
        self._TEMP = TEMP
        self._eps = eps
        self._eps_temp = eps_temp
        self._soc_0 = soc_0
        self._soh_p = soh_p
        self._temp_0 = temp_0
        self._soc_ref_p = soc_ref_p
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
        p_chg_ref: np.ndarray,
        p_dis_ref: np.ndarray,
        p_reg_ref: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the MPC control action.

        Parameters
        ----------
        x_est     : ndarray (5,)   [SOC_est, SOH_est, T_est, V_rc1_est, V_rc2_est]
                    SOH is used as a frozen parameter. V_rc estimates are ignored.
        soc_ref   : ndarray (N+1,)
        p_chg_ref : ndarray (N,)
        p_dis_ref : ndarray (N,)
        p_reg_ref : ndarray (N,)
        u_prev    : ndarray (3,) or None

        Returns
        -------
        u_cmd : ndarray (3,)   [P_chg, P_dis, P_reg]
        """
        N = self.mp.N_mpc
        Nc = self.mp.Nc_mpc
        opti = self._opti
        thp = self.thp

        soc_ref = self._pad(soc_ref, N + 1)
        p_chg_ref = self._pad(p_chg_ref, N)
        p_dis_ref = self._pad(p_dis_ref, N)
        p_reg_ref = self._pad(p_reg_ref, N)

        soc_0_val = float(np.clip(x_est[0], self.bp.SOC_min, self.bp.SOC_max))
        soh_val = float(np.clip(x_est[1], 0.5, 1.0))
        temp_0_val = float(np.clip(x_est[2], thp.T_min - 5.0, thp.T_max + 5.0))

        if u_prev is None:
            u_prev = np.array([p_chg_ref[0], p_dis_ref[0], p_reg_ref[0]])

        # Set parameters
        opti.set_value(self._soc_0, soc_0_val)
        opti.set_value(self._soh_p, soh_val)
        opti.set_value(self._temp_0, temp_0_val)
        opti.set_value(self._soc_ref_p, soc_ref)
        opti.set_value(self._p_chg_ref_p, p_chg_ref)
        opti.set_value(self._p_dis_ref_p, p_dis_ref)
        opti.set_value(self._p_reg_ref_p, p_reg_ref)
        opti.set_value(self._u_prev_p, u_prev)

        # Warm-start
        if self._prev_P_chg is not None:
            opti.set_initial(self._P_chg, self._prev_P_chg)
            opti.set_initial(self._P_dis, self._prev_P_dis)
            opti.set_initial(self._P_reg, self._prev_P_reg)
            opti.set_initial(self._SOC, self._prev_SOC)
            opti.set_initial(self._TEMP, self._prev_TEMP)
        else:
            opti.set_initial(self._P_chg, 0.0)
            opti.set_initial(self._P_dis, 0.0)
            opti.set_initial(self._P_reg, 0.0)
            opti.set_initial(self._SOC, soc_ref)
            opti.set_initial(self._TEMP, temp_0_val)
        opti.set_initial(self._eps, 0.0)
        opti.set_initial(self._eps_temp, 0.0)

        # Solve
        try:
            sol = opti.solve()
            self.last_solve_failed = False

            pc = np.array(sol.value(self._P_chg)).flatten()
            pd = np.array(sol.value(self._P_dis)).flatten()
            pr = np.array(sol.value(self._P_reg)).flatten()
            soc_opt = np.array(sol.value(self._SOC)).flatten()
            temp_opt = np.array(sol.value(self._TEMP)).flatten()

            u_cmd = np.array([pc[0], pd[0], pr[0]])

            # Cache shifted solution for warm-start
            self._prev_P_chg = np.append(pc[1:], pc[-1])
            self._prev_P_dis = np.append(pd[1:], pd[-1])
            self._prev_P_reg = np.append(pr[1:], pr[-1])
            self._prev_SOC = np.append(soc_opt[1:], soc_opt[-1])
            self._prev_TEMP = np.append(temp_opt[1:], temp_opt[-1])

        except RuntimeError as exc:
            logger.warning("MPC solver failed: %s", str(exc)[:200])
            self.last_solve_failed = True
            u_cmd = np.zeros(3)

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
