"""Stochastic Energy Management System (EMS) with regulation market.

Solves a scenario-based NLP every ``dt_ems`` (3 600 s) over a 24-hour
rolling horizon using CasADi Opti / IPOPT.

Decision variables per scenario *s*, per time step *k*
-------------------------------------------------------
  P_chg[s, k], P_dis[s, k], P_reg[s, k]   :  power commands  [kW]
  SOC[s, k],   SOH[s, k]                   :  state trajectories  [-]

Objective
---------
  Maximise   E_s [ energy_revenue + regulation_revenue - degradation_cost ]

Non-anticipativity: first-stage decisions agree across all scenarios.
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from config.parameters import BatteryParams, EMSParams, TimeParams
from models.battery_model import build_casadi_rk4_integrator

logger = logging.getLogger(__name__)


class EconomicEMS:
    """Stochastic economic EMS with energy arbitrage and regulation revenue.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    ep : EMSParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep

        # RK4 integrator at EMS time step (3 600 s)
        self._F_ems = build_casadi_rk4_integrator(bp, tp.dt_ems)

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:
        """Solve the stochastic EMS optimisation.

        Parameters
        ----------
        soc_init : float
        soh_init : float
        energy_scenarios : ndarray, shape (n_scenarios, N_hours)
            Energy prices per scenario [$/kWh].
        reg_scenarios : ndarray, shape (n_scenarios, N_hours)
            Regulation prices per scenario [$/kW/h].
        probabilities : ndarray, shape (n_scenarios,)

        Returns
        -------
        dict with keys:
            P_chg_ref   : ndarray (N,)   hourly charge reference  [kW]
            P_dis_ref   : ndarray (N,)   hourly discharge reference  [kW]
            P_reg_ref   : ndarray (N,)   hourly regulation reference  [kW]
            SOC_ref     : ndarray (N+1,) SOC reference trajectory  [-]
            SOH_ref     : ndarray (N+1,) SOH reference trajectory  [-]
            expected_profit : float  [$]
        """
        bp = self.bp
        ep = self.ep
        N = min(ep.N_ems, energy_scenarios.shape[1])
        S = len(probabilities)

        # Clip initial state to strictly feasible region
        soc_init = float(np.clip(soc_init, bp.SOC_min + 0.001, bp.SOC_max - 0.001))
        soh_init = float(np.clip(soh_init, 0.51, 1.0))

        opti = ca.Opti()

        # ---- Decision variables (per scenario) ----
        P_chg: list[ca.MX] = []
        P_dis: list[ca.MX] = []
        P_reg: list[ca.MX] = []
        SOC: list[ca.MX] = []
        SOH: list[ca.MX] = []
        eps_soc: list[ca.MX] = []   # soft SOC slack

        for _ in range(S):
            P_chg.append(opti.variable(N))
            P_dis.append(opti.variable(N))
            P_reg.append(opti.variable(N))
            SOC.append(opti.variable(N + 1))
            SOH.append(opti.variable(N + 1))
            eps_soc.append(opti.variable(N + 1))

        total_obj = 0.0

        for s in range(S):
            # Initial conditions
            opti.subject_to(SOC[s][0] == soc_init)
            opti.subject_to(SOH[s][0] == soh_init)

            scenario_profit = 0.0

            for k in range(N):
                # Dynamics
                x_k = ca.vertcat(SOC[s][k], SOH[s][k])
                u_k = ca.vertcat(P_chg[s][k], P_dis[s][k], P_reg[s][k])
                x_next = self._F_ems(x_k, u_k)

                opti.subject_to(SOC[s][k + 1] == x_next[0])
                opti.subject_to(SOH[s][k + 1] == x_next[1])

                # Energy arbitrage revenue  [$ for this hour]
                # price [$/kWh] * net_power [kW] * dt [h]
                dt_hours = self.tp.dt_ems / 3600.0
                energy_rev = float(energy_scenarios[s, k]) * (
                    P_dis[s][k] - P_chg[s][k]
                ) * dt_hours

                # Regulation capacity payment  [$ for this hour]
                # reg_price [$/kW/h] * reg_capacity [kW] * dt [h]
                reg_rev = float(reg_scenarios[s, k]) * P_reg[s][k] * dt_hours

                # Degradation cost
                deg_cost = ep.degradation_cost * bp.alpha_deg * (
                    P_chg[s][k] + P_dis[s][k] + P_reg[s][k]
                ) * self.tp.dt_ems

                scenario_profit += energy_rev + reg_rev - deg_cost

            # Terminal penalties
            scenario_profit -= ep.terminal_soc_weight * (
                SOC[s][N] - bp.SOC_terminal
            ) ** 2
            scenario_profit -= ep.terminal_soh_weight * (
                SOH[s][N] - soh_init
            ) ** 2

            # Soft SOC constraint penalty
            for k in range(N + 1):
                scenario_profit -= 1e5 * eps_soc[s][k] ** 2

            # ---- Bounds for this scenario ----
            opti.subject_to(eps_soc[s] >= 0)
            for k in range(N + 1):
                opti.subject_to(SOC[s][k] >= bp.SOC_min - eps_soc[s][k])
                opti.subject_to(SOC[s][k] <= bp.SOC_max + eps_soc[s][k])
            opti.subject_to(opti.bounded(0.5, SOH[s], 1.001))
            opti.subject_to(opti.bounded(0.0, P_chg[s], bp.P_max_kw))
            opti.subject_to(opti.bounded(0.0, P_dis[s], bp.P_max_kw))
            opti.subject_to(
                opti.bounded(0.0, P_reg[s], bp.P_max_kw * ep.regulation_fraction)
            )

            # Power budget: charge + reg <= P_max,  discharge + reg <= P_max
            for k in range(N):
                opti.subject_to(P_chg[s][k] + P_reg[s][k] <= bp.P_max_kw)
                opti.subject_to(P_dis[s][k] + P_reg[s][k] <= bp.P_max_kw)

            # Accumulate expected cost (minimise negative profit)
            total_obj += float(probabilities[s]) * (-scenario_profit)

        # ---- Non-anticipativity (first-stage) ----
        for s in range(1, S):
            opti.subject_to(P_chg[s][0] == P_chg[0][0])
            opti.subject_to(P_dis[s][0] == P_dis[0][0])
            opti.subject_to(P_reg[s][0] == P_reg[0][0])

        opti.minimize(total_obj)

        # ---- Initial guesses ----
        for s in range(S):
            opti.set_initial(SOC[s], np.linspace(soc_init, bp.SOC_terminal, N + 1))
            opti.set_initial(SOH[s], soh_init)
            opti.set_initial(P_chg[s], 0.0)
            opti.set_initial(P_dis[s], 0.0)
            opti.set_initial(P_reg[s], 0.0)
            opti.set_initial(eps_soc[s], 0.0)

        # ---- Solver options ----
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.linear_solver": "mumps",
        }
        opti.solver("ipopt", opts)

        # ---- Solve ----
        try:
            sol = opti.solve()
        except RuntimeError as exc:
            logger.error("EMS solver failed: %s", exc)
            # Return zero references as fallback
            return self._fallback_result(N, soc_init, soh_init)

        # ---- Extract probability-weighted references ----
        p_chg_ref = np.zeros(N)
        p_dis_ref = np.zeros(N)
        p_reg_ref = np.zeros(N)
        soc_ref = np.zeros(N + 1)
        soh_ref = np.zeros(N + 1)

        for s in range(S):
            w = float(probabilities[s])
            p_chg_ref += w * np.array(sol.value(P_chg[s])).flatten()
            p_dis_ref += w * np.array(sol.value(P_dis[s])).flatten()
            p_reg_ref += w * np.array(sol.value(P_reg[s])).flatten()
            soc_ref += w * np.array(sol.value(SOC[s])).flatten()
            soh_ref += w * np.array(sol.value(SOH[s])).flatten()

        expected_profit = float(-sol.value(total_obj))

        logger.info(
            "EMS solved: expected profit = $%.2f  |  "
            "SOC [%.3f -> %.3f]  |  SOH [%.6f -> %.6f]",
            expected_profit,
            soc_ref[0],
            soc_ref[-1],
            soh_ref[0],
            soh_ref[-1],
        )

        return {
            "P_chg_ref": p_chg_ref,
            "P_dis_ref": p_dis_ref,
            "P_reg_ref": p_reg_ref,
            "SOC_ref": soc_ref,
            "SOH_ref": soh_ref,
            "expected_profit": expected_profit,
        }

    # ------------------------------------------------------------------
    #  Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_result(N: int, soc_init: float, soh_init: float) -> dict:
        """Return zero-power references when the solver fails."""
        return {
            "P_chg_ref": np.zeros(N),
            "P_dis_ref": np.zeros(N),
            "P_reg_ref": np.zeros(N),
            "SOC_ref": np.full(N + 1, soc_init),
            "SOH_ref": np.full(N + 1, soh_init),
            "expected_profit": 0.0,
        }
