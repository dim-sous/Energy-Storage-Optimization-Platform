"""Economic MPC.

v5_regulation_activation, Step 3.

**Economic objective** with soft EMS anchor instead of hard reference
tracking. MPC sees forecast prices (energy + FCR capacity, both as
probability-weighted forecast means — never the realized prices) and
trades arbitrage profit against degradation and a soft penalty on
deviating from the EMS SOC plan.

RF1 (2026-04-15): the previous "activation-aware OU persistence
forecast" was removed. It used the latest ground-truth activation
sample, which is a cheat — on real hardware the MPC cannot see the
grid signal at sub-minute granularity. The activation signal lives
inside the plant now (it *is* the BESS controller). MPC plans SOC
as if activation will take its expected value (zero signed mean),
which is correct given that the OU activation process is symmetric
around zero. The MPC's edge over LP/EMS comes from re-planning
against the live EKF state estimate, not from peeking at the grid.

P_reg is treated as **exogenous** (parameter, not decision variable).
The simulator passes ``P_reg_committed`` from the EMS hourly plan,
ZOH-expanded across the MPC horizon, since in execution the PI
controller dispatches the EMS-committed reg power, not the MPC's
choice. This eliminates a redundant decision variable and makes the
MPC's job clear: "given that the EMS committed P_reg, plan chg/dis to
keep the SOC trajectory feasible under the expected activation
realisation, while squeezing whatever arbitrage value is available."
"""

from __future__ import annotations

import logging

import casadi as ca
import numpy as np

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.mpc.tracking import TrackingMPC, _build_2state_integrator

logger = logging.getLogger(__name__)


class EconomicMPC:
    """Profit-maximising MPC with soft EMS anchor (v5 Step 3)."""

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        mp: MPCParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        ep: EMSParams,
        reg_p: RegulationParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.mp = mp
        self.thp = thp
        self.elp = elp
        self.ep = ep
        self.reg_p = reg_p

        self.last_solve_failed = False

        # Warm-start caches
        self._prev_P_chg: np.ndarray | None = None
        self._prev_P_dis: np.ndarray | None = None
        self._prev_P_reg: np.ndarray | None = None
        self._prev_SOC: np.ndarray | None = None
        self._prev_TEMP: np.ndarray | None = None

        # Fallback: if the economic NLP fails, fall back to a tracking MPC
        # solve so we never lose the EMS plan. Built lazily.
        self._tracking_fallback = TrackingMPC(bp, tp, mp, thp, elp)

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
        reg_p = self.reg_p
        ep = self.ep

        opti = ca.Opti()

        # ---- Decision variables ----
        # Note: P_reg is NO LONGER a decision variable. It's set hourly by
        # the EMS and the PI controller dispatches it directly. The MPC
        # plans P_chg/P_dis to keep SOC feasible under the EMS reg plan.
        P_chg = opti.variable(Nc)
        P_dis = opti.variable(Nc)
        SOC = opti.variable(N + 1)
        TEMP = opti.variable(N + 1)
        eps = opti.variable(N + 1)
        eps_temp = opti.variable(N + 1)

        # ---- Parameters (set each solve) ----
        soc_0 = opti.parameter()
        soh_p = opti.parameter()
        temp_0 = opti.parameter()
        soc_ref_p = opti.parameter(N + 1)             # EMS anchor (cross-hour SOC)
        u_prev_p = opti.parameter(2)                  # last [P_chg, P_dis]
        price_e_p = opti.parameter(N)                 # forecast $/kWh per step
        # P_reg committed by EMS, ZOH per MPC step (already exogenous)
        p_reg_committed_p = opti.parameter(N)

        # ---- 2-state integrator with SOH parameter ----
        F_mpc = _build_2state_integrator(
            bp, thp, self.elp, self.tp.dt_mpc, soh_p,
        )

        # Initial conditions
        opti.subject_to(SOC[0] == soc_0)
        opti.subject_to(TEMP[0] == temp_0)

        dt_h = self.tp.dt_mpc / 3600.0          # hours per MPC step
        dt_s = self.tp.dt_mpc                    # seconds per MPC step

        cost = 0.0

        for k in range(N):
            j = min(k, Nc - 1)   # control horizon blocking

            # RF1 (2026-04-15): the activation-aware OU forecast was
            # removed. The expected signed activation over the horizon
            # is zero (symmetric OU process around zero), so there is
            # no SOC pre-positioning term. The expected delivery
            # payment is constant w.r.t. the decision variables
            # (P_reg is exogenous) and has also been dropped from the
            # cost — it cannot influence the optimum.

            # ---- Prediction model inputs ----
            # Reg input is the absolute committed reg power so thermal
            # Joule heating is correctly accounted.
            u_k_eff = ca.vertcat(P_chg[j], P_dis[j], p_reg_committed_p[k])

            x2_k = ca.vertcat(SOC[k], TEMP[k])
            x2_next = F_mpc(x2_k, u_k_eff, soh_p)
            opti.subject_to(SOC[k + 1] == x2_next[0])
            opti.subject_to(TEMP[k + 1] == x2_next[1])

            # ---- Cost terms (cost = -profit on the planning variables) ----
            # Energy arbitrage profit (negative cost = profit)
            cost += -mp.w_e * price_e_p[k] * (P_dis[j] - P_chg[j]) * dt_h

            # Degradation cost on the planned-action chg/dis only
            # (split form matches plant/EMS/accounting). Reg-power
            # degradation is constant w.r.t. our decision variables and
            # is omitted from optimization (added to value report by the
            # simulator).
            P_arb = P_chg[j] + P_dis[j]
            cost += mp.w_deg * ep.degradation_cost * bp.alpha_deg * P_arb * dt_s

            # Soft EMS anchor: keep planned SOC near EMS strategic plan
            cost += mp.Q_soc_anchor * (SOC[k] - soc_ref_p[k]) ** 2

            # Rate-of-change penalty (smoothness)
            if k == 0:
                cost += mp.R_delta_econ * (
                    (P_chg[0] - u_prev_p[0]) ** 2
                    + (P_dis[0] - u_prev_p[1]) ** 2
                )
            elif k < Nc:
                cost += mp.R_delta_econ * (
                    (P_chg[k] - P_chg[k - 1]) ** 2
                    + (P_dis[k] - P_dis[k - 1]) ** 2
                )

        # Terminal anchor (cross-hour alignment with EMS plan)
        cost += mp.Q_terminal_econ * (SOC[N] - soc_ref_p[N]) ** 2

        # Slack penalties
        for k in range(N + 1):
            cost += mp.slack_penalty * eps[k] ** 2
            cost += mp.slack_penalty_temp * eps_temp[k] ** 2

        opti.minimize(cost)

        # ---- Constraints ----
        # Power bounds: P_chg / P_dis are decision variables and bounded
        # by the headroom available after subtracting the committed reg
        # capacity. Use the maximum reg over the horizon for a tight,
        # static bound (conservative but simple).
        opti.subject_to(opti.bounded(0.0, P_chg, bp.P_max_kw))
        opti.subject_to(opti.bounded(0.0, P_dis, bp.P_max_kw))
        opti.subject_to(eps >= 0)
        opti.subject_to(eps_temp >= 0)

        for k in range(N + 1):
            opti.subject_to(SOC[k] >= bp.SOC_min - eps[k])
            opti.subject_to(SOC[k] <= bp.SOC_max + eps[k])
            opti.subject_to(TEMP[k] <= thp.T_max + eps_temp[k])
            opti.subject_to(TEMP[k] >= thp.T_min - eps_temp[k])

        # Power-budget headroom: planned chg/dis must leave room for the
        # committed reg power at the corresponding horizon step. Enforced
        # over the FULL prediction horizon (not just the unblocked control
        # window) — the EMS-committed P_reg can change cross-hour inside
        # the horizon, and the held control values must stay feasible
        # against those later commitments. Otherwise the MPC's predicted
        # SOC trajectory propagates a physically infeasible plan in the
        # blocked region.
        for k in range(N):
            j = min(k, Nc - 1)
            opti.subject_to(P_chg[j] + p_reg_committed_p[k] <= bp.P_max_kw)
            opti.subject_to(P_dis[j] + p_reg_committed_p[k] <= bp.P_max_kw)

        # ---- Solver options (identical to TrackingMPC, including JIT) ----
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
        import os as _os
        if _os.environ.get("BESS_JIT", "1") != "0":
            opts.update({
                "jit": True,
                "compiler": "shell",
                "jit_options": {"flags": ["-O3", "-march=native"], "verbose": False},
            })
        opti.solver("ipopt", opts)

        # ---- Store handles ----
        self._opti = opti
        self._P_chg = P_chg
        self._P_dis = P_dis
        self._SOC = SOC
        self._TEMP = TEMP
        self._eps = eps
        self._eps_temp = eps_temp
        self._soc_0 = soc_0
        self._soh_p = soh_p
        self._temp_0 = temp_0
        self._soc_ref_p = soc_ref_p
        self._u_prev_p = u_prev_p
        self._price_e_p = price_e_p
        self._p_reg_committed_p = p_reg_committed_p

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
        price_e_horizon: np.ndarray,
        p_reg_committed_horizon: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the economic-MPC control action.

        Parameters
        ----------
        x_est                    : ndarray (5,) EKF estimate
        soc_ref                  : ndarray (N+1,) EMS soft anchor
        p_chg_ref/p_dis_ref/p_reg_ref : ndarray (N,) — EMS refs (fallback only)
        price_e_horizon          : ndarray (N,) FORECAST energy price ZOH [$/kWh]
        p_reg_committed_horizon  : ndarray (N,) committed FCR power ZOH from EMS plan [kW]
        u_prev                   : ndarray (3,) — previous u_cmd

        Returns
        -------
        u_cmd : ndarray (3,)  [P_chg, P_dis, P_reg]  (P_reg = committed)
        """
        N = self.mp.N_mpc
        opti = self._opti
        thp = self.thp

        soc_ref = self._pad(soc_ref, N + 1)
        price_e = self._pad(price_e_horizon, N)
        p_reg_committed = self._pad(p_reg_committed_horizon, N)

        soc_0_val = float(np.clip(x_est[0], self.bp.SOC_min, self.bp.SOC_max))
        soh_val = float(np.clip(x_est[1], 0.5, 1.0))
        temp_0_val = float(np.clip(x_est[2], thp.T_min - 5.0, thp.T_max + 5.0))

        if u_prev is None:
            u_prev = np.zeros(2)
        else:
            # Old callers may pass length-3 [chg, dis, reg]; truncate.
            u_prev = np.asarray(u_prev, dtype=float)[:2]

        # Set parameters
        opti.set_value(self._soc_0, soc_0_val)
        opti.set_value(self._soh_p, soh_val)
        opti.set_value(self._temp_0, temp_0_val)
        opti.set_value(self._soc_ref_p, soc_ref)
        opti.set_value(self._u_prev_p, u_prev)
        opti.set_value(self._price_e_p, price_e)
        opti.set_value(self._p_reg_committed_p, p_reg_committed)

        # Warm-start
        if self._prev_P_chg is not None:
            opti.set_initial(self._P_chg, self._prev_P_chg)
            opti.set_initial(self._P_dis, self._prev_P_dis)
            opti.set_initial(self._SOC, self._prev_SOC)
            opti.set_initial(self._TEMP, self._prev_TEMP)
        else:
            opti.set_initial(self._P_chg, 0.0)
            opti.set_initial(self._P_dis, 0.0)
            opti.set_initial(self._SOC, soc_ref)
            opti.set_initial(self._TEMP, temp_0_val)
        opti.set_initial(self._eps, 0.0)
        opti.set_initial(self._eps_temp, 0.0)

        try:
            sol = opti.solve()
            self.last_solve_failed = False

            pc = np.array(sol.value(self._P_chg)).flatten()
            pd = np.array(sol.value(self._P_dis)).flatten()
            soc_opt = np.array(sol.value(self._SOC)).flatten()
            temp_opt = np.array(sol.value(self._TEMP)).flatten()

            # P_reg returned to caller is the EMS-committed value at k=0
            # (MPC does not choose it; it forwards what's committed).
            u_cmd = np.array([pc[0], pd[0], float(p_reg_committed[0])])

            self._prev_P_chg = np.append(pc[1:], pc[-1])
            self._prev_P_dis = np.append(pd[1:], pd[-1])
            self._prev_SOC = np.append(soc_opt[1:], soc_opt[-1])
            self._prev_TEMP = np.append(temp_opt[1:], temp_opt[-1])

        except RuntimeError as exc:
            logger.warning("EconomicMPC failed (%s); falling back to TrackingMPC",
                           str(exc)[:200])
            self.last_solve_failed = True
            # Tracking-MPC fallback so we never lose the EMS plan.
            # Interface drift: TrackingMPC takes a 3-vector u_prev
            # ([P_chg, P_dis, P_reg]); EconomicMPC stores only 2 entries
            # because P_reg is exogenous here. Pad with 0.0 for the
            # tracking call — the appended P_reg slot only feeds the
            # tracking MPC's rate-of-change penalty on the (discarded)
            # P_reg decision variable, so the value doesn't matter.
            u_cmd = self._tracking_fallback.solve(
                x_est=x_est,
                soc_ref=soc_ref,
                p_chg_ref=p_chg_ref,
                p_dis_ref=p_dis_ref,
                p_reg_ref=p_reg_ref,
                u_prev=np.append(u_prev, 0.0) if len(u_prev) == 2 else u_prev,
            )

        return u_cmd

    @staticmethod
    def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) >= target_len:
            return arr[:target_len]
        pad_val = arr[-1] if len(arr) > 0 else 0.0
        return np.concatenate([arr, np.full(target_len - len(arr), pad_val)])
