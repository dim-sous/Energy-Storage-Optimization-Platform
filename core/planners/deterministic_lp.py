"""Deterministic LP energy management — commercial-baseline reference.

This is the **honest industry baseline** for the v5 B2B comparison.  Most
commercial BESS EMS products on the market today ship some form of
deterministic LP / MILP dispatch over a forecast price horizon — no
stochastic programming, no degradation cost in the objective, no thermal
coupling, no second-stage closed-loop controller.  Beating it in the v5
comparison demonstrates that the v5 stack adds *real* value over what a
buyer would otherwise procure.

The LP is given the same forecast information as every other strategy:
the probability-weighted mean of the forecast scenarios passed to
``solve()``. The realized prices are held out and never seen by the
planner; they enter only through the post-hoc accounting ledger. This
matches the base.py protocol contract.

Formulation
-----------
Decision variables (k = 0..N-1):
    P_chg[k], P_dis[k], P_reg[k]   in [0, P_max]   (kW)
    z_plus, z_minus                in [0, +inf)    — L1 terminal slacks

Linear SOC dynamics:
    SOC[k+1] = SOC[k] + (eta_c * P_chg[k] - P_dis[k] / eta_d) * dt_h / E_nom

Constraints:
    SOC_min <= SOC[k] <= SOC_max
    P_chg[k] + P_reg[k] <= P_max
    P_dis[k] + P_reg[k] <= P_max
    Endurance: SOC headroom for P_reg over ``endurance_hours``
    (same form as EconomicEMS endurance constraint).
    Terminal-SOC anchor (equality):
       SOC[N] - SOC_terminal == z_plus - z_minus

Objective (minimize the negative of profit, then add penalties):
    sum_k [ -price_e[k] * (P_dis[k] - P_chg[k]) * dt_h
            -price_reg_cap[k] * P_reg[k] * dt_h
            +deg_cost*alpha_deg*(P_chg[k]+P_dis[k])*dt_ems
            +deg_cost*alpha_deg_reg*P_reg[k]*dt_ems ]
    + TERMINAL_W * (z_plus + z_minus)

The degradation cost and terminal anchor are required for an
apples-to-apples comparison with the stochastic EconomicEMS, which
optimises the same physical objective. Without them, the LP would be
solving a strictly easier problem and produce misleadingly higher
"expected profit" by overcommitting FCR and dumping the battery to
SOC_min at the end of horizon.

Solver: scipy.optimize.linprog (HiGHS).  Single LP solve per EMS re-call.

Returns the same dict shape as ``EconomicEMS.solve`` so the simulator
strategy dispatch can swap them transparently.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linprog

from core.config.parameters import BatteryParams, EMSParams, ThermalParams, TimeParams

logger = logging.getLogger(__name__)


class DeterministicLP:
    """Perfect-foresight rolling-horizon LP dispatch (commercial baseline)."""

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        thp: ThermalParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.thp = thp

    # ------------------------------------------------------------------
    #  Public interface — matches EconomicEMS.solve signature shape
    # ------------------------------------------------------------------

    def solve(
        self,
        soc_init: float,
        soh_init: float,
        t_init: float,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
        vrc1_init: float = 0.0,
        vrc2_init: float = 0.0,
    ) -> dict:
        """Solve the deterministic LP over the horizon.

        Parameters mirror ``EconomicEMS.solve`` for drop-in compatibility.
        Only the **expected** (probability-weighted) price path is used —
        scenarios are collapsed to a single deterministic forecast, which
        is the standard commercial-EMS treatment.
        """
        bp = self.bp
        ep = self.ep
        N = min(ep.N_ems, energy_scenarios.shape[1])
        dt_h = self.tp.dt_ems / 3600.0   # hours per step (=1.0 for hourly EMS)

        # Collapse scenarios to expected price (probability-weighted).
        w = np.asarray(probabilities, dtype=float)
        e_price = np.asarray(energy_scenarios[:, :N], dtype=float).T @ w  # (N,)
        r_price = np.asarray(reg_scenarios[:, :N], dtype=float).T @ w     # (N,)

        # Clip initial state to strictly feasible region (matches EconomicEMS).
        soc_init = float(np.clip(soc_init, bp.SOC_min + 1e-3, bp.SOC_max - 1e-3))

        # ---- Decision vector layout ----
        # [P_chg(N), P_dis(N), P_reg(N), z_plus(1), z_minus(1)]
        # z_plus / z_minus are auxiliary nonneg slacks for the L1
        # terminal-SOC anchor:  SOC[N] - SOC_terminal == z_plus - z_minus.
        # Penalising (z_plus + z_minus) approximates a |·| penalty in
        # pure LP form (no quadratics — scipy linprog can't do them).
        n_var = 3 * N + 2
        i_chg = slice(0, N)
        i_dis = slice(N, 2 * N)
        i_reg = slice(2 * N, 3 * N)
        i_zp = 3 * N            # z_plus  index
        i_zm = 3 * N + 1        # z_minus index

        # Cost coefficients (linprog minimizes c^T x; we maximize profit).
        # Energy + FCR revenue terms (negative cost = positive revenue):
        c = np.zeros(n_var)
        c[i_chg] = +e_price * dt_h     # +cost to charge (negative revenue)
        c[i_dis] = -e_price * dt_h     # discharge revenue
        c[i_reg] = -r_price * dt_h     # capacity payment

        # Degradation cost — same split form as the EMS objective and the
        # simulator's profit accounting (alpha_deg for chg/dis throughput,
        # alpha_deg_reg for shallow symmetric reg cycling). Linear in
        # decisions, so it just adds to the cost coefficients.
        # Units: degradation_cost [$/SOH] * alpha_deg [1/(kW·s)] * dt[s] = $/kW
        deg_unit = ep.degradation_cost * self.tp.dt_ems   # $/kW per LP step
        c[i_chg] += deg_unit * bp.alpha_deg
        c[i_dis] += deg_unit * bp.alpha_deg
        c[i_reg] += deg_unit * bp.alpha_deg_reg

        # Terminal SOC anchor: penalise |SOC[N] - SOC_terminal| via L1.
        # Weight chosen so a 5% terminal drift costs ~$25 — comparable to
        # the EconomicEMS quadratic terminal_soc_weight = 1e4 evaluated
        # at the same drift (1e4 * 0.05^2 = $25).
        TERMINAL_W = 500.0     # $/SOC-unit  (linear)
        c[i_zp] = TERMINAL_W
        c[i_zm] = TERMINAL_W

        # ---- Bounds ----
        bounds = (
            [(0.0, bp.P_max_kw)] * N           # P_chg
            + [(0.0, bp.P_max_kw)] * N         # P_dis
            + [(0.0, bp.P_max_kw)] * N         # P_reg
            + [(0.0, None)] * 2                # z_plus, z_minus (>= 0)
        )

        # ---- Inequality constraints A_ub @ x <= b_ub ----
        # 1. Power budget: P_chg[k] + P_reg[k] <= P_max
        # 2. Power budget: P_dis[k] + P_reg[k] <= P_max
        # 3. SOC upper:    SOC[k] <= SOC_max  for k=1..N
        # 4. SOC lower:   -SOC[k] <= -SOC_min for k=1..N
        # 5. Endurance up: SOC[k] + reg_margin_high <= SOC_max
        # 6. Endurance dn: -SOC[k] + reg_margin_low <= -SOC_min
        #
        # SOC[k] = soc_init + (1/E_nom) * sum_{j<k} (eta_c*P_chg[j]*dt_h
        #                                            - P_dis[j]/eta_d * dt_h)
        E_nom = bp.E_nom_kwh
        eta_c = bp.eta_charge
        eta_d = bp.eta_discharge

        # Build cumulative-sum coefficient matrix M_soc[k, j] for k=1..N
        # so that SOC[k] - soc_init = M_soc[k] @ x_chg + M_dis[k] @ x_dis
        rows_ub = []
        bs_ub = []

        # Power budget: chg + reg <= P_max  and  dis + reg <= P_max
        for k in range(N):
            row = np.zeros(n_var)
            row[i_chg.start + k] = 1.0
            row[i_reg.start + k] = 1.0
            rows_ub.append(row)
            bs_ub.append(bp.P_max_kw)

            row = np.zeros(n_var)
            row[i_dis.start + k] = 1.0
            row[i_reg.start + k] = 1.0
            rows_ub.append(row)
            bs_ub.append(bp.P_max_kw)

        # SOC bounds + endurance, applied to SOC[k] for k=1..N
        # SOC[k] = soc_init + (dt_h/E_nom) * sum_{j=0}^{k-1} (eta_c*chg[j] - dis[j]/eta_d)
        endurance_h = ep.endurance_hours
        # Endurance margins are functions of P_reg[k-1] (the most recent commitment).
        # This matches EconomicEMS logic: "SOC[k+1] >= SOC_min + margin_low" using
        # P_reg[k]. In our indexing, after step j=0..k-1, the relevant reg power
        # is P_reg[k-1].
        for k in range(1, N + 1):
            # SOC[k] <= SOC_max - margin_high(P_reg[k-1])
            #   => (dt_h/E_nom) * sum (eta_c*chg - dis/eta_d) + margin_high <= SOC_max - soc_init
            row = np.zeros(n_var)
            for j in range(k):
                row[i_chg.start + j] = +(dt_h / E_nom) * eta_c
                row[i_dis.start + j] = -(dt_h / E_nom) / eta_d
            row[i_reg.start + (k - 1)] = +endurance_h * eta_c / E_nom
            rows_ub.append(row)
            bs_ub.append(bp.SOC_max - soc_init)

            # -SOC[k] <= -(SOC_min + margin_low(P_reg[k-1]))
            row2 = np.zeros(n_var)
            for j in range(k):
                row2[i_chg.start + j] = -(dt_h / E_nom) * eta_c
                row2[i_dis.start + j] = +(dt_h / E_nom) / eta_d
            row2[i_reg.start + (k - 1)] = +endurance_h / (E_nom * eta_d)
            rows_ub.append(row2)
            bs_ub.append(-(bp.SOC_min - soc_init))

        A_ub = np.array(rows_ub)
        b_ub = np.array(bs_ub)

        # ---- Equality: terminal SOC anchor ----
        # SOC[N] = soc_init + (dt_h/E_nom) * sum_{j=0}^{N-1} (eta_c*chg[j] - dis[j]/eta_d)
        # We want: SOC[N] - SOC_terminal == z_plus - z_minus
        # => sum_chg(coef) - sum_dis(coef) - z_plus + z_minus == SOC_terminal - soc_init
        eq_row = np.zeros(n_var)
        for j in range(N):
            eq_row[i_chg.start + j] = +(dt_h / E_nom) * eta_c
            eq_row[i_dis.start + j] = -(dt_h / E_nom) / eta_d
        eq_row[i_zp] = -1.0
        eq_row[i_zm] = +1.0
        A_eq = eq_row.reshape(1, -1)
        b_eq = np.array([bp.SOC_terminal - soc_init])

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            logger.error("DeterministicLP infeasible/failed: %s", res.message)
            return self._fallback_result(N, soc_init, soh_init, t_init)

        x = res.x
        p_chg_ref = x[i_chg]
        p_dis_ref = x[i_dis]
        p_reg_ref = x[i_reg]

        # Reconstruct SOC trajectory
        soc_ref = np.zeros(N + 1)
        soc_ref[0] = soc_init
        for k in range(N):
            soc_ref[k + 1] = soc_ref[k] + (dt_h / E_nom) * (
                eta_c * p_chg_ref[k] - p_dis_ref[k] / eta_d
            )

        # `expected_profit` is the negated LP objective, which includes the
        # L1 terminal-anchor slack penalty. Used only for the planner's own
        # log line below; the comparative `total_profit` reported across
        # strategies is computed independently by the simulator's ledger.
        expected_profit = float(-res.fun)

        logger.info(
            "DeterministicLP solved: objective=$%.2f  |  SOC [%.3f -> %.3f]  |  N=%d",
            expected_profit, soc_ref[0], soc_ref[-1], N,
        )

        return {
            "P_chg_ref": p_chg_ref,
            "P_dis_ref": p_dis_ref,
            "P_reg_ref": p_reg_ref,
            "SOC_ref": soc_ref,
            "SOH_ref": np.full(N + 1, soh_init),       # LP doesn't model SOH
            "TEMP_ref": np.full(N + 1, t_init),        # LP doesn't model T
            "VRC1_ref": np.zeros(N + 1),
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": expected_profit,
        }

    @staticmethod
    def _fallback_result(N: int, soc_init: float, soh_init: float, t_init: float) -> dict:
        """Idle fallback when the LP fails (should be unreachable in practice)."""
        return {
            "P_chg_ref": np.zeros(N),
            "P_dis_ref": np.zeros(N),
            "P_reg_ref": np.zeros(N),
            "SOC_ref": np.full(N + 1, soc_init),
            "SOH_ref": np.full(N + 1, soh_init),
            "TEMP_ref": np.full(N + 1, t_init),
            "VRC1_ref": np.zeros(N + 1),
            "VRC2_ref": np.zeros(N + 1),
            "expected_profit": 0.0,
        }
