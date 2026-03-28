"""Multi-rate simulation coordinator with FCR regulation activation.

v5_regulation_activation: 4-rate cascade with PI regulation inner loop.

Time scales
-----------
  dt_sim  = 4 s         plant integration  (BatteryPack.step)
  dt_pi   = 4 s         PI regulation controller  (= dt_sim)
  dt_mpc  = 60 s        MPC solve  +  EKF / MHE update
  dt_ems  = 3 600 s     EMS re-solve

Strategies
----------
  FULL:        EMS -> MPC -> PI -> Plant  (complete hierarchy)
  EMS_PI:      EMS -> PI -> Plant         (no MPC trajectory management)
  EMS_CLAMPS:  EMS -> Plant               (open-loop with hard SOC clamps)
  RULE_BASED:  Rule -> Plant              (price-sorted schedule, no optimization)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    RegControllerParams,
    RegulationParams,
    Strategy,
    ThermalParams,
    TimeParams,
)
from data.activation_generator import ActivationSignalGenerator
from ems.economic_ems import EconomicEMS
from estimation.ekf import ExtendedKalmanFilter
from estimation.mhe import MovingHorizonEstimator
from models.battery_model import BatteryPack, BatteryPlant
from mpc.tracking_mpc import TrackingMPC
from pi.regulation_controller import RegulationController
from revenue.regulation_revenue import (
    RegulationAccounting,
    compute_step_revenue,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Reference interpolation
# ---------------------------------------------------------------------------

def interpolate_ems_to_mpc(
    ems_result: dict,
    dt_ems: float,
    dt_mpc: float,
) -> dict:
    """Expand hourly EMS references to MPC resolution via zero-order hold."""
    ratio = int(round(dt_ems / dt_mpc))

    p_chg = np.repeat(ems_result["P_chg_ref"], ratio)
    p_dis = np.repeat(ems_result["P_dis_ref"], ratio)
    p_reg = np.repeat(ems_result["P_reg_ref"], ratio)

    # State: zero-order hold (end-of-hour target held for the full hour)
    soc_mpc = np.repeat(ems_result["SOC_ref"][1:], ratio)
    soh_mpc = np.repeat(ems_result["SOH_ref"][1:], ratio)
    temp_mpc = np.repeat(ems_result["TEMP_ref"][1:], ratio)

    return {
        "P_chg_ref_mpc": p_chg,
        "P_dis_ref_mpc": p_dis,
        "P_reg_ref_mpc": p_reg,
        "SOC_ref_mpc": soc_mpc,
        "SOH_ref_mpc": soh_mpc,
        "TEMP_ref_mpc": temp_mpc,
    }


# ---------------------------------------------------------------------------
#  Rule-based dispatch
# ---------------------------------------------------------------------------

def rule_based_dispatch(
    energy_prices: np.ndarray,
    bp: BatteryParams,
    n_hours: int,
) -> dict:
    """Generate rule-based charge/discharge schedule from price sorting.

    Charges during the cheapest 8 hours, discharges during the most
    expensive 8 hours.  No regulation commitment.
    """
    prices = energy_prices[:n_hours]
    order = np.argsort(prices)

    p_chg = np.zeros(n_hours)
    p_dis = np.zeros(n_hours)

    n_charge = min(8, n_hours // 3)
    n_discharge = min(8, n_hours // 3)

    p_chg[order[:n_charge]] = bp.P_max_kw * 0.8
    p_dis[order[-n_discharge:]] = bp.P_max_kw * 0.8

    soc_ref = np.full(n_hours + 1, bp.SOC_init)
    soh_ref = np.full(n_hours + 1, bp.SOH_init)
    temp_ref = np.full(n_hours + 1, 25.0)

    return {
        "P_chg_ref": p_chg,
        "P_dis_ref": p_dis,
        "P_reg_ref": np.zeros(n_hours),
        "SOC_ref": soc_ref,
        "SOH_ref": soh_ref,
        "TEMP_ref": temp_ref,
        "VRC1_ref": np.zeros(n_hours + 1),
        "VRC2_ref": np.zeros(n_hours + 1),
        "expected_profit": 0.0,
    }


# ---------------------------------------------------------------------------
#  Multi-rate simulator
# ---------------------------------------------------------------------------

class MultiRateSimulator:
    """Coordinates plant, EMS, MPC, PI, EKF, and MHE at their respective rates.

    v5: adds PI inner loop, activation signal, regulation accounting,
    and strategy-dependent dispatch.
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        mp: MPCParams,
        ekf_p: EKFParams,
        mhe_p: MHEParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        reg_ctrl_p: RegControllerParams,
        reg_p: RegulationParams,
        strategy: Strategy = Strategy.FULL,
        pp: PackParams | None = None,
        run_mhe: bool = False,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.mp = mp
        self.thp = thp
        self.elp = elp
        self.reg_ctrl_p = reg_ctrl_p
        self.reg_p = reg_p
        self.strategy = strategy
        self.pp = pp
        self.run_mhe = run_mhe

        # Plant
        if pp is not None:
            self.plant = BatteryPack(bp, tp, thp, elp, pp)
            self.n_cells = pp.n_cells
        else:
            self.plant = BatteryPlant(bp, tp, thp, elp)
            self.n_cells = 1

        # Controllers (created even if strategy doesn't use them — lightweight)
        self.ems = EconomicEMS(bp, tp, ep, thp, elp)
        self.mpc = TrackingMPC(bp, tp, mp, thp, elp)
        self.reg_ctrl = RegulationController(bp, reg_ctrl_p, tp.dt_pi)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)
        self.mhe: MovingHorizonEstimator | None = None
        if run_mhe:
            self.mhe = MovingHorizonEstimator(bp, tp, mhe_p, thp, elp)

        # Activation signal generator
        self.activation_gen = ActivationSignalGenerator(reg_p, tp.dt_pi)

    # ------------------------------------------------------------------
    #  Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:
        """Execute the full multi-rate closed-loop simulation."""
        bp = self.bp
        tp = self.tp
        ep = self.ep
        thp = self.thp
        strategy = self.strategy
        use_ems = strategy != Strategy.RULE_BASED
        use_mpc = strategy == Strategy.FULL
        use_pi = strategy in (Strategy.FULL, Strategy.EMS_PI)

        # Timing
        total_seconds = int(tp.sim_hours * 3600)
        steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)   # 15
        steps_per_ems = int(tp.dt_ems / tp.dt_sim)    # 900
        N_sim_steps = int(total_seconds / tp.dt_sim)
        N_mpc_steps = int(total_seconds / tp.dt_mpc)

        # Generate activation signal for entire simulation
        activation_signal_full = self.activation_gen.generate(N_sim_steps)

        # Pre-allocate storage (pack-level, dt_sim resolution)
        soc_true = np.zeros(N_sim_steps + 1)
        soh_true = np.zeros(N_sim_steps + 1)
        temp_true = np.zeros(N_sim_steps + 1)
        vrc1_true = np.zeros(N_sim_steps + 1)
        vrc2_true = np.zeros(N_sim_steps + 1)
        vterm_true = np.zeros(N_sim_steps + 1)

        # Estimator states (dt_mpc resolution)
        soc_ekf = np.zeros(N_mpc_steps + 1)
        soh_ekf = np.zeros(N_mpc_steps + 1)
        temp_ekf = np.zeros(N_mpc_steps + 1)
        vrc1_ekf = np.zeros(N_mpc_steps + 1)
        vrc2_ekf = np.zeros(N_mpc_steps + 1)
        soc_mhe = np.zeros(N_mpc_steps + 1)
        soh_mhe = np.zeros(N_mpc_steps + 1)
        temp_mhe = np.zeros(N_mpc_steps + 1)
        vrc1_mhe = np.zeros(N_mpc_steps + 1)
        vrc2_mhe = np.zeros(N_mpc_steps + 1)

        power_applied = np.zeros((N_sim_steps, 3))  # at dt_sim resolution now
        power_mpc_base = np.zeros((N_mpc_steps, 3))

        # Timing instrumentation
        mpc_solve_times = np.zeros(N_mpc_steps)
        est_solve_times = np.zeros(N_mpc_steps)
        mpc_solver_failures = 0

        # Reference tracking
        soc_ref_at_mpc = np.zeros(N_mpc_steps)
        power_ref_at_mpc = np.zeros((N_mpc_steps, 3))

        # Regulation tracking (dt_sim resolution)
        power_delivered_log = np.zeros(N_sim_steps)
        activation_log = activation_signal_full.copy()

        # EMS reference storage (for plotting)
        ems_soc_refs: list[np.ndarray] = []

        # Cell-level arrays (only when multi-cell)
        n_cells = self.n_cells
        has_cells = n_cells > 1
        if has_cells:
            cell_socs = np.zeros((n_cells, N_sim_steps + 1))
            cell_sohs = np.zeros((n_cells, N_sim_steps + 1))
            cell_temps = np.zeros((n_cells, N_sim_steps + 1))
            cell_vrc1s = np.zeros((n_cells, N_sim_steps + 1))
            cell_vrc2s = np.zeros((n_cells, N_sim_steps + 1))
            balancing_power_log = np.zeros((n_cells, N_mpc_steps))

        # Initialise
        x_true = self.plant.get_state()
        soc_true[0] = x_true[0]
        soh_true[0] = x_true[1]
        temp_true[0] = x_true[2]
        vrc1_true[0] = x_true[3]
        vrc2_true[0] = x_true[4]
        vterm_true[0] = self.plant.get_terminal_voltage()

        if has_cells:
            cs = self.plant.get_cell_states()
            cell_socs[:, 0] = cs[:, 0]
            cell_sohs[:, 0] = cs[:, 1]
            cell_temps[:, 0] = cs[:, 2]
            cell_vrc1s[:, 0] = cs[:, 3]
            cell_vrc2s[:, 0] = cs[:, 4]

        ekf_est = self.ekf.get_estimate()
        soc_ekf[0] = ekf_est[0]
        soh_ekf[0] = ekf_est[1]
        temp_ekf[0] = ekf_est[2]
        vrc1_ekf[0] = ekf_est[3]
        vrc2_ekf[0] = ekf_est[4]

        if self.run_mhe:
            mhe_est = self.mhe.get_estimate()
            soc_mhe[0] = mhe_est[0]
            soh_mhe[0] = mhe_est[1]
            temp_mhe[0] = mhe_est[2]
            vrc1_mhe[0] = mhe_est[3]
            vrc2_mhe[0] = mhe_est[4]

        # Current commands
        u_mpc = np.zeros(3)   # MPC base setpoint
        u_actual = np.zeros(3)  # actual power to plant (from PI or direct)
        P_reg_committed = 0.0

        # Interpolated MPC-resolution references
        soc_ref_mpc_local = np.full(N_mpc_steps + 1, bp.SOC_init)
        soh_ref_mpc_local = np.full(N_mpc_steps + 1, bp.SOH_init)
        temp_ref_mpc_local = np.full(N_mpc_steps + 1, thp.T_init)
        p_chg_ref_mpc_local = np.zeros(N_mpc_steps)
        p_dis_ref_mpc_local = np.zeros(N_mpc_steps)
        p_reg_ref_mpc_local = np.zeros(N_mpc_steps)

        mpc_ref_base = 0
        mpc_idx = 0

        # Profit tracking
        cum_energy_profit = 0.0
        cum_deg_cost = 0.0
        energy_profit_arr = np.zeros(N_mpc_steps)
        deg_cost_arr = np.zeros(N_mpc_steps)

        # Regulation accounting
        accounting = RegulationAccounting()
        reg_accounting_arr = np.zeros((N_sim_steps, 4))  # cap, del, pen, score

        for sim_step in range(N_sim_steps):
            # ===========================================================
            #  EMS update  (every dt_ems = 3 600 s)
            # ===========================================================
            if sim_step % steps_per_ems == 0:
                ems_hour = sim_step // steps_per_ems

                if use_ems:
                    x_est = self.ekf.get_estimate()

                    remaining_hours = min(
                        ep.N_ems, energy_scenarios.shape[1] - ems_hour
                    )
                    if remaining_hours < 1:
                        remaining_hours = 1

                    e_scen = energy_scenarios[:, ems_hour:ems_hour + remaining_hours]
                    r_scen = reg_scenarios[:, ems_hour:ems_hour + remaining_hours]

                    if e_scen.shape[1] < ep.N_ems:
                        pad_w = ep.N_ems - e_scen.shape[1]
                        e_scen = np.pad(e_scen, ((0, 0), (0, pad_w)), mode="edge")
                        r_scen = np.pad(r_scen, ((0, 0), (0, pad_w)), mode="edge")

                    logger.info(
                        "EMS solve at t=%d s (hour %d), SOC=%.3f, SOH=%.6f, T=%.1f",
                        sim_step * tp.dt_sim, ems_hour,
                        x_est[0], x_est[1], x_est[2],
                    )

                    ems_result = self.ems.solve(
                        soc_init=x_est[0],
                        soh_init=x_est[1],
                        t_init=x_est[2],
                        vrc1_init=x_est[3],
                        vrc2_init=x_est[4],
                        energy_scenarios=e_scen,
                        reg_scenarios=r_scen,
                        probabilities=probabilities,
                    )
                else:
                    # Rule-based
                    n_hours = min(24, energy_scenarios.shape[1] - ems_hour)
                    if n_hours < 1:
                        n_hours = 1
                    ems_result = rule_based_dispatch(
                        energy_scenarios[0, ems_hour:ems_hour + n_hours],
                        bp, n_hours,
                    )

                P_reg_committed = float(ems_result["P_reg_ref"][0])
                ems_soc_refs.append(ems_result["SOC_ref"].copy())

                # Blending
                if ems_hour > 0:
                    off = min(mpc_ref_base, len(p_chg_ref_mpc_local) - 1)
                    prev_p_chg_end = float(p_chg_ref_mpc_local[off])
                    prev_p_dis_end = float(p_dis_ref_mpc_local[off])
                    prev_p_reg_end = float(p_reg_ref_mpc_local[off])

                refs = interpolate_ems_to_mpc(ems_result, tp.dt_ems, tp.dt_mpc)
                mpc_ref_base = 0

                soc_ref_mpc_local = refs["SOC_ref_mpc"]
                soh_ref_mpc_local = refs["SOH_ref_mpc"]
                temp_ref_mpc_local = refs["TEMP_ref_mpc"]
                p_chg_ref_mpc_local = refs["P_chg_ref_mpc"]
                p_dis_ref_mpc_local = refs["P_dis_ref_mpc"]
                p_reg_ref_mpc_local = refs["P_reg_ref_mpc"]

                if ems_hour > 0:
                    Nb = min(self.mp.n_blend_steps, len(p_chg_ref_mpc_local))
                    alpha = np.linspace(1.0 / (Nb + 1), 1.0, Nb)
                    p_chg_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_chg_end
                        + alpha * p_chg_ref_mpc_local[:Nb]
                    )
                    p_dis_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_dis_end
                        + alpha * p_dis_ref_mpc_local[:Nb]
                    )
                    p_reg_ref_mpc_local[:Nb] = (
                        (1.0 - alpha) * prev_p_reg_end
                        + alpha * p_reg_ref_mpc_local[:Nb]
                    )

            # ===========================================================
            #  MPC + Estimation update  (every dt_mpc = 60 s)
            # ===========================================================
            if sim_step % steps_per_mpc == 0:
                y_meas = self.plant.get_measurement()

                t0_est = time.perf_counter()
                if sim_step > 0:
                    ekf_est = self.ekf.step(u_actual, y_meas)
                    if self.run_mhe:
                        mhe_est = self.mhe.step(u_actual, y_meas)
                else:
                    ekf_est = self.ekf.get_estimate()
                    if self.run_mhe:
                        mhe_est = self.mhe.get_estimate()
                if mpc_idx < N_mpc_steps:
                    est_solve_times[mpc_idx] = time.perf_counter() - t0_est

                if mpc_idx < N_mpc_steps:
                    soc_ekf[mpc_idx] = ekf_est[0]
                    soh_ekf[mpc_idx] = ekf_est[1]
                    temp_ekf[mpc_idx] = ekf_est[2]
                    vrc1_ekf[mpc_idx] = ekf_est[3]
                    vrc2_ekf[mpc_idx] = ekf_est[4]
                    if self.run_mhe:
                        soc_mhe[mpc_idx] = mhe_est[0]
                        soh_mhe[mpc_idx] = mhe_est[1]
                        temp_mhe[mpc_idx] = mhe_est[2]
                        vrc1_mhe[mpc_idx] = mhe_est[3]
                        vrc2_mhe[mpc_idx] = mhe_est[4]

                if use_mpc:
                    off = mpc_ref_base
                    N_pred = self.mp.N_mpc

                    soc_win = self._extract_ref(soc_ref_mpc_local, off, N_pred + 1)
                    pc_win = self._extract_ref(p_chg_ref_mpc_local, off, N_pred)
                    pd_win = self._extract_ref(p_dis_ref_mpc_local, off, N_pred)
                    pr_win = self._extract_ref(p_reg_ref_mpc_local, off, N_pred)

                    if mpc_idx < N_mpc_steps:
                        soc_ref_at_mpc[mpc_idx] = soc_win[0]
                        power_ref_at_mpc[mpc_idx] = [pc_win[0], pd_win[0], pr_win[0]]

                    t0_mpc = time.perf_counter()
                    u_mpc = self.mpc.solve(
                        x_est=ekf_est,
                        soc_ref=soc_win,
                        p_chg_ref=pc_win,
                        p_dis_ref=pd_win,
                        p_reg_ref=pr_win,
                        u_prev=u_mpc,
                    )
                    if mpc_idx < N_mpc_steps:
                        mpc_solve_times[mpc_idx] = time.perf_counter() - t0_mpc
                    if self.mpc.last_solve_failed:
                        mpc_solver_failures += 1
                else:
                    # No MPC: use EMS reference directly
                    off = min(mpc_ref_base, len(p_chg_ref_mpc_local) - 1)
                    u_mpc = np.array([
                        p_chg_ref_mpc_local[off],
                        p_dis_ref_mpc_local[off],
                        p_reg_ref_mpc_local[off],
                    ])

                if mpc_idx < N_mpc_steps:
                    power_mpc_base[mpc_idx] = u_mpc

                    # Log balancing power
                    if has_cells:
                        balancing_power_log[:, mpc_idx] = self.plant.get_balancing_power()

                    # Energy profit accounting (at MPC resolution)
                    ems_hour_now = sim_step // steps_per_ems
                    if ems_hour_now < energy_scenarios.shape[1]:
                        price_e = float(energy_scenarios[0, ems_hour_now])
                    else:
                        price_e = float(energy_scenarios[0, -1])

                    dt_h = tp.dt_mpc / 3600.0
                    e_profit = price_e * (u_mpc[1] - u_mpc[0]) * dt_h
                    d_cost = (
                        ep.degradation_cost
                        * bp.alpha_deg
                        * (u_mpc[0] + u_mpc[1] + u_mpc[2])
                        * tp.dt_mpc
                    )
                    cum_energy_profit += e_profit
                    cum_deg_cost += d_cost
                    energy_profit_arr[mpc_idx] = e_profit
                    deg_cost_arr[mpc_idx] = d_cost

                mpc_ref_base += 1
                mpc_idx += 1

            # ===========================================================
            #  PI + Plant step  (every dt_sim = dt_pi = 4 s)
            # ===========================================================
            activation = activation_signal_full[sim_step]

            if use_pi:
                u_actual, P_delivered = self.reg_ctrl.compute(
                    P_chg_base=u_mpc[0],
                    P_dis_base=u_mpc[1],
                    P_reg_committed=P_reg_committed,
                    activation_signal=activation,
                    soc_current=ekf_est[0],
                )
            elif strategy == Strategy.EMS_CLAMPS:
                # Open-loop with hard SOC clamps
                P_reg_demand = activation * P_reg_committed
                P_chg = u_mpc[0]
                P_dis = u_mpc[1]

                if P_reg_demand > 0:
                    P_dis += P_reg_demand
                else:
                    P_chg += abs(P_reg_demand)

                # Hard SOC clamps
                soc_now = soc_true[sim_step]
                if soc_now <= bp.SOC_min + 0.01:
                    P_dis = 0.0
                if soc_now >= bp.SOC_max - 0.01:
                    P_chg = 0.0

                P_chg = np.clip(P_chg, 0.0, bp.P_max_kw)
                P_dis = np.clip(P_dis, 0.0, bp.P_max_kw)

                P_delivered = (P_dis - u_mpc[1]) - (P_chg - u_mpc[0])
                u_actual = np.array([P_chg, P_dis, P_reg_committed])
            else:
                # Rule-based: no regulation delivery
                u_actual = np.array([u_mpc[0], u_mpc[1], 0.0])
                P_delivered = 0.0

            # Regulation revenue accounting (every PI step)
            ems_hour_now = sim_step // steps_per_ems
            if ems_hour_now < reg_scenarios.shape[1]:
                price_r = float(reg_scenarios[0, ems_hour_now])
            else:
                price_r = float(reg_scenarios[0, -1])

            cap_rev, del_rev, pen, is_ok = compute_step_revenue(
                P_reg_committed, activation, P_delivered,
                price_r, self.reg_p, tp.dt_pi,
            )
            is_active = abs(activation) > 1e-6
            accounting.update(cap_rev, del_rev, pen, is_ok, is_active)
            reg_accounting_arr[sim_step] = [cap_rev, del_rev, pen, float(is_ok)]

            # Plant step
            x_new, _ = self.plant.step(u_actual)
            soc_true[sim_step + 1] = x_new[0]
            soh_true[sim_step + 1] = x_new[1]
            temp_true[sim_step + 1] = x_new[2]
            vrc1_true[sim_step + 1] = x_new[3]
            vrc2_true[sim_step + 1] = x_new[4]
            vterm_true[sim_step + 1] = self.plant.get_terminal_voltage()

            power_applied[sim_step] = u_actual
            power_delivered_log[sim_step] = P_delivered

            # Log cell-level states
            if has_cells:
                cs = self.plant.get_cell_states()
                cell_socs[:, sim_step + 1] = cs[:, 0]
                cell_sohs[:, sim_step + 1] = cs[:, 1]
                cell_temps[:, sim_step + 1] = cs[:, 2]
                cell_vrc1s[:, sim_step + 1] = cs[:, 3]
                cell_vrc2s[:, sim_step + 1] = cs[:, 4]

        # Trim estimator arrays
        soc_ekf = soc_ekf[:mpc_idx]
        soh_ekf = soh_ekf[:mpc_idx]
        temp_ekf = temp_ekf[:mpc_idx]
        vrc1_ekf = vrc1_ekf[:mpc_idx]
        vrc2_ekf = vrc2_ekf[:mpc_idx]
        soc_mhe = soc_mhe[:mpc_idx]
        soh_mhe = soh_mhe[:mpc_idx]
        temp_mhe = temp_mhe[:mpc_idx]
        vrc1_mhe = vrc1_mhe[:mpc_idx]
        vrc2_mhe = vrc2_mhe[:mpc_idx]

        # Total profit
        total_profit = (
            cum_energy_profit
            + accounting.net_regulation_profit
            - cum_deg_cost
        )

        logger.info(
            "Simulation complete [%s]: profit=$%.2f "
            "(energy=$%.2f, reg_net=$%.2f, deg=$%.2f), "
            "delivery_score=%.1f%%, SOH loss=%.4f%%",
            strategy.value,
            total_profit,
            cum_energy_profit,
            accounting.net_regulation_profit,
            cum_deg_cost,
            accounting.delivery_score * 100,
            (soh_true[0] - soh_true[-1]) * 100,
        )

        result = {
            # Strategy
            "strategy": strategy.value,
            # Plant (dt_sim resolution)
            "time_sim": np.arange(N_sim_steps + 1) * tp.dt_sim,
            "soc_true": soc_true,
            "soh_true": soh_true,
            "temp_true": temp_true,
            "vrc1_true": vrc1_true,
            "vrc2_true": vrc2_true,
            "vterm_true": vterm_true,
            # Estimators (dt_mpc resolution)
            "time_mpc": np.arange(len(soc_ekf)) * tp.dt_mpc,
            "soc_ekf": soc_ekf,
            "soh_ekf": soh_ekf,
            "temp_ekf": temp_ekf,
            "vrc1_ekf": vrc1_ekf,
            "vrc2_ekf": vrc2_ekf,
            "soc_mhe": soc_mhe,
            "soh_mhe": soh_mhe,
            "temp_mhe": temp_mhe,
            "vrc1_mhe": vrc1_mhe,
            "vrc2_mhe": vrc2_mhe,
            # Applied power (dt_sim resolution)
            "power_applied": power_applied,
            "power_mpc_base": power_mpc_base[:mpc_idx],
            # Activation & regulation (dt_sim resolution)
            "activation_signal": activation_log,
            "power_delivered": power_delivered_log,
            "reg_accounting": reg_accounting_arr,
            # Regulation summary
            "delivery_score": accounting.delivery_score,
            "capacity_revenue": accounting.capacity_revenue,
            "delivery_revenue": accounting.delivery_revenue,
            "penalty_cost": accounting.penalty_cost,
            "net_regulation_profit": accounting.net_regulation_profit,
            # Profit
            "energy_profit_total": cum_energy_profit,
            "energy_profit": energy_profit_arr[:mpc_idx],
            "deg_cost_total": cum_deg_cost,
            "deg_cost": deg_cost_arr[:mpc_idx],
            "total_profit": total_profit,
            "soh_degradation": soh_true[0] - soh_true[-1],
            # EMS references (for plotting)
            "ems_soc_refs": ems_soc_refs,
            # Scenario prices (probability-weighted expected prices)
            "prices_energy": (probabilities[:, None] * energy_scenarios).sum(axis=0),
            "prices_reg": (probabilities[:, None] * reg_scenarios).sum(axis=0),
            # Timing
            "mpc_solve_times": mpc_solve_times[:mpc_idx],
            "est_solve_times": est_solve_times[:mpc_idx],
            "mpc_solver_failures": mpc_solver_failures,
            # Reference tracking
            "soc_ref_at_mpc": soc_ref_at_mpc[:mpc_idx],
            "power_ref_at_mpc": power_ref_at_mpc[:mpc_idx],
        }

        # Cell-level data
        if has_cells:
            result["cell_socs"] = cell_socs
            result["cell_sohs"] = cell_sohs
            result["cell_temps"] = cell_temps
            result["cell_vrc1s"] = cell_vrc1s
            result["cell_vrc2s"] = cell_vrc2s
            result["balancing_power"] = balancing_power_log[:, :mpc_idx]
            result["soc_imbalance"] = (
                np.max(cell_socs, axis=0) - np.min(cell_socs, axis=0)
            )
            result["n_cells"] = n_cells

        return result

    @staticmethod
    def _extract_ref(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
        """Extract a window from *arr* starting at *offset*, padding if needed."""
        end = offset + length
        if end <= len(arr):
            return arr[offset:end].copy()
        available = arr[offset:].copy()
        pad_val = available[-1] if len(available) > 0 else 0.0
        return np.concatenate([available, np.full(length - len(available), pad_val)])
