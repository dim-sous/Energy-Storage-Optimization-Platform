"""Multi-rate simulation coordinator.

Time scales
-----------
  dt_sim  = 1 s         plant integration  (BatteryPlant.step)
  dt_mpc  = 60 s        MPC solve  +  EKF / MHE update
  dt_ems  = 3 600 s     EMS re-solve

Coordination
------------
  for each 1 s plant step:
      apply current MPC command to plant
      if  time % dt_mpc == 0:
          get measurement
          run EKF predict + update
          run MHE step
          run MPC solve with estimated state
      if  time % dt_ems == 0:
          run EMS with current state estimate
          interpolate new references
"""

from __future__ import annotations

import logging

import numpy as np

from config.parameters import (
    BatteryParams,
    EKFParams,
    EMSParams,
    MHEParams,
    MPCParams,
    TimeParams,
)
from data.price_generator import PriceGenerator
from ems.economic_ems import EconomicEMS
from estimation.ekf import ExtendedKalmanFilter
from estimation.mhe import MovingHorizonEstimator
from models.battery_model import BatteryPlant
from mpc.tracking_mpc import TrackingMPC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Reference interpolation
# ---------------------------------------------------------------------------

def interpolate_ems_to_mpc(
    ems_result: dict,
    dt_ems: float,
    dt_mpc: float,
) -> dict:
    """Interpolate hourly EMS references to MPC resolution.

    Power references use **zero-order hold** (constant within each hour).
    State references (SOC, SOH) use **linear interpolation** between
    hourly knot points for smooth tracking targets.

    Parameters
    ----------
    ems_result : dict
        Output from ``EconomicEMS.solve()``.
    dt_ems : float  [s]
    dt_mpc : float  [s]

    Returns
    -------
    dict with keys  ``*_mpc``  at MPC resolution.
    """
    ratio = int(round(dt_ems / dt_mpc))  # 60

    # Power: zero-order hold
    p_chg = np.repeat(ems_result["P_chg_ref"], ratio)
    p_dis = np.repeat(ems_result["P_dis_ref"], ratio)
    p_reg = np.repeat(ems_result["P_reg_ref"], ratio)

    # State: linear interpolation between hourly knots
    soc_hourly = ems_result["SOC_ref"]         # (N+1,)
    soh_hourly = ems_result["SOH_ref"]         # (N+1,)
    N_hours = len(soc_hourly) - 1
    N_mpc_pts = N_hours * ratio + 1

    t_hourly = np.arange(N_hours + 1, dtype=np.float64)
    t_mpc = np.linspace(0.0, N_hours, N_mpc_pts)

    soc_mpc = np.interp(t_mpc, t_hourly, soc_hourly)
    soh_mpc = np.interp(t_mpc, t_hourly, soh_hourly)

    return {
        "P_chg_ref_mpc": p_chg,
        "P_dis_ref_mpc": p_dis,
        "P_reg_ref_mpc": p_reg,
        "SOC_ref_mpc": soc_mpc,
        "SOH_ref_mpc": soh_mpc,
    }


# ---------------------------------------------------------------------------
#  Multi-rate simulator
# ---------------------------------------------------------------------------

class MultiRateSimulator:
    """Coordinates plant, EMS, MPC, EKF, and MHE at their respective rates.

    Parameters
    ----------
    bp : BatteryParams
    tp : TimeParams
    ep : EMSParams
    mp : MPCParams
    ekf_p : EKFParams
    mhe_p : MHEParams
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        ep: EMSParams,
        mp: MPCParams,
        ekf_p: EKFParams,
        mhe_p: MHEParams,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.ep = ep
        self.mp = mp

        # Sub-components
        self.plant = BatteryPlant(bp, tp)
        self.ems = EconomicEMS(bp, tp, ep)
        self.mpc = TrackingMPC(bp, tp, mp)
        self.ekf = ExtendedKalmanFilter(bp, tp, ekf_p)
        self.mhe = MovingHorizonEstimator(bp, tp, mhe_p)

    # ------------------------------------------------------------------
    #  Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        energy_scenarios: np.ndarray,
        reg_scenarios: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:
        """Execute the full multi-rate closed-loop simulation.

        Parameters
        ----------
        energy_scenarios : ndarray (n_scenarios, n_hours)
        reg_scenarios    : ndarray (n_scenarios, n_hours)
        probabilities    : ndarray (n_scenarios,)

        Returns
        -------
        dict   (see below for keys)
        """
        bp = self.bp
        tp = self.tp
        ep = self.ep

        # Timing
        total_seconds = int(tp.sim_hours * 3600)
        steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)            # 60
        steps_per_ems = int(tp.dt_ems / tp.dt_sim)            # 3600
        N_sim_steps = int(total_seconds / tp.dt_sim)           # 86 400
        N_mpc_steps = int(total_seconds / tp.dt_mpc)           # 1 440
        N_ems_steps = int(total_seconds / tp.dt_ems)           # 24

        # Pre-allocate storage
        soc_true = np.zeros(N_sim_steps + 1)
        soh_true = np.zeros(N_sim_steps + 1)
        soc_ekf = np.zeros(N_mpc_steps + 1)
        soh_ekf = np.zeros(N_mpc_steps + 1)
        soc_mhe = np.zeros(N_mpc_steps + 1)
        soh_mhe = np.zeros(N_mpc_steps + 1)
        power_applied = np.zeros((N_mpc_steps, 3))  # [P_chg, P_dis, P_reg]

        # EMS-level reference storage (for plotting)
        ems_p_chg_refs: list[np.ndarray] = []
        ems_p_dis_refs: list[np.ndarray] = []
        ems_p_reg_refs: list[np.ndarray] = []
        ems_soc_refs: list[np.ndarray] = []

        # Initialise
        x_true = self.plant.get_state()
        soc_true[0] = x_true[0]
        soh_true[0] = x_true[1]

        ekf_est = self.ekf.get_estimate()
        soc_ekf[0] = ekf_est[0]
        soh_ekf[0] = ekf_est[1]

        mhe_est = self.mhe.get_estimate()
        soc_mhe[0] = mhe_est[0]
        soh_mhe[0] = mhe_est[1]

        # Current control command (held between MPC updates)
        u_current = np.zeros(3)

        # Interpolated MPC-resolution references (set by EMS)
        soc_ref_mpc = np.full(N_mpc_steps + 1, bp.SOC_init)
        soh_ref_mpc = np.full(N_mpc_steps + 1, bp.SOH_init)
        p_chg_ref_mpc = np.zeros(N_mpc_steps)
        p_dis_ref_mpc = np.zeros(N_mpc_steps)
        p_reg_ref_mpc = np.zeros(N_mpc_steps)

        # Offset into the interpolated references (reset every EMS solve)
        mpc_ref_base = 0

        mpc_idx = 0
        cum_profit = 0.0
        cum_profit_arr = np.zeros(N_mpc_steps)
        energy_profit_arr = np.zeros(N_mpc_steps)
        reg_profit_arr = np.zeros(N_mpc_steps)
        deg_cost_arr = np.zeros(N_mpc_steps)

        for sim_step in range(N_sim_steps):
            t_sec = sim_step * tp.dt_sim

            # ===========================================================
            #  EMS update  (every dt_ems = 3 600 s)
            # ===========================================================
            if sim_step % steps_per_ems == 0:
                ems_hour = sim_step // steps_per_ems
                x_est = self.ekf.get_estimate()

                remaining_hours = min(
                    ep.N_ems, energy_scenarios.shape[1] - ems_hour
                )
                if remaining_hours < 1:
                    remaining_hours = 1

                e_scen = energy_scenarios[:, ems_hour : ems_hour + remaining_hours]
                r_scen = reg_scenarios[:, ems_hour : ems_hour + remaining_hours]

                # Pad scenarios if shorter than N_ems
                if e_scen.shape[1] < ep.N_ems:
                    pad_w = ep.N_ems - e_scen.shape[1]
                    e_scen = np.pad(e_scen, ((0, 0), (0, pad_w)), mode="edge")
                    r_scen = np.pad(r_scen, ((0, 0), (0, pad_w)), mode="edge")

                logger.info(
                    "EMS solve at t=%d s (hour %d), SOC=%.3f, SOH=%.6f",
                    sim_step, ems_hour, x_est[0], x_est[1],
                )

                ems_result = self.ems.solve(
                    soc_init=x_est[0],
                    soh_init=x_est[1],
                    energy_scenarios=e_scen,
                    reg_scenarios=r_scen,
                    probabilities=probabilities,
                )

                # Save EMS references for plotting
                ems_p_chg_refs.append(ems_result["P_chg_ref"].copy())
                ems_p_dis_refs.append(ems_result["P_dis_ref"].copy())
                ems_p_reg_refs.append(ems_result["P_reg_ref"].copy())
                ems_soc_refs.append(ems_result["SOC_ref"].copy())

                # Save old power reference endpoint before overwriting.
                # Used for short blend at the EMS boundary to smooth
                # the power reference transition (avoids MPC spikes).
                if ems_hour > 0:
                    off = min(mpc_ref_base, len(p_chg_ref_mpc_local) - 1)
                    prev_p_chg_end = float(p_chg_ref_mpc_local[off])
                    prev_p_dis_end = float(p_dis_ref_mpc_local[off])
                    prev_p_reg_end = float(p_reg_ref_mpc_local[off])

                # Interpolate to MPC resolution
                refs = interpolate_ems_to_mpc(ems_result, tp.dt_ems, tp.dt_mpc)

                # Compute how many MPC steps remain in this EMS window
                mpc_ref_base = 0

                soc_ref_mpc_local = refs["SOC_ref_mpc"]
                soh_ref_mpc_local = refs["SOH_ref_mpc"]
                p_chg_ref_mpc_local = refs["P_chg_ref_mpc"]
                p_dis_ref_mpc_local = refs["P_dis_ref_mpc"]
                p_reg_ref_mpc_local = refs["P_reg_ref_mpc"]

                # Short blend: ramp power references from the old plan's
                # endpoint to the new plan over n_blend_steps.  State
                # references (SOC/SOH) are already smooth from linear
                # interpolation and don't need blending.  Beyond the
                # blend region the MPC sees the full new plan, including
                # the fresh terminal target — critical for Q_terminal.
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
                # Get measurement
                y_meas = self.plant.get_measurement()

                if sim_step > 0:
                    # EKF step
                    ekf_est = self.ekf.step(u_current, y_meas)
                    # MHE step
                    mhe_est = self.mhe.step(u_current, y_meas)
                else:
                    ekf_est = self.ekf.get_estimate()
                    mhe_est = self.mhe.get_estimate()

                soc_ekf[mpc_idx] = ekf_est[0]
                soh_ekf[mpc_idx] = ekf_est[1]
                soc_mhe[mpc_idx] = mhe_est[0]
                soh_mhe[mpc_idx] = mhe_est[1]

                # Extract reference windows for MPC
                off = mpc_ref_base
                N_pred = self.mp.N_mpc

                soc_win = self._extract_ref(soc_ref_mpc_local, off, N_pred + 1)
                soh_win = self._extract_ref(soh_ref_mpc_local, off, N_pred + 1)
                pc_win = self._extract_ref(p_chg_ref_mpc_local, off, N_pred)
                pd_win = self._extract_ref(p_dis_ref_mpc_local, off, N_pred)
                pr_win = self._extract_ref(p_reg_ref_mpc_local, off, N_pred)

                # MPC solve (EKF estimate as state feedback)
                u_current = self.mpc.solve(
                    x_est=ekf_est,
                    soc_ref=soc_win,
                    soh_ref=soh_win,
                    p_chg_ref=pc_win,
                    p_dis_ref=pd_win,
                    p_reg_ref=pr_win,
                    u_prev=u_current,
                )

                # Log applied power
                if mpc_idx < N_mpc_steps:
                    power_applied[mpc_idx] = u_current

                    # Profit calculation
                    ems_hour_now = sim_step // steps_per_ems
                    if ems_hour_now < energy_scenarios.shape[1]:
                        price_e = float(energy_scenarios[0, ems_hour_now])
                        price_r = float(reg_scenarios[0, ems_hour_now])
                    else:
                        price_e = float(energy_scenarios[0, -1])
                        price_r = float(reg_scenarios[0, -1])

                    dt_h = tp.dt_mpc / 3600.0
                    e_profit = price_e * (u_current[1] - u_current[0]) * dt_h
                    r_profit = price_r * u_current[2] * dt_h
                    d_cost = (
                        self.ep.degradation_cost
                        * bp.alpha_deg
                        * (u_current[0] + u_current[1] + u_current[2])
                        * tp.dt_mpc
                    )
                    cum_profit += e_profit + r_profit - d_cost
                    cum_profit_arr[mpc_idx] = cum_profit
                    energy_profit_arr[mpc_idx] = e_profit
                    reg_profit_arr[mpc_idx] = r_profit
                    deg_cost_arr[mpc_idx] = d_cost

                mpc_ref_base += 1
                mpc_idx += 1

            # ===========================================================
            #  Plant step  (every dt_sim = 1 s)
            # ===========================================================
            x_new, _ = self.plant.step(u_current)
            soc_true[sim_step + 1] = x_new[0]
            soh_true[sim_step + 1] = x_new[1]

        # Trim estimator arrays to actual number of entries written
        soc_ekf = soc_ekf[:mpc_idx]
        soh_ekf = soh_ekf[:mpc_idx]
        soc_mhe = soc_mhe[:mpc_idx]
        soh_mhe = soh_mhe[:mpc_idx]

        logger.info(
            "Simulation complete: profit=$%.2f, SOH loss=%.4f%%",
            cum_profit,
            (soh_true[0] - soh_true[-1]) * 100,
        )

        return {
            # Plant (dt_sim resolution)
            "time_sim": np.arange(N_sim_steps + 1) * tp.dt_sim,
            "soc_true": soc_true,
            "soh_true": soh_true,
            # Estimators (dt_mpc resolution)
            "time_mpc": np.arange(len(soc_ekf)) * tp.dt_mpc,
            "soc_ekf": soc_ekf,
            "soh_ekf": soh_ekf,
            "soc_mhe": soc_mhe,
            "soh_mhe": soh_mhe,
            # Applied power (dt_mpc resolution)
            "power_applied": power_applied[:mpc_idx],
            # Profit
            "cumulative_profit": cum_profit_arr[:mpc_idx],
            "energy_profit": energy_profit_arr[:mpc_idx],
            "reg_profit": reg_profit_arr[:mpc_idx],
            "deg_cost": deg_cost_arr[:mpc_idx],
            "total_profit": cum_profit,
            # Degradation
            "soh_degradation": soh_true[0] - soh_true[-1],
            # EMS references (for plotting)
            "ems_p_chg_refs": ems_p_chg_refs,
            "ems_p_dis_refs": ems_p_dis_refs,
            "ems_p_reg_refs": ems_p_reg_refs,
            "ems_soc_refs": ems_soc_refs,
            # Scenario prices (first scenario for plotting)
            "prices_energy": energy_scenarios[0],
            "prices_reg": reg_scenarios[0],
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ref(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
        """Extract a window from *arr* starting at *offset*, padding if needed."""
        end = offset + length
        if end <= len(arr):
            return arr[offset:end].copy()
        available = arr[offset:].copy()
        pad_val = available[-1] if len(available) > 0 else 0.0
        return np.concatenate([available, np.full(length - len(available), pad_val)])
