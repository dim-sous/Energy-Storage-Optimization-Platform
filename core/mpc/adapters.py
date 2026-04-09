"""MPC adapters — uniform interface for the linear simulator.

Both `TrackingMPC` and `EconomicMPC` have varied `solve()` signatures
because they take different inputs (tracking refs vs forecast prices).
The linear simulator expects a single shape:

    setpoint_pnet, setpoint_preg, failed = adapter.solve_setpoint(
        state_est, plan, forecast_e, probs,
        sim_step, steps_per_ems, steps_per_mpc, u_prev,
    )

Each adapter wraps one underlying MPC and converts its (chg, dis, reg)
output into a single signed `P_net = dis - chg`.

RF1 (2026-04-15): neither adapter takes `recent_activation` anymore.
Activation tracking lives in the plant; the MPC plans against the
expected activation magnitude (constant) instead of the most recent
sample.
"""

from __future__ import annotations

import numpy as np

from core.mpc.economic import EconomicMPC
from core.mpc.tracking import TrackingMPC
from core.planners.plan import Plan


def _window(arr: np.ndarray, offset: int, length: int) -> np.ndarray:
    """Slice arr[offset:offset+length], padding with the last value."""
    end = offset + length
    if end <= len(arr):
        return arr[offset:end].copy()
    available = arr[offset:].copy()
    pad_val = available[-1] if len(available) > 0 else 0.0
    return np.concatenate([available, np.full(length - len(available), pad_val)])


class TrackingMPCAdapter:
    """Adapt TrackingMPC to the linear simulator's uniform interface."""

    def __init__(self, mpc: TrackingMPC) -> None:
        self._mpc = mpc

    @property
    def last_solve_failed(self) -> bool:
        return self._mpc.last_solve_failed

    def solve_setpoint(
        self,
        state_est: np.ndarray,
        plan: Plan,
        forecast_e: np.ndarray,        # (n_scenarios, n_hours)
        probabilities: np.ndarray,
        sim_step: int,
        steps_per_ems: int,
        steps_per_mpc: int,
        u_prev_3: np.ndarray,          # (3,) [chg, dis, reg]
    ) -> tuple[float, float, bool]:
        # Build per-MPC-step references by ZOH-expanding the hourly plan.
        # The MPC needs N+1 SOC refs and N power refs at MPC resolution.
        N = self._mpc.mp.N_mpc
        steps_per_hour = steps_per_ems  # ems_step = 1 hour
        ratio = steps_per_hour // steps_per_mpc  # MPC steps per hour

        # Plan.p_*_hourly are arrays of length n_hours from the planner
        # We expand them to MPC resolution. The plan started at plan.start_step.
        offset_in_plan = (sim_step - plan.start_step) // steps_per_hour

        # Build hourly source arrays from the plan's signed P_net split into chg/dis
        n_plan_hours = len(plan.p_net_hourly)
        plan_chg = np.where(plan.p_net_hourly < 0, -plan.p_net_hourly, 0.0)
        plan_dis = np.where(plan.p_net_hourly > 0, plan.p_net_hourly, 0.0)

        # Repeat each hourly value `ratio` times to get the MPC-resolution series
        chg_mpc = np.repeat(plan_chg, ratio)
        dis_mpc = np.repeat(plan_dis, ratio)
        reg_mpc = np.repeat(plan.p_reg_hourly, ratio)
        # SOC ref: end-of-hour values (skip initial), one per MPC step
        soc_mpc = np.repeat(plan.soc_ref_hourly[1:], ratio)

        # MPC offset within the expanded series
        mpc_off = offset_in_plan * ratio
        soc_win = _window(soc_mpc, mpc_off, N + 1)
        pc_win = _window(chg_mpc, mpc_off, N)
        pd_win = _window(dis_mpc, mpc_off, N)
        pr_win = _window(reg_mpc, mpc_off, N)

        u_cmd_3 = self._mpc.solve(
            x_est=state_est,
            soc_ref=soc_win,
            p_chg_ref=pc_win,
            p_dis_ref=pd_win,
            p_reg_ref=pr_win,
            p_reg_committed_horizon=pr_win,
            u_prev=u_prev_3,
        )
        # P_reg in u_cmd_3 is already the EMS-committed value at k=0
        # (TrackingMPC no longer decides P_reg — it's exogenous, just
        # like in EconomicMPC).
        p_net = float(u_cmd_3[1] - u_cmd_3[0])
        p_reg = float(u_cmd_3[2])
        return p_net, p_reg, self._mpc.last_solve_failed


class EconomicMPCAdapter:
    """Adapt EconomicMPC to the linear simulator's uniform interface."""

    def __init__(self, mpc: EconomicMPC) -> None:
        self._mpc = mpc

    @property
    def last_solve_failed(self) -> bool:
        return self._mpc.last_solve_failed

    def solve_setpoint(
        self,
        state_est: np.ndarray,
        plan: Plan,
        forecast_e: np.ndarray,        # (n_scenarios, n_hours)
        probabilities: np.ndarray,
        sim_step: int,
        steps_per_ems: int,
        steps_per_mpc: int,
        u_prev_3: np.ndarray,
    ) -> tuple[float, float, bool]:
        N = self._mpc.mp.N_mpc
        steps_per_hour = steps_per_ems
        ratio = steps_per_hour // steps_per_mpc

        # Plan-derived references at MPC resolution
        offset_in_plan = (sim_step - plan.start_step) // steps_per_hour
        plan_chg = np.where(plan.p_net_hourly < 0, -plan.p_net_hourly, 0.0)
        plan_dis = np.where(plan.p_net_hourly > 0, plan.p_net_hourly, 0.0)
        chg_mpc = np.repeat(plan_chg, ratio)
        dis_mpc = np.repeat(plan_dis, ratio)
        reg_mpc = np.repeat(plan.p_reg_hourly, ratio)
        soc_mpc = np.repeat(plan.soc_ref_hourly[1:], ratio)

        mpc_off = offset_in_plan * ratio
        soc_win = _window(soc_mpc, mpc_off, N + 1)
        pc_win = _window(chg_mpc, mpc_off, N)
        pd_win = _window(dis_mpc, mpc_off, N)
        pr_win = _window(reg_mpc, mpc_off, N)

        # Forecast-mean energy price horizon at MPC resolution
        # ZOH-expand from hourly forecast mean
        forecast_mean_e = forecast_e.T @ probabilities  # (n_hours,)
        e_mpc = np.repeat(forecast_mean_e, ratio)
        price_e_horizon = _window(e_mpc, mpc_off, N)

        u_cmd_3 = self._mpc.solve(
            x_est=state_est,
            soc_ref=soc_win,
            p_chg_ref=pc_win,
            p_dis_ref=pd_win,
            p_reg_ref=pr_win,
            price_e_horizon=price_e_horizon,
            p_reg_committed_horizon=pr_win,
            u_prev=u_prev_3,
        )
        p_net = float(u_cmd_3[1] - u_cmd_3[0])
        p_reg = float(u_cmd_3[2])
        return p_net, p_reg, self._mpc.last_solve_failed
