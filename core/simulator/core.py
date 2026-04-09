"""Linear multi-rate simulator — single loop, no strategy branches.

This is the v5 refactor's centrepiece. Replaces the legacy 500+ line
`MultiRateSimulator.run()` monolith with a ~150-line linear loop that
reads top-to-bottom: planner → mpc → pi → plant → ledger.

Strategy-specific behaviour is parameterised via the `Strategy` recipe
(`core/simulator/strategy.py`). The loop has no `if strategy == FOO`
branches anywhere — every dispatch is via duck-typed method calls.

Multi-rate cadence
------------------
- dt_ems = 3600 s   → planner re-solves
- dt_mpc =   60 s   → estimator update + (optional) MPC solve
- dt_pi  =    4 s   → PI compute (or open-loop dispatch) + plant.step()

Power representation
--------------------
The simulator uses the **signed** ``(P_net, P_reg)`` interface
throughout. Wash trades and budget violations are eliminated by
construction in the plant and PI controller (see Bug A/B/C fixes
in [backlog.md](../../backlog.md) item 0).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from core.accounting.ledger import compute_ledger
from core.config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    PackParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.estimators.ekf import ExtendedKalmanFilter
from core.markets.activation import ActivationSignalGenerator
from core.physics.plant import BatteryPack, BatteryPlant
from core.planners.plan import Plan
from core.simulator.strategy import Strategy
from core.simulator.traces import SimTraces

logger = logging.getLogger(__name__)


def run_simulation(
    strategy: Strategy,
    forecast_e: np.ndarray,           # (n_scenarios, n_hours_total) [$/kWh]
    forecast_r: np.ndarray,           # (n_scenarios, n_hours_total) [$/kW/h]
    probabilities: np.ndarray,        # (n_scenarios,)
    realized_e_prices: np.ndarray,    # (n_hours_total,) [$/kWh] — accounting only
    realized_r_prices: np.ndarray,    # (n_hours_total,) [$/kW/h] — accounting only
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    mp: MPCParams,
    ekf_p: EKFParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    reg_p: RegulationParams,
    pp: PackParams | None = None,
) -> dict:
    """Execute a strategy end-to-end and return the result dict.

    The strategy decides what's wired together (planner / mpc); this
    function decides WHEN each layer runs (the multi-rate cadence) and
    routes data between them. Activation tracking lives in the plant.
    """
    # ---- Setup ----
    use_pack = pp is not None
    plant = BatteryPack(bp, tp, thp, elp, pp) if use_pack else BatteryPlant(bp, tp, thp, elp)
    n_cells = pp.n_cells if use_pack else 1

    ekf = ExtendedKalmanFilter(bp, tp, ekf_p, thp, elp)

    activation_gen = ActivationSignalGenerator(reg_p, tp.dt_pi)
    total_seconds = int(tp.sim_hours * 3600)
    N_sim = int(total_seconds / tp.dt_sim)
    N_mpc = int(total_seconds / tp.dt_mpc)
    steps_per_mpc = int(tp.dt_mpc / tp.dt_sim)
    steps_per_ems = int(tp.dt_ems / tp.dt_sim)
    activation = activation_gen.generate(N_sim)

    traces = SimTraces(n_sim_steps=N_sim, n_mpc_steps=N_mpc, n_cells=n_cells)
    cells = plant.get_cell_states() if use_pack else None
    traces.record_initial_state(plant.get_state(), plant.get_terminal_voltage(), cells)

    state_est = ekf.get_estimate()
    traces.soc_ekf[0] = state_est[0]
    traces.soh_ekf[0] = state_est[1]
    traces.temp_ekf[0] = state_est[2]
    traces.vrc1_ekf[0] = state_est[3]
    traces.vrc2_ekf[0] = state_est[4]

    plan: Plan | None = None
    setpoint_pnet = 0.0
    setpoint_preg = 0.0
    # Last applied (P_chg, P_dis, P_reg) in the EKF/MPC's 3-vector
    # convention. Updated each PI step from u_applied; consumed by
    # both the EKF (predict step) and the MPC (rate-of-change penalty).
    u_prev_3 = np.zeros(3)
    mpc_idx = 0

    # ---- Main loop ----
    for k in range(N_sim):
        # 1. Hourly: planner re-solves
        if k % steps_per_ems == 0:
            ems_hour = k // steps_per_ems
            n_remaining = max(1, min(ep.N_ems, forecast_e.shape[1] - ems_hour))
            e_window = forecast_e[:, ems_hour:ems_hour + n_remaining]
            r_window = forecast_r[:, ems_hour:ems_hour + n_remaining]
            # Pad scenario windows to N_ems if short
            if e_window.shape[1] < ep.N_ems:
                pad = ep.N_ems - e_window.shape[1]
                e_window = np.pad(e_window, ((0, 0), (0, pad)), mode="edge")
                r_window = np.pad(r_window, ((0, 0), (0, pad)), mode="edge")

            plan_dict = strategy.planner.solve(
                soc_init=state_est[0],
                soh_init=state_est[1],
                t_init=state_est[2],
                energy_scenarios=e_window,
                reg_scenarios=r_window,
                probabilities=probabilities,
                vrc1_init=state_est[3],
                vrc2_init=state_est[4],
            )
            plan = Plan.from_planner_dict(plan_dict, start_step=k)
            traces.ems_soc_refs.append(plan.soc_ref_hourly.copy())

        # 2. Per-minute: estimator + (optional) MPC
        if k % steps_per_mpc == 0:
            t0_est = time.perf_counter()
            y_meas = plant.get_measurement()
            if k > 0:
                state_est = ekf.step(u_prev_3, y_meas)
            est_solve_time = time.perf_counter() - t0_est

            # MPC solve (or use plan setpoint directly)
            mpc_solve_time = 0.0
            mpc_failed = False
            if strategy.mpc is not None:
                t0_mpc = time.perf_counter()
                # RF1 (2026-04-15): MPC no longer receives recent_activation.
                # Activation lives in the plant; MPC plans against the
                # expected (zero signed mean) activation.
                setpoint_pnet, setpoint_preg, mpc_failed = strategy.mpc.solve_setpoint(
                    state_est=state_est,
                    plan=plan,
                    forecast_e=e_window,
                    probabilities=probabilities,
                    sim_step=k,
                    steps_per_ems=steps_per_ems,
                    steps_per_mpc=steps_per_mpc,
                    u_prev_3=u_prev_3,
                )
                mpc_solve_time = time.perf_counter() - t0_mpc
            else:
                setpoint_pnet, setpoint_preg = plan.setpoint_at(k, steps_per_ems)

            soc_anchor = plan.soc_anchor_at(k, steps_per_ems)
            traces.record_mpc(
                m=mpc_idx,
                ekf_state=state_est,
                setpoint_pnet=setpoint_pnet,
                setpoint_preg=setpoint_preg,
                soc_anchor=soc_anchor,
                solve_time_s=mpc_solve_time,
                est_time_s=est_solve_time,
                solver_failed=mpc_failed,
            )
            mpc_idx += 1

        # 3. Per-step: hand the held setpoint straight to the plant.
        # RF1 (2026-04-15): activation tracking lives in the plant. The
        # 2026-04-15 cleanup deleted the strategy-layer PI controller
        # entirely (it was empirically ceremonial post-RF1).
        activation_k = float(activation[k])
        u_command = np.array([setpoint_pnet, setpoint_preg])

        # 4. Plant integrates with the real activation sample and reports
        # the actually-applied power + actually-delivered FCR power.
        x_new, y_meas, u_applied, p_delivered = plant.step(
            u_command, activation_k=activation_k,
        )

        # 5. Record (use u_applied so accounting matches reality)
        cells_now = plant.get_cell_states() if use_pack else None
        traces.record_step(
            k=k,
            u_applied=u_applied,
            p_delivered=p_delivered,
            x_new=x_new,
            vterm_new=plant.get_terminal_voltage(),
            activation_k=activation_k,
            p_reg_committed_k=setpoint_preg,
            cells=cells_now,
        )

        # 6. Update u_prev_3 for the next EKF / MPC call. The EKF takes
        # the (chg, dis, reg) shape, so decompose the signed P_net into
        # non-negative (chg, dis) — exactly one is nonzero per row.
        p_net_applied = u_applied[0]
        p_reg_applied = u_applied[1]
        if p_net_applied >= 0:
            u_prev_3 = np.array([0.0, p_net_applied, p_reg_applied])
        else:
            u_prev_3 = np.array([-p_net_applied, 0.0, p_reg_applied])

    # Fill the last EKF slot (loop wrote slots 0..N_mpc-1; final slot
    # would otherwise stay zero and confuse the audit / visualization).
    traces.soc_ekf[N_mpc] = state_est[0]
    traces.soh_ekf[N_mpc] = state_est[1]
    traces.temp_ekf[N_mpc] = state_est[2]
    traces.vrc1_ekf[N_mpc] = state_est[3]
    traces.vrc2_ekf[N_mpc] = state_est[4]

    # ---- Accounting (pure function over the trace) ----
    return compute_ledger(
        traces=traces,
        realized_e_prices=realized_e_prices,
        realized_r_prices=realized_r_prices,
        bp=bp,
        tp=tp,
        ep=ep,
        reg_p=reg_p,
        strategy_name=strategy.name,
        strategy_metadata=strategy.metadata,
    )
