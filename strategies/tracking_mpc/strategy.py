"""TRACKING_MPC — sanity check: stochastic EMS + tracking MPC + PI.

NOT pitch-visible. The "old v5 stack". Demonstrates how a tracking-only
MPC compares to the new economic MPC. If they're equal, the economic
formulation isn't doing useful work.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    RegControllerParams,
    ThermalParams,
    TimeParams,
)
from core.mpc.adapters import TrackingMPCAdapter
from core.mpc.tracking import TrackingMPC
from core.pi.regulation import RegulationController
from core.planners.stochastic_ems import EconomicEMS
from core.simulator.strategy import Strategy


def make_strategy(
    bp: BatteryParams,
    tp: TimeParams,
    ep: EMSParams,
    mp: MPCParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    reg_ctrl_p: RegControllerParams,
    pi_enabled: bool = True,
    **_unused,
) -> Strategy:
    suffix = "" if pi_enabled else "_no_pi"
    label = "Tracking MPC (sanity)" if pi_enabled else "Tracking MPC no-PI (sanity)"
    return Strategy(
        name=f"tracking_mpc{suffix}",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=TrackingMPCAdapter(TrackingMPC(bp, tp, mp, thp, elp)),
        pi=RegulationController(bp, reg_ctrl_p, tp.dt_pi),
        pi_enabled=pi_enabled,
        metadata={
            "label": label,
            "pitch_visible": False,
            "description": "EMS + tracking MPC (Q_soc-dominated) + PI. Old v5 stack.",
        },
    )
