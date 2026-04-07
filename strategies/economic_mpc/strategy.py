"""ECONOMIC_MPC — the v5 product. Stochastic EMS + activation-aware economic MPC + PI.

This is the production v5 strategy and the one pitched in the B2B deck.
The economic MPC has two information edges over `deterministic_lp`:

1. **Closed-loop state feedback** via the PI controller — handles
   activation signal modulation in real time, which open-loop dispatchers
   physically cannot do.
2. **Activation-aware OU forecasting** — the MPC sees the most recent
   activation sample and forecasts persistence over its 1-hour horizon,
   pre-positioning SOC headroom for the upcoming dispatch.
"""

from __future__ import annotations

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    EMSParams,
    MPCParams,
    RegControllerParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.mpc.adapters import EconomicMPCAdapter
from core.mpc.economic import EconomicMPC
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
    reg_p: RegulationParams,
    pi_enabled: bool = True,
    **_unused,
) -> Strategy:
    suffix = "" if pi_enabled else "_no_pi"
    label = "Economic MPC (v5)" if pi_enabled else "Economic MPC (no PI)"
    return Strategy(
        name=f"economic_mpc{suffix}",
        planner=EconomicEMS(bp, tp, ep, thp, elp),
        mpc=EconomicMPCAdapter(EconomicMPC(bp, tp, mp, thp, elp, ep, reg_p)),
        pi=RegulationController(bp, reg_ctrl_p, tp.dt_pi),
        pi_enabled=pi_enabled,
        metadata={
            "label": label,
            "pitch_visible": pi_enabled,
            "description": "Stochastic EMS + activation-aware economic MPC + PI. v5 product.",
        },
    )
