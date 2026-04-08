"""Plan — uniform setpoint container produced by every planner.

The simulator's linear loop reads from a `Plan` to find the current
hour's setpoint. Planners return their internal dict (the legacy
shape with `P_chg_ref / P_dis_ref / P_reg_ref / SOC_ref / ...`); the
simulator wraps that dict into a `Plan` once per re-solve.

Power is normalised here to the **signed** convention `P_net > 0 = discharge`
so the rest of the simulator (PI controller, plant) can use a single
value instead of the legacy (chg, dis) tuple. Wash trades are impossible
to express in this representation.

Wow Factor 1 (2026-04-15): `Plan` now also carries the per-scenario
second-stage trajectories from the EMS stochastic solve, so the MPC
layer can pick which scenario to anchor against instead of tracking
the probability-weighted average. See docs/wow_factor_1_design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ScenarioPlan:
    """One scenario's second-stage trajectory from a stochastic solve.

    For deterministic planners (DeterministicLP, rule-based) a Plan
    carries a single ScenarioPlan with probability 1.0.
    """
    p_net: np.ndarray    # (n_hours,)   [kW, signed]
    p_reg: np.ndarray    # (n_hours,)   [kW, >= 0]
    soc:   np.ndarray    # (n_hours+1,) [0..1]
    soh:   np.ndarray    # (n_hours+1,) [0..1]
    temp:  np.ndarray    # (n_hours+1,) [degC]


@dataclass(frozen=True)
class Plan:
    """One hourly dispatch plan, valid from sim_step `start_step` onward.

    Averaged fields (`p_net_hourly`, `p_reg_hourly`, `soc_ref_hourly`) are
    the probability-weighted means across scenarios. Non-MPC strategies
    read only these and ignore the per-scenario `scenarios` / `probabilities`
    fields, so they are unaffected by Wow Factor 1.
    """
    p_net_hourly: np.ndarray          # (n_hours,)   [kW, signed]
    p_reg_hourly: np.ndarray          # (n_hours,)   [kW, >= 0]
    soc_ref_hourly: np.ndarray        # (n_hours+1,) [0..1]
    start_step: int                   # sim_step at which hour 0 begins
    expected_profit: float            # planner's forecast-evaluated profit
    scenarios: tuple[ScenarioPlan, ...] = field(default_factory=tuple)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    @classmethod
    def from_planner_dict(cls, d: dict, start_step: int) -> "Plan":
        """Wrap a planner's output dict into a Plan.

        Converts (P_chg_ref, P_dis_ref) into a single signed P_net using
        ``P_net = P_dis - P_chg``. Both must be non-negative in the
        input dict (planners produce non-negative chg/dis).

        Reads the per-scenario arrays from ``d["scenarios_*"]`` if
        present; otherwise constructs a single degenerate scenario from
        the averaged fields (used by fallback paths and any planner that
        hasn't been updated to the scenario-aware interface).
        """
        p_chg = np.asarray(d["P_chg_ref"], dtype=float)
        p_dis = np.asarray(d["P_dis_ref"], dtype=float)
        p_net = p_dis - p_chg
        p_reg = np.asarray(d["P_reg_ref"], dtype=float)
        soc_ref = np.asarray(d["SOC_ref"], dtype=float)
        soh_ref = np.asarray(d.get("SOH_ref", np.ones(len(soc_ref))), dtype=float)
        temp_ref = np.asarray(d.get("TEMP_ref", np.full(len(soc_ref), 25.0)), dtype=float)

        # Per-scenario trajectories (Wow Factor 1). If a planner did not
        # populate them, fall back to a single degenerate scenario that
        # equals the averaged fields — keeps non-updated planners working.
        if "scenarios_p_chg" in d:
            s_p_chg = np.asarray(d["scenarios_p_chg"], dtype=float)    # (S, N)
            s_p_dis = np.asarray(d["scenarios_p_dis"], dtype=float)    # (S, N)
            s_p_reg = np.asarray(d["scenarios_p_reg"], dtype=float)    # (S, N)
            s_soc   = np.asarray(d["scenarios_soc"],   dtype=float)    # (S, N+1)
            s_soh   = np.asarray(d["scenarios_soh"],   dtype=float)    # (S, N+1)
            s_temp  = np.asarray(d["scenarios_temp"],  dtype=float)    # (S, N+1)
            probs = np.asarray(d["probabilities"], dtype=float)
            S = s_p_chg.shape[0]
            scenarios = tuple(
                ScenarioPlan(
                    p_net=s_p_dis[s] - s_p_chg[s],
                    p_reg=s_p_reg[s],
                    soc=s_soc[s],
                    soh=s_soh[s],
                    temp=s_temp[s],
                )
                for s in range(S)
            )
        else:
            scenarios = (
                ScenarioPlan(
                    p_net=p_net,
                    p_reg=p_reg,
                    soc=soc_ref,
                    soh=soh_ref,
                    temp=temp_ref,
                ),
            )
            probs = np.array([1.0])

        return cls(
            p_net_hourly=p_net,
            p_reg_hourly=p_reg,
            soc_ref_hourly=soc_ref,
            start_step=start_step,
            expected_profit=float(d.get("expected_profit", 0.0)),
            scenarios=scenarios,
            probabilities=probs,
        )

    def setpoint_at(self, sim_step: int, steps_per_hour: int) -> tuple[float, float]:
        """Return (P_net, P_reg) for the given sim_step (ZOH within hour)."""
        h = (sim_step - self.start_step) // steps_per_hour
        h = max(0, min(h, len(self.p_net_hourly) - 1))
        return float(self.p_net_hourly[h]), float(self.p_reg_hourly[h])

    def soc_anchor_at(self, sim_step: int, steps_per_hour: int) -> float:
        """End-of-current-hour SOC target (the EMS strategic anchor)."""
        h = (sim_step - self.start_step) // steps_per_hour + 1
        h = max(0, min(h, len(self.soc_ref_hourly) - 1))
        return float(self.soc_ref_hourly[h])
