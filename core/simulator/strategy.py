"""Strategy — composition recipe for the linear simulator.

A Strategy is a frozen dataclass that names which planner, MPC, and PI
controller the simulator should use. The simulator's main loop has zero
strategy-specific branches: it just calls `strategy.planner.solve(...)`,
optionally `strategy.mpc.solve(...)`, optionally `strategy.pi.compute(...)`.

Adding a new strategy means writing a new file under `strategies/<name>/`
that returns a `Strategy(...)` instance. No simulator changes required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


class _PlannerLike(Protocol):
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
    ) -> dict: ...


class _MPCLike(Protocol):
    """Loose protocol — MPC implementations have varied signatures.
    The simulator handles the dispatch via duck-typing on attribute names.
    Required: a callable named `solve` returning a 3-vector
    [P_chg, P_dis, P_reg].
    Optional: `last_solve_failed` flag.
    """
    def solve(self, *args, **kwargs) -> np.ndarray: ...


class _PILike(Protocol):
    def compute(
        self,
        setpoint_pnet: float,
        p_reg_committed: float,
        activation_signal: float,
        soc_current: float,
    ) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class Strategy:
    """A composition of (planner, mpc, pi) with metadata.

    Any of `mpc` and `pi` can be `None`. Strategies that omit `mpc` use
    the planner's hourly setpoint directly at every minute. Strategies
    that omit `pi` apply activation modulation in an open-loop way
    (the simulator dispatches them via the `open_loop_dispatch` helper).
    """
    name: str
    planner: _PlannerLike
    mpc: Optional[_MPCLike] = None
    pi: Optional[_PILike] = None
    pi_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
