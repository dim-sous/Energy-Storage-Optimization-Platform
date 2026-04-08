"""SOC-safety wrapper for the EMS/MPC power setpoint.

RF1 (2026-04-15): this module's job is now much smaller. Activation
tracking has moved into the plant (it is the BESS controller on real
hardware), so PI no longer sees or modulates the activation signal.

What PI does now:
  1. Reduce the committed FCR capacity ``p_reg_committed`` when SOC
     is approaching either bound — the battery cannot safely deliver
     a full FCR commitment when it has almost no headroom in the
     direction the grid might pull. This is a *unilateral contract
     reduction*: the ledger's capacity revenue drops with the reduced
     commitment and the penalty math sees a smaller "demanded" amount,
     so worst-case we give up some revenue to avoid a bigger penalty.
  2. Apply a small SOC recovery bias to the base setpoint: nudge P_net
     so that SOC drifts toward ``SOC_terminal`` over long timescales.
     Small gain (design default 0.05 pre-RF1, tuned to 0.005 in step C).

What PI no longer does (moved to the plant or deleted):
  - Read or modulate ``activation_signal`` — plant handles it.
  - Compute ``p_delivered`` — plant returns it.
  - Enforce ``|P_net| + P_reg <= P_max`` — plant enforces it.

Runs at dt_pi = 4 s.
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import BatteryParams, RegControllerParams


class RegulationController:
    """SOC-safety wrapper around the EMS/MPC setpoint (RF1)."""

    def __init__(
        self,
        bp: BatteryParams,
        reg_params: RegControllerParams,
        dt: float = 4.0,
    ) -> None:
        self._bp = bp
        self._rp = reg_params
        self._dt = dt

    def compute(
        self,
        setpoint_pnet: float,
        p_reg_committed: float,
        soc_current: float,
    ) -> np.ndarray:
        """Apply SOC-safety scaling and recovery bias to the base setpoint.

        Parameters
        ----------
        setpoint_pnet : float
            MPC / EMS base net power setpoint [kW, signed].
            > 0 = discharge, < 0 = charge.
        p_reg_committed : float
            FCR capacity committed by the EMS for the current hour [kW, >= 0].
        soc_current : float
            Current SOC estimate (from EKF).

        Returns
        -------
        u_command : ndarray, shape (2,)
            ``[P_net_setpoint_biased, P_reg_committed_clamped]`` ready to
            pass to ``plant.step(u, activation_k)``. The plant will apply
            activation on top of this.
        """
        bp = self._bp
        rp = self._rp
        P_max = bp.P_max_kw

        # ---- 1. SOC safety: reduce the committed capacity near bounds ----
        # Any activation — up or down — could push SOC further into a
        # bound when SOC is already close to it. Scale p_reg_committed
        # symmetrically so the plant has less to deliver in the danger
        # zone. The ledger will see the reduced commitment via u_applied,
        # so capacity revenue drops with this — we trade revenue for
        # avoiding larger penalties.
        p_reg_safe = self._soc_clamp_committed(p_reg_committed, soc_current)

        # ---- 2. SOC recovery bias on the base setpoint ----
        # Small proportional pull toward SOC_terminal, fired every step
        # (not just when activation is idle). recovery > 0 means "want to
        # charge more", i.e. a NEGATIVE addition to P_net.
        p_recovery = rp.recovery_gain * (bp.SOC_terminal - soc_current) * P_max
        p_net_biased = setpoint_pnet - p_recovery

        # ---- 3. Defensive absolute-power clip ----
        # The plant will also clip, but bounding here keeps u_command
        # interpretable and the trace symmetric.
        p_net_clipped = float(np.clip(p_net_biased, -P_max, P_max))

        return np.array([p_net_clipped, p_reg_safe])

    def _soc_clamp_committed(self, p_reg: float, soc: float) -> float:
        """Linearly scale the committed FCR capacity toward zero as SOC
        approaches either bound. Symmetric in SOC — any activation can
        be the direction that hurts, so we protect both sides."""
        rp = self._rp

        if soc <= rp.soc_cutoff_low or soc >= rp.soc_cutoff_high:
            return 0.0
        if soc < rp.soc_safety_low:
            scale = (soc - rp.soc_cutoff_low) / (rp.soc_safety_low - rp.soc_cutoff_low)
            return p_reg * scale
        if soc > rp.soc_safety_high:
            scale = (rp.soc_cutoff_high - soc) / (rp.soc_cutoff_high - rp.soc_safety_high)
            return p_reg * scale
        return p_reg

    def reset(self) -> None:
        """Reset controller state (no-op)."""
        pass
