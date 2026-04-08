"""Nonlinear 5-state battery model with 2RC equivalent circuit and multi-cell pack.

v4_electrical_rc_model: extends the 3-state thermal model with a 2RC
equivalent circuit.  Each cell gains two voltage states (V_rc1, V_rc2)
representing charge-transfer and diffusion transients.

Provides:
  1. CasADi symbolic dynamics (shared by EMS, MPC, EKF, MHE).
  2. A numpy-based high-fidelity plant for closed-loop simulation.
  3. OCV polynomial and quadratic current solver (CasADi + numpy).

State vector   x = [SOC, SOH, T, V_rc1, V_rc2]
Input vector   u = [P_charge, P_discharge, P_reg]   (all >= 0, kW)
Measurement    y = [SOC_measured, T_measured, V_term_measured]

Continuous-time dynamics
------------------------
  dSOC/dt  = (eta_c * P_chg - P_dis / eta_d) / (SOH * E_nom * 3600)   [1/s]
  dSOH/dt  = -kappa(T) * (alpha_deg*(P_chg+P_dis) + alpha_deg_reg*|P_reg|)  [1/s]
  dT/dt    = (I^2 * R_total_dc - h_cool * (T - T_amb)) / C_thermal     [degC/s]
  dV_rc1/dt = -V_rc1 / tau_1  +  I / C1                                 [V/s]
  dV_rc2/dt = -V_rc2 / tau_2  +  I / C2                                 [V/s]

  where kappa(T) = exp(E_a / R_gas * (1/T_ref_K - 1/T_K))    (Arrhenius)
        V_term   = OCV(SOC) - V_rc1 - V_rc2 - I * R0          [V]
        I        = quadratic_solve(P_net, V_oc_eff, R0)        [A]
        P_net    = P_dis - P_chg                                [kW]

Sign convention: I > 0 for discharge, I < 0 for charge.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from core.config.parameters import (
    BatteryParams,
    ElectricalParams,
    PackParams,
    ThermalParams,
    TimeParams,
)


# ---------------------------------------------------------------------------
#  OCV polynomial  (single NMC cell, ~3.0–4.19 V)
# ---------------------------------------------------------------------------

def ocv_cell_numpy(soc: float | np.ndarray, elp: ElectricalParams) -> float | np.ndarray:
    """Evaluate single-cell OCV polynomial (numpy, Horner's method).

    Parameters
    ----------
    soc : float or ndarray
        State of charge in [0, 1].
    elp : ElectricalParams

    Returns
    -------
    OCV in volts (single cell).
    """
    c = elp.ocv_coeffs
    result = c[-1]
    for k in range(len(c) - 2, -1, -1):
        result = result * soc + c[k]
    return result


def ocv_cell_casadi(soc: ca.MX, elp: ElectricalParams) -> ca.MX:
    """Evaluate single-cell OCV polynomial (CasADi symbolic, Horner's method).

    Parameters
    ----------
    soc : ca.MX   Symbolic SOC.
    elp : ElectricalParams

    Returns
    -------
    ca.MX : symbolic OCV [V].
    """
    c = elp.ocv_coeffs
    result = c[-1]
    for k in range(len(c) - 2, -1, -1):
        result = result * soc + c[k]
    return result


def ocv_pack_numpy(
    soc: float | np.ndarray, elp: ElectricalParams, n_modules: int = 4,
) -> float | np.ndarray:
    """Pack-level OCV = n_modules * n_series_cells * OCV_cell(SOC)."""
    return n_modules * elp.n_series_cells * ocv_cell_numpy(soc, elp)


def ocv_pack_casadi(
    soc: ca.MX, elp: ElectricalParams, n_modules: int = 4,
) -> ca.MX:
    """Pack-level OCV = n_modules * n_series_cells * OCV_cell(SOC)."""
    return n_modules * elp.n_series_cells * ocv_cell_casadi(soc, elp)


# ---------------------------------------------------------------------------
#  Quadratic current solver:  R0 * I^2 - V_oc_eff * I + P_net_W = 0
# ---------------------------------------------------------------------------

def compute_current_numpy(
    P_net_kw: float, V_oc_eff: float, R0: float,
) -> tuple[float, float]:
    """Solve for battery current and terminal voltage (numpy).

    Parameters
    ----------
    P_net_kw : float
        Net power [kW], positive = discharge.  P_net = P_dis - P_chg.
    V_oc_eff : float
        Effective open-circuit voltage = OCV(SOC) - V_rc1 - V_rc2  [V].
    R0 : float
        Series resistance [Ohm].

    Returns
    -------
    I : float       Current [A], positive = discharge.
    V_term : float  Terminal voltage [V].
    """
    P_W = P_net_kw * 1000.0
    disc = V_oc_eff ** 2 - 4.0 * R0 * P_W
    disc = max(disc, 0.0)
    I = (V_oc_eff - np.sqrt(disc)) / (2.0 * R0)
    V_term = V_oc_eff - I * R0
    return float(I), float(V_term)


def compute_current_casadi(
    P_net_kw: ca.MX, V_oc_eff: ca.MX, R0: float,
) -> tuple[ca.MX, ca.MX]:
    """Solve for battery current and terminal voltage (CasADi symbolic).

    Uses a linear approximation I = P / V_oc_eff (ignoring the R0*I^2 term
    in the power equation) to keep the NLP smooth and well-conditioned for
    IPOPT.  The error is negligible: at 100 A and R0=0.005 Ohm, the voltage
    drop is 0.5 V out of ~800 V (0.06%).  The numpy plant model uses the
    exact quadratic solve.

    Parameters
    ----------
    P_net_kw : ca.MX   Net power [kW], positive = discharge.
    V_oc_eff : ca.MX   OCV - V_rc1 - V_rc2  [V].
    R0       : float   Series resistance [Ohm].

    Returns
    -------
    I      : ca.MX   Current [A].
    V_term : ca.MX   Terminal voltage [V].
    """
    V_safe = ca.fmax(ca.fabs(V_oc_eff), 100.0)
    I = P_net_kw * 1000.0 / V_safe
    V_term = V_oc_eff - I * R0
    return I, V_term


# ---------------------------------------------------------------------------
#  CasADi symbolic dynamics  (reused by EMS, MPC, EKF, MHE)
# ---------------------------------------------------------------------------

def build_casadi_dynamics(
    bp: BatteryParams, thp: ThermalParams, elp: ElectricalParams,
    n_modules: int = 4,
) -> ca.Function:
    """Return a CasADi Function  f(x, u) -> x_dot  (continuous-time ODE).

    Parameters
    ----------
    bp  : BatteryParams
    thp : ThermalParams
    elp : ElectricalParams
    n_modules : int   Number of modules in series (for OCV scaling).

    Returns
    -------
    ca.Function  with signature  (x[5], u[3]) -> x_dot[5]
    """
    x = ca.MX.sym("x", 5)     # [SOC, SOH, T, V_rc1, V_rc2]
    u = ca.MX.sym("u", 3)     # [P_chg, P_dis, P_reg]

    SOC, SOH, T = x[0], x[1], x[2]
    V_rc1, V_rc2 = x[3], x[4]
    P_chg, P_dis, P_reg = u[0], u[1], u[2]

    # ---- SOC dynamics (unchanged from v1) ----
    E_eff_kws = SOH * bp.E_nom_kwh * 3600.0          # effective capacity [kW*s]
    dSOC_dt = (bp.eta_charge * P_chg - P_dis / bp.eta_discharge) / E_eff_kws

    # ---- Thermally-coupled degradation ----
    T_ref_K = thp.T_ref + 273.15                      # [K]
    T_K = T + 273.15                                   # [K]
    kappa = ca.exp(thp.E_a / thp.R_gas * (1.0 / T_ref_K - 1.0 / T_K))

    P_arb = P_chg + P_dis                              # arbitrage throughput [kW]
    P_reg_abs = ca.fabs(P_reg)                         # committed reg capacity [kW]
    # Split degradation: arbitrage chg/dis throughput + OU-cycling wear proxy
    # Scale by 1/n_modules: each cell in the series pack handles its share.
    dSOH_dt = -kappa * (bp.alpha_deg * P_arb + bp.alpha_deg_reg * P_reg_abs) / n_modules

    # ---- Current from quadratic solve (2RC model) ----
    OCV_pack = ocv_pack_casadi(SOC, elp, n_modules)
    V_oc_eff = OCV_pack - V_rc1 - V_rc2
    P_net = P_dis - P_chg                              # [kW], positive = discharge
    I_net, _ = compute_current_casadi(P_net, V_oc_eff, elp.R0)

    # ---- Thermal dynamics ----
    # Heating from actual physical current only (|P_net|), not the
    # committed reg capacity. The OU-cycling wear from reg dispatch is
    # captured in dSOH via alpha_deg_reg above.
    V_safe = ca.fmax(ca.fabs(V_oc_eff), 100.0)
    I_thermal = ca.fabs(P_net) * 1000.0 / V_safe
    Q_joule = I_thermal ** 2 * elp.R_total_dc          # Joule heating [W]
    Q_cool = thp.h_cool * (T - thp.T_ambient)         # cooling [W]
    dT_dt = (Q_joule - Q_cool) / thp.C_thermal        # [degC/s]

    # ---- RC dynamics ----
    dV_rc1_dt = -V_rc1 / elp.tau_1 + I_net / elp.C1   # [V/s]
    dV_rc2_dt = -V_rc2 / elp.tau_2 + I_net / elp.C2   # [V/s]

    x_dot = ca.vertcat(dSOC_dt, dSOH_dt, dT_dt, dV_rc1_dt, dV_rc2_dt)
    return ca.Function("battery_ode", [x, u], [x_dot], ["x", "u"], ["x_dot"])


def build_casadi_dynamics_3state(
    bp: BatteryParams, thp: ThermalParams, elp: ElectricalParams,
    n_modules: int = 4, expected_activation_frac: float = 0.0,
) -> ca.Function:
    """Return a 3-state CasADi ODE  f(x, u) -> x_dot  for the EMS layer.

    Omits V_rc1/V_rc2 dynamics (they decay to zero within one EMS step)
    while still using the OCV polynomial for improved thermal accuracy
    over v3's constant-voltage model.

    When expected_activation_frac > 0, the SOC dynamics include the expected
    energy drain from regulation delivery.  Symmetric activation (equal UP/DOWN)
    causes a net SOC loss of (eta_c - 1/eta_d) * E[|a|] * P_reg / E_eff per
    second, reflecting round-trip efficiency losses during delivery.

    Parameters
    ----------
    bp  : BatteryParams
    thp : ThermalParams
    elp : ElectricalParams
    n_modules : int
    expected_activation_frac : float
        Expected absolute activation signal E[|a|] from the Markov chain
        stationary distribution.  0.0 disables the regulation throughput
        cost (backward-compatible with v4).

    Returns
    -------
    ca.Function  with signature  (x[3], u[3]) -> x_dot[3]
    """
    x = ca.MX.sym("x", 3)     # [SOC, SOH, T]
    u = ca.MX.sym("u", 3)     # [P_chg, P_dis, P_reg]

    SOC, SOH, T = x[0], x[1], x[2]
    P_chg, P_dis, P_reg = u[0], u[1], u[2]

    # ---- SOC dynamics ----
    E_eff_kws = SOH * bp.E_nom_kwh * 3600.0
    dSOC_dt = (bp.eta_charge * P_chg - P_dis / bp.eta_discharge) / E_eff_kws

    # Expected SOC drain from regulation delivery efficiency losses.
    # Symmetric activation: half charge (stores eta_c * P), half discharge
    # (uses P / eta_d).  Net per unit delivered:
    #   eta_c - 1/eta_d  ≈  0.95 - 1.053  =  -0.103  (always negative)
    if expected_activation_frac > 0.0:
        eta_loss = bp.eta_charge - 1.0 / bp.eta_discharge   # < 0
        dSOC_dt += eta_loss * expected_activation_frac * P_reg / E_eff_kws

    # ---- Thermally-coupled degradation ----
    T_ref_K = thp.T_ref + 273.15
    T_K = T + 273.15
    kappa = ca.exp(thp.E_a / thp.R_gas * (1.0 / T_ref_K - 1.0 / T_K))
    P_arb = P_chg + P_dis
    P_reg_abs = ca.fabs(P_reg)
    # Split degradation (see 5-state version above for derivation).
    dSOH_dt = -kappa * (bp.alpha_deg * P_arb + bp.alpha_deg_reg * P_reg_abs) / n_modules

    # ---- Thermal dynamics (OCV-based current, V_rc=0 at hourly resolution) ----
    # Heating from actual physical current only (|P_net|).
    OCV_pack = ocv_pack_casadi(SOC, elp, n_modules)
    V_safe = ca.fmax(OCV_pack, 100.0)
    P_net = P_dis - P_chg
    I_thermal = ca.fabs(P_net) * 1000.0 / V_safe
    Q_joule = I_thermal ** 2 * elp.R_total_dc
    Q_cool = thp.h_cool * (T - thp.T_ambient)
    dT_dt = (Q_joule - Q_cool) / thp.C_thermal

    x_dot = ca.vertcat(dSOC_dt, dSOH_dt, dT_dt)
    return ca.Function("battery_ode_3s", [x, u], [x_dot], ["x", "u"], ["x_dot"])


def build_casadi_rk4_integrator_3state(
    bp: BatteryParams, thp: ThermalParams, elp: ElectricalParams,
    dt: float, n_modules: int = 4, expected_activation_frac: float = 0.0,
) -> ca.Function:
    """Return a single-step RK4 integrator for the 3-state EMS model.

    No sub-stepping needed — all 3-state dynamics are slow (tau >> dt_ems).

    Returns
    -------
    ca.Function  with signature  (x[3], u[3]) -> x_next[3]
    """
    f = build_casadi_dynamics_3state(bp, thp, elp, n_modules, expected_activation_frac)

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 3)

    k1 = f(x, u)
    k2 = f(x + dt / 2.0 * k1, u)
    k3 = f(x + dt / 2.0 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step_3s", [x, u], [x_next], ["x", "u"], ["x_next"])


def build_casadi_rk4_integrator(
    bp: BatteryParams, thp: ThermalParams, elp: ElectricalParams,
    dt: float, n_modules: int = 4, max_sub: int | None = None,
) -> ca.Function:
    """Return a multi-sub-step RK4 integrator  F(x, u) -> x_next.

    Sub-stepping ensures numerical stability when dt exceeds the fastest
    RC time constant.  RK4 stability requires dt_sub < ~2.78 * tau_min.

    Parameters
    ----------
    bp  : BatteryParams
    thp : ThermalParams
    elp : ElectricalParams
    dt  : float   Integration time step [s].
    n_modules : int
    max_sub : int or None
        Cap on the number of RK4 sub-steps.  For long time steps (e.g.
        dt_ems = 3600 s) where RC transients decay within the step,
        capping sub-steps avoids a bloated CasADi graph.  If None,
        uses as many sub-steps as needed for strict RK4 stability.

    Returns
    -------
    ca.Function  with signature  (x[5], u[3]) -> x_next[5]
    """
    f = build_casadi_dynamics(bp, thp, elp, n_modules)

    # Sub-stepping for RK4 stability: dt_sub < 2.5 * tau_min
    tau_min = min(elp.tau_1, elp.tau_2)
    max_stable_dt = 2.5 * tau_min
    n_sub = max(1, int(np.ceil(dt / max_stable_dt)))
    if max_sub is not None:
        n_sub = min(n_sub, max_sub)
    dt_sub = dt / n_sub

    x = ca.MX.sym("x", 5)
    u = ca.MX.sym("u", 3)

    x_curr = x
    for _ in range(n_sub):
        k1 = f(x_curr, u)
        k2 = f(x_curr + dt_sub / 2.0 * k1, u)
        k3 = f(x_curr + dt_sub / 2.0 * k2, u)
        k4 = f(x_curr + dt_sub * k3, u)
        x_curr = x_curr + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step", [x, u], [x_curr], ["x", "u"], ["x_next"])


def build_casadi_measurement(
    elp: ElectricalParams, n_modules: int = 4,
) -> ca.Function:
    """Return a CasADi Function  h(x, u) -> y_pred  (measurement model).

    y = [SOC, T, V_term]  where V_term = OCV(SOC) - V_rc1 - V_rc2 - I*R0.

    Parameters
    ----------
    elp : ElectricalParams
    n_modules : int

    Returns
    -------
    ca.Function  with signature  (x[5], u[3]) -> y[3]
    """
    x = ca.MX.sym("x", 5)
    u = ca.MX.sym("u", 3)

    SOC = x[0]
    T = x[2]
    V_rc1, V_rc2 = x[3], x[4]
    P_chg, P_dis = u[0], u[1]

    OCV_pack = ocv_pack_casadi(SOC, elp, n_modules)
    V_oc_eff = OCV_pack - V_rc1 - V_rc2
    P_net = P_dis - P_chg
    _, V_term = compute_current_casadi(P_net, V_oc_eff, elp.R0)

    y = ca.vertcat(SOC, T, V_term)
    return ca.Function("meas_model", [x, u], [y], ["x", "u"], ["y"])


# ---------------------------------------------------------------------------
#  Numpy plant model for simulation
# ---------------------------------------------------------------------------

class BatteryPlant:
    """High-fidelity 5-state BESS plant with 2RC circuit, integrated at
    ``dt_sim`` resolution.

    The plant uses RK4 internally and generates noisy measurements for
    SOC, Temperature, and terminal voltage.  SOH is a hidden state.

    Parameters
    ----------
    bp  : BatteryParams
    tp  : TimeParams
    thp : ThermalParams
    elp : ElectricalParams
    n_modules : int   Number of modules (for OCV scaling).
    seed : int        Random seed for measurement noise.
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        n_modules: int = 4,
        seed: int = 42,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.thp = thp
        self.elp = elp
        self.n_modules = n_modules
        self._rng = np.random.default_rng(seed)

        # True state  [SOC, SOH, T, V_rc1, V_rc2]
        self._x = np.array(
            [bp.SOC_init, bp.SOH_init, thp.T_init,
             elp.V_rc1_init, elp.V_rc2_init],
            dtype=np.float64,
        )

        # Measurement noise standard deviations
        self._meas_std_soc = 0.01
        self._meas_std_temp = 0.5
        self._meas_std_volt = elp.sigma_v_meas

        # Cache last current for measurement model
        self._last_I_net: float = 0.0

    # ---- continuous-time ODE (numpy) ----
    def _ode(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        SOC, SOH, T = x[0], x[1], x[2]
        V_rc1, V_rc2 = x[3], x[4]
        P_chg, P_dis, P_reg = u[0], u[1], u[2]
        thp = self.thp
        elp = self.elp

        # SOC
        E_eff = SOH * self.bp.E_nom_kwh * 3600.0
        dSOC = (self.bp.eta_charge * P_chg - P_dis / self.bp.eta_discharge) / E_eff

        # Arrhenius factor
        T_ref_K = thp.T_ref + 273.15
        T_K = T + 273.15
        kappa = np.exp(thp.E_a / thp.R_gas * (1.0 / T_ref_K - 1.0 / T_K))

        # SOH (split throughput: arbitrage chg/dis vs reg-cycling proxy)
        # alpha_deg     : wear from actual chg/dis throughput
        # alpha_deg_reg : OU-cycling wear proxy from committed reg capacity
        P_arb = P_chg + P_dis
        P_reg_abs = abs(P_reg)
        dSOH = -kappa * (self.bp.alpha_deg * P_arb + self.bp.alpha_deg_reg * P_reg_abs)

        # Current from quadratic solve (uses actual net power)
        OCV_p = ocv_pack_numpy(SOC, elp, self.n_modules)
        V_oc_eff = OCV_p - V_rc1 - V_rc2
        P_net = P_dis - P_chg
        I_net, _ = compute_current_numpy(P_net, V_oc_eff, elp.R0)

        # Thermal heating: |P_net| only (single physical current).
        # The committed P_reg is NOT added here — heating is from actual
        # current flow, not from the contract. The OU-cycling wear from
        # reg dispatch is captured in dSOH via alpha_deg_reg above.
        V_oc_abs = max(abs(V_oc_eff), 100.0)
        I_thermal = abs(P_net) * 1000.0 / V_oc_abs
        Q_joule = I_thermal ** 2 * elp.R_total_dc
        Q_cool = thp.h_cool * (T - thp.T_ambient)
        dT = (Q_joule - Q_cool) / thp.C_thermal

        # RC dynamics
        dV_rc1 = -V_rc1 / elp.tau_1 + I_net / elp.C1
        dV_rc2 = -V_rc2 / elp.tau_2 + I_net / elp.C2

        return np.array([dSOC, dSOH, dT, dV_rc1, dV_rc2])

    # ---- single RK4 step (numpy) ----
    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._ode(x, u)
        k2 = self._ode(x + dt / 2.0 * k1, u)
        k3 = self._ode(x + dt / 2.0 * k2, u)
        k4 = self._ode(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- public interface ----
    def step(
        self,
        u: np.ndarray,
        activation_k: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Integrate one ``dt_sim`` step.

        v5 refactor: takes a 2-vector ``[P_net_signed, P_reg]`` instead of
        the legacy ``(P_chg, P_dis, P_reg)`` triple. ``P_net > 0`` means
        discharge, ``P_net < 0`` means charge. This eliminates wash trades
        by construction (Bug A) — the battery only ever does net power.

        RF1 (2026-04-15): the plant now tracks FCR activation internally
        given the committed ``P_reg``. Callers pass the current activation
        sample as ``activation_k`` and receive the actually-delivered FCR
        power as a new return value. Strategies no longer see
        ``activation_k`` — the plant is the BESS controller.

        The plant enforces the power budget ``|P_net| + P_reg <= P_max``
        (Bug B fix) and the SOC headroom (clamps charge so SOC stays
        below SOC_max, clamps discharge so SOC stays above SOC_min). It
        returns the **actually-applied** ``u_applied`` so the simulator's
        accounting always reflects what the battery physically did
        (Bug C fix).

        Parameters
        ----------
        u : ndarray, shape (2,)
            [P_net_setpoint, P_reg_committed]   (kW)
            P_net > 0 = discharge, P_net < 0 = charge.
            P_reg >= 0 always (committed FCR capacity, not realized
            dispatch).
        activation_k : float, default 0.0
            Current FCR activation sample in [-1, +1]. The plant
            computes ``P_net_with_activation = P_net_setpoint +
            activation_k * P_reg_committed`` before clipping and
            integration. Default 0.0 means "no activation," which
            reproduces the pre-RF1 plant behavior when the caller
            pre-added activation to ``u[0]``.

        Returns
        -------
        x_new : ndarray, shape (5,)
            Updated true state [SOC, SOH, T, V_rc1, V_rc2].
        y_meas : ndarray, shape (3,)
            Noisy measurement [SOC_meas, T_meas, V_term_meas].
        u_applied : ndarray, shape (2,)
            [P_net_applied, P_reg_applied] — the values actually applied
            after pre-integration clipping. Use these for downstream
            accounting, NOT the input ``u``.
        p_delivered : float
            Signed FCR power delivered this step, computed as
            ``P_net_applied - P_net_setpoint``. Positive = up-reg
            (discharge); negative = down-reg (charge). If the plant
            clipped activation demand (SOC limit, P_max hit), this will
            fall short of ``activation_k * P_reg_committed`` and the
            ledger's penalty math fires.
        """
        bp = self.bp
        dt = self.tp.dt_sim
        dt_h = dt / 3600.0
        SOC = self._x[0]
        SOH = self._x[1]
        E_eff = SOH * bp.E_nom_kwh  # effective capacity [kWh]

        # ---- Local helper: clip P_net to both P_max and SOC headroom ----
        # Used twice: once for the base setpoint alone (to find the base
        # dispatch the battery can actually do), and once for the setpoint
        # plus activation (to find the total dispatch). The delta between
        # them is the FCR portion actually delivered.
        def _clip_with_soc(p: float) -> float:
            p = float(np.clip(p, -bp.P_max_kw, bp.P_max_kw))
            if p > 0.0:
                max_dis = (SOC - bp.SOC_min) * E_eff * bp.eta_discharge / dt_h
                return min(p, max(0.0, max_dis))
            if p < 0.0:
                max_chg = (bp.SOC_max - SOC) * E_eff / (bp.eta_charge * dt_h)
                return max(p, -max(0.0, max_chg))
            return 0.0

        # ---- 1. Clip the base setpoint (no activation yet) ----
        # P_reg_committed is a CONTRACT, not a separate power channel.
        # The battery has ONE physical current = P_net at any instant.
        P_net_setpoint = float(u[0])
        P_reg = float(np.clip(u[1], 0.0, bp.P_max_kw))
        P_net_setpoint_clipped = _clip_with_soc(P_net_setpoint)

        # ---- 2. Add activation on top of the (already-clipped) base ----
        # Plant-internal activation tracking (RF1). On real hardware this
        # is the BESS controller's droop loop running at kHz cadence.
        # Two-pass clipping correctly attributes SOC/P_max clipping to
        # the *activation* portion, not the base dispatch: any shortfall
        # in p_delivered below activation_k * P_reg is real FCR
        # undelivery, not an artifact of the base schedule hitting a
        # bound.
        desired_activation_delta = activation_k * P_reg
        P_net = _clip_with_soc(P_net_setpoint_clipped + desired_activation_delta)

        # ---- 3. Decompose to (chg, dis, reg) for the internal ODE ----
        # Exactly one of (P_chg, P_dis) is nonzero — no wash trade possible.
        if P_net >= 0.0:
            P_chg, P_dis = 0.0, P_net
        else:
            P_chg, P_dis = -P_net, 0.0

        u_for_ode = np.array([P_chg, P_dis, P_reg])
        x_new = self._rk4_step(self._x, u_for_ode, dt)

        # ---- State saturation (safety, should rarely trigger) ----
        x_new[0] = np.clip(x_new[0], bp.SOC_min, bp.SOC_max)
        x_new[1] = np.clip(x_new[1], 0.5, 1.0)
        x_new[2] = np.clip(x_new[2], -20.0, 80.0)
        v_rc_max = 50.0
        x_new[3] = np.clip(x_new[3], -v_rc_max, v_rc_max)
        x_new[4] = np.clip(x_new[4], -v_rc_max, v_rc_max)

        # Cache current for measurement
        OCV_p = ocv_pack_numpy(x_new[0], self.elp, self.n_modules)
        V_oc_eff = OCV_p - x_new[3] - x_new[4]
        self._last_I_net, _ = compute_current_numpy(P_net, V_oc_eff, self.elp.R0)

        self._x = x_new.copy()
        y_meas = self.get_measurement()
        u_applied = np.array([P_net, P_reg])
        # p_delivered is the FCR power the plant actually served — i.e.
        # the difference between total dispatch and the clipped base
        # dispatch. Signed. The ledger uses |p_delivered| vs
        # |activation * P_reg| to compute delivery score and penalty.
        p_delivered = float(P_net - P_net_setpoint_clipped)
        return self._x.copy(), y_meas, u_applied, p_delivered

    def get_measurement(self) -> np.ndarray:
        """Return noisy [SOC, Temperature, V_term] measurement.

        Returns
        -------
        y_meas : ndarray, shape (3,)
            [SOC_measured, T_measured, V_term_measured]
        """
        SOC, T = self._x[0], self._x[2]
        V_rc1, V_rc2 = self._x[3], self._x[4]

        OCV_p = ocv_pack_numpy(SOC, self.elp, self.n_modules)
        V_term = OCV_p - V_rc1 - V_rc2 - self._last_I_net * self.elp.R0

        noise_soc = self._rng.normal(0.0, self._meas_std_soc)
        noise_temp = self._rng.normal(0.0, self._meas_std_temp)
        noise_volt = self._rng.normal(0.0, self._meas_std_volt)

        soc_meas = float(np.clip(SOC + noise_soc, 0.0, 1.0))
        t_meas = T + noise_temp
        v_meas = V_term + noise_volt

        return np.array([soc_meas, t_meas, v_meas])

    def get_terminal_voltage(self) -> float:
        """Return current terminal voltage (no noise)."""
        SOC = self._x[0]
        V_rc1, V_rc2 = self._x[3], self._x[4]
        OCV_p = ocv_pack_numpy(SOC, self.elp, self.n_modules)
        return float(OCV_p - V_rc1 - V_rc2 - self._last_I_net * self.elp.R0)

    def get_state(self) -> np.ndarray:
        """Return the true state [SOC, SOH, T, V_rc1, V_rc2] (for logging only)."""
        return self._x.copy()

    def reset(
        self,
        soc: float | None = None,
        soh: float | None = None,
        temp: float | None = None,
        vrc1: float | None = None,
        vrc2: float | None = None,
    ) -> None:
        """Reset plant state."""
        self._x[0] = soc if soc is not None else self.bp.SOC_init
        self._x[1] = soh if soh is not None else self.bp.SOH_init
        self._x[2] = temp if temp is not None else self.thp.T_init
        self._x[3] = vrc1 if vrc1 is not None else self.elp.V_rc1_init
        self._x[4] = vrc2 if vrc2 is not None else self.elp.V_rc2_init
        self._last_I_net = 0.0


# ---------------------------------------------------------------------------
#  Multi-cell pack model
# ---------------------------------------------------------------------------

class BatteryPack:
    """Multi-cell battery pack with active balancing and per-cell 2RC circuits.

    Wraps *N* ``BatteryPlant`` instances, each with per-cell scaled
    parameters including manufacturing variation.

    Pack-level aggregation
    ----------------------
    SOC_pack  = mean(cell SOCs)      — most representative for optimizer
    SOH_pack  = min(cell SOHs)       — weakest-link industry standard
    T_pack    = max(cell temps)      — thermal safety (hottest cell)
    V_rc1_pack = sum(cell V_rc1s)    — series connection
    V_rc2_pack = sum(cell V_rc2s)    — series connection
    """

    def __init__(
        self,
        bp: BatteryParams,
        tp: TimeParams,
        thp: ThermalParams,
        elp: ElectricalParams,
        pp: PackParams,
        seed: int = 42,
    ) -> None:
        self.bp = bp
        self.tp = tp
        self.thp = thp
        self.elp = elp
        self.pp = pp
        self._rng_meas = np.random.default_rng(seed)
        rng_cells = np.random.default_rng(pp.seed)

        n = pp.n_cells

        # Per-cell variation factors (deterministic from pp.seed)
        cap_factors = 1.0 + rng_cells.uniform(-pp.capacity_spread, pp.capacity_spread, n)
        res_factors = 1.0 + rng_cells.uniform(-pp.resistance_spread, pp.resistance_spread, n)
        deg_factors = 1.0 + rng_cells.uniform(-pp.degradation_spread, pp.degradation_spread, n)
        soc_offsets = rng_cells.uniform(-pp.initial_soc_spread, pp.initial_soc_spread, n)

        # Create per-cell BatteryPlant instances with scaled parameters
        self.cells: list[BatteryPlant] = []
        for i in range(n):
            cell_bp = BatteryParams(
                E_nom_kwh=bp.E_nom_kwh / n * cap_factors[i],
                P_max_kw=bp.P_max_kw / n,
                SOC_min=bp.SOC_min,
                SOC_max=bp.SOC_max,
                SOC_init=float(np.clip(
                    bp.SOC_init + soc_offsets[i], bp.SOC_min, bp.SOC_max,
                )),
                SOH_init=bp.SOH_init,
                SOC_terminal=bp.SOC_terminal,
                eta_charge=bp.eta_charge,
                eta_discharge=bp.eta_discharge,
                alpha_deg=bp.alpha_deg * deg_factors[i],
                alpha_deg_reg=bp.alpha_deg_reg * deg_factors[i],
            )
            cell_thp = ThermalParams(
                R_internal=thp.R_internal / n * res_factors[i],
                C_thermal=thp.C_thermal / n,
                h_cool=thp.h_cool / n,
                T_ambient=thp.T_ambient,
                T_init=thp.T_init,
                T_max=thp.T_max,
                T_min=thp.T_min,
                V_nominal=thp.V_nominal / n,
                E_a=thp.E_a,
                R_gas=thp.R_gas,
                T_ref=thp.T_ref,
            )
            # Per-cell electrical params: R scaled by cell count and variation,
            # time constants preserved (intrinsic to cell chemistry)
            rf = res_factors[i]
            cell_elp = ElectricalParams(
                R0=elp.R0 / n * rf,
                R1=elp.R1 / n * rf,
                tau_1=elp.tau_1,
                R2=elp.R2 / n * rf,
                tau_2=elp.tau_2,
                n_series_cells=elp.n_series_cells,
                V_min_cell=elp.V_min_cell,
                V_max_cell=elp.V_max_cell,
                V_rc1_init=elp.V_rc1_init,
                V_rc2_init=elp.V_rc2_init,
                ocv_coeffs=elp.ocv_coeffs,
                sigma_v_meas=elp.sigma_v_meas / n,
            )
            self.cells.append(BatteryPlant(
                cell_bp, tp, cell_thp, cell_elp,
                n_modules=1,  # each cell IS one module
                seed=seed + i,
            ))

        # Last-applied balancing power per cell
        self._balancing_power = np.zeros(n)

        # Pack-level measurement noise
        self._meas_std_soc = 0.01
        self._meas_std_temp = 0.5
        self._meas_std_volt = elp.sigma_v_meas

    # ---- public interface (matches BatteryPlant) ----

    def step(
        self,
        u: np.ndarray,
        activation_k: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Integrate one dt_sim step for all cells.

        v5 refactor: 2-vector input ``[P_net_signed, P_reg]``. Wash trades
        are impossible by construction. The pack enforces the global
        budget ``|P_net| + P_reg <= P_max`` and SOC headroom (using the
        worst-case cell SOC), then distributes ``P_net / n`` to each cell.
        Active balancing is layered on top per cell as a small chg/dis
        delta with zero net pack-level effect.

        Parameters
        ----------
        u : ndarray, shape (2,)
            Pack-level [P_net_setpoint, P_reg_committed]  (kW).
        activation_k : float, default 0.0
            Current FCR activation sample in [-1, +1]. RF1: plant
            applies activation at the pack boundary before the equal
            split to cells.

        Returns
        -------
        x_pack : ndarray, shape (5,)
            [SOC_mean, SOH_min, T_max, V_rc1_sum, V_rc2_sum].
        y_meas : ndarray, shape (3,)
            Noisy [SOC_pack, T_pack, V_term_pack].
        u_applied : ndarray, shape (2,)
            [P_net_applied, P_reg_applied] — actually-applied pack-level
            power post-clipping.
        p_delivered : float
            Signed FCR power delivered at pack level (= P_net_applied -
            P_net_setpoint).
        """
        n = self.pp.n_cells
        bp = self.bp
        dt = self.tp.dt_sim
        dt_h = dt / 3600.0

        # ---- 1. Runtime power limit + SOC headroom (worst-case cell) ----
        # SOC headroom uses the worst-case cell so the weakest cell stays in bounds.
        cell_states = self.get_cell_states()
        soc_min_cell = float(np.min(cell_states[:, 0]))
        soc_max_cell = float(np.max(cell_states[:, 0]))
        soh_min_cell = float(np.min(cell_states[:, 1]))
        E_eff_pack = soh_min_cell * bp.E_nom_kwh

        def _clip_with_soc(p: float) -> float:
            p = float(np.clip(p, -bp.P_max_kw, bp.P_max_kw))
            if p > 0.0:
                max_dis = (soc_min_cell - bp.SOC_min) * E_eff_pack * bp.eta_discharge / dt_h
                return min(p, max(0.0, max_dis))
            if p < 0.0:
                max_chg = (bp.SOC_max - soc_max_cell) * E_eff_pack / (bp.eta_charge * dt_h)
                return max(p, -max(0.0, max_chg))
            return 0.0

        P_net_setpoint = float(u[0])
        P_reg = float(np.clip(u[1], 0.0, bp.P_max_kw))
        # Two-pass clip: first the base setpoint, then the setpoint plus
        # activation. Delta between them = FCR portion the plant actually
        # served (attribution fix for p_delivered).
        P_net_setpoint_clipped = _clip_with_soc(P_net_setpoint)
        P_net = _clip_with_soc(P_net_setpoint_clipped + activation_k * P_reg)

        # ---- 2. Decompose to chg/dis at the pack level ----
        if P_net >= 0.0:
            P_chg_pack, P_dis_pack = 0.0, P_net
        else:
            P_chg_pack, P_dis_pack = -P_net, 0.0

        # ---- 3. Equal split to cells ----
        P_chg_cell = P_chg_pack / n
        P_dis_cell = P_dis_pack / n
        P_reg_cell = P_reg / n

        # ---- 4. Active balancing (zero-sum delta on top of base) ----
        if self.pp.balancing_enabled:
            cell_socs = cell_states[:, 0]
            soc_avg = np.mean(cell_socs)
            bal = self.pp.balancing_gain * (soc_avg - cell_socs)
            bal = np.clip(bal, -self.pp.max_balancing_power, self.pp.max_balancing_power)
            bal -= np.mean(bal)  # enforce zero-sum
            self._balancing_power = bal.copy()
        else:
            self._balancing_power = np.zeros(n)

        # ---- 5. Step each cell with its (P_net_cell, P_reg_cell) ----
        # Balancing power adds a tiny chg/dis delta per cell. Since each
        # cell's plant uses the new (P_net, P_reg) interface, we feed it
        # a signed P_net = P_dis_cell - P_chg_cell + balancing_delta.
        for i, cell in enumerate(self.cells):
            p_bal = self._balancing_power[i]
            # Cell's net power = base + balancing
            # base = P_dis_cell - P_chg_cell  (one of them is zero)
            cell_p_net = P_dis_cell - P_chg_cell + p_bal
            cell.step(np.array([cell_p_net, P_reg_cell]))

        x_pack = self.get_state()
        y_meas = self._make_measurement(x_pack)
        u_applied = np.array([P_net, P_reg])
        p_delivered = float(P_net - P_net_setpoint_clipped)
        return x_pack, y_meas, u_applied, p_delivered

    def get_state(self) -> np.ndarray:
        """Return aggregated pack state [SOC_mean, SOH_min, T_max, V_rc1_sum, V_rc2_sum]."""
        cs = self.get_cell_states()
        return np.array([
            np.mean(cs[:, 0]),    # SOC mean
            np.min(cs[:, 1]),     # SOH min (weakest-link)
            np.max(cs[:, 2]),     # T max (thermal safety)
            np.sum(cs[:, 3]),     # V_rc1 sum (series)
            np.sum(cs[:, 4]),     # V_rc2 sum (series)
        ])

    def get_measurement(self) -> np.ndarray:
        """Return noisy [SOC_pack, T_pack, V_term_pack] measurement."""
        return self._make_measurement(self.get_state())

    def get_cell_states(self) -> np.ndarray:
        """Return (n_cells, 5) array: [SOC, SOH, T, V_rc1, V_rc2] per cell."""
        return np.array([c.get_state() for c in self.cells])

    def get_balancing_power(self) -> np.ndarray:
        """Return last-applied balancing power per cell, shape (n_cells,)."""
        return self._balancing_power.copy()

    def get_terminal_voltage(self) -> float:
        """Return pack terminal voltage = sum of cell terminal voltages."""
        return float(sum(c.get_terminal_voltage() for c in self.cells))

    def reset(
        self,
        soc: float | None = None,
        soh: float | None = None,
        temp: float | None = None,
        vrc1: float | None = None,
        vrc2: float | None = None,
    ) -> None:
        """Reset all cells."""
        n = self.pp.n_cells
        cell_vrc1 = vrc1 / n if vrc1 is not None else None
        cell_vrc2 = vrc2 / n if vrc2 is not None else None
        for cell in self.cells:
            cell.reset(soc=soc, soh=soh, temp=temp, vrc1=cell_vrc1, vrc2=cell_vrc2)

    # ---- internal helpers ----

    def _make_measurement(self, x_pack: np.ndarray) -> np.ndarray:
        noise_soc = self._rng_meas.normal(0.0, self._meas_std_soc)
        noise_temp = self._rng_meas.normal(0.0, self._meas_std_temp)
        noise_volt = self._rng_meas.normal(0.0, self._meas_std_volt)

        v_term_pack = self.get_terminal_voltage()

        return np.array([
            float(np.clip(x_pack[0] + noise_soc, 0.0, 1.0)),
            x_pack[2] + noise_temp,
            v_term_pack + noise_volt,
        ])
