"""Configuration parameters for the hierarchical BESS control platform.

All physical units are SI-consistent and explicitly documented:
  - Energy:  kWh
  - Power:   kW
  - Time:    seconds (except sim_hours)
  - Price:   $/kWh  (energy),  $/kW/h  (regulation)
  - SOC/SOH: dimensionless  [0, 1]
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryParams:
    """Physical parameters of the battery energy storage system with degradation.

    Degradation model
    -----------------
    dSOH/dt = -alpha_deg * (P_chg + P_dis + |P_reg|)

    With alpha_deg = 2.78e-9  [1/(kW*s)]:
      One full 200 kWh cycle at 100 kW takes ~14 400 s.
      SOH loss per cycle = 2.78e-9 * 100 * 14400 = 0.004 (0.4 %).
      ~50 full cycles to reach 80 % SOH (accelerated for demo visibility).
      In a 24 h simulation with ~2 equivalent cycles, SOH drops ~0.8 %.
    """

    E_nom_kwh: float = 200.0           # Nominal energy capacity  [kWh]
    P_max_kw: float = 100.0            # Maximum charge / discharge power  [kW]
    SOC_min: float = 0.10              # Minimum allowable state of charge  [-]
    SOC_max: float = 0.90              # Maximum allowable state of charge  [-]
    SOC_init: float = 0.50             # Initial state of charge  [-]
    SOH_init: float = 1.00             # Initial state of health  [-]
    SOC_terminal: float = 0.50         # Terminal SOC target for EMS  [-]
    eta_charge: float = 0.95           # Charging efficiency  [-]
    eta_discharge: float = 0.95        # Discharging efficiency  [-]
    alpha_deg: float = 2.78e-9         # Degradation rate  [1/(kW*s)]


@dataclass(frozen=True)
class TimeParams:
    """Time discretisation for multi-rate control.

    Every time quantity is in **seconds** except ``sim_hours``.
    """

    dt_ems: float = 3600.0             # EMS sampling period  [s]
    dt_mpc: float = 60.0               # MPC sampling period  [s]
    dt_estimator: float = 60.0         # Estimator sampling period  [s]  (= dt_mpc)
    dt_sim: float = 1.0                # Plant simulation step  [s]
    sim_hours: float = 24.0            # Total simulation duration  [hours]


@dataclass(frozen=True)
class EMSParams:
    """Parameters for the stochastic Energy Management System optimizer."""

    N_ems: int = 24                    # Planning horizon  [hours / steps at dt_ems]
    Nc_ems: int = 24                   # Control horizon  (= N_ems)
    n_scenarios: int = 5               # Number of price scenarios
    regulation_fraction: float = 0.3   # Max fraction of P_max for regulation  [-]
    degradation_cost: float = 50.0     # Cost of SOH loss  [$/unit SOH lost]
    terminal_soc_weight: float = 1e4   # Terminal SOC deviation penalty
    terminal_soh_weight: float = 1e4   # Terminal SOH deviation penalty


@dataclass(frozen=True)
class MPCParams:
    """Tuning parameters for the nonlinear tracking MPC."""

    N_mpc: int = 60                    # Prediction horizon  [steps at dt_mpc]
    Nc_mpc: int = 10                   # Control horizon  [steps at dt_mpc]
    Q_soc: float = 1e4                 # SOC tracking weight
    Q_soh: float = 1e2                 # SOH tracking weight
    R_power: float = 1.0               # Power reference tracking weight (per input)
    R_delta: float = 10.0              # Control rate-of-change penalty
    Q_terminal: float = 1e5            # Terminal SOC penalty
    slack_penalty: float = 1e6         # Soft SOC constraint violation penalty


@dataclass(frozen=True)
class EKFParams:
    """Tuning parameters for the Extended Kalman Filter."""

    # Process noise covariance  Q  (2x2 diagonal)
    q_soc: float = 1e-6               # SOC process noise variance
    q_soh: float = 1e-12              # SOH process noise variance (extremely slow dynamics)

    # Measurement noise covariance  R  (scalar — only SOC measured)
    r_soc_meas: float = 1e-4          # SOC measurement noise variance (sigma ~ 0.01)

    # Initial state error covariance  P_0  (2x2 diagonal)
    p0_soc: float = 1e-3              # Initial SOC uncertainty
    p0_soh: float = 1e-2              # Initial SOH uncertainty (larger — unknown)


@dataclass(frozen=True)
class MHEParams:
    """Tuning parameters for Moving Horizon Estimation."""

    N_mhe: int = 30                    # Estimation window  [steps at dt_estimator]

    # Arrival cost weights  (inverse of prior covariance)
    arrival_soc: float = 1e3           # Weight on SOC arrival cost
    arrival_soh: float = 1e4           # Weight on SOH arrival cost (high — SOH barely observable)

    # Stage cost weights
    w_soc_meas: float = 1e4            # Measurement residual weight
    w_process_soc: float = 1e2         # Process disturbance weight (SOC)
    w_process_soh: float = 1e8         # Process disturbance weight (SOH, very high — penalise SOH noise)
