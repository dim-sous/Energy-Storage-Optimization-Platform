"""Four-panel result visualisation for the hierarchical BESS platform.

Panel layout (2 rows x 2 columns)
----------------------------------
  [0,0] SOC: true vs EKF vs MHE       |  [0,1] SOH: true vs EKF vs MHE
  [1,0] Net grid power + regulation    |  [1,1] Cumulative profit breakdown
        + energy price overlay

Design notes
------------
- Net grid power panel uses green/red fill to show selling/buying
  periods, with regulation power as an orange step line and a
  dual y-axis energy price overlay for arbitrage context.
- Large fonts and thick lines for readability in presentations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from config.parameters import BatteryParams


# ---------------------------------------------------------------------------
#  Global style constants
# ---------------------------------------------------------------------------
_TITLE_SIZE = 16
_SUPTITLE_SIZE = 18
_LABEL_SIZE = 13
_TICK_SIZE = 11
_LEGEND_SIZE = 10
_LW_TRUE = 2.0          # true / primary curves
_LW_EST = 1.5           # estimator curves
_LW_REF = 2.5           # EMS reference (step)
_LW_MPC = 1.0           # MPC applied (dense, thinner)
_ALPHA_REF = 0.45        # transparency for reference step lines


def _stepify(
    t: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand (t, y) into step-plot coordinates for use with fill_between."""
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    t_s = np.empty(2 * n)
    y_s = np.empty(2 * n)
    t_s[0::2] = t
    t_s[1::2] = np.append(t[1:], t[-1] + dt)
    y_s[0::2] = y
    y_s[1::2] = y
    return t_s, y_s


def plot_results(
    sim: dict,
    bp: BatteryParams,
    save_path: str = "results.png",
) -> None:
    """Generate the four-panel summary figure.

    Parameters
    ----------
    sim : dict
        Output from ``MultiRateSimulator.run()``.
    bp : BatteryParams
    save_path : str
        File path for the saved figure.
    """
    plt.rcParams.update({
        "font.size": _TICK_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "legend.fontsize": _LEGEND_SIZE,
    })

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(
        "Hierarchical BESS Control \u2014 24 h Multi-Rate Simulation",
        fontsize=_SUPTITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )

    t_sim_h = sim["time_sim"] / 3600.0           # plant time in hours
    t_mpc_h = sim["time_mpc"] / 3600.0           # estimator / MPC time in hours

    # ==================================================================
    #  Panel [0,0] — SOC: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 0]
    ax.plot(t_sim_h, sim["soc_true"], color="0.35", linewidth=_LW_TRUE,
            label="True SOC")
    ax.plot(t_mpc_h, sim["soc_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF estimate")
    ax.plot(t_mpc_h, sim["soc_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE estimate")
    ax.axhspan(bp.SOC_min, bp.SOC_max, alpha=0.08, color="green",
               label=f"SOC limits [{bp.SOC_min:.0%}\u2013{bp.SOC_max:.0%}]")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.set_title("State of Charge Estimation")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel [0,1] — SOH: true vs EKF vs MHE
    # ==================================================================
    ax = axes[0, 1]
    ax.plot(t_sim_h, sim["soh_true"], color="0.35", linewidth=_LW_TRUE,
            label="True SOH")
    ax.plot(t_mpc_h, sim["soh_ekf"], color="tab:blue", linewidth=_LW_EST,
            label="EKF estimate")
    ax.plot(t_mpc_h, sim["soh_mhe"], color="tab:red", linewidth=_LW_EST,
            linestyle="--", label="MHE estimate")
    ax.set_ylabel("SOH [-]")
    ax.legend(loc="lower left")
    ax.set_title("State of Health Estimation")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Shared computations for power / profit panels
    # ==================================================================
    n_mpc = len(sim["power_applied"])
    dt_mpc_s = (sim["time_mpc"][1] - sim["time_mpc"][0]
                if len(sim["time_mpc"]) > 1 else 60.0)
    t_pow = np.arange(n_mpc) * dt_mpc_s / 3600.0

    # Net grid power: positive = selling (discharge), negative = buying (charge)
    net_grid_power = sim["power_applied"][:, 1] - sim["power_applied"][:, 0]

    # Regulation power (always >= 0)
    reg_power = sim["power_applied"][:, 2]

    # Stitch EMS first-stage references into hourly time series
    ems_p_chg = sim.get("ems_p_chg_refs", [])
    ems_p_dis = sim.get("ems_p_dis_refs", [])
    ems_p_reg = sim.get("ems_p_reg_refs", [])
    all_chg, all_dis, all_reg = [], [], []
    if ems_p_chg:
        for pc, pd, pr in zip(ems_p_chg, ems_p_dis, ems_p_reg):
            all_chg.append(pc[0])
            all_dis.append(pd[0])
            all_reg.append(pr[0])
    t_ems_h = np.arange(len(all_chg)) if all_chg else np.array([])

    # Prices
    prices_e = sim.get("prices_energy", None)
    n_price_hours = 0
    t_price = np.array([])
    if prices_e is not None:
        n_price_hours = min(len(prices_e), int(t_sim_h[-1]) + 1)
        t_price = np.arange(n_price_hours)

    # ==================================================================
    #  Panel [1,0] — Net Grid Power + Regulation + Price Overlay
    # ==================================================================
    ax = axes[1, 0]

    # Green/red fill for sell/buy regions
    ts, ys = _stepify(t_pow, net_grid_power)
    ax.fill_between(ts, ys, 0, where=(ys >= 0),
                    color="tab:green", alpha=0.25, label="Selling to grid",
                    interpolate=True)
    ax.fill_between(ts, ys, 0, where=(ys < 0),
                    color="tab:red", alpha=0.25, label="Buying from grid",
                    interpolate=True)
    ax.plot(ts, ys, color="k", linewidth=1.0, label="Net grid power")

    # Regulation power (MPC applied)
    ax.step(t_pow, reg_power, where="post",
            color="tab:orange", linewidth=_LW_MPC, label="P_reg applied")

    # EMS references
    if all_chg:
        ems_net = np.array(all_dis) - np.array(all_chg)
        ax.step(t_ems_h, ems_net, where="post",
                color="0.3", linewidth=_LW_REF, alpha=0.5,
                linestyle="--", label="EMS net ref")
    if all_reg:
        ax.step(t_ems_h, all_reg, where="post",
                color="tab:orange", linewidth=_LW_REF, alpha=_ALPHA_REF,
                linestyle="--", label="EMS P_reg ref")

    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Power [kW]  (+ sell / \u2212 buy)")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Grid Power Exchange and Regulation Reserve")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # Dual y-axis: energy price overlay
    if prices_e is not None:
        ax2 = ax.twinx()
        ax2.step(t_price, prices_e[:n_price_hours], where="post",
                 color="tab:purple", linewidth=1.0, alpha=0.4, linestyle="-.")
        ax2.set_ylabel("Energy Price [$/kWh]", color="tab:purple", alpha=0.6)
        ax2.tick_params(axis="y", colors="tab:purple", labelsize=_TICK_SIZE)

    # ==================================================================
    #  Panel [1,1] — Cumulative profit breakdown
    # ==================================================================
    ax = axes[1, 1]
    n_prof = len(sim["cumulative_profit"])
    t_prof = np.arange(n_prof) * dt_mpc_s / 3600.0

    cum_energy = np.cumsum(sim["energy_profit"])
    cum_reg = np.cumsum(sim["reg_profit"])
    cum_deg = np.cumsum(sim["deg_cost"])

    ax.plot(t_prof, cum_energy, color="tab:blue", linewidth=_LW_EST,
            label="Energy arbitrage")
    ax.plot(t_prof, cum_reg, color="tab:orange", linewidth=_LW_EST,
            label="Regulation revenue")
    ax.plot(t_prof, -cum_deg, color="tab:red", linewidth=_LW_EST,
            label="Degradation cost")
    ax.plot(t_prof, sim["cumulative_profit"], color="k", linewidth=_LW_TRUE,
            label=f"Net profit (${sim['total_profit']:.2f})")
    ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
    ax.set_ylabel("Cumulative [$]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper left")
    ax.set_title("Revenue Breakdown")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
