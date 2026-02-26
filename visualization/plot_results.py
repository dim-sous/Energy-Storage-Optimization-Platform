"""Six-panel result visualisation for the hierarchical BESS platform.

Panel layout (3 rows x 2 columns)
----------------------------------
  [1] SOC: true vs EKF vs MHE       |  [2] SOH: true vs EKF vs MHE
  [3] Power dispatch (MPC + EMS)     |  [4] Energy & regulation prices
  [5] Cumulative profit breakdown    |  [6] SOH degradation over time

Design notes
------------
- Large fonts and thick lines for readability in presentations.
- MPC applied power and EMS reference power overlaid in one panel
  so the viewer can judge tracking quality at a glance.
- Market prices plotted alongside so energy-market professionals
  can correlate dispatch decisions with price signals.
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


def plot_results(
    sim: dict,
    bp: BatteryParams,
    save_path: str = "results.png",
) -> None:
    """Generate the six-panel summary figure.

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

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(
        "Hierarchical BESS Control \u2014 24 h Multi-Rate Simulation",
        fontsize=_SUPTITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )

    t_sim_h = sim["time_sim"] / 3600.0           # plant time in hours
    t_mpc_h = sim["time_mpc"] / 3600.0           # estimator / MPC time in hours

    # ==================================================================
    #  Panel 1 — SOC: true vs EKF vs MHE
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
    #  Panel 2 — SOH: true vs EKF vs MHE
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
    #  Panel 3 — Combined power: MPC applied + EMS reference
    # ==================================================================
    ax = axes[1, 0]
    n_mpc = len(sim["power_applied"])
    dt_mpc_s = sim["time_mpc"][1] - sim["time_mpc"][0] if len(sim["time_mpc"]) > 1 else 60.0
    t_pow = np.arange(n_mpc) * dt_mpc_s / 3600.0

    # MPC applied power (solid, thinner — high-frequency detail)
    ax.step(t_pow, sim["power_applied"][:, 0], where="post",
            color="tab:green", linewidth=_LW_MPC, label="P_chg applied")
    ax.step(t_pow, sim["power_applied"][:, 1], where="post",
            color="tab:blue", linewidth=_LW_MPC, label="P_dis applied")
    ax.step(t_pow, sim["power_applied"][:, 2], where="post",
            color="tab:orange", linewidth=_LW_MPC, label="P_reg applied")

    # EMS hourly references (thick, transparent step — the plan)
    ems_p_chg = sim.get("ems_p_chg_refs", [])
    ems_p_dis = sim.get("ems_p_dis_refs", [])
    ems_p_reg = sim.get("ems_p_reg_refs", [])

    if ems_p_chg:
        # Stitch all hourly EMS solves into one time series
        all_chg, all_dis, all_reg = [], [], []
        for pc, pd, pr in zip(ems_p_chg, ems_p_dis, ems_p_reg):
            all_chg.append(pc[0])    # only the first-stage (applied) value
            all_dis.append(pd[0])
            all_reg.append(pr[0])
        t_ems_h = np.arange(len(all_chg))

        ax.step(t_ems_h, all_chg, where="post",
                color="tab:green", linewidth=_LW_REF, alpha=_ALPHA_REF,
                linestyle="-", label="P_chg EMS ref")
        ax.step(t_ems_h, all_dis, where="post",
                color="tab:blue", linewidth=_LW_REF, alpha=_ALPHA_REF,
                linestyle="-", label="P_dis EMS ref")
        ax.step(t_ems_h, all_reg, where="post",
                color="tab:orange", linewidth=_LW_REF, alpha=_ALPHA_REF,
                linestyle="-", label="P_reg EMS ref")

    ax.axhline(0, color="k", linewidth=0.4)
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Power Dispatch: MPC Tracking vs EMS Reference")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel 4 — Energy and regulation prices
    # ==================================================================
    ax = axes[1, 1]
    prices_e = sim.get("prices_energy", None)
    prices_r = sim.get("prices_reg", None)

    if prices_e is not None:
        n_price_hours = min(len(prices_e), int(t_sim_h[-1]) + 1)
        t_price = np.arange(n_price_hours)
        ax.step(t_price, prices_e[:n_price_hours], where="post",
                color="tab:blue", linewidth=_LW_TRUE, label="Energy price [$/kWh]")
    if prices_r is not None:
        n_price_hours = min(len(prices_r), int(t_sim_h[-1]) + 1)
        t_price = np.arange(n_price_hours)
        ax.step(t_price, prices_r[:n_price_hours], where="post",
                color="tab:orange", linewidth=_LW_TRUE,
                linestyle="--", label="Regulation price [$/kW/h]")

    ax.set_ylabel("Price [$/kWh or $/kW/h]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="upper right")
    ax.set_title("Market Prices (Scenario 1)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # ==================================================================
    #  Panel 5 — Cumulative profit breakdown
    # ==================================================================
    ax = axes[2, 0]
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

    # ==================================================================
    #  Panel 6 — Degradation tracking
    # ==================================================================
    ax = axes[2, 1]
    soh_loss = (sim["soh_true"][0] - sim["soh_true"]) * 100  # percent
    ax.plot(t_sim_h, soh_loss, color="tab:purple", linewidth=_LW_TRUE)
    ax.set_ylabel("SOH Loss [%]")
    ax.set_xlabel("Time [h]")
    ax.set_title(
        f"Battery Degradation (total: {sim['soh_degradation']*100:.4f}%)"
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved to {save_path}")
