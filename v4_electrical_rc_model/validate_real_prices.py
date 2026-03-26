"""Real-price validation for v4_electrical_rc_model.

Runs the v4 optimizer against real European day-ahead prices (DE-LU zone)
and compares performance to a rule-based baseline dispatch strategy.

Usage
-----
    uv run python v4_electrical_rc_model/validate_real_prices.py

Outputs
-------
    results/v4_real_price_validation.png   — summary plots
    results/v4_real_price_validation.json  — raw metrics per day + statistics
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.parameters import (
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    ThermalParams,
    TimeParams,
)
from data.real_price_loader import RealPriceLoader
from simulation.simulator import MultiRateSimulator

# ---------------------------------------------------------------------------
#  Logging — quieter for batch runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
N_DAYS = 84              # All available days (full Q1 2024)
ENERGY_CSV = PROJECT_ROOT / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = PROJECT_ROOT / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = PROJECT_ROOT.parent / "results"


# ===========================================================================
#  Rule-based baseline
# ===========================================================================

def rule_based_profit(
    energy_prices_24h: np.ndarray,
    bp: BatteryParams,
) -> tuple[float, np.ndarray]:
    """Simulate a simple threshold-based dispatch over 24 hours.

    Strategy
    --------
    - Sort hours by price.
    - Charge at full power during the cheapest hours.
    - Discharge at full power during the most expensive hours.
    - Respect SOC limits [SOC_min, SOC_max].

    Returns
    -------
    profit : float [$]
    hourly_power : ndarray, shape (24, 3)
        [P_charge, P_discharge, P_reg] per hour in kW.
    """
    n_hours = len(energy_prices_24h)
    P_max = bp.P_max_kw           # kW
    E_nom = bp.E_nom_kwh          # kWh
    eta = bp.eta_charge           # charge/discharge efficiency
    soc_min = bp.SOC_min
    soc_max = bp.SOC_max

    sorted_hours = np.argsort(energy_prices_24h)

    usable_kwh = (soc_max - soc_min) * E_nom
    hours_needed = usable_kwh / (P_max * eta)
    n_charge = int(np.ceil(hours_needed))
    n_discharge = int(np.ceil(hours_needed))

    charge_hours = set(sorted_hours[:n_charge])
    discharge_hours = set(sorted_hours[-n_discharge:])

    overlap = charge_hours & discharge_hours
    charge_hours -= overlap
    discharge_hours -= overlap

    soc = 0.5
    profit = 0.0
    hourly_power = np.zeros((n_hours, 3))  # P_chg, P_dis, P_reg

    for h in range(n_hours):
        price = energy_prices_24h[h]

        if h in charge_hours and soc < soc_max:
            energy_in = min(P_max * eta, (soc_max - soc) * E_nom)
            soc += energy_in / E_nom
            cost = (energy_in / eta) * price
            profit -= cost
            hourly_power[h, 0] = P_max  # Charging

        elif h in discharge_hours and soc > soc_min:
            energy_out = min(P_max * eta, (soc - soc_min) * E_nom)
            soc -= energy_out / E_nom
            revenue = energy_out * eta * price
            profit += revenue
            hourly_power[h, 1] = P_max  # Discharging

    return profit, hourly_power


def simulate_baseline_degradation(
    hourly_power: np.ndarray,
    bp: BatteryParams,
    tp: TimeParams,
    thp: ThermalParams,
    elp: ElectricalParams,
    pp: PackParams,
) -> float:
    """Run baseline power schedule through the real battery plant model.

    Feeds the rule-based hourly commands into the BatteryPack at dt_sim
    resolution and returns the actual SOH degradation.

    Returns
    -------
    float : SOH degradation (positive, e.g. 0.0001 = 0.01%)
    """
    from models.battery_model import BatteryPack

    # Use coarse dt for baseline — power only changes hourly, no need for 10s resolution
    tp_coarse = dataclasses.replace(tp, dt_sim=60.0)
    pack = BatteryPack(bp, tp_coarse, thp, elp, pp)
    soh_init = pack.get_state()[1]  # SOH_min (weakest-link)

    dt_sim = tp_coarse.dt_sim
    total_seconds = int(24 * 3600)
    n_steps = int(total_seconds / dt_sim)  # 1440 steps instead of 8640
    steps_per_hour = int(3600 / dt_sim)

    for step_idx in range(n_steps):
        hour = min(step_idx // steps_per_hour, 23)
        u = hourly_power[hour]  # [P_chg, P_dis, P_reg]
        pack.step(u)

    soh_final = pack.get_state()[1]

    return float(soh_init - soh_final)


# ===========================================================================
#  Per-day worker (runs in a subprocess)
# ===========================================================================

def _run_single_day(args: tuple) -> dict:
    """Simulate one day (MPC + baseline). Designed for multiprocessing.Pool."""
    (day_idx, energy_scen, reg_scen, probs,
     day_prices_24, bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp) = args

    # Suppress CasADi solver output in worker processes
    logging.disable(logging.WARNING)

    simulator = MultiRateSimulator(bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp)

    t0 = time.perf_counter()
    results = simulator.run(energy_scen, reg_scen, probs)
    wall_time = time.perf_counter() - t0

    baseline_profit, baseline_power = rule_based_profit(day_prices_24, bp)

    # Run baseline power schedule through the real plant model
    baseline_soh_deg = simulate_baseline_degradation(
        baseline_power, bp, tp, thp, elp, pp,
    )

    return {
        "day_idx": day_idx,
        "mpc_profit": float(results["total_profit"]),
        "baseline_profit": float(baseline_profit),
        "soh_deg": float(results["soh_degradation"]),
        "baseline_soh_deg": float(baseline_soh_deg),
        "max_temp": float(np.max(results["temp_true"])),
        "avg_solve_ms": float(np.mean(results["mpc_solve_times"])) * 1000,
        "solver_failures": int(results.get("solver_failures", 0)),
        "wall_time": wall_time,
    }


# ===========================================================================
#  Main validation loop
# ===========================================================================

def run_validation() -> dict:
    """Run v4 optimizer + baseline on N_DAYS of real European prices."""

    # ---- Configuration ----
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    mhe_p = MHEParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    pp = PackParams()

    # ==================================================================
    #  CALIBRATED PARAMETER OVERRIDES
    #
    #  Reference cell: Samsung SDI 94Ah prismatic NMC (utility ESS)
    #  Pack: 216s × 3p = 648 cells, 794.9 V nominal, 224 kWh nameplate
    #
    #  Sources:
    #    [1] Samsung SDI ESS prismatic cell datasheet (94 Ah, R_dc ~ 1 mΩ)
    #    [2] Schmalstieg et al., J. Power Sources 257 (2014) 325–334
    #        "From accelerated aging tests to a lifetime prediction model:
    #         Analyzing lithium-ion batteries" — NMC cycle aging
    #    [3] Ecker et al., J. Electrochem. Soc. 162 (2015) A1836
    #        "Calendar and cycle life study of Li(NiMnCo)O2-based 18650"
    #    [4] EIS-derived 2RC split: ~45% R0, ~30% R1, ~25% R2
    #        (Waag et al., J. Power Sources 2013)
    # ==================================================================

    # ---- Electrical: pack resistance from cell datasheet [1] ----
    # R_cell = 1.0 mΩ (25°C, BOL). Pack: 216s × 1.0mΩ / 3p = 72 mΩ
    # 2RC split from EIS literature [4]
    elp = dataclasses.replace(elp,
        R0=0.0324,   # 45% of 72 mΩ — ohmic (electrolyte + contacts)
        R1=0.0216,   # 30% — charge transfer / double-layer
        R2=0.0180,   # 25% — solid-state diffusion
    )
    # Also update ThermalParams.R_internal to match
    thp = dataclasses.replace(thp,
        R_internal=elp.R0 + elp.R1 + elp.R2,  # 0.072 Ω
        h_cool=150.0,    # Forced-air cooling, typical utility BESS [W/K]
        C_thermal=300_000.0,  # Larger thermal mass for 648-cell pack [J/K]
    )

    # ---- Degradation: calibrated to published NMC aging data [2,3] ----
    # Target: ~2.5%/year at 25°C, 1 cycle/day, 80% DOD
    # alpha_deg = 0.025 / (50 kW × 8h × 3600 s/h × 365 d) ≈ 4.76e-11
    bp = dataclasses.replace(bp,
        alpha_deg=4.76e-11,  # [1/(kW·s)] — calibrated to 2.5%/year
    )

    # ---- EMS: calibrated degradation cost + REAL regulation ----
    # The EMS formula: d_cost = degradation_cost * alpha_deg * P * dt
    # Calibrated so marginal degradation is ~$6/MWh ($0.006/kWh), giving:
    #   - Round-trip degradation cost: ~$12.5/MWh
    #   - Realistic daily margins of $2–3/day on 200 kWh system
    #   - Annual degradation cost: ~$900/year (2.5% SOH × $36,473)
    # This matches published BESS project economics where degradation
    # is a significant but not prohibitive fraction of arbitrage revenue.
    #
    # Regulation: ENABLED with real FCR capacity prices from SMARD.
    ep = dataclasses.replace(ep,
        degradation_cost=36_500.0,   # $/unit SOH — calibrated to ~$6/MWh marginal
    )

    # ---- Speed overrides for batch validation ----
    # Coarser time steps: 5-min MPC (288 solves/day vs 1440), 10s plant.
    # EMS stays hourly. Economics driven by EMS; MPC tracks references.
    tp = dataclasses.replace(tp, dt_mpc=300.0, dt_estimator=300.0, dt_sim=10.0)
    mp = dataclasses.replace(mp, N_mpc=12, Nc_mpc=4)
    mhe_p = dataclasses.replace(mhe_p, N_mhe=min(mhe_p.N_mhe, 6))

    # ---- Load real prices ----
    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=123)
    stats = loader.price_stats
    print("=" * 70)
    print("  REAL-PRICE VALIDATION  [v4_electrical_rc_model]")
    print("=" * 70)
    print(f"  Data source:  German day-ahead (DE-LU), EPEX SPOT via Energy-Charts")
    print(f"  Period:       Q1 2024 ({stats['n_days']} days, {stats['n_hours']} hours)")
    print(f"  Price range:  {stats['min_eur_mwh']:.1f} to {stats['max_eur_mwh']:.1f} EUR/MWh")
    print(f"  Mean price:   {stats['mean_eur_mwh']:.1f} EUR/MWh")
    print(f"  Neg. prices:  {stats['pct_negative']:.1f}%")
    print(f"  Days to run:  {N_DAYS}")
    print(f"  --- Calibrated Parameters ---")
    print(f"  Cell ref:     Samsung SDI 94Ah NMC prismatic")
    print(f"  R_pack:       {elp.R0+elp.R1+elp.R2:.4f} Ohm (R0={elp.R0:.4f}, R1={elp.R1:.4f}, R2={elp.R2:.4f})")
    print(f"  Cooling:      {thp.h_cool:.0f} W/K (forced air)")
    print(f"  alpha_deg:    {bp.alpha_deg:.2e} (~2.5%/year, Schmalstieg 2014)")
    print(f"  deg_cost:     {ep.degradation_cost:,.0f} $/unit SOH")
    reg_src = stats.get("reg_data", "unknown")
    print(f"  Regulation:   {reg_src}")
    if "reg_mean_eur_mw_h" in stats:
        print(f"  FCR mean:     {stats['reg_mean_eur_mw_h']:.2f} EUR/MW/h")
    print(f"  Revenue:      Energy arbitrage + FCR capacity")
    print("=" * 70)
    print()

    # ---- Sample days ----
    day_indices = loader.sample_day_indices(N_DAYS)
    day_indices.sort()

    # ---- Build job arguments (serialisable) ----
    jobs = []
    for day_idx in day_indices:
        energy_scen, reg_scen, probs = loader.generate_scenarios_for_day(
            day_idx=int(day_idx),
            n_hours=int(tp.sim_hours) + ep.N_ems,
            n_scenarios=ep.n_scenarios,
        )
        day_prices_24 = loader.get_day(int(day_idx))
        jobs.append((int(day_idx), energy_scen, reg_scen, probs,
                      day_prices_24, bp, tp, ep, mp, ekf_p, mhe_p, thp, elp, pp))

    # ---- Run in parallel (one process per CPU core) ----
    n_workers = min(N_DAYS, multiprocessing.cpu_count(), 3)  # Cap at 3 to avoid OOM on WSL
    print(f"  Running {N_DAYS} days across {n_workers} workers ...\n")
    t0_all = time.perf_counter()

    with multiprocessing.Pool(n_workers) as pool:
        day_results = pool.map(_run_single_day, jobs)

    wall_total = time.perf_counter() - t0_all
    print(f"\n  All {N_DAYS} days completed in {wall_total:.0f}s "
          f"({wall_total/N_DAYS:.0f}s effective per day)\n")

    # ---- Unpack results ----
    mpc_profits = []
    baseline_profits = []
    soh_degradations = []
    baseline_soh_degradations = []
    max_temps = []
    mpc_solve_times_avg = []
    solver_failures_list = []
    day_labels = []

    for r in day_results:
        day_labels.append(r["day_idx"])
        mpc_profits.append(r["mpc_profit"])
        baseline_profits.append(r["baseline_profit"])
        soh_degradations.append(r["soh_deg"])
        baseline_soh_degradations.append(r["baseline_soh_deg"])
        max_temps.append(r["max_temp"])
        mpc_solve_times_avg.append(r["avg_solve_ms"])
        solver_failures_list.append(r["solver_failures"])

        adv = r["mpc_profit"] - r["baseline_profit"]
        print(
            f"  Day {r['day_idx']:3d}  "
            f"MPC ${r['mpc_profit']:6.2f}  vs  Baseline ${r['baseline_profit']:6.2f}  "
            f"(+${adv:5.2f})  SOH opt-{r['soh_deg']*100:.4f}% base-{r['baseline_soh_deg']*100:.4f}%  "
            f"[{r['wall_time']:.0f}s]"
        )

    # ---- Aggregate statistics ----
    mpc_arr = np.array(mpc_profits)
    base_arr = np.array(baseline_profits)
    advantage_arr = mpc_arr - base_arr

    summary = {
        "data_source": "German day-ahead (DE-LU), EPEX SPOT, Q1 2024",
        "revenue_streams": "Energy arbitrage + FCR capacity (both real data)",
        "n_days_simulated": N_DAYS,
        "calibration": {
            "cell_reference": "Samsung SDI 94Ah NMC prismatic",
            "R_pack_ohm": elp.R0 + elp.R1 + elp.R2,
            "h_cool_W_per_K": thp.h_cool,
            "alpha_deg": bp.alpha_deg,
            "degradation_cost_per_unit_soh": ep.degradation_cost,
            "sources": [
                "Samsung SDI ESS prismatic cell datasheet (94 Ah)",
                "Schmalstieg et al., J. Power Sources 257 (2014) 325-334",
                "Ecker et al., J. Electrochem. Soc. 162 (2015) A1836",
                "Waag et al., J. Power Sources 2013 (EIS 2RC split)",
            ],
        },
        "price_stats": stats,
        "mpc_optimizer": {
            "profit_mean": float(np.mean(mpc_arr)),
            "profit_median": float(np.median(mpc_arr)),
            "profit_std": float(np.std(mpc_arr)),
            "profit_p5": float(np.percentile(mpc_arr, 5)),
            "profit_p95": float(np.percentile(mpc_arr, 95)),
            "profit_min": float(np.min(mpc_arr)),
            "profit_max": float(np.max(mpc_arr)),
            "profitable_days_pct": float(np.mean(mpc_arr > 0) * 100),
        },
        "rule_based_baseline": {
            "profit_mean": float(np.mean(base_arr)),
            "profit_median": float(np.median(base_arr)),
            "profit_std": float(np.std(base_arr)),
            "profit_p5": float(np.percentile(base_arr, 5)),
            "profit_p95": float(np.percentile(base_arr, 95)),
            "profit_min": float(np.min(base_arr)),
            "profit_max": float(np.max(base_arr)),
            "profitable_days_pct": float(np.mean(base_arr > 0) * 100),
        },
        "mpc_vs_baseline": {
            "advantage_mean": float(np.mean(advantage_arr)),
            "advantage_median": float(np.median(advantage_arr)),
            "mpc_wins_pct": float(np.mean(advantage_arr > 0) * 100),
            "advantage_total_over_n_days": float(np.sum(advantage_arr)),
        },
        "operational": {
            "soh_degradation_mean_pct": float(np.mean(soh_degradations) * 100),
            "soh_degradation_max_pct": float(np.max(soh_degradations) * 100),
            "baseline_soh_degradation_mean_pct": float(np.mean(baseline_soh_degradations) * 100),
            "baseline_soh_degradation_max_pct": float(np.max(baseline_soh_degradations) * 100),
            "max_temp_worst_case_degC": float(np.max(max_temps)),
            "avg_mpc_solve_ms": float(np.mean(mpc_solve_times_avg)),
            "total_solver_failures": int(np.sum(solver_failures_list)),
        },
        "per_day": {
            "day_indices": [int(d) for d in day_labels],
            "mpc_profits": [float(p) for p in mpc_profits],
            "baseline_profits": [float(p) for p in baseline_profits],
            "soh_degradations": [float(d) for d in soh_degradations],
            "baseline_soh_degradations": [float(d) for d in baseline_soh_degradations],
            "max_temps": [float(t) for t in max_temps],
        },
    }

    return summary


# ===========================================================================
#  Plotting
# ===========================================================================

def plot_validation(summary: dict, save_path: str) -> None:
    """Generate validation summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_day = summary["per_day"]
    mpc = np.array(per_day["mpc_profits"])
    base = np.array(per_day["baseline_profits"])
    advantage = mpc - base
    sohs = np.array(per_day["soh_degradations"]) * 100
    temps = np.array(per_day["max_temps"])
    n = len(mpc)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "v4 Real-Price Validation — German Day-Ahead + FCR (DE-LU), Q1 2024\n"
        "Calibrated params (Samsung SDI 94Ah) · Real energy + real FCR capacity prices",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # --- 1. Profit comparison bar chart ---
    ax = axes[0, 0]
    w = 0.35
    ax.bar(x - w/2, mpc, w, label="MPC Optimizer", color="#3b82f6", alpha=0.85)
    ax.bar(x + w/2, base, w, label="Rule-Based Baseline", color="#94a3b8", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Day (sorted by index)")
    ax.set_ylabel("Daily Profit [$]")
    ax.set_title("Daily Profit: MPC vs Rule-Based Baseline")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::max(1, n//10)])

    # --- 2. Advantage histogram ---
    ax = axes[0, 1]
    ax.hist(advantage, bins=15, color="#10b981", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(advantage), color="#ef4444", linewidth=2,
               linestyle="--", label=f"Mean: +${np.mean(advantage):.2f}")
    ax.axvline(0, color="gray", linewidth=1, linestyle=":")
    ax.set_xlabel("MPC Advantage over Baseline [$]")
    ax.set_ylabel("Count")
    ax.set_title(
        f"MPC Advantage Distribution (wins {summary['mpc_vs_baseline']['mpc_wins_pct']:.0f}% of days)"
    )
    ax.legend(fontsize=9)

    # --- 3. Profit distribution (box + swarm) ---
    ax = axes[1, 0]
    bp_data = ax.boxplot(
        [mpc, base],
        labels=["MPC Optimizer", "Rule-Based"],
        patch_artist=True,
        widths=0.5,
    )
    bp_data["boxes"][0].set_facecolor("#3b82f6")
    bp_data["boxes"][0].set_alpha(0.3)
    bp_data["boxes"][1].set_facecolor("#94a3b8")
    bp_data["boxes"][1].set_alpha(0.3)
    # Overlay individual points
    for i, (data, color) in enumerate(zip([mpc, base], ["#3b82f6", "#64748b"])):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   alpha=0.5, s=20, color=color, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Daily Profit [$]")
    ax.set_title("Profit Distribution Across Real Market Days")

    # --- 4. SOH degradation vs max temp ---
    ax = axes[1, 1]
    sc = ax.scatter(temps, sohs, c=mpc, cmap="RdYlGn", s=40, alpha=0.8,
                    edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax, label="MPC Profit [$]")
    ax.set_xlabel("Max Temperature [°C]")
    ax.set_ylabel("SOH Degradation [%/day]")
    ax.set_title("Degradation vs Temperature (color = profit)")

    # --- Summary text box ---
    s = summary
    text = (
        f"MPC: ${s['mpc_optimizer']['profit_mean']:.2f} ± "
        f"${s['mpc_optimizer']['profit_std']:.2f}/day  "
        f"[P5=${s['mpc_optimizer']['profit_p5']:.2f}, "
        f"P95=${s['mpc_optimizer']['profit_p95']:.2f}]\n"
        f"Baseline: ${s['rule_based_baseline']['profit_mean']:.2f} ± "
        f"${s['rule_based_baseline']['profit_std']:.2f}/day\n"
        f"MPC wins {s['mpc_vs_baseline']['mpc_wins_pct']:.0f}% of days  |  "
        f"Avg advantage: +${s['mpc_vs_baseline']['advantage_mean']:.2f}/day  |  "
        f"SOH loss: {s['operational']['soh_degradation_mean_pct']:.4f}%/day  |  "
        f"Solver failures: {s['operational']['total_solver_failures']}"
    )
    fig.text(0.5, 0.01, text, ha="center", fontsize=10,
             style="italic", color="#475569",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                       edgecolor="#cbd5e1", alpha=0.9))

    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {save_path}")


# ===========================================================================
#  Entry point
# ===========================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_validation()

    # ---- Save JSON ----
    json_path = RESULTS_DIR / "v4_real_price_validation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics saved: {json_path}")

    # ---- Print summary ----
    s = summary
    print()
    print("=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Data:  {s['data_source']}")
    print(f"  Days:  {s['n_days_simulated']}")
    print()
    print(f"  {'':30s}  {'MPC Optimizer':>15s}  {'Rule-Based':>15s}")
    print(f"  {'─'*30}  {'─'*15}  {'─'*15}")
    print(f"  {'Mean daily profit':30s}  ${s['mpc_optimizer']['profit_mean']:13.2f}  ${s['rule_based_baseline']['profit_mean']:13.2f}")
    print(f"  {'Median daily profit':30s}  ${s['mpc_optimizer']['profit_median']:13.2f}  ${s['rule_based_baseline']['profit_median']:13.2f}")
    print(f"  {'Std deviation':30s}  ${s['mpc_optimizer']['profit_std']:13.2f}  ${s['rule_based_baseline']['profit_std']:13.2f}")
    print(f"  {'P5 (worst-case)':30s}  ${s['mpc_optimizer']['profit_p5']:13.2f}  ${s['rule_based_baseline']['profit_p5']:13.2f}")
    print(f"  {'P95 (best-case)':30s}  ${s['mpc_optimizer']['profit_p95']:13.2f}  ${s['rule_based_baseline']['profit_p95']:13.2f}")
    print(f"  {'Profitable days':30s}  {s['mpc_optimizer']['profitable_days_pct']:13.0f}%  {s['rule_based_baseline']['profitable_days_pct']:13.0f}%")
    print()
    print(f"  MPC advantage (mean):     +${s['mpc_vs_baseline']['advantage_mean']:.2f}/day")
    print(f"  MPC wins:                 {s['mpc_vs_baseline']['mpc_wins_pct']:.0f}% of days")
    print(f"  Total advantage ({N_DAYS}d):   +${s['mpc_vs_baseline']['advantage_total_over_n_days']:.2f}")
    print()
    print(f"  Avg SOH degradation:      {s['operational']['soh_degradation_mean_pct']:.4f}% / day")
    print(f"  Worst-case max temp:      {s['operational']['max_temp_worst_case_degC']:.1f} °C")
    print(f"  Avg MPC solve time:       {s['operational']['avg_mpc_solve_ms']:.0f} ms")
    print(f"  Total solver failures:    {s['operational']['total_solver_failures']}")
    print("=" * 70)

    # ---- Plot ----
    plot_path = str(RESULTS_DIR / "v4_real_price_validation.png")
    plot_validation(summary, plot_path)


if __name__ == "__main__":
    main()
