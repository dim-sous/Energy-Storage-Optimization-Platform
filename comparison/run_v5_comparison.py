"""84-day strategy comparison for v5_regulation_activation on real German data.

Runs four control strategies through v5's MultiRateSimulator on real
EPEX SPOT day-ahead + SMARD FCR prices (Q1 2024, 84 days):

  1. Rule-Based      — price-sorted schedule, no regulation
  2. EMS + Clamps    — EMS + open-loop hard SOC clamps (industry standard)
  3. EMS + PI        — EMS + advanced PI controller, no MPC
  4. Full Optimizer  — EMS + MPC + PI (complete hierarchy)

All strategies share identical conditions per day: same realized prices,
same forecast scenarios (realized day excluded), same activation signal
(fixed seed per strategy instance).

Saves structured results to results/v5_comparison.json for the
presentation generator.

Usage:
    uv run python comparison/run_v5_comparison.py
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
V5_ROOT = REPO_ROOT / "v5_regulation_activation"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from config.parameters import (  # noqa: E402
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    RegControllerParams,
    RegulationParams,
    Strategy,
    ThermalParams,
    TimeParams,
)
from data.real_price_loader import RealPriceLoader  # noqa: E402
from simulation.simulator import MultiRateSimulator  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Data paths (real prices live in v4's data directory)
ENERGY_CSV = REPO_ROOT / "v4_electrical_rc_model" / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = REPO_ROOT / "v4_electrical_rc_model" / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = REPO_ROOT / "results"

# Strategy order: simplest to most complex
STRATEGY_ORDER = [
    Strategy.RULE_BASED,
    Strategy.EMS_CLAMPS,
    Strategy.EMS_PI,
    Strategy.FULL,
]
STRAT_NAMES = [s.value for s in STRATEGY_ORDER]

STRATEGY_LABELS = {
    "rule_based": "Rule-Based",
    "ems_clamps": "Industry Standard",
    "ems_pi": "Advanced PI",
    "full": "Full Optimizer",
}


# =========================================================================
#  Per-day worker (runs all 4 strategies for one day)
# =========================================================================

def _run_single_day(args: tuple) -> dict:
    """Run all 4 strategies for one day. Designed for Pool.map."""
    (day_idx, energy_scenarios, reg_scenarios, probabilities,
     bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
     reg_ctrl_p, reg_p, pp) = args

    logging.disable(logging.WARNING)

    day_result = {"day_idx": day_idx}

    for strategy in STRATEGY_ORDER:
        t0 = time.perf_counter()

        simulator = MultiRateSimulator(
            bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
            reg_ctrl_p, reg_p, strategy, pp, run_mhe=False,
        )
        results = simulator.run(energy_scenarios, reg_scenarios, probabilities)
        del simulator
        gc.collect()

        mpc_t = results.get("mpc_solve_times", np.array([]))
        day_result[strategy.value] = {
            "total_profit": float(results["total_profit"]),
            "energy_profit": float(results["energy_profit_total"]),
            "capacity_revenue": float(results["capacity_revenue"]),
            "delivery_revenue": float(results["delivery_revenue"]),
            "penalty_cost": float(results["penalty_cost"]),
            "net_regulation_profit": float(results["net_regulation_profit"]),
            "deg_cost": float(results["deg_cost_total"]),
            "delivery_score": float(results["delivery_score"]),
            "soh_degradation": float(results["soh_degradation"]),
            "final_soc": float(results["soc_true"][-1]),
            "final_soh": float(results["soh_true"][-1]),
            "mpc_solver_failures": int(results.get("mpc_solver_failures", 0)),
            "avg_mpc_solve_time_s": float(np.mean(mpc_t)) if len(mpc_t) > 0 else 0.0,
            "max_mpc_solve_time_s": float(np.max(mpc_t)) if len(mpc_t) > 0 else 0.0,
            "max_temp_degC": float(np.max(results["temp_true"])),
            "wall_time": time.perf_counter() - t0,
        }

    return day_result


# =========================================================================
#  Results aggregation and printing
# =========================================================================

def aggregate_results(all_days: list[dict]) -> dict[str, dict]:
    """Aggregate per-day results into arrays keyed by strategy."""
    agg = {s: {
        "profits": [], "energy_profits": [], "capacity_revenues": [],
        "delivery_revenues": [], "penalty_costs": [], "reg_net_profits": [],
        "deg_costs": [], "delivery_scores": [], "soh_degs": [],
        "wall_times": [], "mpc_failures": [],
    } for s in STRAT_NAMES}

    for day in all_days:
        for s in STRAT_NAMES:
            d = day[s]
            agg[s]["profits"].append(d["total_profit"])
            agg[s]["energy_profits"].append(d["energy_profit"])
            agg[s]["capacity_revenues"].append(d["capacity_revenue"])
            agg[s]["delivery_revenues"].append(d["delivery_revenue"])
            agg[s]["penalty_costs"].append(d["penalty_cost"])
            agg[s]["reg_net_profits"].append(d["net_regulation_profit"])
            agg[s]["deg_costs"].append(d["deg_cost"])
            agg[s]["delivery_scores"].append(d["delivery_score"])
            agg[s]["soh_degs"].append(d["soh_degradation"])
            agg[s]["wall_times"].append(d["wall_time"])
            agg[s]["mpc_failures"].append(d["mpc_solver_failures"])

    return agg


def print_results(agg: dict[str, dict], n_days: int) -> None:
    """Print revenue breakdown, profit statistics, and timing."""
    def _row(label: str, values: list, fmt: str = "14.2f") -> None:
        print(f"  {label:22s}", end="")
        for v in values:
            if isinstance(v, int):
                print(f"  {v:14d}", end="")
            elif isinstance(v, str):
                print(f"  {v:>14s}", end="")
            else:
                print(f"  {v:{fmt}}", end="")
        print()

    def _sep() -> None:
        print(f"  {'─' * 22}" + f"  {'─' * 14}" * len(STRAT_NAMES))

    def _header() -> None:
        print(f"  {'':22s}", end="")
        for s in STRAT_NAMES:
            print(f"  {STRATEGY_LABELS[s]:>14s}", end="")
        print()
        _sep()

    # Revenue breakdown (mean $/day)
    print(f"\n  Revenue Breakdown (mean $/day, {n_days} days):")
    _header()
    for label, key, sign in [
        ("Energy revenue", "energy_profits", 1),
        ("Capacity revenue", "capacity_revenues", 1),
        ("Delivery revenue", "delivery_revenues", 1),
        ("Penalty cost", "penalty_costs", -1),
        ("Degradation cost", "deg_costs", -1),
    ]:
        _row(label, [sign * np.mean(agg[s][key]) for s in STRAT_NAMES])
    _sep()
    _row("Net profit", [np.mean(agg[s]["profits"]) for s in STRAT_NAMES])

    # Profit distribution
    print(f"\n  Profit Distribution ($/day):")
    _header()
    for label, fn in [
        ("Mean", np.mean),
        ("Median", np.median),
        ("Std (day-to-day)", np.std),
        ("P5", lambda a: np.percentile(a, 5)),
        ("P95", lambda a: np.percentile(a, 95)),
        ("Worst day", np.min),
        ("Best day", np.max),
    ]:
        _row(label, [fn(agg[s]["profits"]) for s in STRAT_NAMES])
    _row("Loss days", [int(np.sum(np.array(agg[s]["profits"]) < 0)) for s in STRAT_NAMES])

    # Delivery & degradation
    print(f"\n  Delivery & Degradation:")
    _header()
    _row("Avg delivery score",
         [f"{np.mean(agg[s]['delivery_scores'])*100:.1f}%" for s in STRAT_NAMES])
    _row("SOH %/day",
         [np.mean(agg[s]["soh_degs"]) * 100 for s in STRAT_NAMES], "14.5f")
    _row("MPC failures (total)",
         [int(np.sum(agg[s]["mpc_failures"])) for s in STRAT_NAMES])

    # Advantage
    opt = np.array(agg["full"]["profits"])
    rb = np.array(agg["rule_based"]["profits"])
    ind = np.array(agg["ems_clamps"]["profits"])
    print(f"\n  Full Optimizer vs Rule-Based:")
    adv_rb = opt - rb
    print(f"    Advantage:  ${adv_rb.mean():.2f}/day  "
          f"({(adv_rb > 0).mean() * 100:.0f}% win rate)")
    print(f"    Annual (200 kWh): ${adv_rb.mean() * 365:.0f}")
    print(f"    Annual (10 MWh):  ${adv_rb.mean() * 365 * 50:,.0f}")
    print(f"    Annual (50 MWh):  ${adv_rb.mean() * 365 * 250:,.0f}")

    print(f"\n  Full Optimizer vs Industry Standard:")
    adv_ind = opt - ind
    print(f"    Advantage:  ${adv_ind.mean():.2f}/day  "
          f"({(adv_ind > 0).mean() * 100:.0f}% win rate)")
    print(f"    Annual (50 MWh):  ${adv_ind.mean() * 365 * 250:,.0f}")

    # Wall time
    print(f"\n  Wall Time (mean s/day):")
    _header()
    _row("Mean", [np.mean(agg[s]["wall_times"]) for s in STRAT_NAMES])
    _row("Max", [np.max(agg[s]["wall_times"]) for s in STRAT_NAMES])


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run 84-day comparison on real German market data."""
    N_DAYS = 84        # Full Q1 2024
    N_FORECAST = 5     # Forecast scenarios per day (realized excluded)

    # Parameters
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    mhe_p = MHEParams()
    thp = ThermalParams()
    elp = ElectricalParams()
    pp = PackParams()
    reg_ctrl_p = RegControllerParams()
    reg_p = RegulationParams()

    # Load real prices
    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)
    n_days = min(N_DAYS, loader.n_days)
    n_hours_total = int(tp.sim_hours) + ep.N_ems

    stats = loader.price_stats
    print("=" * 78)
    print("  V5 STRATEGY COMPARISON — Real German Market Data")
    print("=" * 78)
    print(f"  Data:       EPEX SPOT DE-LU + SMARD FCR, Q1 2024 ({stats['n_days']} days)")
    print(f"  Battery:    {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Days:       {n_days}")
    print(f"  Forecasts:  {N_FORECAST} other days per run (realized NOT in set)")
    print(f"  Strategies: {', '.join(STRATEGY_LABELS[s.value] for s in STRATEGY_ORDER)}")
    print(f"  Total sims: {n_days * len(STRATEGY_ORDER)}")
    print("=" * 78)
    print()

    # Build jobs: one per day
    jobs = []
    for day_idx in range(n_days):
        energy_scen, reg_scen, probs = loader.generate_scenarios_for_day(
            day_idx, n_hours=n_hours_total, n_scenarios=N_FORECAST,
        )
        jobs.append((
            day_idx, energy_scen, reg_scen, probs,
            bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
            reg_ctrl_p, reg_p, pp,
        ))

    # Run with multiprocessing
    n_workers = min(len(jobs), max(1, multiprocessing.cpu_count() - 1), 2)
    print(f"  Running {len(jobs)} days x {len(STRATEGY_ORDER)} strategies "
          f"across {n_workers} workers...")
    print(f"  Estimated runtime: ~{n_days * 6 / n_workers / 60:.0f} hours "
          f"(FULL strategy dominates at ~5 min/day)\n")
    t0 = time.perf_counter()

    all_days: list[dict] = []
    with multiprocessing.Pool(n_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_run_single_day, jobs, chunksize=1), 1
        ):
            all_days.append(result)
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(jobs) - i)
            day_idx = result["day_idx"]
            full_profit = result["full"]["total_profit"]
            full_score = result["full"]["delivery_score"]
            print(
                f"  [{i:3d}/{len(jobs)}] "
                f"Day {day_idx:2d} done  "
                f"profit=${full_profit:6.2f}  "
                f"delivery={full_score*100:5.1f}%  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                flush=True,
            )

    wall = time.perf_counter() - t0
    print(f"\n\n  Done in {wall:.0f}s ({wall / 60:.1f} min, "
          f"{wall / n_days:.1f}s/day)\n")

    # Sort by day index for consistent output
    all_days.sort(key=lambda d: d["day_idx"])

    # Aggregate
    agg = aggregate_results(all_days)
    print_results(agg, n_days)

    # Build per-day JSON summary
    per_day_json = []
    for day in all_days:
        entry = {"day_idx": day["day_idx"]}
        for s in STRAT_NAMES:
            entry[s] = day[s]
        per_day_json.append(entry)

    # Mean scalars across all days (for presentation)
    mean_scalars = {}
    for s in STRAT_NAMES:
        mean_scalars[s] = {
            "total_profit": float(np.mean(agg[s]["profits"])),
            "energy_profit": float(np.mean(agg[s]["energy_profits"])),
            "capacity_revenue": float(np.mean(agg[s]["capacity_revenues"])),
            "delivery_revenue": float(np.mean(agg[s]["delivery_revenues"])),
            "penalty_cost": float(np.mean(agg[s]["penalty_costs"])),
            "net_regulation_profit": float(np.mean(agg[s]["reg_net_profits"])),
            "deg_cost": float(np.mean(agg[s]["deg_costs"])),
            "delivery_score": float(np.mean(agg[s]["delivery_scores"])),
            "soh_degradation": float(np.mean(agg[s]["soh_degs"])),
            "mpc_solver_failures": int(np.sum(agg[s]["mpc_failures"])),
            "avg_mpc_solve_time_s": float(np.mean(agg[s]["wall_times"])),
            "loss_days": int(np.sum(np.array(agg[s]["profits"]) < 0)),
            "win_rate_vs_rule_based": float(
                np.mean(np.array(agg[s]["profits"]) > np.array(agg["rule_based"]["profits"]))
            ) if s != "rule_based" else None,
        }

    # Per-day profit arrays (for charts)
    daily_profits = {s: [d[s]["total_profit"] for d in all_days] for s in STRAT_NAMES}
    daily_delivery = {s: [d[s]["delivery_score"] for d in all_days] for s in STRAT_NAMES}

    comparison = {
        "meta": {
            "version": "v5_regulation_activation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "EPEX SPOT DE-LU + SMARD FCR, Q1 2024",
            "n_days": n_days,
            "n_forecast_scenarios": N_FORECAST,
            "E_nom_kwh": bp.E_nom_kwh,
            "P_max_kw": bp.P_max_kw,
            "sim_hours": tp.sim_hours,
            "dt_ems_s": tp.dt_ems,
            "dt_mpc_s": tp.dt_mpc,
            "dt_pi_s": tp.dt_pi,
            "n_cells": pp.n_cells,
            "SOC_min": bp.SOC_min,
            "SOC_max": bp.SOC_max,
            "strategy_labels": STRATEGY_LABELS,
            "wall_time_total_s": round(wall, 1),
        },
        "strategies": mean_scalars,
        "daily_profits": daily_profits,
        "daily_delivery": daily_delivery,
        "per_day": per_day_json,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "v5_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=float)

    print(f"\n  Saved: {out_path}")
    print(f"  Size:  {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
