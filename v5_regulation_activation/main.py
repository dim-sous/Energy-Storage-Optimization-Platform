"""v5_regulation_activation — FCR regulation delivery with MPC necessity.

Adds real-time regulation delivery with stochastic activation signals at
4s resolution.  A fast PI controller tracks the grid's activation signal
while MPC manages SOC/thermal trajectory to ensure headroom.

Control hierarchy:
    EMS (3600s) -> MPC (60s) -> PI Controller (4s) -> Plant (4s)
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

VERSION_TAG = "v5_regulation_activation"

# Ensure version folder root is importable (and ONLY this folder)
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
    RegControllerParams,
    RegulationParams,
    Strategy,
    ThermalParams,
    TimeParams,
)
from data.price_generator import PriceGenerator
from simulation.simulator import MultiRateSimulator
from visualization.plot_results import plot_results

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the hierarchical BESS control simulation with regulation."""
    # ---- CLI args ----
    parser = argparse.ArgumentParser(description="v5 regulation activation")
    parser.add_argument(
        "--strategy", type=str, default="full",
        choices=[s.value for s in Strategy],
        help="Control strategy (default: full)",
    )
    parser.add_argument(
        "--mhe", action="store_true",
        help="Enable MHE estimator (default: OFF to save compute)",
    )
    args = parser.parse_args()
    strategy = Strategy(args.strategy)
    run_mhe = args.mhe

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
    reg_ctrl_p = RegControllerParams()
    reg_p = RegulationParams()

    logger.info("=" * 62)
    logger.info("  BESS HIERARCHICAL CONTROL PLATFORM  [%s]", VERSION_TAG)
    logger.info("  Strategy: %s", strategy.value)
    logger.info("=" * 62)
    logger.info("  Battery:    %d kWh / %d kW", bp.E_nom_kwh, bp.P_max_kw)
    logger.info("  SOC range:  [%.2f, %.2f]", bp.SOC_min, bp.SOC_max)
    logger.info("  --- Electrical (2RC) ---")
    logger.info("  R_total:    %.4f Ohm", elp.R_total_dc)
    logger.info("  --- Regulation (v5) ---")
    logger.info("  dt_reg:     %d s", tp.dt_pi)
    logger.info("  SOC safety: [%.2f, %.2f] cutoff [%.2f, %.2f]",
                reg_ctrl_p.soc_safety_low, reg_ctrl_p.soc_safety_high,
                reg_ctrl_p.soc_cutoff_low, reg_ctrl_p.soc_cutoff_high)
    logger.info("  Penalty:    %.1fx capacity price", reg_p.penalty_mult)
    logger.info("  Delivery $: %.3f $/kWh", reg_p.price_activation)
    logger.info("  --- Pack ---")
    logger.info("  n_cells:    %d", pp.n_cells)
    logger.info("  --- Timing ---")
    logger.info("  dt_ems=%ds  dt_mpc=%ds  dt_pi=%ds  dt_sim=%ds",
                tp.dt_ems, tp.dt_mpc, tp.dt_pi, tp.dt_sim)
    logger.info("  Sim hours:  %d h", tp.sim_hours)
    logger.info("=" * 62)

    # ---- Price scenarios ----
    n_hours_total = int(tp.sim_hours) + ep.N_ems
    price_gen = PriceGenerator(seed=42)
    energy_scen, reg_scen, probs = price_gen.generate_scenarios(
        n_hours=n_hours_total,
        n_scenarios=ep.n_scenarios,
    )
    logger.info(
        "Price scenarios generated: %d scenarios x %d hours",
        energy_scen.shape[0], energy_scen.shape[1],
    )

    # ---- Multi-rate simulation ----
    simulator = MultiRateSimulator(
        bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
        reg_ctrl_p, reg_p, strategy, pp, run_mhe=run_mhe,
    )
    results = simulator.run(energy_scen, reg_scen, probs)

    # ---- Visualisation ----
    results_dir = PROJECT_ROOT.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = str(results_dir / f"{VERSION_TAG}_results.png")
    plot_results(results, bp, thp, elp, pp, save_path=plot_path)

    # ---- Save raw results ----
    array_data = {}
    scalar_data = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            array_data[k] = v
        elif isinstance(v, (int, float, str, bool)):
            scalar_data[k] = v

    npz_path = results_dir / f"{VERSION_TAG}_results.npz"
    np.savez(npz_path, **array_data)

    scalars_path = results_dir / f"{VERSION_TAG}_scalars.json"
    with open(scalars_path, "w") as f:
        json.dump(scalar_data, f, indent=2)

    logger.info("Results saved to %s", npz_path)

    # ---- Compute standard metrics for cross-version comparison ----
    metrics = {"version": VERSION_TAG}

    # SOC tracking RMSE (true vs EMS reference, at MPC resolution)
    soc_ref_mpc = results["soc_ref_at_mpc"]
    soc_true_mpc = results["soc_true"][::int(tp.dt_mpc / tp.dt_sim)]
    n_cmp = min(len(soc_ref_mpc), len(soc_true_mpc))
    if n_cmp > 0:
        metrics["rmse_soc_tracking"] = float(np.sqrt(
            np.mean((soc_true_mpc[:n_cmp] - soc_ref_mpc[:n_cmp]) ** 2)))

    # Power tracking RMSE
    pow_ref = results["power_ref_at_mpc"]
    pow_base = results["power_mpc_base"]
    n_p = min(len(pow_ref), len(pow_base))
    if n_p > 0:
        net_ref = pow_ref[:n_p, 1] - pow_ref[:n_p, 0]
        net_app = pow_base[:n_p, 1] - pow_base[:n_p, 0]
        metrics["rmse_power_tracking"] = float(np.sqrt(np.mean((net_app - net_ref) ** 2)))

    # Estimation RMSE (EKF, MHE)
    soc_t_at_mpc = soc_true_mpc[:len(results["soc_ekf"])]
    soh_t_at_mpc = results["soh_true"][::int(tp.dt_mpc / tp.dt_sim)][:len(results["soh_ekf"])]
    temp_t_at_mpc = results["temp_true"][::int(tp.dt_mpc / tp.dt_sim)][:len(results["temp_ekf"])]

    metrics["rmse_soc_ekf"] = float(np.sqrt(np.mean((soc_t_at_mpc - results["soc_ekf"]) ** 2)))
    metrics["rmse_soh_ekf"] = float(np.sqrt(np.mean((soh_t_at_mpc - results["soh_ekf"]) ** 2)))
    metrics["rmse_temp_ekf"] = float(np.sqrt(np.mean((temp_t_at_mpc - results["temp_ekf"]) ** 2)))
    if run_mhe and np.any(results["soc_mhe"] != 0):
        metrics["rmse_soc_mhe"] = float(np.sqrt(np.mean((soc_t_at_mpc - results["soc_mhe"]) ** 2)))
        metrics["rmse_soh_mhe"] = float(np.sqrt(np.mean((soh_t_at_mpc - results["soh_mhe"]) ** 2)))
        metrics["rmse_temp_mhe"] = float(np.sqrt(np.mean((temp_t_at_mpc - results["temp_mhe"]) ** 2)))

    # Economic
    metrics["total_profit_usd"] = results["total_profit"]
    metrics["energy_profit_usd"] = results["energy_profit_total"]
    metrics["capacity_revenue_usd"] = results["capacity_revenue"]
    metrics["delivery_revenue_usd"] = results["delivery_revenue"]
    metrics["penalty_cost_usd"] = results["penalty_cost"]
    metrics["net_regulation_profit_usd"] = results["net_regulation_profit"]
    metrics["total_degradation_cost_usd"] = results["deg_cost_total"]
    metrics["delivery_score"] = results["delivery_score"]

    # Computational
    mpc_t = results["mpc_solve_times"]
    est_t = results["est_solve_times"]
    metrics["avg_mpc_solve_time_s"] = float(np.mean(mpc_t)) if len(mpc_t) > 0 else 0.0
    metrics["max_mpc_solve_time_s"] = float(np.max(mpc_t)) if len(mpc_t) > 0 else 0.0
    metrics["mpc_solver_failures"] = int(results.get("mpc_solver_failures", 0))
    metrics["avg_estimator_solve_time_s"] = float(np.mean(est_t)) if len(est_t) > 0 else 0.0
    metrics["max_estimator_solve_time_s"] = float(np.max(est_t)) if len(est_t) > 0 else 0.0

    # State
    metrics["final_soc"] = float(results["soc_true"][-1])
    metrics["final_soh"] = float(results["soh_true"][-1])
    metrics["soh_degradation_pct"] = float(results["soh_degradation"] * 100)

    # Pack-level
    if "cell_socs" in results:
        metrics["n_cells"] = results["n_cells"]
        soc_imb = results["soc_imbalance"]
        metrics["max_soc_imbalance_pct"] = float(np.max(soc_imb) * 100)
        metrics["avg_soc_imbalance_pct"] = float(np.mean(soc_imb) * 100)
        cell_sohs_end = results["cell_sohs"][:, -1]
        metrics["soh_spread_pct"] = float((np.max(cell_sohs_end) - np.min(cell_sohs_end)) * 100)
        metrics["max_temp_spread_degC"] = float(
            np.max(np.max(results["cell_temps"], axis=0) - np.min(results["cell_temps"], axis=0)))
        bal_pow = results.get("balancing_power")
        if bal_pow is not None:
            metrics["balancing_energy_kwh"] = float(np.sum(np.abs(bal_pow)) * tp.dt_mpc / 3600.0)

    metrics_path = results_dir / f"{VERSION_TAG}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ---- Summary ----
    print()
    print("=" * 62)
    print(f"  RESULTS SUMMARY  [{VERSION_TAG}]  strategy={strategy.value}")
    print("=" * 62)
    print(f"  Battery:          {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Simulation:       {tp.sim_hours:.0f} hours")
    print(f"  --- Profit ---")
    print(f"  Total profit:     ${results['total_profit']:.2f}")
    print(f"  Energy profit:    ${results['energy_profit_total']:.2f}")
    print(f"  Reg net profit:   ${results['net_regulation_profit']:.2f}")
    print(f"    Capacity rev:   ${results['capacity_revenue']:.2f}")
    print(f"    Delivery rev:   ${results['delivery_revenue']:.2f}")
    print(f"    Penalty cost:   ${results['penalty_cost']:.2f}")
    print(f"  Degradation cost: ${results['deg_cost_total']:.2f}")
    print(f"  --- Regulation ---")
    print(f"  Delivery score:   {results['delivery_score']*100:.1f}%")
    print(f"  --- Battery ---")
    print(f"  SOH degradation:  {results['soh_degradation']*100:.4f}%")
    print(f"  Final SOC:        {results['soc_true'][-1]:.4f}")
    print(f"  Final SOH:        {results['soh_true'][-1]:.6f}")
    print(f"  Max  Temp:        {np.max(results['temp_true']):.2f} degC")
    vterm = results.get("vterm_true")
    if vterm is not None:
        print(f"  V_term range:     [{np.min(vterm[1:]):.1f}, {np.max(vterm[1:]):.1f}] V")
    mpc_t = results.get("mpc_solve_times")
    if mpc_t is not None and len(mpc_t) > 0:
        print(f"  Avg MPC solve:    {np.mean(mpc_t)*1000:.1f} ms")
        print(f"  Max MPC solve:    {np.max(mpc_t)*1000:.1f} ms")
        print(f"  MPC failures:     {results.get('mpc_solver_failures', 0)}")
    est_t = results.get("est_solve_times")
    if est_t is not None and len(est_t) > 0:
        print(f"  Avg Est solve:    {np.mean(est_t)*1000:.1f} ms")
    print("=" * 62)


if __name__ == "__main__":
    main()
