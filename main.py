"""Battery Energy Storage Optimisation Platform — entry point.

Runs the full hierarchical nonlinear control pipeline:
  1. Initialise all parameter dataclasses
  2. Generate stochastic price scenarios
  3. Run multi-rate simulation  (EMS + MPC + EKF + MHE)
  4. Visualise and report results
"""

from __future__ import annotations

import logging
import pathlib
import sys

# Ensure project root is importable regardless of working directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.parameters import (
    BatteryParams,
    EKFParams,
    EMSParams,
    MHEParams,
    MPCParams,
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
    """Orchestrate the hierarchical BESS control simulation."""
    # ---- Configuration ----
    bp = BatteryParams()
    tp = TimeParams()
    ep = EMSParams()
    mp = MPCParams()
    ekf_p = EKFParams()
    mhe_p = MHEParams()

    logger.info("=" * 62)
    logger.info("  BESS HIERARCHICAL CONTROL PLATFORM")
    logger.info("=" * 62)
    logger.info("  Battery:    %d kWh / %d kW", bp.E_nom_kwh, bp.P_max_kw)
    logger.info("  SOC range:  [%.2f, %.2f]", bp.SOC_min, bp.SOC_max)
    logger.info("  SOH init:   %.4f", bp.SOH_init)
    logger.info("  alpha_deg:  %.2e  [1/(kW*s)]", bp.alpha_deg)
    logger.info("  dt_ems:     %d s", tp.dt_ems)
    logger.info("  dt_mpc:     %d s", tp.dt_mpc)
    logger.info("  dt_sim:     %d s", tp.dt_sim)
    logger.info("  Sim hours:  %d h", tp.sim_hours)
    logger.info("  EMS:  N=%d  scenarios=%d", ep.N_ems, ep.n_scenarios)
    logger.info("  MPC:  N=%d  Nc=%d", mp.N_mpc, mp.Nc_mpc)
    logger.info("  MHE:  N=%d", mhe_p.N_mhe)
    logger.info("=" * 62)

    # ---- Price scenarios ----
    # Need sim_hours + N_ems hours of lookahead for the last EMS solve
    n_hours_total = int(tp.sim_hours) + ep.N_ems
    price_gen = PriceGenerator(seed=42)
    energy_scen, reg_scen, probs = price_gen.generate_scenarios(
        n_hours=n_hours_total,
        n_scenarios=ep.n_scenarios,
    )
    logger.info(
        "Price scenarios generated: %d scenarios x %d hours",
        energy_scen.shape[0],
        energy_scen.shape[1],
    )

    # ---- Multi-rate simulation ----
    simulator = MultiRateSimulator(bp, tp, ep, mp, ekf_p, mhe_p)
    results = simulator.run(energy_scen, reg_scen, probs)

    # ---- Visualisation ----
    plot_results(results, bp, save_path="results.png")

    # ---- Summary ----
    print()
    print("=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  Battery:          {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Simulation:       {tp.sim_hours:.0f} hours")
    print(f"  Total profit:     ${results['total_profit']:.2f}")
    print(f"  SOH degradation:  {results['soh_degradation']*100:.4f}%")
    print(f"  Final SOC:        {results['soc_true'][-1]:.4f}")
    print(f"  Final SOH:        {results['soh_true'][-1]:.6f}")
    print(f"  EKF final SOC:    {results['soc_ekf'][-1]:.4f}")
    print(f"  EKF final SOH:    {results['soh_ekf'][-1]:.6f}")
    print(f"  MHE final SOC:    {results['soc_mhe'][-1]:.4f}")
    print(f"  MHE final SOH:    {results['soh_mhe'][-1]:.6f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
