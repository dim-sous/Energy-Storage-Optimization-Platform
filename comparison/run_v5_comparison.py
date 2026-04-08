"""Strategy comparison for v5 on real German market data.

Runs the six configured strategies through the linear simulator core
on real EPEX SPOT day-ahead + SMARD FCR prices (Q1 2024).

Pitch-visible (B2B):
  1. rule_based       — naive price-sorted dispatch, no FCR
  2. deterministic_lp — commercial-baseline rolling-horizon LP
  3. economic_mpc     — v5 product (stochastic EMS + economic MPC + PI)

Internal sanity checks (NOT in pitch deck):
  4. ems_clamps       — stochastic EMS + open-loop dispatch
  5. ems_pi           — stochastic EMS + PI (no MPC)
  6. tracking_mpc     — stochastic EMS + tracking MPC + PI (old v5 stack)

All strategies share identical conditions per day: same realized prices,
same forecast scenarios (realized day held out), same activation seed.

Saves structured results to results/v5_comparison.json.

Usage:
    uv run python comparison/run_v5_comparison.py            # 1 day (quick)
    uv run python comparison/run_v5_comparison.py --full     # 84 days
    uv run python comparison/run_v5_comparison.py -n 10      # custom day count
    uv run python comparison/run_v5_comparison.py --days 0,3,41
"""

from __future__ import annotations

import argparse
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config.parameters import (  # noqa: E402
    BatteryParams,
    EKFParams,
    ElectricalParams,
    EMSParams,
    MHEParams,
    MPCParams,
    PackParams,
    RegControllerParams,
    RegulationParams,
    ThermalParams,
    TimeParams,
)
from core.markets.price_loader import RealPriceLoader  # noqa: E402
from core.simulator.core import run_simulation  # noqa: E402

# Strategy recipes — each module exposes `make_strategy(**params) -> Strategy`
from strategies.deterministic_lp.strategy import make_strategy as _ms_lp  # noqa: E402
from strategies.economic_mpc.strategy import make_strategy as _ms_econ  # noqa: E402
from strategies.ems_clamps.strategy import make_strategy as _ms_clamps  # noqa: E402
from strategies.ems_pi.strategy import make_strategy as _ms_pi  # noqa: E402
from strategies.rule_based.strategy import make_strategy as _ms_rb  # noqa: E402
from strategies.tracking_mpc.strategy import make_strategy as _ms_track  # noqa: E402


def _ms_econ_no_pi(**kw):
    return _ms_econ(pi_enabled=False, **kw)


def _ms_track_no_pi(**kw):
    return _ms_track(pi_enabled=False, **kw)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Data paths (real prices live in v4's data directory)
ENERGY_CSV = REPO_ROOT / "archive" / "v4_electrical_rc_model" / "data" / "real_prices" / "de_day_ahead_2024_q1.csv"
REG_CSV = REPO_ROOT / "archive" / "v4_electrical_rc_model" / "data" / "real_prices" / "de_fcr_regulation_2024_q1.csv"
RESULTS_DIR = REPO_ROOT / "results"

# Strategy registry. The order is also the print/plot order.
# Pitch deck renders only the "pitch_visible" subset.
STRATEGY_FACTORIES = [
    ("rule_based",       _ms_rb),
    ("ems_clamps",       _ms_clamps),
    ("ems_pi",           _ms_pi),
    ("deterministic_lp", _ms_lp),
    ("tracking_mpc",     _ms_track),
    ("economic_mpc",     _ms_econ),
]
STRAT_NAMES = [name for name, _ in STRATEGY_FACTORIES]

STRATEGY_LABELS = {
    "rule_based":       "Rule-Based",
    "ems_clamps":       "Stochastic EMS (sanity)",
    "ems_pi":           "EMS + PI (sanity)",
    "deterministic_lp": "Commercial Baseline",
    "tracking_mpc":     "Tracking MPC (sanity)",
    "economic_mpc":     "Economic MPC (v5)",
}

# Strategies shown in the B2B pitch deck.
PITCH_VISIBLE = {"rule_based", "deterministic_lp", "economic_mpc"}

# Extended strategy registry used by the --big experiment.
# The full ladder + PI on/off counterfactuals for the two MPC strategies.
BIG_STRATEGY_FACTORIES = [
    ("rule_based",          _ms_rb),
    ("deterministic_lp",    _ms_lp),
    ("ems_clamps",          _ms_clamps),
    ("ems_pi",              _ms_pi),
    ("tracking_mpc",        _ms_track),
    ("tracking_mpc_no_pi",  _ms_track_no_pi),
    ("economic_mpc",        _ms_econ),
    ("economic_mpc_no_pi",  _ms_econ_no_pi),
]
BIG_STRATEGY_LABELS = {
    "rule_based":          "Rule-Based",
    "deterministic_lp":    "Det. LP",
    "ems_clamps":          "EMS clamps",
    "ems_pi":              "EMS+PI",
    "tracking_mpc":        "Trk MPC+PI",
    "tracking_mpc_no_pi":  "Trk MPC no-PI",
    "economic_mpc":        "Econ MPC+PI",
    "economic_mpc_no_pi":  "Econ MPC no-PI",
}

# Strategies whose first-day-per-subset traces we persist for visualization.
TRACE_PERSIST_STRATEGIES = {
    "deterministic_lp", "ems_pi", "economic_mpc", "economic_mpc_no_pi",
}


# =========================================================================
#  Per-day worker (runs all 4 strategies for one day)
# =========================================================================

def _run_single_day(args: tuple) -> dict:
    """Run all configured strategies for one day. Designed for Pool.map."""
    (day_idx, forecast_e, forecast_r, probabilities,
     realized_e_prices, realized_r_prices,
     bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
     reg_ctrl_p, reg_p, pp) = args

    logging.disable(logging.WARNING)

    params = dict(
        bp=bp, tp=tp, ep=ep, mp=mp, ekf_p=ekf_p, mhe_p=mhe_p,
        thp=thp, elp=elp, reg_ctrl_p=reg_ctrl_p, reg_p=reg_p, pp=pp,
    )

    day_result = {"day_idx": day_idx}

    for name, factory in STRATEGY_FACTORIES:
        t0 = time.perf_counter()
        strategy = factory(**params)

        results = run_simulation(
            strategy=strategy,
            forecast_e=forecast_e,
            forecast_r=forecast_r,
            probabilities=probabilities,
            realized_e_prices=realized_e_prices,
            realized_r_prices=realized_r_prices,
            **params,
        )
        del strategy
        gc.collect()

        mpc_t = results.get("mpc_solve_times", np.array([]))
        day_result[name] = {
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
#  Big experiment: per-(subset, day) worker
# =========================================================================

def _compute_non_numeric_metrics(results: dict, bp) -> dict:
    """Compute non-numeric metrics from a finished simulation result.

    All quantities are computable from the existing trace columns; this
    function does not modify the simulator. Smoothness, constraint touches,
    and thermal envelope are the things that distinguish strategies in the
    benign-regime case where profits tie.
    """
    # Signed P_net at PI cadence (column 0 of the (N, 2) signed array).
    p_net = results["power_applied_signed"][:, 0]
    soc = results["soc_true"]
    temp = results["temp_true"]
    P_max = bp.P_max_kw

    # Smoothness: total variation (sum of |dP_net|), and dispatch sign flips.
    dp = np.diff(p_net)
    total_variation_kw = float(np.sum(np.abs(dp)))
    sign_changes = int(np.sum(np.diff(np.sign(p_net)) != 0))

    # Constraint touches.
    n_p_max_touches = int(np.sum(np.abs(p_net) > 0.99 * P_max))
    n_soc_low_touches = int(np.sum(soc < bp.SOC_min + 0.01))
    n_soc_high_touches = int(np.sum(soc > bp.SOC_max - 0.01))

    # Thermal envelope.
    max_temp = float(np.max(temp))
    frac_above_35 = float(np.mean(temp > 35.0))

    return {
        "total_variation_kw": total_variation_kw,
        "sign_changes": sign_changes,
        "n_p_max_touches": n_p_max_touches,
        "n_soc_low_touches": n_soc_low_touches,
        "n_soc_high_touches": n_soc_high_touches,
        "max_temp_degC": max_temp,
        "frac_time_above_35C": frac_above_35,
    }


def _persist_raw_traces(results: dict, out_path: pathlib.Path) -> None:
    """Save a slim subset of trace arrays for plotting later."""
    np.savez_compressed(
        out_path,
        time_sim=results["time_sim"],
        soc_true=results["soc_true"],
        soh_true=results["soh_true"],
        temp_true=results["temp_true"],
        power_applied_signed=results["power_applied_signed"],
        activation_signal=results["activation_signal"],
        power_delivered=results["power_delivered"],
        prices_energy=results["prices_energy"],
        prices_reg=results["prices_reg"],
    )


def _run_big_job(args: tuple) -> dict:
    """Run all BIG_STRATEGY_FACTORIES strategies for one (subset, day).

    Args tuple:
      (subset_id, day_idx, persist_first_day,
       forecast_e, forecast_r, probabilities,
       realized_e, realized_r,
       bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
       reg_ctrl_p, reg_p, pp,
       trace_dir)

    `reg_p` is per-subset (carries the sigma_mhz_mult override).
    `persist_first_day` is True only for the first day of each subset; the
    worker saves traces for the strategies in TRACE_PERSIST_STRATEGIES.
    """
    (subset_id, day_idx, persist_first_day,
     forecast_e, forecast_r, probabilities,
     realized_e, realized_r,
     bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
     reg_ctrl_p, reg_p, pp, trace_dir) = args

    logging.disable(logging.WARNING)

    params = dict(
        bp=bp, tp=tp, ep=ep, mp=mp, ekf_p=ekf_p, mhe_p=mhe_p,
        thp=thp, elp=elp, reg_ctrl_p=reg_ctrl_p, reg_p=reg_p, pp=pp,
    )

    day_result = {"subset_id": subset_id, "day_idx": day_idx}

    for name, factory in BIG_STRATEGY_FACTORIES:
        t0 = time.perf_counter()
        strategy = factory(**params)

        results = run_simulation(
            strategy=strategy,
            forecast_e=forecast_e,
            forecast_r=forecast_r,
            probabilities=probabilities,
            realized_e_prices=realized_e,
            realized_r_prices=realized_r,
            **params,
        )

        mpc_t = results.get("mpc_solve_times", np.array([]))
        nm = _compute_non_numeric_metrics(results, bp)

        day_result[name] = {
            # Numeric / financial
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
            "wall_time": time.perf_counter() - t0,
            # Non-numeric
            **nm,
        }

        if persist_first_day and name in TRACE_PERSIST_STRATEGIES:
            out_path = trace_dir / f"{subset_id}_{name}_day{day_idx:02d}.npz"
            _persist_raw_traces(results, out_path)

        del strategy, results
        gc.collect()

    return day_result


# =========================================================================
#  Big experiment: subset selection + driver
# =========================================================================

def _score_days(loader) -> list[dict]:
    """Per-day spread/volatility metrics computed once from real prices."""
    scores = []
    for d in range(loader.n_days):
        e = loader.get_day(d)                 # $/kWh, shape (24,)
        # Reuse the loader's private regulation array
        if loader._has_real_reg:
            r = loader._daily_reg[d]
        else:
            r = np.zeros(24)
        intraday_spread = float(e.max() - e.min())
        reg_volatility = float(r.std())
        scores.append({
            "day_idx": d,
            "intraday_spread_usd_kwh": intraday_spread,
            "reg_volatility_usd_kw_h": reg_volatility,
        })
    return scores


def _pick_day_indices(scores: list[dict], kind: str, n: int) -> list[int]:
    """Pick day indices for a subset.

    kind in {calm, volatile}: rank by intraday spread (low / high).
    """
    if kind == "calm":
        ranked = sorted(scores, key=lambda s: s["intraday_spread_usd_kwh"])
    elif kind == "volatile":
        ranked = sorted(scores, key=lambda s: -s["intraday_spread_usd_kwh"])
    else:
        raise ValueError(f"unknown day-set kind: {kind}")
    return [s["day_idx"] for s in ranked[:n]]


def _aggregate_big(all_jobs: list[dict], strat_names: list[str]) -> dict:
    """Group raw worker outputs by (subset_id, strategy)."""
    agg: dict = {}
    for r in all_jobs:
        sid = r["subset_id"]
        agg.setdefault(sid, {s: [] for s in strat_names})
        for s in strat_names:
            agg[sid][s].append(r[s])
    return agg


def _print_big_summary(agg: dict, subsets: list[dict], strat_names: list[str]) -> None:
    """Per-subset table of headline numbers + non-numeric metrics."""
    labels = BIG_STRATEGY_LABELS

    def _fmt(v, w=12, prec=2):
        if isinstance(v, (int, np.integer)):
            return f"{int(v):>{w}d}"
        if isinstance(v, float):
            return f"{v:>{w}.{prec}f}"
        return f"{str(v):>{w}}"

    print()
    print("=" * 110)
    print("  BIG EXPERIMENT — SUMMARY")
    print("=" * 110)

    for sub in subsets:
        sid = sub["id"]
        rows = agg[sid]
        n_days = len(rows[strat_names[0]])

        print(f"\n  ── Subset: {sid}  ({sub['description']}, {n_days} days) ──")

        # Header
        print(f"    {'metric':28s}", end="")
        for s in strat_names:
            print(f"  {labels[s]:>14s}", end="")
        print()
        print("    " + "─" * 28 + ("  " + "─" * 14) * len(strat_names))

        rowdefs = [
            ("Net profit ($/d)",      "total_profit",       float, 2),
            ("Energy rev ($/d)",      "energy_profit",      float, 2),
            ("Capacity rev ($/d)",    "capacity_revenue",   float, 2),
            ("Delivery rev ($/d)",    "delivery_revenue",   float, 3),
            ("Penalty cost ($/d)",    "penalty_cost",       float, 3),
            ("Deg cost ($/d)",        "deg_cost",           float, 4),
            ("Delivery score (%)",    "delivery_score",     float, 3),
            ("SOH %/day",             "soh_degradation",    float, 5),
            ("TV(P_net) (kW total)",  "total_variation_kw", float, 0),
            ("Sign changes (#/d)",    "sign_changes",       int,   0),
            ("P_max touches (#/d)",   "n_p_max_touches",    int,   0),
            ("SOC-low touches",       "n_soc_low_touches",  int,   0),
            ("SOC-high touches",      "n_soc_high_touches", int,   0),
            ("Max temp (°C)",         "max_temp_degC",      float, 1),
            ("MPC solve (s/step)",    "avg_mpc_solve_time_s", float, 4),
            ("Wall time (s/day)",     "wall_time",          float, 1),
        ]

        for label, key, _t, prec in rowdefs:
            print(f"    {label:28s}", end="")
            for s in strat_names:
                vals = [d[key] for d in rows[s]]
                if key == "delivery_score":
                    v = float(np.mean(vals)) * 100
                elif key == "soh_degradation":
                    v = float(np.mean(vals)) * 100
                else:
                    v = float(np.mean(vals))
                print(f"  {_fmt(v, 14, prec)}", end="")
            print()

    # Cross-subset profit table (calm vs volatile vs stressed)
    print("\n  ── Cross-subset profit ($/day mean) ──")
    print(f"    {'strategy':28s}", end="")
    for sub in subsets:
        print(f"  {sub['id']:>14s}", end="")
    print()
    print("    " + "─" * 28 + ("  " + "─" * 14) * len(subsets))
    for s in strat_names:
        print(f"    {labels[s]:28s}", end="")
        for sub in subsets:
            vals = [d["total_profit"] for d in agg[sub["id"]][s]]
            print(f"  {float(np.mean(vals)):14.2f}", end="")
        print()

    # PI on/off views (the user-requested counterfactual).
    print("\n  ── PI on/off counterfactuals (mean profit, $/day) ──")
    print(f"    {'comparison':40s}", end="")
    for sub in subsets:
        print(f"  {sub['id']:>14s}", end="")
    print()
    print("    " + "─" * 40 + ("  " + "─" * 14) * len(subsets))
    for label, on_key, off_key in [
        ("economic_mpc:  PI on  − PI off", "economic_mpc", "economic_mpc_no_pi"),
        ("tracking_mpc:  PI on  − PI off", "tracking_mpc", "tracking_mpc_no_pi"),
    ]:
        print(f"    {label:40s}", end="")
        for sub in subsets:
            on = float(np.mean([d["total_profit"] for d in agg[sub["id"]][on_key]]))
            off = float(np.mean([d["total_profit"] for d in agg[sub["id"]][off_key]]))
            print(f"  {(on - off):+14.3f}", end="")
        print()


def _run_big_experiment(n_days_per_subset: int) -> None:
    """Run the 3-subset x 8-strategy big experiment.

    Subsets:
      - calm:     N lowest intraday-spread days, sigma_mhz_mult = 1.0
      - volatile: N highest intraday-spread days, sigma_mhz_mult = 1.0
      - stressed: same N calm days, sigma_mhz_mult = 2.0  (activation stress)
    """
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
    reg_p_nominal = RegulationParams()                                # sigma_mhz_mult=1.0
    reg_p_stressed = RegulationParams(sigma_mhz_mult=2.0)              # 2x activation stress

    loader = RealPriceLoader(ENERGY_CSV, reg_csv=REG_CSV, seed=42)
    n_hours_total = int(tp.sim_hours) + ep.N_ems

    # ----- Pick subsets -----
    scores = _score_days(loader)
    calm_idx = _pick_day_indices(scores, "calm", n_days_per_subset)
    volatile_idx = _pick_day_indices(scores, "volatile", n_days_per_subset)

    subsets = [
        {
            "id": "calm",
            "description": "5 lowest-spread days, nominal activation",
            "day_indices": calm_idx,
            "reg_p": reg_p_nominal,
        },
        {
            "id": "volatile",
            "description": "5 highest-spread days, nominal activation",
            "day_indices": volatile_idx,
            "reg_p": reg_p_nominal,
        },
        {
            "id": "stressed",
            "description": "calm days with 2x OU sigma activation",
            "day_indices": calm_idx,
            "reg_p": reg_p_stressed,
        },
    ]

    # Trace dir for raw npz files
    trace_dir = RESULTS_DIR / "v5_big_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build flat job list -----
    jobs = []
    for sub in subsets:
        for i, day_idx in enumerate(sub["day_indices"]):
            forecast_e, forecast_r, probs, realized_e, realized_r = (
                loader.generate_scenarios_for_day(
                    day_idx, n_hours=n_hours_total, n_scenarios=5,
                )
            )
            jobs.append((
                sub["id"], day_idx, (i == 0),
                forecast_e, forecast_r, probs,
                realized_e, realized_r,
                bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
                reg_ctrl_p, sub["reg_p"], pp, trace_dir,
            ))

    strat_names = [n for n, _ in BIG_STRATEGY_FACTORIES]
    n_workers = min(len(jobs), max(1, multiprocessing.cpu_count() - 1), 2)

    print("=" * 110)
    print("  V5 BIG EXPERIMENT — Real German Market Data, 3 subsets x 8 strategies")
    print("=" * 110)
    print(f"  Battery:    {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Subsets:    {[s['id'] for s in subsets]}")
    print(f"  Days/subset:{n_days_per_subset}")
    print(f"  Strategies: {len(strat_names)}")
    print(f"  Total sims: {len(jobs) * len(strat_names)}")
    print(f"  Workers:    {n_workers}")
    print(f"  Calm days:      {calm_idx}")
    print(f"  Volatile days:  {volatile_idx}")
    print("=" * 110)
    print()

    t0 = time.perf_counter()
    all_results: list[dict] = []
    with multiprocessing.Pool(n_workers) as pool:
        for i, r in enumerate(
            pool.imap_unordered(_run_big_job, jobs, chunksize=1), 1
        ):
            all_results.append(r)
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(jobs) - i)
            sid = r["subset_id"]
            day = r["day_idx"]
            head = r.get("economic_mpc", {}).get("total_profit", float("nan"))
            print(
                f"  [{i:3d}/{len(jobs)}]  {sid:10s} day {day:2d}  "
                f"econ_mpc=${head:6.2f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                flush=True,
            )

    wall = time.perf_counter() - t0
    print(f"\n  Done in {wall:.0f}s ({wall/60:.1f} min)\n")

    # Sort within each subset by day_idx
    all_results.sort(key=lambda r: (r["subset_id"], r["day_idx"]))

    agg = _aggregate_big(all_results, strat_names)
    _print_big_summary(agg, subsets, strat_names)

    # Build the JSON payload — both per-day and per-subset means.
    out_payload = {
        "meta": {
            "version": "v5_big_experiment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "EPEX SPOT DE-LU + SMARD FCR, Q1 2024",
            "n_days_per_subset": n_days_per_subset,
            "E_nom_kwh": bp.E_nom_kwh,
            "P_max_kw": bp.P_max_kw,
            "sim_hours": tp.sim_hours,
            "subsets": [
                {
                    "id": s["id"],
                    "description": s["description"],
                    "day_indices": s["day_indices"],
                    "sigma_mhz_mult": s["reg_p"].sigma_mhz_mult,
                }
                for s in subsets
            ],
            "strategy_labels": BIG_STRATEGY_LABELS,
            "trace_dir": str(trace_dir.relative_to(REPO_ROOT)),
            "wall_time_total_s": round(wall, 1),
        },
        "per_subset_means": {
            sid: {
                s: {
                    k: float(np.mean([d[k] for d in rows[s]]))
                    for k in rows[s][0].keys()
                    if isinstance(rows[s][0][k], (int, float, np.integer, np.floating))
                }
                for s in strat_names
            }
            for sid, rows in agg.items()
        },
        "per_day": [
            {
                "subset_id": r["subset_id"],
                "day_idx": r["day_idx"],
                **{s: r[s] for s in strat_names},
            }
            for r in all_results
        ],
    }

    out_path = RESULTS_DIR / "v5_big_experiment.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Traces: {trace_dir} ({len(list(trace_dir.glob('*.npz')))} files)")


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

    # ---- Pitch comparison: economic_mpc vs rule_based and deterministic_lp ----
    if "economic_mpc" in agg and "deterministic_lp" in agg:
        econ = np.array(agg["economic_mpc"]["profits"])
        lp = np.array(agg["deterministic_lp"]["profits"])

        if "rule_based" in agg:
            rb = np.array(agg["rule_based"]["profits"])
            print(f"\n  [PITCH] Economic MPC vs Rule-Based:")
            adv = econ - rb
            print(f"    Advantage:  ${adv.mean():.2f}/day  "
                  f"({(adv > 0).mean() * 100:.0f}% win rate)")
            print(f"    Annual (200 kWh): ${adv.mean() * 365:.0f}")
            print(f"    Annual (50 MWh):  ${adv.mean() * 365 * 250:,.0f}")

        print(f"\n  [PITCH] Economic MPC vs Commercial Baseline (LP):")
        adv = econ - lp
        pct = adv.mean() / max(abs(lp.mean()), 1e-6) * 100
        print(f"    Advantage:  ${adv.mean():.2f}/day "
              f"({pct:+.1f}% vs LP, {(adv > 0).mean() * 100:.0f}% win rate)")
        print(f"    Annual (50 MWh):  ${adv.mean() * 365 * 250:,.0f}")

    # ---- Sanity comparison: economic vs tracking MPC ----
    if "tracking_mpc" in agg and "economic_mpc" in agg:
        econ = np.array(agg["economic_mpc"]["profits"])
        trk = np.array(agg["tracking_mpc"]["profits"])
        print(f"\n  [SANITY] Economic MPC vs Tracking MPC:")
        adv = econ - trk
        print(f"    Advantage:  ${adv.mean():.2f}/day  "
              f"({(adv > 0).mean() * 100:.0f}% win rate)")

    # Wall time
    print(f"\n  Wall Time (mean s/day):")
    _header()
    _row("Mean", [np.mean(agg[s]["wall_times"]) for s in STRAT_NAMES])
    _row("Max", [np.max(agg[s]["wall_times"]) for s in STRAT_NAMES])


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run strategy comparison on real German market data."""
    global STRATEGY_FACTORIES, STRAT_NAMES
    parser = argparse.ArgumentParser(description="v5 strategy comparison")
    parser.add_argument("--full", action="store_true", help="Run all 84 days")
    parser.add_argument("-n", type=int, default=None, help="Number of contiguous days from day 0")
    parser.add_argument(
        "--days", type=str, default=None,
        help="Comma-separated specific day indices (e.g. '3,41,86'). Overrides -n/--full.",
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated strategy names to run (default: all). "
             "Valid: " + ",".join(STRAT_NAMES),
    )
    parser.add_argument(
        "--big", action="store_true",
        help="Run the big experiment: 3 subsets (calm/volatile/stressed) "
             "x 8 strategies (full ladder + PI on/off MPC variants).",
    )
    parser.add_argument(
        "--big-n", type=int, default=5,
        help="Days per subset for --big (default 5).",
    )
    args = parser.parse_args()

    if args.big:
        _run_big_experiment(args.big_n)
        return

    if args.strategies is not None:
        wanted = [s.strip() for s in args.strategies.split(",")]
        unknown = [s for s in wanted if s not in STRAT_NAMES]
        if unknown:
            parser.error(f"unknown strategies: {unknown}. Valid: {STRAT_NAMES}")
        STRATEGY_FACTORIES = [(n, f) for (n, f) in STRATEGY_FACTORIES if n in wanted]
        STRAT_NAMES = [n for n in STRAT_NAMES if n in wanted]

    if args.days is not None:
        DAY_INDICES = [int(d) for d in args.days.split(",")]
        N_DAYS = len(DAY_INDICES)
    else:
        DAY_INDICES = None
        if args.n is not None:
            N_DAYS = args.n
        elif args.full:
            N_DAYS = 84
        else:
            N_DAYS = 1

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
    if DAY_INDICES is None:
        DAY_INDICES = list(range(min(N_DAYS, loader.n_days)))
    n_days = len(DAY_INDICES)
    n_hours_total = int(tp.sim_hours) + ep.N_ems

    stats = loader.price_stats
    print("=" * 78)
    print("  V5 STRATEGY COMPARISON — Real German Market Data")
    print("=" * 78)
    print(f"  Data:       EPEX SPOT DE-LU + SMARD FCR, Q1 2024 ({stats['n_days']} days)")
    print(f"  Battery:    {bp.E_nom_kwh:.0f} kWh / {bp.P_max_kw:.0f} kW")
    print(f"  Days:       {n_days}")
    print(f"  Forecasts:  {N_FORECAST} other days per run "
          f"(realized day held out — fixed v5 info-leak)")
    print(f"  Strategies: {', '.join(STRATEGY_LABELS[name] for name, _ in STRATEGY_FACTORIES)}")
    print(f"  Total sims: {n_days * len(STRATEGY_FACTORIES)}")
    print("=" * 78)
    print()

    # Build jobs: one per requested day
    jobs = []
    for day_idx in DAY_INDICES:
        forecast_e, forecast_r, probs, realized_e, realized_r = (
            loader.generate_scenarios_for_day(
                day_idx, n_hours=n_hours_total, n_scenarios=N_FORECAST,
            )
        )
        jobs.append((
            day_idx, forecast_e, forecast_r, probs,
            realized_e, realized_r,
            bp, tp, ep, mp, ekf_p, mhe_p, thp, elp,
            reg_ctrl_p, reg_p, pp,
        ))

    # Run with multiprocessing
    n_workers = min(len(jobs), max(1, multiprocessing.cpu_count() - 1), 2)
    print(f"  Running {len(jobs)} days x {len(STRATEGY_FACTORIES)} strategies "
          f"across {n_workers} workers...")
    print(f"  Estimated runtime: ~{n_days * 6 / n_workers / 60:.0f} hours "
          f"(MPC strategies dominate at ~3 min/day with JIT)\n")
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
            headline_key = "economic_mpc" if "economic_mpc" in result else "tracking_mpc"
            head_profit = result[headline_key]["total_profit"]
            head_score = result[headline_key]["delivery_score"]
            print(
                f"  [{i:3d}/{len(jobs)}] "
                f"Day {day_idx:2d} done  "
                f"{headline_key}: profit=${head_profit:6.2f}  "
                f"delivery={head_score*100:5.1f}%  "
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
            "version": "v5_core_refactor",
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
            "pitch_visible": sorted(PITCH_VISIBLE),
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
