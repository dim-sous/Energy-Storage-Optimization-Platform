# v5_regulation_activation — Regulation Delivery with MPC Necessity

Extends the v4 electrical model with **real-time regulation delivery**. The system operator sends stochastic activation signals at 4-second resolution that the BESS must follow while maintaining SOC, thermal, and voltage constraints.

Adds a **feedforward regulation controller** between MPC and the plant, creating a 4-level control hierarchy:

    EMS (3600s) → MPC (60s) → Regulation Controller (4s) → Plant (4s)

## What Changed from v4

| Component | v4 | v5 |
|-----------|----|----|
| **Control hierarchy** | EMS → MPC → Plant | EMS → MPC → RegCtrl → Plant |
| **Simulation dt** | 5 s | 4 s (matches activation resolution) |
| **Activation signals** | Not modeled | Stochastic ±P_reg at 4s (PJM RegD-style) |
| **Regulation controller** | None | Feedforward with SOC safety clamp + recovery bias |
| **Revenue model** | Arbitrage + capacity | Arbitrage + capacity + delivery − penalties |
| **EMS SOC constraints** | Fixed SOC bounds | Physics-based endurance constraint (30 min sustained activation) |
| **MPC** | 3-state (SOC, SOH, T) | 2-state (SOC, T) + SOH frozen as parameter |
| **MPC objective** | SOC + SOH + T tracking + power tracking | SOC tracking + power tracking only |
| **Strategy comparison** | Single strategy | `--strategy full|ems_only|no_regulation` |

**Unchanged**: 5-state plant (SOC, SOH, T, V_rc1, V_rc2), 2RC circuit, OCV polynomial, pack architecture (4 cells, active balancing), EKF/MHE estimation.

## Key Concept: MPC Necessity

This version demonstrates that MPC is **indispensable** for real-time grid service delivery:

- **EMS-only** dispatch cannot react to stochastic activation signals → SOC constraint violations, delivery failures, penalties
- **EMS+MPC** provides closed-loop feedback → smooth delivery, constraint satisfaction, higher net profit

Run the formal comparison with `--strategy ems_only` vs `--strategy full`.

## MPC Simplification

The v5 MPC uses a **2-state prediction model** (SOC, T) with SOH frozen as a parameter:

- **Removed from MPC**: SOH state (changes <0.001/hour), temperature tracking term (soft constraint handles safety), headroom cost (EMS owns this via endurance constraint)
- **Kept**: SOC tracking (closed-loop feedback), power tracking (economic timing from EMS), rate-of-change penalty (smooth control)
- **Result**: 24% faster solves (174→132 ms avg) with equivalent control quality

## Regulation Delivery Model

The system operator sends a normalized activation signal `a(t) ∈ [-1, 1]` at 4-second intervals (PJM RegD-style centrally dispatched signal). The BESS must deliver:

    P_reg(t) = a(t) × P_committed

where `P_committed` is the capacity bid from the EMS (hourly). A feedforward controller applies this directly to the plant power setpoint.

**SOC safety zones** (linear curtailment):
- Below SOC 0.15: linearly reduce up-regulation (discharge) to zero at SOC 0.12
- Above SOC 0.85: linearly reduce down-regulation (charge) to zero at SOC 0.88

**SOC recovery bias**: during low-activation periods (`|activation| < 0.05`), a small power bias nudges SOC toward 0.50 (gain=0.05, ~1 kW at SOC=0.30).

> **Note:** This models a centrally dispatched signal (like PJM RegD or AEMO FCAS), not a European FCR droop response where the battery measures local grid frequency and responds autonomously via a droop curve.

## EMS Endurance Constraint

The EMS enforces a physics-based SOC endurance constraint (per ENTSO-E FCR 30-minute requirement):

    SOC[k+1] >= SOC_min + P_reg[k] × endurance_hours / (E_nom × η_discharge)
    SOC[k+1] <= SOC_max - P_reg[k] × endurance_hours × η_charge / E_nom

At P_reg=30 kW: lower margin=0.079, upper margin=0.071. The constraint is soft (penalty 1e5) to avoid infeasibility. Applied to end-of-hour SOC (`k+1`) so the EMS accounts for planned charge/discharge within each hour.

## Revenue Structure

| Component | Description |
|-----------|-------------|
| Energy arbitrage | Buy low / sell high on day-ahead prices |
| Capacity revenue | Payment for committed regulation capacity ($/kW/h) |
| Delivery revenue | Payment for energy delivered during activation ($/kWh) |
| Penalty cost | Penalty for under-delivery vs committed capacity (3× capacity price) |
| Degradation cost | Battery wear from all power flows |

## Performance (24h simulation)

| Metric | Value |
|--------|-------|
| Total profit | $22.66 |
| Energy profit | $6.22 |
| Net regulation profit | $16.97 |
| Delivery score | 95.3% |
| MPC solver failures | 0 |
| Avg MPC solve | 154 ms |
| SOH degradation | 0.34% |

## Module Structure

```
v5_regulation_activation/
├── main.py                       # Entry point with --strategy and --mhe flags
├── config/
│   └── parameters.py             # All params: Battery, Thermal, Electrical, MPC, EMS, RegCtrl, etc.
├── models/
│   └── battery_model.py          # 5-state CasADi + numpy dynamics, 3-state EMS model
├── ems/
│   └── economic_ems.py           # Stochastic NLP with endurance constraint
├── mpc/
│   └── tracking_mpc.py           # 2-state MPC (SOC, T) with SOH as parameter
├── pi/
│   └── regulation_controller.py  # Feedforward regulation controller with SOC safety + recovery
├── estimation/
│   ├── ekf.py                    # 5-state, 3-measurement EKF
│   └── mhe.py                    # 5-state MHE (optional, --mhe flag)
├── simulation/
│   └── simulator.py              # 4-level multi-rate coordinator
├── data/
│   ├── price_generator.py        # Stochastic energy + regulation price scenarios
│   ├── activation_generator.py   # PJM RegD-style Markov chain activation signal
│   └── real_price_loader.py      # Load real market price data
├── revenue/
│   └── regulation_revenue.py     # Delivery revenue, penalty, and score calculation
├── visualization/
│   └── plot_results.py           # Results plotting with regulation-specific panels
└── stress_test.py                # 20 stress tests (14 inherited from v4 + 6 v5-specific)
```

## Running

```bash
# From repository root
uv run python v5_regulation_activation/main.py                    # full strategy (default)
uv run python v5_regulation_activation/main.py --strategy ems_only
uv run python v5_regulation_activation/main.py --strategy no_regulation

# With MHE estimator (slower)
uv run python v5_regulation_activation/main.py --mhe

# Run stress tests (20 tests)
uv run python v5_regulation_activation/stress_test.py
```

## Status

**In development** — MPC simplified, endurance constraint implemented, 20/20 stress tests pass. Pending gate review. See `backlog.md` for gate process.
