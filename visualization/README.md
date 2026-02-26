# visualization/ — Result Visualization

Generates a 6-panel summary figure (`results.png`) from the simulation output.

---

## Panel Layout (3 x 2 grid)

```
┌────────────────────────────┬────────────────────────────┐
│  [1] SOC Estimation        │  [2] SOH Estimation        │
│  True vs EKF vs MHE        │  True vs EKF vs MHE        │
├────────────────────────────┼────────────────────────────┤
│  [3] Power Dispatch        │  [4] Market Prices         │
│  MPC applied + EMS ref     │  Energy + Regulation       │
├────────────────────────────┼────────────────────────────┤
│  [5] Revenue Breakdown     │  [6] Battery Degradation   │
│  Energy + Reg − Deg = Net  │  SOH loss over time        │
└────────────────────────────┴────────────────────────────┘
```

### Panel 1 — SOC Estimation
- **True SOC** (dark gray, thick): ground truth from the plant at 1 s resolution
- **EKF estimate** (blue): recursive Kalman filter at 60 s resolution
- **MHE estimate** (red, dashed): optimization-based at 60 s resolution
- **SOC limits** (green band): operational window [10%, 90%]

### Panel 2 — SOH Estimation
- Same color scheme as Panel 1
- Shows the slow degradation trend and how well estimators track the hidden SOH state
- EKF and MHE estimates should converge toward the true SOH within ~2 hours

### Panel 3 — Power Dispatch (Combined)
This panel overlays MPC applied power and EMS reference power to visualize tracking quality:
- **Thin solid lines**: MPC actually applied power (60 s resolution)
  - Green = P_chg, Blue = P_dis, Orange = P_reg
- **Thick transparent lines**: EMS reference (hourly first-stage decisions)
  - Same color scheme, with transparency to distinguish from MPC

Good tracking shows the thin MPC lines closely following the thick EMS reference steps.

### Panel 4 — Market Prices
- **Energy price** (blue, solid): $/kWh energy market price
- **Regulation price** (orange, dashed): $/kW/h regulation capacity price
- Shows the first scenario's prices used in profit calculation
- Provides context for understanding dispatch decisions in Panel 3

### Panel 5 — Revenue Breakdown
- **Energy arbitrage** (blue): cumulative revenue from buy-low-sell-high
- **Regulation revenue** (orange): cumulative capacity payments
- **Degradation cost** (red, negative): cumulative battery wear cost
- **Net profit** (black, thick): total revenue minus costs
- Final net profit value shown in legend

### Panel 6 — Battery Degradation
- SOH loss in percent over time
- Shows the degradation rate increasing during high-power periods

---

## Style

- Large fonts (16pt titles, 13pt labels, 11pt ticks, 10pt legends) for readability
- Thick lines (2.0pt true curves, 1.5pt estimates, 2.5pt EMS references)
- High DPI output (180) for crisp rendering
- Figure size: 20 x 16 inches

---

## Usage

```python
from visualization.plot_results import plot_results
from config.parameters import BatteryParams

plot_results(sim_results, BatteryParams(), save_path="results.png")
```

The `sim` dictionary must contain the keys returned by `MultiRateSimulator.run()` (see `simulation/README.md` for the full schema).
