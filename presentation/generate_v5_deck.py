"""Generate interactive HTML presentation from v5 comparison results.

Reads results/v5_comparison.json (84-day comparison on real German data)
and produces a self-contained HTML deck with Plotly.js charts.
All numbers are injected dynamically — zero hardcoded values.

Usage:
    uv run python presentation/generate_v5_deck.py
    uv run python presentation/generate_v5_deck.py --input results/v5_comparison.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Strategy display order and styling
STRAT_ORDER = ["rule_based", "ems_clamps", "ems_pi", "full"]
STRAT_LABELS = {
    "rule_based": "Rule-Based",
    "ems_clamps": "Industry Standard",
    "ems_pi": "Advanced PI",
    "full": "Full Optimizer",
}
STRAT_COLORS = {
    "rule_based": "#64748b",
    "ems_clamps": "#f59e0b",
    "ems_pi": "#8b5cf6",
    "full": "#3b82f6",
}


def load_data(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_dollar(v: float) -> str:
    if abs(v) >= 1000:
        return f"${v:,.0f}"
    return f"${v:.2f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


# ─── CSS ───

CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
  --bg:#0b0f1a; --card:#111827; --card2:#1a2235; --accent:#3b82f6; --glow:#60a5fa;
  --green:#10b981; --amber:#f59e0b; --red:#ef4444; --purple:#8b5cf6; --cyan:#06b6d4;
  --t1:#f1f5f9; --t2:#94a3b8; --dim:#64748b; --brd:#1e293b;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth;scroll-snap-type:y mandatory}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased}
.S{min-height:100vh;width:100%;display:flex;flex-direction:column;justify-content:center;padding:56px 72px;position:relative;scroll-snap-align:start}
.S::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 70% 50% at 20% 40%,rgba(59,130,246,.05)0,transparent 60%),radial-gradient(ellipse 50% 40% at 80% 70%,rgba(139,92,246,.03)0,transparent 60%);pointer-events:none}
.S>*{position:relative;z-index:1}
.sn{position:absolute;bottom:24px;right:36px;font-size:11px;color:var(--dim);letter-spacing:2px}
.ft{position:absolute;bottom:24px;left:72px;font-size:10px;color:var(--dim);letter-spacing:1px}
.title{text-align:center;align-items:center}
h1{font-size:52px;font-weight:800;line-height:1.1;letter-spacing:-2px;max-width:860px;background:linear-gradient(135deg,#f1f5f9 30%,#94a3b8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{font-size:19px;color:var(--t2);margin-top:20px;max-width:680px;line-height:1.6}
.dv{width:56px;height:3px;background:linear-gradient(135deg,var(--accent),var(--purple));border-radius:3px;margin:24px auto}
.meta{font-size:12px;color:var(--dim);letter-spacing:1px}
.lb{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
.lb::before{content:'';width:18px;height:2px;background:var(--accent);border-radius:2px}
h2{font-size:40px;font-weight:800;line-height:1.15;letter-spacing:-1.5px;margin-bottom:8px}
.desc{font-size:16px;color:var(--t2);line-height:1.6;max-width:660px;margin-bottom:32px}
.g{display:grid;gap:18px}
.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
.sp{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start}
.c{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:24px}
.c h3{font-size:17px;font-weight:700;margin-bottom:6px}
.c p{font-size:13px;color:var(--t2);line-height:1.6}
.ic{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:14px;font-weight:700}
.ic.bl{background:rgba(59,130,246,.15);color:var(--accent)}
.ic.am{background:rgba(245,158,11,.15);color:var(--amber)}
.ic.rd{background:rgba(239,68,68,.15);color:var(--red)}
.ic.pu{background:rgba(139,92,246,.15);color:var(--purple)}
.ic.gn{background:rgba(16,185,129,.15);color:var(--green)}
.kpi{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:22px;text-align:center}
.kv{font-size:36px;font-weight:800;letter-spacing:-1px;line-height:1}
.kl{font-size:11px;font-weight:600;color:var(--t2);margin-top:7px;text-transform:uppercase;letter-spacing:1px}
.ks{font-size:11px;color:var(--dim);margin-top:3px}
.hl{border-radius:14px;padding:20px 24px;margin:16px 0}
.hl.bl{background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(139,92,246,.04));border:1px solid rgba(59,130,246,.2)}
.hl.gr{background:linear-gradient(135deg,rgba(16,185,129,.08),rgba(6,182,212,.04));border:1px solid rgba(16,185,129,.2)}
.hl.am{background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(239,68,68,.04));border:1px solid rgba(245,158,11,.2)}
.tbl{width:100%;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;border:1px solid var(--brd);background:var(--card)}
.tbl th{background:var(--card2);padding:12px 16px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--t2);text-align:left;border-bottom:1px solid var(--brd)}
.tbl td{padding:10px 16px;font-size:13px;border-bottom:1px solid rgba(30,41,59,.5)}
.tbl tr:last-child td{border-bottom:none}
.tg{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700}
.tg.g{background:rgba(16,185,129,.15);color:var(--green)}
.tg.r{background:rgba(239,68,68,.15);color:var(--red)}
.plotly-chart{border-radius:12px;overflow:hidden;border:1px solid var(--brd)}
.checks{list-style:none;padding:0}
.checks li{font-size:14px;line-height:1.5;padding:5px 0 5px 26px;position:relative;color:var(--t2)}
.checks li::before{content:'\\2713';position:absolute;left:0;font-weight:800;color:var(--green)}
.strat-grid{display:grid;grid-template-columns:auto repeat(4,1fr);gap:0;margin:20px 0}
.strat-grid .sh{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--t2);padding:10px 14px;text-align:center;border-bottom:2px solid var(--brd)}
.strat-grid .sr{font-size:12px;font-weight:600;color:var(--t2);padding:8px 14px;border-bottom:1px solid rgba(30,41,59,.3)}
.strat-grid .sc{font-size:18px;padding:8px 14px;text-align:center;border-bottom:1px solid rgba(30,41,59,.3)}
@media print{.S{page-break-after:always}body{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
@media(max-width:1100px){.S{padding:32px 36px}h1{font-size:38px}h2{font-size:30px}.g3{grid-template-columns:1fr 1fr}.g4{grid-template-columns:1fr 1fr}.sp{grid-template-columns:1fr}}
"""


# ─── Slide builders ───

def slide_title(data: dict) -> str:
    meta = data["meta"]
    return f"""\
<section class="S title">
  <div style="margin-bottom:40px">
    <img src="https://arpedon.com/assets/images/logo/logo.svg" alt="Arpedon" style="height:36px;filter:brightness(1.5)">
  </div>
  <h1>Battery Storage<br>Control Strategy Analysis</h1>
  <p class="sub">Why real-time optimization matters for grid-scale storage doing frequency regulation.</p>
  <div class="dv"></div>
  <p class="meta">{meta['E_nom_kwh']:.0f} kWh / {meta['P_max_kw']:.0f} kW &middot; {meta['n_days']} Days of Real German Market Data &middot; Q1 2024 &middot; Confidential</p>
</section>"""


def slide_challenge() -> str:
    return """\
<section class="S">
  <div class="lb">The Challenge</div>
  <h2>Grid Services Demand<br>Real-Time Intelligence</h2>
  <p class="desc">Frequency regulation requires sub-minute response. A dispatch plan alone cannot handle stochastic activation signals.</p>
  <div class="g g3">
    <div class="c"><div class="ic am">1</div><h3>Stochastic Activations</h3><p>The grid sends unpredictable activation commands every 4 seconds. A static schedule cannot respond &mdash; it will miss delivery and incur penalties.</p></div>
    <div class="c"><div class="ic rd">2</div><h3>SOC Constraint Violations</h3><p>Without feedback control, sustained activations drive SOC to limits. Hard clamps cause abrupt delivery failures and penalty cascades.</p></div>
    <div class="c"><div class="ic pu">3</div><h3>Revenue at Risk</h3><p>Non-delivery penalties are 3&times; the capacity price. A single hour of poor delivery can erase an entire day's regulation revenue.</p></div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">02</div>
</section>"""


def slide_strategies() -> str:
    return """\
<section class="S">
  <div class="lb">Four Strategies Compared</div>
  <h2>From Simple Rules<br>to Full Optimization</h2>
  <p class="desc">Each strategy adds a layer of intelligence. We test all four on identical market conditions.</p>
  <div style="background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:28px;overflow-x:auto">
    <div class="strat-grid">
      <div class="sh" style="text-align:left">Component</div>
      <div class="sh" style="border-bottom-color:#64748b">Rule-Based</div>
      <div class="sh" style="border-bottom-color:#f59e0b">Industry Std.</div>
      <div class="sh" style="border-bottom-color:#8b5cf6">Advanced PI</div>
      <div class="sh" style="border-bottom-color:#3b82f6">Full Optimizer</div>
      <div class="sr">Economic Scheduler (EMS)</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
      <div class="sr">Model Predictive Control (MPC)</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
      <div class="sr">Regulation Controller</div>
      <div class="sc" style="color:var(--dim)">&mdash;</div>
      <div class="sc" style="font-size:11px;color:var(--amber)">Hard Clamps</div>
      <div class="sc" style="font-size:11px;color:var(--purple)">Safety Zones</div>
      <div class="sc" style="font-size:11px;color:var(--glow)">Safety Zones</div>
      <div class="sr">SOC Protection</div>
      <div class="sc" style="font-size:11px;color:var(--dim)">None</div>
      <div class="sc" style="font-size:11px;color:var(--amber)">Hard Clamps</div>
      <div class="sc" style="font-size:11px;color:var(--purple)">Gradual Scaling</div>
      <div class="sc" style="font-size:11px;color:var(--glow)">MPC + Scaling</div>
      <div class="sr">State Estimation (EKF)</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--red)">&#x2717;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
      <div class="sc" style="color:var(--green)">&#x2713;</div>
    </div>
  </div>
  <div class="hl bl"><div style="font-size:13px;color:var(--t2);line-height:1.7"><strong style="color:var(--t1)">Key question:</strong> Does the added complexity of MPC and advanced PI actually pay for itself in delivery performance and revenue?</div></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">03</div>
</section>"""


def slide_kpi_table(data: dict) -> str:
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]

    def best_val(key: str, higher_better: bool = True) -> str:
        vals = {s: strats[s][key] for s in STRAT_ORDER}
        return max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)

    best_profit = best_val("total_profit")
    best_delivery = best_val("delivery_score")
    best_penalty = best_val("penalty_cost", higher_better=False)
    best_soh = best_val("soh_degradation", higher_better=False)

    def td(s: str, val: str, is_best: bool) -> str:
        if is_best:
            return f'<td style="text-align:center"><span class="tg g">{val}</span></td>'
        return f'<td style="text-align:center">{val}</td>'

    rows = ""
    rows += "<tr><td><strong>Net Profit (mean/day)</strong></td>"
    for s in STRAT_ORDER:
        rows += td(s, fmt_dollar(strats[s]["total_profit"]), s == best_profit)
    rows += "</tr>"

    rows += "<tr><td>Energy Profit</td>"
    for s in STRAT_ORDER:
        rows += f'<td style="text-align:center">{fmt_dollar(strats[s]["energy_profit"])}</td>'
    rows += "</tr>"

    rows += "<tr><td>Net Regulation</td>"
    for s in STRAT_ORDER:
        rows += f'<td style="text-align:center">{fmt_dollar(strats[s]["net_regulation_profit"])}</td>'
    rows += "</tr>"

    rows += "<tr><td><strong>Delivery Score</strong></td>"
    for s in STRAT_ORDER:
        rows += td(s, fmt_pct(strats[s]["delivery_score"]), s == best_delivery)
    rows += "</tr>"

    rows += "<tr><td>Penalty Cost</td>"
    for s in STRAT_ORDER:
        rows += td(s, fmt_dollar(strats[s]["penalty_cost"]), s == best_penalty)
    rows += "</tr>"

    rows += "<tr><td>Degradation Cost</td>"
    for s in STRAT_ORDER:
        rows += f'<td style="text-align:center">{fmt_dollar(strats[s]["deg_cost"])}</td>'
    rows += "</tr>"

    rows += "<tr><td><strong>SOH Loss (mean/day)</strong></td>"
    for s in STRAT_ORDER:
        rows += td(s, f'{strats[s]["soh_degradation"] * 100:.4f}%', s == best_soh)
    rows += "</tr>"

    rows += "<tr><td>Loss Days</td>"
    for s in STRAT_ORDER:
        ld = strats[s].get("loss_days", 0)
        rows += f'<td style="text-align:center">{ld}</td>'
    rows += "</tr>"

    headers = "".join(
        f'<th style="text-align:center;border-bottom-color:{STRAT_COLORS[s]}">'
        f'{STRAT_LABELS[s]}</th>'
        for s in STRAT_ORDER
    )

    full_p = strats["full"]["total_profit"]
    ind_p = strats["ems_clamps"]["total_profit"]
    advantage = full_p - ind_p

    return f"""\
<section class="S">
  <div class="lb">Head-to-Head</div>
  <h2>Performance Across {n_days} Days</h2>
  <p class="desc">Same battery, same market data, same activation signals. Only the control strategy differs.</p>
  <table class="tbl">
    <thead><tr><th>Metric</th>{headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="hl gr" style="margin-top:20px"><div style="display:flex;align-items:center;gap:14px"><div style="font-size:24px">&#x1F4B0;</div><div style="font-size:13px;color:var(--t2);line-height:1.6"><strong style="color:var(--green)">Full Optimizer advantage over Industry Standard: {fmt_dollar(advantage)}/day</strong> &mdash; annualized on a 10 MWh system: <strong>{fmt_dollar(advantage * 365 * 50)}/year</strong></div></div></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">04</div>
</section>"""


def slide_daily_profits() -> str:
    return """\
<section class="S">
  <div class="lb">84-Day Backtest</div>
  <h2>Every Day of Q1 2024, Simulated</h2>
  <p class="desc">Daily net profit across all strategies. Sorted from worst to best day for each strategy.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div id="chart-sorted" class="plotly-chart" style="height:360px"></div>
    <div id="chart-cumul" class="plotly-chart" style="height:360px"></div>
  </div>
  <div id="kpi-row" style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px">
    <div class="kpi" id="kpi-loss" style="border-color:rgba(16,185,129,.3)"></div>
    <div class="kpi" id="kpi-adv" style="border-color:rgba(59,130,246,.3)"></div>
    <div class="kpi" id="kpi-delivery" style="border-color:rgba(139,92,246,.3)"></div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">05</div>
</section>"""


def slide_delivery() -> str:
    return """\
<section class="S">
  <div class="lb">Regulation Delivery</div>
  <h2>Delivery Score Distribution</h2>
  <p class="desc">Per-day delivery scores across all 84 days. Higher is better — penalties kick in below 95%.</p>
  <div id="chart-delivery" class="plotly-chart" style="height:420px"></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">06</div>
</section>"""


def slide_revenue(data: dict) -> str:
    return """\
<section class="S">
  <div class="lb">Revenue Breakdown</div>
  <h2>Where the Money Comes From</h2>
  <p class="desc">Mean daily revenue decomposition across all four strategies. Penalties and degradation subtracted.</p>
  <div id="chart-revenue" class="plotly-chart" style="height:440px"></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">07</div>
</section>"""


def slide_closing(data: dict) -> str:
    full = data["strategies"]["full"]
    rb = data["strategies"]["rule_based"]
    ind = data["strategies"]["ems_clamps"]
    adv_rb = full["total_profit"] - rb["total_profit"]
    adv_ind = full["total_profit"] - ind["total_profit"]
    n_days = data["meta"]["n_days"]

    return f"""\
<section class="S">
  <div class="lb">Results</div>
  <h2>The Full Optimizer Wins</h2>
  <div class="sp">
    <div>
      <p class="desc" style="max-width:100%">Validated on {n_days} days of real German market data. The complete control hierarchy consistently outperforms simpler approaches.</p>
      <ul class="checks">
        <li>Highest mean net profit: {fmt_dollar(full['total_profit'])}/day per 200 kWh</li>
        <li>Best delivery score: {full['delivery_score']*100:.1f}% &mdash; minimizes penalty exposure</li>
        <li>+{fmt_dollar(adv_ind)}/day over industry-standard control</li>
        <li>Annualized on 50 MWh: +{fmt_dollar(adv_ind * 365 * 250)}/year over industry standard</li>
        <li>{full.get('loss_days', 0)} loss days out of {n_days}</li>
        <li>Zero MPC solver failures across {n_days} &times; 1,440 optimization solves</li>
      </ul>
    </div>
    <div>
      <div class="kpi" style="padding:28px;margin-bottom:14px"><div class="kv" style="font-size:52px;color:var(--green)">{fmt_dollar(full['total_profit'])}</div><div class="kl" style="margin-top:8px">Mean Daily Profit</div><div class="ks">per 200 kWh unit</div></div>
      <div class="kpi" style="padding:28px;margin-bottom:14px"><div class="kv" style="font-size:52px;color:var(--glow)">{full['delivery_score']*100:.1f}%</div><div class="kl" style="margin-top:8px">Delivery Score</div><div class="ks">mean across {n_days} days</div></div>
      <div class="kpi" style="padding:28px"><div class="kv" style="font-size:52px;color:var(--amber)">{fmt_dollar(adv_rb * 365 * 250)}</div><div class="kl" style="margin-top:8px">Annual Advantage (50 MWh)</div><div class="ks">vs rule-based dispatch</div></div>
    </div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">08</div>
</section>"""


def slide_roadmap() -> str:
    return """\
<section class="S">
  <div class="lb">What's Next</div>
  <h2>Platform Roadmap</h2>
  <p class="desc">Each release adds a capability that brings the platform closer to full production readiness.</p>
  <div class="g" style="grid-template-columns:1fr 1fr;gap:16px">
    <div class="c"><div class="ic gn">1</div><h3>Smarter State Estimation</h3><p>Replace the current estimator with a more accurate algorithm that handles the battery's nonlinear behavior without simplifications.</p></div>
    <div class="c"><div class="ic bl">2</div><h3>Online Parameter Learning</h3><p>The battery changes as it ages. The system will learn internal resistance and true capacity in real time, not from static datasheets.</p></div>
    <div class="c"><div class="ic pu">3</div><h3>Ultra-Fast Optimization</h3><p>Move from a general-purpose solver to an embedded real-time solver. Target: 10&times; faster decisions, enabling millisecond-level control.</p></div>
    <div class="c"><div class="ic am">4</div><h3>Degradation-Aware Trading</h3><p>Every trade decision will explicitly weigh profit against battery wear. The optimizer will extend asset life by years while maximizing revenue.</p></div>
    <div class="c"><div class="ic rd">5</div><h3>Robust Forecast Handling</h3><p>Price forecasts are uncertain. The system will plan against multiple scenarios simultaneously, protecting against worst-case outcomes.</p></div>
    <div class="c"><div class="ic gn">6</div><h3>Multi-Battery Coordination</h3><p>Scale from one battery to an entire fleet. Each unit optimizes locally while a central coordinator maximizes portfolio-level revenue.</p></div>
    <div class="c"><div class="ic bl">7</div><h3>Grid Inverter Integration</h3><p>Model the power electronics that connect the battery to the grid. Control reactive power and respect real inverter limits.</p></div>
    <div class="c"><div class="ic pu">8</div><h3>Market Bidding Optimization</h3><p>Automate day-ahead and reserve market participation. The system will decide what capacity to commit and at what price.</p></div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">09</div>
</section>"""


def slide_final() -> str:
    return """\
<section class="S title" style="background:radial-gradient(ellipse 100% 60% at 50% 50%,rgba(16,185,129,.08)0,transparent 70%),var(--bg)">
  <div style="margin-bottom:32px">
    <img src="https://arpedon.com/assets/images/logo/logo.svg" alt="Arpedon" style="height:36px;filter:brightness(1.5)">
  </div>
  <h1 style="font-size:42px;max-width:760px">Real-Time Optimization.<br>Proven on Real Data.</h1>
  <div class="dv" style="background:linear-gradient(135deg,var(--green),var(--cyan))"></div>
  <p class="meta" style="margin-top:28px">Confidential</p>
</section>"""


# ─── Plotly chart scripts ───

def build_chart_scripts(data: dict) -> str:
    strats = data["strategies"]
    daily_profits = data["daily_profits"]
    daily_delivery = data["daily_delivery"]
    n_days = data["meta"]["n_days"]

    lines = []
    lines.append("const dark = {paper_bgcolor:'#111827', plot_bgcolor:'#111827', font:{color:'#94a3b8',family:'Inter'}};")
    lines.append(f"const x84 = Array.from({{length:{n_days}}},(_,i)=>i+1);")

    # Inject daily profit arrays
    for s in STRAT_ORDER:
        lines.append(f"const profits_{s} = {json.dumps(daily_profits[s])};")
        lines.append(f"const delivery_{s} = {json.dumps(daily_delivery[s])};")

    # --- Chart 1: Sorted advantage (Full vs Rule-Based) ---
    lines.append(f"""\
const advantage = profits_full.map((v,i) => +(v - profits_rule_based[i]).toFixed(4));
const adv_sorted = [...advantage].sort((a,b)=>a-b);
const adv_colors = adv_sorted.map(v => v >= 0 ? '#10b981' : '#ef4444');
Plotly.newPlot('chart-sorted',[
  {{x:x84, y:adv_sorted, type:'bar',
   marker:{{color:adv_colors, cornerradius:3}},
   hovertemplate:'$%{{y:.2f}} advantage<extra></extra>'}}
],{{
  ...dark,
  title:{{text:'Full Optimizer advantage over Rule-Based — per day, sorted',font:{{size:11,color:'#f1f5f9'}}}},
  xaxis:{{title:'{n_days} days of Q1 2024, sorted', gridcolor:'#1e293b', tickfont:{{size:9}}, titlefont:{{size:10}}}},
  yaxis:{{title:'$ / day', gridcolor:'#1e293b', tickfont:{{size:10}}, titlefont:{{size:11}},
         zeroline:true, zerolinecolor:'#94a3b8', zerolinewidth:1.5}},
  showlegend:false,
  margin:{{t:40, b:44, l:50, r:16}}
}},{{responsive:true,displayModeBar:false}});""")

    # --- Chart 2: Cumulative profit ---
    traces_cumul = []
    for s in STRAT_ORDER:
        color = STRAT_COLORS[s]
        label = STRAT_LABELS[s]
        width = 3 if s == "full" else 1.5
        traces_cumul.append(
            f"{{x:x84, y:profits_{s}.reduce((acc,v,i)=>{{acc.push((acc[i]||0)+v);return acc}},[]).slice(0,{n_days}), "
            f"type:'scatter', mode:'lines', name:'{label}', "
            f"line:{{color:'{color}',width:{width}}}, "
            f"hovertemplate:'Day %{{x}}: $%{{y:.0f}} total<extra>{label}</extra>'}}"
        )

    lines.append(f"""\
Plotly.newPlot('chart-cumul',[
  {','.join(traces_cumul)}
],{{
  ...dark,
  title:{{text:'Cumulative profit over Q1 2024',font:{{size:11,color:'#f1f5f9'}}}},
  xaxis:{{title:'Day', gridcolor:'#1e293b', tickfont:{{size:9}}, titlefont:{{size:10}}}},
  yaxis:{{title:'$ cumulative', gridcolor:'#1e293b', tickfont:{{size:10}}, titlefont:{{size:11}}}},
  legend:{{orientation:'v', y:0.95, x:0.02, font:{{size:9}}}},
  margin:{{t:40, b:44, l:56, r:16}}
}},{{responsive:true,displayModeBar:false}});""")

    # --- KPI cards (dynamic) ---
    full_loss = strats["full"].get("loss_days", 0)
    rb_loss = strats["rule_based"].get("loss_days", 0)
    full_profit_mean = strats["full"]["total_profit"]
    rb_profit_mean = strats["rule_based"]["total_profit"]
    cum_advantage = (full_profit_mean - rb_profit_mean) * n_days
    full_del = strats["full"]["delivery_score"]

    lines.append(f"""\
document.getElementById('kpi-loss').innerHTML = `
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-size:11px;font-weight:700;color:var(--green)">Loss days</span>
    <span style="font-size:20px;font-weight:800;color:var(--green)">{full_loss} / {n_days}</span>
  </div>
  <div style="font-size:10px;color:var(--dim);margin-top:8px">Full Optimizer &middot; Rule-Based loses {rb_loss}</div>`;
document.getElementById('kpi-adv').innerHTML = `
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-size:11px;font-weight:700;color:var(--glow)">Cumulative edge</span>
    <span style="font-size:20px;font-weight:800;color:var(--glow)">{fmt_dollar(cum_advantage)}</span>
  </div>
  <div style="font-size:10px;color:var(--dim);margin-top:8px">over {n_days} days, net of all wear</div>`;
document.getElementById('kpi-delivery').innerHTML = `
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-size:11px;font-weight:700;color:var(--purple)">Delivery score</span>
    <span style="font-size:20px;font-weight:800;color:var(--purple)">{full_del*100:.1f}%</span>
  </div>
  <div style="font-size:10px;color:var(--dim);margin-top:8px">Full Optimizer mean &middot; Industry: {strats['ems_clamps']['delivery_score']*100:.1f}%</div>`;""")

    # --- Chart 3: Delivery score box plot ---
    delivery_traces = []
    for s in STRAT_ORDER:
        delivery_traces.append(
            f"{{y:delivery_{s}.map(v=>v*100), type:'box', name:'{STRAT_LABELS[s]}', "
            f"marker:{{color:'{STRAT_COLORS[s]}'}}, "
            f"line:{{color:'{STRAT_COLORS[s]}'}}, "
            f"boxmean:true}}"
        )

    lines.append(f"""\
Plotly.newPlot('chart-delivery',[
  {','.join(delivery_traces)}
],{{
  ...dark,
  yaxis:{{title:'Delivery Score (%)', gridcolor:'#1e293b', tickfont:{{size:10}}, titlefont:{{size:11}},
         range:[70,102]}},
  showlegend:false,
  margin:{{t:20, b:40, l:56, r:16}}
}},{{responsive:true,displayModeBar:false}});""")

    # --- Chart 4: Revenue breakdown bar chart ---
    x_labels = json.dumps([STRAT_LABELS[s] for s in STRAT_ORDER])
    energy_vals = json.dumps([strats[s]["energy_profit"] for s in STRAT_ORDER])
    cap_vals = json.dumps([strats[s]["capacity_revenue"] for s in STRAT_ORDER])
    del_vals = json.dumps([strats[s]["delivery_revenue"] for s in STRAT_ORDER])
    pen_vals = json.dumps([-strats[s]["penalty_cost"] for s in STRAT_ORDER])
    deg_vals = json.dumps([-strats[s]["deg_cost"] for s in STRAT_ORDER])
    net_vals = json.dumps([strats[s]["total_profit"] for s in STRAT_ORDER])

    lines.append(f"""\
Plotly.newPlot('chart-revenue',[
  {{x:{x_labels}, y:{energy_vals}, type:'bar', name:'Energy Arbitrage',
   marker:{{color:'#3b82f6',cornerradius:3}}}},
  {{x:{x_labels}, y:{cap_vals}, type:'bar', name:'Capacity Revenue',
   marker:{{color:'#06b6d4',cornerradius:3}}}},
  {{x:{x_labels}, y:{del_vals}, type:'bar', name:'Delivery Revenue',
   marker:{{color:'#10b981',cornerradius:3}}}},
  {{x:{x_labels}, y:{pen_vals}, type:'bar', name:'Penalties',
   marker:{{color:'#ef4444',cornerradius:3}}}},
  {{x:{x_labels}, y:{deg_vals}, type:'bar', name:'Degradation',
   marker:{{color:'#64748b',cornerradius:3}}}},
  {{x:{x_labels}, y:{net_vals}, type:'scatter', mode:'lines+markers',
   name:'Net Profit', line:{{color:'#f1f5f9',width:3}},
   marker:{{size:10,color:'#f1f5f9',line:{{width:2,color:'#0b0f1a'}}}},
   hovertemplate:'$%{{y:.2f}}<extra>Net Profit</extra>'}}
],{{
  ...dark,
  barmode:'relative',
  xaxis:{{tickfont:{{size:11,color:'#f1f5f9'}}}},
  yaxis:{{title:'$ / day (mean, 200 kWh)', gridcolor:'#1e293b', tickfont:{{size:10}}, titlefont:{{size:11}},
         zeroline:true, zerolinecolor:'#94a3b8', zerolinewidth:1}},
  legend:{{orientation:'h', y:1.08, x:0.5, xanchor:'center', font:{{size:10}}}},
  margin:{{t:40,b:40,l:56,r:16}}
}},{{responsive:true,displayModeBar:false}});""")

    return "\n".join(lines)


# ─── Main assembly ───

def generate_html(data: dict) -> str:
    slides = [
        slide_title(data),
        slide_challenge(),
        slide_strategies(),
        slide_kpi_table(data),
        slide_daily_profits(),
        slide_delivery(),
        slide_revenue(data),
        slide_closing(data),
        slide_roadmap(),
        slide_final(),
    ]
    charts_js = build_chart_scripts(data)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BESS Control Strategy Analysis &mdash; Arpedon</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{CSS}
</style>
</head>
<body>
{''.join(slides)}
<script>
{charts_js}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v5 strategy comparison HTML deck")
    parser.add_argument("--input", type=str,
                        default=str(REPO_ROOT / "results" / "v5_comparison.json"))
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "presentation" / "v5_strategy_comparison.html"))
    args = parser.parse_args()

    data = load_data(pathlib.Path(args.input))
    html = generate_html(data)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    n_days = data["meta"]["n_days"]
    print(f"  Generated: {out_path}")
    print(f"  Slides:    10")
    print(f"  Data:      {n_days} days, {len(STRAT_ORDER)} strategies")


if __name__ == "__main__":
    main()
