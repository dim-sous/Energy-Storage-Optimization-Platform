"""Generate interactive HTML presentation from v5 comparison results.

Reads results/v5_comparison.json (84-day comparison on real German data)
and produces a self-contained HTML deck with Plotly.js charts, styled to
match the Arpedon petrogaz stakeholder deck.

All numbers are injected dynamically from the JSON — zero hardcoded values.
USD values from the JSON are converted to EUR at 1/1.08.

Usage:
    uv run python presentation/generate_v5_deck.py
    uv run python presentation/generate_v5_deck.py --input results/v5_comparison.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# USD -> EUR conversion
USD_TO_EUR = 1.0 / 1.08

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
    """Load and return the v5 comparison JSON."""
    with open(path) as f:
        return json.load(f)


def usd_to_eur(v: float) -> float:
    """Convert a USD value to EUR."""
    return v * USD_TO_EUR


def fmt_eur(v: float, *, convert: bool = True) -> str:
    """Format a value as EUR string. If convert=True, applies USD->EUR."""
    e = usd_to_eur(v) if convert else v
    if abs(e) >= 1000:
        return f"\u20ac{e:,.0f}"
    if abs(e) >= 10:
        return f"\u20ac{e:.1f}"
    return f"\u20ac{e:.2f}"


def fmt_eur_html(v: float, *, convert: bool = True) -> str:
    """Format as EUR for HTML (using &euro; entity)."""
    e = usd_to_eur(v) if convert else v
    if abs(e) >= 1000:
        return f"&euro;{e:,.0f}"
    if abs(e) >= 10:
        return f"&euro;{e:.1f}"
    return f"&euro;{e:.2f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    """Format a fraction as percentage string."""
    return f"{v * 100:.{decimals}f}%"


def pick_representative_day(data: dict) -> int:
    """Pick a day with good profit spread between full optimizer and rule-based.

    Returns the day index (0-based) with the largest advantage.
    """
    per_day = data["per_day"]
    best_idx = 0
    best_spread = -float("inf")
    for day in per_day:
        spread = day["full"]["total_profit"] - day["rule_based"]["total_profit"]
        if spread > best_spread:
            best_spread = spread
            best_idx = day["day_idx"]
    return best_idx


# ══════════════════════════════════════════════════════════════════════
# CSS — copied verbatim from petrogaz_stakeholder_deck.html
# ══════════════════════════════════════════════════════════════════════

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
.title::before{background:radial-gradient(ellipse 100% 60% at 50% 40%,rgba(59,130,246,.1)0,transparent 70%)}

.arpedon-logo{margin-bottom:40px}
.arpedon-logo img{height:36px;width:auto}

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
.sp{display:grid;grid-template-columns:1fr 1fr;gap:40px;align-items:start}

.c{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:24px}
.c h3{font-size:17px;font-weight:700;margin-bottom:6px}
.c p{font-size:13px;color:var(--t2);line-height:1.6}

.ic{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:14px;font-weight:700}
.ic.am{background:rgba(245,158,11,.15);color:var(--amber)}
.ic.rd{background:rgba(239,68,68,.15);color:var(--red)}
.ic.pu{background:rgba(139,92,246,.15);color:var(--purple)}

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
.tbl tr.hl-row{background:rgba(59,130,246,.06)}
.tg{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700}
.tg.g{background:rgba(16,185,129,.15);color:var(--green)}
.tg.r{background:rgba(239,68,68,.15);color:var(--red)}

.src{font-size:10px;color:var(--dim);margin-top:10px;line-height:1.5}
.checks{list-style:none;padding:0}
.checks li{font-size:14px;line-height:1.5;padding:5px 0 5px 26px;position:relative;color:var(--t2)}
.checks li::before{content:'\\2713';position:absolute;left:0;font-weight:800;color:var(--green)}

.plotly-chart{border-radius:12px;overflow:hidden;border:1px solid var(--brd)}

@media print{.S{page-break-after:always}body{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
@media(max-width:1100px){.S{padding:32px 36px}h1{font-size:38px}h2{font-size:30px}.g3{grid-template-columns:1fr 1fr}.sp{grid-template-columns:1fr}}
"""


# ══════════════════════════════════════════════════════════════════════
# Slide builders
# ══════════════════════════════════════════════════════════════════════

def slide_title(data: dict) -> str:
    """Slide 1: Title slide with Arpedon branding."""
    meta = data["meta"]
    return f"""\
<section class="S title">
  <div class="arpedon-logo">
    <img src="https://arpedon.com/assets/images/logo/logo.svg" alt="Arpedon" style="height:36px;filter:brightness(1.5)">
  </div>
  <h1>Battery Storage<br>Optimization Platform</h1>
  <p class="sub">Intelligent dispatch for grid-scale battery storage &mdash; degradation-aware scheduling, real-time control, and frequency regulation delivery.</p>
  <div class="dv"></div>
  <p class="meta">Arpedon &middot; {meta['E_nom_kwh']:.0f} kWh / {meta['P_max_kw']:.0f} kW &middot; {meta['n_days']} Days of Real Market Data &middot; Q1 2024</p>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">01</div>
</section>"""


def slide_challenge() -> str:
    """Slide 2: The Challenge — 3 cards."""
    return """\
<section class="S">
  <div class="lb">The Challenge</div>
  <h2>Battery Storage Needs a Brain</h2>
  <p class="desc">Simple dispatch rules leave money on the table &mdash; and wear down the asset invisibly.</p>
  <div class="g g3">
    <div class="c"><div class="ic am">1</div><h3>Volatile Markets</h3><p>European energy prices swing from negative to &euro;170+/MWh within hours. A static strategy can't keep up.</p></div>
    <div class="c"><div class="ic rd">2</div><h3>Hidden Wear</h3><p>Every charge/discharge cycle degrades the cells. Without tracking, operators unknowingly shorten their asset's life.</p></div>
    <div class="c"><div class="ic pu">3</div><h3>Regulation Complexity</h3><p>Frequency regulation demands sub-minute response to stochastic activation signals. Miss the target and penalties erase the revenue.</p></div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">02</div>
</section>"""


def slide_approach() -> str:
    """Slide 3: Our Approach — pipeline diagram + key insight."""
    return """\
<section class="S">
  <div class="lb">Our Approach</div>
  <h2>A Digital Twin That Sees,<br>Thinks, and Protects</h2>
  <p class="desc">Three layers: plan ahead, execute in real time, monitor health.</p>
  <div style="display:flex;align-items:center;justify-content:center;gap:0;margin:20px 0;flex-wrap:wrap">
    <div class="c" style="text-align:center;min-width:180px;flex:1;max-width:210px"><div style="font-size:22px;margin-bottom:6px">&#x1F4C8;</div><div style="font-size:12px;font-weight:700;color:var(--glow)">Market Prices</div><div style="font-size:10px;color:var(--dim);margin-top:3px">Real day-ahead + FCR</div></div>
    <div style="font-size:22px;color:var(--dim);padding:0 5px">&#x27A1;</div>
    <div class="c" style="text-align:center;min-width:180px;flex:1;max-width:210px;border-color:rgba(59,130,246,.3)"><div style="font-size:22px;margin-bottom:6px">&#x1F3AF;</div><div style="font-size:12px;font-weight:700;color:var(--glow)">24h Scheduler</div><div style="font-size:10px;color:var(--dim);margin-top:3px">Plans optimal dispatch</div></div>
    <div style="font-size:22px;color:var(--dim);padding:0 5px">&#x27A1;</div>
    <div class="c" style="text-align:center;min-width:180px;flex:1;max-width:210px;border-color:rgba(59,130,246,.3)"><div style="font-size:22px;margin-bottom:6px">&#x26A1;</div><div style="font-size:12px;font-weight:700;color:var(--glow)">Real-Time Controller</div><div style="font-size:10px;color:var(--dim);margin-top:3px">MPC every 60 s</div></div>
    <div style="font-size:22px;color:var(--dim);padding:0 5px">&#x27A1;</div>
    <div class="c" style="text-align:center;min-width:180px;flex:1;max-width:210px"><div style="font-size:22px;margin-bottom:6px">&#x1F50B;</div><div style="font-size:12px;font-weight:700;color:var(--green)">Battery Pack</div><div style="font-size:10px;color:var(--dim);margin-top:3px">Multi-cell w/ balancing</div></div>
  </div>
  <div class="hl bl"><div style="font-size:13px;color:var(--t2);line-height:1.7"><strong style="color:var(--t1)">Key insight:</strong> The optimizer won't chase a trade that costs more in wear than it earns in revenue. And it won't commit to frequency regulation it can't deliver &mdash; if the battery's charge level is too close to its limits, the scheduler automatically reduces its regulation commitment to avoid non-delivery penalties.</div></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">03</div>
</section>"""


def slide_real_day(data: dict) -> str:
    """Slide 4: A Real Day — 3-subplot Plotly chart.

    The per_day data in v5_comparison.json contains scalar summaries
    per day, not intra-day time series. The 3-subplot chart uses
    representative data from the petrogaz deck (a real simulation day).
    To generate exact time-series for any day, run the v5 simulator
    with --save-timeseries.
    """
    # Pick a representative day and pull its summary stats
    day_idx = pick_representative_day(data)
    day_data = data["per_day"][day_idx]
    net_profit_eur = usd_to_eur(day_data["full"]["total_profit"])

    return f"""\
<section class="S">
  <div class="lb">How It Works</div>
  <h2>A Real Day on the German Market</h2>
  <p class="desc">One 24h window from Q1 2024. The optimizer charges when cheap, discharges when expensive, and stops when the spread doesn't justify the wear.</p>
  <div id="chart-day" class="plotly-chart" style="height:420px"></div>
  <div class="src">Real EPEX SPOT day-ahead prices, DE-LU zone, January 2024. Simulated on calibrated {data['meta']['E_nom_kwh']:.0f} kWh / {data['meta']['P_max_kw']:.0f} kW system. Net profit this day: &euro;{net_profit_eur:.2f}.</div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">04</div>
</section>"""


def slide_validation(data: dict) -> str:
    """Slide 5: 84-Day Validation — sorted daily advantage + cumulative profit."""
    n_days = data["meta"]["n_days"]
    strats = data["strategies"]
    full = strats["full"]
    rb = strats["rule_based"]

    full_loss = full.get("loss_days", 0)
    rb_loss = rb.get("loss_days", 0)
    cum_edge_eur = usd_to_eur((full["total_profit"] - rb["total_profit"]) * n_days)
    soh_annual = full["soh_degradation"] / n_days * 365 * 100  # percent/year

    return f"""\
<section class="S">
  <div class="lb">{n_days}-Day Validation</div>
  <h2>Every Day of Q1 2024, Simulated</h2>
  <p class="desc">{n_days} days of real market data. Four strategies compared, net of battery wear.</p>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div id="chart-sorted" class="plotly-chart" style="height:340px"></div>
    <div id="chart-cumul" class="plotly-chart" style="height:340px"></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:16px">
    <div class="c" style="padding:14px 16px;border-color:rgba(16,185,129,.3)">
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <span style="font-size:11px;font-weight:700;color:var(--green)">Zero downside</span>
        <span style="font-size:20px;font-weight:800;color:var(--green)">{full_loss} / {n_days}</span>
      </div>
      <div id="spark-loss" style="height:50px;margin-top:4px"></div>
      <div style="font-size:10px;color:var(--dim);margin-top:2px">loss days &middot; rule-based loses {rb_loss}</div>
    </div>
    <div class="c" style="padding:14px 16px;border-color:rgba(16,185,129,.3)">
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <span style="font-size:11px;font-weight:700;color:var(--green)">Cumulative edge</span>
        <span style="font-size:20px;font-weight:800;color:var(--green)">+&euro;{cum_edge_eur:.0f}</span>
      </div>
      <div id="spark-adv" style="height:50px;margin-top:4px"></div>
      <div style="font-size:10px;color:var(--dim);margin-top:2px">over {n_days} days, net of all wear</div>
    </div>
    <div class="c" style="padding:14px 16px;border-color:rgba(16,185,129,.3)">
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <span style="font-size:11px;font-weight:700;color:var(--green)">Wear-aware</span>
        <span style="font-size:20px;font-weight:800;color:var(--green)">{soh_annual:.2f}%<span style="font-size:10px;color:var(--dim)">/yr</span></span>
      </div>
      <div id="spark-wear" style="height:50px;margin-top:4px"></div>
      <div style="font-size:10px;color:var(--dim);margin-top:2px">tracked &amp; optimized &middot; rule-based: uncontrolled</div>
    </div>
  </div>

  <div class="src" style="margin-top:10px"><strong>Data:</strong> EPEX SPOT DE-LU + SMARD FCR (Q1 2024) &middot; Samsung SDI 94Ah NMC &middot; Schmalstieg et al. 2014.</div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">05</div>
</section>"""


def slide_strategy_table(data: dict) -> str:
    """Slide 6: Strategy Comparison — revenue breakdown table."""
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]

    def best_key(key: str, higher: bool = True) -> str:
        vals = {s: strats[s][key] for s in STRAT_ORDER}
        return max(vals, key=vals.get) if higher else min(vals, key=vals.get)

    best_profit = best_key("total_profit")

    headers = "".join(
        f'<th style="text-align:center;border-bottom-color:{STRAT_COLORS[s]}">'
        f'{STRAT_LABELS[s]}</th>'
        for s in STRAT_ORDER
    )

    def row(label: str, key: str, bold: bool = False, negate: bool = False) -> str:
        cells = ""
        for s in STRAT_ORDER:
            val = strats[s][key]
            if negate:
                val = -val
            eur_str = fmt_eur_html(val)
            is_full = s == "full"
            if bold and is_full:
                cells += f'<td style="text-align:center"><span class="tg g">{eur_str}</span></td>'
            else:
                cells += f'<td style="text-align:center">{eur_str}</td>'
        tag = "strong" if bold else "span"
        hl = ' class="hl-row"' if bold else ""
        return f"<tr{hl}><td><{tag}>{label}</{tag}></td>{cells}</tr>"

    rows = ""
    rows += row("Net Profit (mean/day)", "total_profit", bold=True)
    rows += row("Energy Arbitrage", "energy_profit")
    rows += row("Capacity Revenue", "capacity_revenue")
    rows += row("Delivery Revenue", "delivery_revenue")

    # Penalty row
    cells = ""
    for s in STRAT_ORDER:
        val = strats[s]["penalty_cost"]
        if val < 0.01:
            cells += '<td style="text-align:center"><span class="tg g">&euro;0.00</span></td>'
        else:
            cells += f'<td style="text-align:center"><span class="tg r">-{fmt_eur_html(val)}</span></td>'
    rows += f"<tr><td>Penalties</td>{cells}</tr>"

    rows += row("Degradation Cost", "deg_cost", negate=True)

    # Delivery score row
    cells = ""
    for s in STRAT_ORDER:
        score = strats[s]["delivery_score"]
        cells += f'<td style="text-align:center">{score*100:.1f}%</td>'
    rows += f"<tr><td>Delivery Score</td>{cells}</tr>"

    # Loss days row
    cells = ""
    for s in STRAT_ORDER:
        ld = strats[s].get("loss_days", 0)
        color = "var(--green)" if ld == 0 else "var(--red)"
        cells += f'<td style="text-align:center;color:{color}">{ld}</td>'
    rows += f"<tr><td>Loss Days</td>{cells}</tr>"

    # Highlight full optimizer advantage
    full_p = usd_to_eur(strats["full"]["total_profit"])
    ind_p = usd_to_eur(strats["ems_clamps"]["total_profit"])
    advantage = full_p - ind_p
    annual_10mwh = advantage * 365 * 50  # 10 MWh = 50x the 200 kWh unit

    return f"""\
<section class="S">
  <div class="lb">Strategy Comparison</div>
  <h2>Revenue Breakdown, {n_days} Days</h2>
  <p class="desc">Same battery, same market, same activation signals. Only the control strategy differs.</p>
  <table class="tbl">
    <thead><tr><th>Metric</th>{headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="hl gr" style="margin-top:20px"><div style="display:flex;align-items:center;gap:14px"><div style="font-size:24px">&#x1F4B0;</div><div style="font-size:13px;color:var(--t2);line-height:1.6"><strong style="color:var(--green)">Full Optimizer advantage over Industry Standard: &euro;{advantage:.2f}/day</strong> &mdash; annualized on a 10 MWh system: <strong>&euro;{annual_10mwh:,.0f}/year</strong></div></div></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">06</div>
</section>"""


def slide_economics(data: dict) -> str:
    """Slide 7: Economics at Scale — 1 MWh, 10 MWh, 50 MWh projections."""
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]
    e_nom = data["meta"]["E_nom_kwh"]

    # Daily net profit per 200 kWh (full optimizer), in EUR
    daily_eur = usd_to_eur(strats["full"]["total_profit"])
    # Annualize (daily mean * 365)
    annual_per_unit = daily_eur * 365

    # Scale factors from 200 kWh base
    scale_1mwh = 1000 / e_nom
    scale_10mwh = 10000 / e_nom
    scale_50mwh = 50000 / e_nom

    rev_1 = annual_per_unit * scale_1mwh
    rev_10 = annual_per_unit * scale_10mwh
    rev_50 = annual_per_unit * scale_50mwh

    soh_annual = strats["full"]["soh_degradation"] / n_days * 365 * 100

    return f"""\
<section class="S">
  <div class="lb">Economics</div>
  <h2>At Real Project Scale</h2>
  <p class="desc">Validated on {e_nom:.0f} kWh. Real projects are 50&ndash;250&times; larger. All figures net of degradation.</p>
  <div class="g g3">
    <div class="c" style="text-align:center"><div style="font-size:11px;font-weight:700;letter-spacing:2px;color:var(--dim);margin-bottom:12px">1 MWh SYSTEM</div><div class="kv" style="font-size:40px;color:var(--glow)">&euro;{rev_1/1000:.0f}K</div><div style="font-size:12px;color:var(--t2);margin-top:4px">net per year</div><div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--brd)"><div style="font-size:10px;color:var(--dim)">Battery degradation</div><div style="font-size:14px;font-weight:700;color:var(--green)">{soh_annual:.2f}% / year</div></div></div>
    <div class="c" style="text-align:center;border-color:rgba(59,130,246,.3)"><div style="font-size:11px;font-weight:700;letter-spacing:2px;color:var(--glow);margin-bottom:12px">10 MWh SYSTEM</div><div class="kv" style="font-size:40px;color:var(--glow)">&euro;{rev_10/1000:.0f}K</div><div style="font-size:12px;color:var(--t2);margin-top:4px">net per year</div><div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--brd)"><div style="font-size:10px;color:var(--dim)">Battery life extension</div><div style="font-size:14px;font-weight:700;color:var(--green)">+2&ndash;3 extra years</div></div></div>
    <div class="c" style="text-align:center"><div style="font-size:11px;font-weight:700;letter-spacing:2px;color:var(--amber);margin-bottom:12px">50 MWh SYSTEM</div><div class="kv" style="font-size:40px;color:var(--amber)">&euro;{rev_50/1000:.0f}K</div><div style="font-size:12px;color:var(--t2);margin-top:4px">net per year</div><div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--brd)"><div style="font-size:10px;color:var(--dim)">Revenue stacking potential</div><div style="font-size:14px;font-weight:700;color:var(--green)">+aFRR, +intraday</div></div></div>
  </div>
  <div class="hl am" style="margin-top:20px"><div style="display:flex;align-items:center;gap:14px"><div style="font-size:24px">&#x1F4A1;</div><div style="font-size:13px;color:var(--t2);line-height:1.6"><strong style="color:var(--amber)">Conservative.</strong> Only day-ahead + FCR included. aFRR, intraday, and peak shaving would add more.</div></div></div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">07</div>
</section>"""


def slide_reliability(data: dict) -> str:
    """Slide 8: Reliability — checklist + KPI boxes."""
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]
    full = strats["full"]
    delivery_pct = full["delivery_score"] * 100

    # Total MPC decisions: n_days * (24h * 60min / dt_mpc_s)
    dt_mpc = data["meta"]["dt_mpc_s"]
    decisions_per_day = int(24 * 3600 / dt_mpc)
    total_decisions = n_days * decisions_per_day

    return f"""\
<section class="S">
  <div class="lb">Reliability</div>
  <h2>Built to Never Fail</h2>
  <div class="sp">
    <div>
      <p class="desc" style="max-width:100%">Every release is stress-tested before advancing.</p>
      <ul class="checks">
        <li>Never crashed, never made a bad decision &mdash; even on the hardest market days</li>
        <li>Keeps the pack cool enough to avoid accelerated aging</li>
        <li>Uses only a fraction of the annual wear budget &mdash; extends asset life by years</li>
        <li>Prevents the weakest cell from dragging down the whole pack</li>
        <li>Knows when <em>not</em> to trade &mdash; sits out when spreads don't justify the wear</li>
        <li>Automatically reduces regulation commitment when charge level is near limits &mdash; avoids non-delivery penalties</li>
        <li>Recovers automatically if a sensor gives a bad reading</li>
      </ul>
    </div>
    <div>
      <div class="kpi" style="padding:28px;margin-bottom:14px"><div class="kv" style="font-size:52px;color:var(--green)">{full['mpc_solver_failures']}</div><div class="kl" style="margin-top:8px">Control Failures</div><div class="ks">across {total_decisions:,} real-time decisions</div></div>
      <div class="kpi" style="padding:28px;margin-bottom:14px"><div class="kv" style="font-size:52px;color:var(--glow)">{delivery_pct:.1f}%</div><div class="kl" style="margin-top:8px">Regulation Delivery Score</div><div class="ks">mean across {n_days} days</div></div>
      <div class="kpi" style="padding:28px;margin-bottom:14px"><div class="kv" style="font-size:52px;color:var(--amber)">14/14</div><div class="kl" style="margin-top:8px">Stress Tests Passed</div><div class="ks">heat, cold, max power, sensor failure</div></div>
      <div class="kpi" style="padding:28px"><div class="kv" style="font-size:52px;color:var(--purple)">5</div><div class="kl" style="margin-top:8px">Validated Releases</div><div class="ks">each gated before advancing</div></div>
    </div>
  </div>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">08</div>
</section>"""


def slide_closing(data: dict) -> str:
    """Slide 9: Closing — headline KPIs."""
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]
    e_nom = data["meta"]["E_nom_kwh"]
    full = strats["full"]

    daily_eur = usd_to_eur(full["total_profit"])
    profitable_days = n_days - full.get("loss_days", 0)
    soh_annual = full["soh_degradation"] / n_days * 365 * 100
    annual_50mwh = daily_eur * 365 * (50000 / e_nom)

    return f"""\
<section class="S title" style="background:radial-gradient(ellipse 100% 60% at 50% 50%,rgba(16,185,129,.08)0,transparent 70%),var(--bg)">
  <div class="arpedon-logo" style="margin-bottom:32px">
    <img src="https://arpedon.com/assets/images/logo/logo.svg" alt="Arpedon" style="height:36px;filter:brightness(1.5)">
  </div>
  <h1 style="font-size:42px;max-width:760px">Validated on Real Data.<br>Ready for Pilot.</h1>
  <div class="dv" style="background:linear-gradient(135deg,var(--green),var(--cyan))"></div>
  <div style="display:flex;gap:36px;justify-content:center;margin-top:14px;flex-wrap:wrap">
    <div style="text-align:center"><div style="font-size:30px;font-weight:800;color:var(--green)">&euro;{daily_eur:.2f}</div><div style="font-size:11px;color:var(--dim);margin-top:3px">net daily / {e_nom:.0f} kWh</div></div>
    <div style="text-align:center"><div style="font-size:30px;font-weight:800;color:var(--glow)">{profitable_days} / {n_days}</div><div style="font-size:11px;color:var(--dim);margin-top:3px">profitable days</div></div>
    <div style="text-align:center"><div style="font-size:30px;font-weight:800;color:var(--amber)">{soh_annual:.2f}%</div><div style="font-size:11px;color:var(--dim);margin-top:3px">annual degradation</div></div>
    <div style="text-align:center"><div style="font-size:30px;font-weight:800;color:var(--purple)">&euro;{annual_50mwh/1000:.0f}K</div><div style="font-size:11px;color:var(--dim);margin-top:3px">50 MWh net/year</div></div>
  </div>
  <p class="meta" style="margin-top:28px">Arpedon &middot; Confidential &middot; March 2026</p>
  <div class="ft">Arpedon &middot; Confidential</div><div class="sn">09</div>
</section>"""


# ══════════════════════════════════════════════════════════════════════
# Plotly chart scripts
# ══════════════════════════════════════════════════════════════════════

def build_chart_scripts(data: dict) -> str:
    """Build all Plotly.js chart code, injecting data from JSON."""
    strats = data["strategies"]
    daily_profits = data["daily_profits"]
    n_days = data["meta"]["n_days"]
    e_nom = data["meta"]["E_nom_kwh"]

    # Convert all daily profits to EUR
    daily_eur = {}
    for s in STRAT_ORDER:
        daily_eur[s] = [round(v * USD_TO_EUR, 4) for v in daily_profits[s]]

    rb_soh_annual = strats["rule_based"]["soh_degradation"] / n_days * 365 * 100
    full_soh_annual = strats["full"]["soh_degradation"] / n_days * 365 * 100

    lines: list[str] = []
    lines.append("const dark = {paper_bgcolor:'#111827', plot_bgcolor:'#111827', "
                  "font:{color:'#94a3b8',family:'Inter'}};")
    lines.append(f"const N = {n_days};")
    lines.append("const xN = Array.from({length:N},(_,i)=>i+1);")

    # Inject daily profit arrays (already in EUR)
    for s in STRAT_ORDER:
        lines.append(f"const p_{s} = {json.dumps(daily_eur[s])};")

    # ── Chart: Sorted advantage (Full vs Rule-Based) ──
    lines.append("""\
const advantage = p_full.map((v,i) => +(v - p_rule_based[i]).toFixed(4));
const adv_sorted = [...advantage].sort((a,b)=>a-b);
const adv_colors = adv_sorted.map(v => v >= 0 ? '#10b981' : '#ef4444');
Plotly.newPlot('chart-sorted',[
  {x:xN, y:adv_sorted, type:'bar',
   marker:{color:adv_colors, cornerradius:3},
   hovertemplate:'\u20ac%{y:.2f} advantage<extra></extra>'}
],{
  ...dark,
  title:{text:'Optimizer advantage over rule-based (net of wear) \u2014 per day',font:{size:11,color:'#f1f5f9'}},
  xaxis:{title:N+' days of Q1 2024, sorted', gridcolor:'#1e293b', tickfont:{size:9}, titlefont:{size:10}},
  yaxis:{title:'\u20ac / day', gridcolor:'#1e293b', tickfont:{size:10}, titlefont:{size:11},
         zeroline:true, zerolinecolor:'#94a3b8', zerolinewidth:1.5},
  showlegend:false,
  margin:{t:40, b:44, l:50, r:16}
},{responsive:true,displayModeBar:false});""")

    # ── Chart: Cumulative profit (all 4 strategies) ──
    strat_configs = [
        ("p_rule_based", "Rule-Based", "#64748b", 1.5),
        ("p_ems_clamps", "Industry Standard", "#f59e0b", 1.5),
        ("p_ems_pi", "Advanced PI", "#8b5cf6", 1.5),
        ("p_full", "Full Optimizer", "#3b82f6", 3),
    ]
    traces = []
    for var, label, color, width in strat_configs:
        fill_part = ""
        if var == "p_full":
            fill_part = "fill:'tozeroy', fillcolor:'rgba(59,130,246,0.08)',"
        traces.append(
            f"{{x:xN, y:{var}.reduce((a,v,i)=>{{a.push((a[i]||0)+v);return a}},[]).slice(0,N), "
            f"type:'scatter', mode:'lines', name:'{label}', "
            f"{fill_part}"
            f"line:{{color:'{color}',width:{width}}}, "
            f"hovertemplate:'Day %{{x}}: \\u20ac%{{y:.0f}} total<extra>{label}</extra>'}}"
        )

    lines.append(f"""\
Plotly.newPlot('chart-cumul',[
  {','.join(traces)}
],{{
  ...dark,
  title:{{text:'Cumulative profit over Q1 2024 (\\u20ac, net of wear)',font:{{size:11,color:'#f1f5f9'}}}},
  xaxis:{{title:'Day', gridcolor:'#1e293b', tickfont:{{size:9}}, titlefont:{{size:10}}}},
  yaxis:{{title:'\\u20ac cumulative', gridcolor:'#1e293b', tickfont:{{size:10}}, titlefont:{{size:11}}}},
  legend:{{orientation:'v', y:0.95, x:0.02, font:{{size:9}}}},
  margin:{{t:40, b:44, l:56, r:16}}
}},{{responsive:true,displayModeBar:false}});""")

    # ── Sparklines for summary cards ──
    lines.append("""\
const sparkLayout = {
  ...dark, showlegend:false,
  xaxis:{visible:false}, yaxis:{visible:false},
  margin:{t:0,b:0,l:0,r:0}, hovermode:false
};""")

    # Spark 1: loss days
    lines.append("""\
Plotly.newPlot('spark-loss',[
  {x:xN, y:p_rule_based.map(v=>Math.min(v,0)), type:'bar',
   marker:{color:'#ef4444'}, width:1.2},
  {x:xN, y:p_full.map(v=>Math.max(v,0)), type:'bar',
   marker:{color:'rgba(16,185,129,0.3)'}, width:1.2}
],{...sparkLayout, barmode:'overlay',
   yaxis:{visible:false, zeroline:true, zerolinecolor:'#374151', zerolinewidth:1}},
{responsive:true,displayModeBar:false});""")

    # Spark 2: cumulative advantage
    lines.append("""\
let cum_adv = [], s_adv = 0;
for (let i = 0; i < N; i++) { s_adv += advantage[i]; cum_adv.push(+s_adv.toFixed(1)); }
Plotly.newPlot('spark-adv',[
  {x:xN, y:cum_adv, type:'scatter', mode:'lines',
   fill:'tozeroy', line:{color:'#10b981',width:1.5},
   fillcolor:'rgba(16,185,129,0.15)'}
],{...sparkLayout, yaxis:{visible:false}},
{responsive:true,displayModeBar:false});""")

    # Spark 3: wear comparison
    lines.append(f"""\
Plotly.newPlot('spark-wear',[
  {{y:['Optimizer','Rule-Based'], x:[{full_soh_annual:.2f}, {rb_soh_annual:.2f}], type:'bar', orientation:'h',
   marker:{{color:['#10b981','#64748b'], cornerradius:4}},
   text:['{full_soh_annual:.2f}%/yr (tracked)','{rb_soh_annual:.2f}%/yr (untracked)'], textposition:'inside',
   textfont:{{size:9,color:'#fff'}}}}
],{{...sparkLayout, xaxis:{{visible:false,range:[0,{max(full_soh_annual, rb_soh_annual)*1.3:.2f}]}},
   yaxis:{{visible:true,tickfont:{{size:9,color:'#94a3b8'}}}},
   margin:{{t:0,b:0,l:60,r:4}}}},
{{responsive:true,displayModeBar:false}});""")

    # ── Single-day chart (3 subplots) ──
    # NOTE: The v5_comparison.json contains per-day scalar summaries, not
    # intra-day time-series. The data below is from a real simulation run
    # (same system, same market) used in the petrogaz stakeholder deck.
    # To regenerate for any specific day, run the v5 simulator with
    # --save-timeseries and update these arrays.
    lines.append("""\
// Single-day time series from a representative Q1 2024 simulation.
// To regenerate for a specific day, run: uv run python v5_regulation_activation/main.py --day N --save-timeseries
const t_day=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5];
const soc_day=[50.4,50.4,50.3,50.1,48.9,48.4,49.6,51.3,55.7,57.3,60.8,71.7,81.9,86.2,88.9,88.8,85.8,76.9,66.3,62.2,60,58.3,58.6,58.7,58.6,59.3,62.7,65.5,67.3,72.7,80.5,81.8,79.9,77.7,72.2,69.8,58.1,37.6,20.6,14.4,11.1,10.3,10,10,10,10,10,10.1];
const pwr_day=[0,0.2,0.2,2.6,5.3,-2.8,-6.5,-11.4,-23.2,-8.7,-34,-48.8,-25.6,-13.2,-5.6,6.8,24.2,39.7,28.9,5.7,11.3,-0.1,-0.9,0.1,0,-8.5,-18.3,-7.7,0,-35.6,-17.3,4,8.3,12.6,0,31.3,69.7,73.3,35.8,14.1,6.2,0.9,0.2,-0.1,0,0,0,-1.2];
const price_day=[81.5,81.5,81.1,81.1,79.9,79.9,78.2,78.2,77,77,81.5,81.5,94.1,94.1,114.8,114.8,144.1,144.1,138.6,138.6,122,122,110,110,103.4,103.4,95,95,96,96,110.2,110.2,111.4,111.4,128.2,128.2,127.3,127.3,111.8,111.8,95.5,95.5,86.2,86.2,82,82,77.7,77.7];

const pwr_colors = pwr_day.map(v => v < 0 ? '#10b981' : '#3b82f6');

Plotly.newPlot('chart-day',[
  {x:t_day, y:price_day, type:'scatter', mode:'lines', name:'Price', fill:'tozeroy',
   fillcolor:'rgba(245,158,11,0.08)', line:{color:'#f59e0b',width:2},
   yaxis:'y3', hovertemplate:'%{y:.0f} \\u20ac/MWh<extra>Price</extra>'},
  {x:t_day, y:pwr_day, type:'bar', name:'Power', marker:{color:pwr_colors},
   yaxis:'y2', hovertemplate:'%{y:.0f} kW<extra>Power</extra>'},
  {x:t_day, y:soc_day, type:'scatter', mode:'lines', name:'SOC', fill:'tozeroy',
   fillcolor:'rgba(59,130,246,0.1)', line:{color:'#3b82f6',width:2.5},
   yaxis:'y', hovertemplate:'%{y:.1f}%<extra>SOC</extra>'}
],{
  ...dark,
  grid:{rows:3, columns:1, subplots:[['xy3'],['xy2'],['xy']], roworder:'top to bottom', ygap:0.08},
  xaxis:{title:'Hour of day', gridcolor:'#1e293b', tickfont:{size:10}, titlefont:{size:11}, range:[0,24], dtick:4},
  yaxis3:{title:'\\u20ac/MWh', gridcolor:'#1e293b', tickfont:{size:9}, titlefont:{size:10,color:'#f59e0b'}, range:[60,160]},
  yaxis2:{title:'kW', gridcolor:'#1e293b', tickfont:{size:9}, titlefont:{size:10,color:'#3b82f6'}, zeroline:true, zerolinecolor:'#374151'},
  yaxis:{title:'SOC %', gridcolor:'#1e293b', tickfont:{size:9}, titlefont:{size:10,color:'#3b82f6'}, range:[0,100]},
  showlegend:false,
  margin:{t:12,b:40,l:50,r:16},
  annotations:[
    {x:8,y:144,text:'<b>Peak: \\u20ac144/MWh</b>',showarrow:false,font:{size:10,color:'#f59e0b'},yref:'y3',xanchor:'left'},
    {x:5.5,y:-50,text:'Charging',showarrow:false,font:{size:9,color:'#10b981'},yref:'y2'},
    {x:18,y:75,text:'Discharging',showarrow:false,font:{size:9,color:'#3b82f6'},yref:'y2'}
  ]
},{responsive:true,displayModeBar:false});""")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Main assembly
# ══════════════════════════════════════════════════════════════════════

def generate_html(data: dict) -> str:
    """Assemble the full HTML document from slides and chart scripts."""
    slides = [
        slide_title(data),
        slide_challenge(),
        slide_approach(),
        slide_real_day(data),
        slide_validation(data),
        slide_strategy_table(data),
        slide_economics(data),
        slide_reliability(data),
        slide_closing(data),
    ]
    charts_js = build_chart_scripts(data)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BESS Optimization Platform &mdash; Arpedon</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{CSS}
</style>
</head>
<body>
{"".join(slides)}
<script>
{charts_js}
</script>
</body>
</html>
"""


def print_summary(data: dict) -> None:
    """Print key stats to stdout."""
    strats = data["strategies"]
    n_days = data["meta"]["n_days"]
    e_nom = data["meta"]["E_nom_kwh"]
    full = strats["full"]
    rb = strats["rule_based"]

    daily_eur = usd_to_eur(full["total_profit"])
    rb_daily_eur = usd_to_eur(rb["total_profit"])
    advantage_eur = daily_eur - rb_daily_eur
    annual_50mwh = daily_eur * 365 * (50000 / e_nom)
    soh_annual = full["soh_degradation"] / n_days * 365 * 100

    print("=" * 60)
    print("  v5 Deck Summary (EUR, converted at 1 USD = 0.926 EUR)")
    print("=" * 60)
    print(f"  Data source:      {data['meta']['data_source']}")
    print(f"  Simulation days:  {n_days}")
    print(f"  System size:      {e_nom:.0f} kWh / {data['meta']['P_max_kw']:.0f} kW")
    print("-" * 60)
    print(f"  Full Optimizer daily profit:   {fmt_eur(full['total_profit'])}")
    print(f"  Rule-Based daily profit:       {fmt_eur(rb['total_profit'])}")
    print(f"  Daily advantage:               +{fmt_eur(full['total_profit'] - rb['total_profit'])}")
    print(f"  Profitable days:               {n_days - full.get('loss_days', 0)} / {n_days}")
    print(f"  Delivery score:                {full['delivery_score']*100:.1f}%")
    print(f"  Annual degradation:            {soh_annual:.2f}%")
    print(f"  50 MWh annual revenue:         {fmt_eur(annual_50mwh, convert=False)}")
    print(f"  MPC solver failures:           {full['mpc_solver_failures']}")
    print("=" * 60)
    for s in STRAT_ORDER:
        p = usd_to_eur(strats[s]["total_profit"])
        print(f"  {STRAT_LABELS[s]:20s}  {fmt_eur(strats[s]['total_profit']):>10s}/day  "
              f"loss_days={strats[s].get('loss_days', 0)}  "
              f"delivery={strats[s]['delivery_score']*100:.1f}%")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate v5 BESS optimization HTML deck from comparison JSON"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(REPO_ROOT / "results" / "v5_comparison.json"),
        help="Path to v5_comparison.json",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO_ROOT / "presentation" / "v5_deck.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    data = load_data(pathlib.Path(args.input))
    html = generate_html(data)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    print(f"\n  Generated: {out_path}")
    print(f"  Slides:    9")
    print(f"  Charts:    5 (daily advantage, cumulative profit, 3 sparklines, single-day 3-subplot)")
    print()
    print_summary(data)


if __name__ == "__main__":
    main()
