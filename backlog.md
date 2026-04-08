# Backlog — Active Issues & Future Work

> **Frozen versions (v1–v4)** live in [archive/](archive/). They are not part
> of active development and should not be modified.
>
> **Historical gate reports** for v1–v5 are in [archive/gate_reports.md](archive/gate_reports.md).

---

## Active issues

### 0. v5 simulator: hard audit findings (2026-04-06)

A strict audit of v5 simulator + strategies turned up four real bugs in the
execution layer between the optimizers and the plant. **The relative ranking
across LP / EMS_PI / FULL / FULL_ECON is probably still meaningful** because
all suffer the bugs in roughly equal proportion, but **absolute numbers and
the rule_based baseline are wrong**.

**Audits that PASSED:**
- Information leak — planners only see forecast scenarios, never realized prices
- EKF observability — uses noisy measurements, not true state
- Multi-rate cadence — correct array lengths
- Per-strategy semantic checks — rule_based has P_reg=0 etc.

**Bug A — Wash trades from PI activation modulation.**
The PI controller in `pi/regulation_controller.py` only nets P_chg/P_dis when
their sum exceeds P_max. Otherwise both stay nonzero. The plant then computes
`dSOC = (eta_c*P_chg - P_dis/eta_d)/E_eff`, applying round-trip losses to BOTH
directions independently. Energy silently leaks to the wash trade.
*Example*: MPC plans `P_chg=21.7, P_dis=0`. Activation demands `P_dis=10`. PI
sets `[21.7, 10, P_reg]`. Plant computes `(0.95*21.7 - 10/0.95)/E_eff = 10.1/E_eff`
instead of the correct net `(0.95*11.7)/E_eff = 11.1/E_eff` — ~9% efficiency
loss on the wash portion.
*Affected*: every PI-using strategy and the EMS_CLAMPS open-loop path.

**Bug B — Power budget violations.**
The PI controller adds activation modulation on top of MPC chg/dis without
rechecking `P_chg + P_reg <= P_max` or `P_dis + P_reg <= P_max`. The plant
doesn't enforce the budget either — it clips each input to [0, P_max]
independently. The battery is silently commanded above its rated power.
*Counts on day 3*: deterministic_lp 6845 chg+reg violations / 5177 dis+reg;
EMS_CLAMPS 6958 / 5240; EMS_PI 9216 / 10247; FULL 13350 / 14639;
FULL_ECON 8363 / 9877. `P_total` used for thermal Joule heating routinely
exceeds 100 kW, inflating temperature and degradation.

**Bug C — `power_applied` records commands, not actuals.** [Critical]
The plant has pre-integration limiting (it clips P_chg/P_dis to SOC headroom),
but `MultiRateSimulator` records the unclipped commanded values into
`power_applied`. All downstream accounting bills strategies for power that
the battery never accepted.
*Smoking gun*: rule_based hour 4 commands `P_chg=80 kW` continuously. Predicted
SOC end = 1.264 (impossible). Actual = 0.900 (clamped). Strategy is credited
with ~80 kWh of cheap-price charging that physically never happened.
*Severity*: rule_based severe (SOC error 0.375); LP/EMS/MPC mild (~0.0025).

**Bug D — Multi-cell pack scaling vs alpha_deg calibration.**
`alpha_deg` was calibrated for "1 FCE/day → 1.37 %/yr at the PACK level", but
the multi-cell `BatteryPack` distributes pack power equally across `n_cells`
cells, so each cell sees `P_pack/n`. The pack reports MIN(cell_SOH). Net
effect: pack SOH/yr is reported at ~3.6× lower than the calibration target.
*Treatment*: kept as a feature per user — model mismatch is real, MPC should
plan around it. Relative SOH ranking across strategies is unchanged. Absolute
SOH numbers should be flagged in any external deck.

**Bug E — alpha_deg planner asymmetry across strategies (post-refactor, 2026-04-07).**
The planners are not consistent with each other on the degradation cost they
optimize against, and none of them match the ledger:
- `deterministic_lp` ([core/planners/deterministic_lp.py:143-144](core/planners/deterministic_lp.py#L143-L144))
  uses **hard** `bp.alpha_deg` (no `n_modules` division) → matches the ledger.
- `EconomicEMS` (used by `ems_clamps`, `ems_pi`) builds the CasADi 3-state
  integrator at [core/planners/stochastic_ems.py:76](core/planners/stochastic_ems.py#L76),
  which divides by `n_modules` at [core/physics/plant.py:203](core/physics/plant.py#L203)
  → plans against **soft** rate (~1/n).
- `tracking_mpc` ([core/mpc/tracking.py:58](core/mpc/tracking.py#L58)) and
  `economic_mpc` ([core/mpc/economic.py:141](core/mpc/economic.py#L141)) both
  consume the same CasADi model → plan against **soft** rate.
- `core/accounting/ledger.py:102` bills all strategies at **hard** rate
  (`bp.alpha_deg * |p_net|`, no `n_modules` division).

*Effect*: every EMS/MPC strategy sees an internal degradation cost ~4× too low
and is structurally incentivized to over-cycle, while LP and rule_based pay an
honest internal cost. **This biases the strategy progression analysis** —
"economic_mpc beats LP on profit" cannot be cleanly attributed to better
control vs. cheating on the internal cost. Must be resolved (or explicitly
neutralized) before any pitch claim about MPC superiority is defensible.

**Concern F — PI recovery bias contaminates MPC plans.**
PI controller at [core/pi/regulation.py:94-100](core/pi/regulation.py#L94-L100)
applies a pure-proportional pull toward `SOC_terminal` whenever the activation
signal is idle (`|activation| < 1e-6`):
`p_recovery = recovery_gain * (SOC_terminal - SOC) * P_max`. This is summed
into the same `p_net` channel as the MPC setpoint at line 103, so during quiet
windows PI **overrides MPC's economic SOC arc** and flattens it toward
`SOC_terminal`. PI also has no integral, no derivative, no rate limiting.
*Effect on the comparison*: PI-using strategies (`ems_pi`, `tracking_mpc`,
`economic_mpc`) get a hidden SOC-flattener that open-loop strategies
(`rule_based`, `deterministic_lp`, `ems_clamps`) do not. This is the
"PI roughs up the smooth MPC plan" effect — it both helps PI strategies on
SOC-safety metrics and hurts them on economic metrics in ways neither side
of the comparison can see.
*Treatment*: add a `pi_enabled: bool` flag to `Strategy` so each MPC strategy
can be run with PI on vs. off. The counterfactual isolates PI's contribution.

**Concern G — open-loop dispatch lacks PI's SOC safety clamp.**
[core/simulator/core.py:56-75](core/simulator/core.py#L56-L75) `_open_loop_dispatch`
does not replicate the SOC safety scaling at
[core/pi/regulation.py:122-141](core/pi/regulation.py#L122-L141). Open-loop
strategies (`rule_based`, `deterministic_lp`, `ems_clamps`) can deliver
regulation demand at unsafe SOC, while PI-using strategies cannot. Minor
asymmetry, but biases SOC-violation counts in the comparison.

**Concern H — current simulation regime is too benign for MPC differentiation
(2026-04-07).** Diagnosis from the post-refactor 7-day comparison
([results/v5_comparison.json](results/v5_comparison.json)): on a 200 kWh /
100 kW battery against Q1 2024 German FCR + day-ahead, the realized FCR
activation signal (OU process σ=18 mHz, [core/markets/activation.py](core/markets/activation.py))
demands only ~2-3 kW typical and ~26 kW peak — out of 100 kW available.
Result: penalty cost = $0.00 for every strategy, delivery score = 100% for
every strategy, capacity revenue dominates (~94% of profit), and LP and
economic_mpc tie at $41.34 vs $41.71/day (+0.9%). The experiment cannot
demonstrate MPC value as configured — there is no infeasibility to recover
from, and the FCR bid is determined by the endurance constraint, not by
optimization. *Treatment*: requires a stress-test regime (volatile day
selection, doubled activation σ, smaller battery, or tighter SOC bounds)
to surface the differentiation. Tracked as the "big experiment" that should
include calm vs volatile vs stressed-activation subsets.

**MPC pipeline audit findings (2026-04-07).** Investigating why the big
experiment showed `ems_pi` beating both `tracking_mpc` and `economic_mpc` by
$0.20–$0.70/day across calm/volatile/stressed regimes. End-to-end audit of
[core/mpc/economic.py](core/mpc/economic.py),
[core/mpc/tracking.py](core/mpc/tracking.py),
[core/mpc/adapters.py](core/mpc/adapters.py),
[core/simulator/core.py](core/simulator/core.py),
[core/planners/stochastic_ems.py](core/planners/stochastic_ems.py).
**Diagnosis: the MPC layer is not solving a problem distinct from the EMS
planner — it is a redundant copy at faster cadence with strong anchor
penalties pinning it to the EMS plan.** The hierarchical EMS→MPC→PI design
is not realised in code. Findings:

- **F1 — Capacity revenue is missing from both MPC cost functions**
  ([economic.py:156-224](core/mpc/economic.py#L156-L224),
  [tracking.py:162-206](core/mpc/tracking.py#L162-L206)).
  In a hierarchical design this is *correct* (P_reg is locked by EMS
  commitment, so capacity revenue is constant w.r.t. MPC decisions), but
  it must be paired with a **delivery feasibility constraint or penalty**
  to give MPC a reason to protect the SOC headroom needed to deliver the
  commitment. That constraint is missing — see F4.

- **F2 — TrackingMPC's P_reg is a dead variable**
  ([tracking.py:136](core/mpc/tracking.py#L136),
  [tracking.py:212](core/mpc/tracking.py#L212),
  [adapters.py:99](core/mpc/adapters.py#L99)).
  `P_reg = opti.variable(Nc)` with bounds `0 <= P_reg <= 0.3*P_max` (an
  arbitrary fraction unrelated to the EMS commitment), tracked toward the
  EMS reference via `R_power * (P_reg - p_reg_ref)^2`, then **the solved
  value is discarded by the adapter** which substitutes the EMS reference.
  Pure dead weight that consumes solver state and distorts neighbouring
  variables via the rate penalty `dP_reg^2` and the SOC dynamics.

- **F3 — Anchor penalties iron-rod the MPC to the EMS plan**
  (`Q_soc_anchor = 10`, `Q_terminal_econ = 1000`,
  [parameters.py:264-280](core/config/parameters.py#L264-L280)).
  Per-step quadratic penalty plus a strong terminal pin together leave
  the MPC essentially no degrees of freedom to deviate from the EMS
  trajectory over a 1-hour horizon. **The MPC is architecturally
  incapable of beating the EMS plan**; the best it can do is exactly
  match it, and any solver noise pushes it slightly worse. This matches
  the empirical observation: MPC strategies are consistently $0.20–$0.70/day
  *worse* than `ems_pi`, and the gap correlates with how much the MPC
  perturbs the dispatch (see TV(P_net) metrics).

- **F4 — EconomicMPC's activation forecast is decoupled from any cost
  incentive** ([economic.py:166-179](core/mpc/economic.py#L166-L179)).
  `recent_activation` decays exponentially over the horizon and is fed
  into the SOC dynamics as a predicted P_reg dispatch, but **the cost
  function does not penalise failing to deliver the committed P_reg**.
  So the MPC's response to "activation will happen" is just a SOC nudge
  competing against the F3 anchor penalty — and the anchor wins. The
  claimed "activation-aware OU forecasting" pitch differentiator
  ([strategies/economic_mpc/strategy.py:9-12](strategies/economic_mpc/strategy.py#L9-L12))
  is implemented in the dynamics but not in the objective.

- **F5 — Solver-failure fallback path is unsafe**
  ([economic.py:383-395](core/mpc/economic.py#L383-L395),
  [tracking.py:364-367](core/mpc/tracking.py#L364-L367)). EconomicMPC
  falls back to TrackingMPC; TrackingMPC's own failure path returns
  `u_cmd = np.zeros(3)`. The simulator passes those zeros to PI without
  detecting the failure or reverting to the EMS plan. We got lucky on
  this run (`mpc_solver_failures = 0` across all 120 sims), but it is a
  landmine for any future regime that stresses the solver.

- **F6 — `soc_anchor` is computed and discarded**
  ([core/simulator/core.py:193](core/simulator/core.py#L193)).
  The simulator computes `plan.soc_anchor_at(...)` and records it in
  traces for instrumentation, but **does not pass it to
  `solve_setpoint`**. The MPC instead reconstructs its own SOC reference
  window from `plan.soc_ref_hourly`. Functionally fine, but a tell that
  the call site was rewritten without cleaning up the dead computation.

**Empirical confirmation**: in the big experiment, `ems_pi` beats both
MPCs by $0.20–$0.70/day across all regimes; `tracking_mpc` is the only
strategy that breaks under activation stress (280 P_max touches/day,
98.3% delivery, 42% faster SOH degradation). Both findings are predicted
by F1+F3+F4: the MPCs are not protecting deliverability and have no room
to do anything genuinely smarter than the EMS plan.

**Architectural implication.** The current MPC is solving the same
problem as the EMS at finer cadence and is therefore strictly redundant.
For the EMS+MPC hierarchy to be defensible, the MPC must have a
**distinct job**: ensure delivery feasibility and short-term constraint
satisfaction *as state realises*, given EMS-locked commitments. That
requires (in priority order):
  (a) A soft delivery-feasibility penalty in the MPC cost: penalise SOC
      trajectories where the committed `P_reg` cannot be served under
      the activation forecast.
  (b) Loosen `Q_soc_anchor` (10 → ~1) and `Q_terminal_econ` (1000 → ~100)
      so the MPC has room to deviate when an opportunity or threat
      justifies it.
  (c) Remove `P_reg` as a decision variable from `TrackingMPC`; treat it
      as a parameter equal to the EMS commitment (mirror EconomicMPC).
  (d) Add an EMS-plan fallback in the simulator on solver failure
      instead of accepting `np.zeros(3)`.
  (e) Verify (and add if missing) hard thermal and voltage envelope
      constraints in MPC — these are things the EMS does *not* model,
      and they would be one of the few things MPC genuinely contributes
      that EMS cannot. If they are missing, MPC has even less reason
      to exist than the audit already shows.

**Pitch implication.** The honest framing for v5 going forward is
*EMS = profit engine, MPC = reliability/feasibility engine, PI = tracker*.
MPC's value should be measured on delivery score, constraint
satisfaction, SOH preservation under stress, and recovery from
disturbances — **not** on profit improvement over `ems_pi` in benign
regimes. None of the v5 deck claims should rest on "MPC squeezes more
profit"; they should rest on "MPC ensures the EMS-committed revenue is
actually realised despite real-world disturbance."

---

**Simulator actuation realism audit (2026-04-15).** Triggered by the
observation that `deterministic_lp` (no PI, no MPC, no controller of any
kind) hits 100% delivery score in the big experiment. End-to-end audit
of [core/simulator/core.py](core/simulator/core.py),
[core/pi/regulation.py](core/pi/regulation.py),
[core/mpc/economic.py](core/mpc/economic.py),
[core/physics/plant.py](core/physics/plant.py),
[core/estimators/ekf.py](core/estimators/ekf.py),
[core/markets/activation.py](core/markets/activation.py).

**Diagnosis: the simulator hands strategies the raw, ground-truth,
sub-second activation signal with zero latency and zero noise.** Every
delivery-related result is on a cheating baseline.

The cheat is concentrated in exactly one channel and one operation:

- **`activation[k]` is read raw at every PI step (~4 s).**
  - [core/simulator/core.py:207](core/simulator/core.py#L207)
    `activation_k = float(activation[k])`
  - Passed unchanged to `_open_loop_dispatch` for `rule_based`,
    `deterministic_lp`, `ems_clamps` ([core.py:216-219](core/simulator/core.py#L216-L219))
  - Passed unchanged to `RegulationController.compute(activation_signal=activation_k, ...)`
    for `ems_pi`, `tracking_mpc`, `economic_mpc` ([core.py:209-214](core/simulator/core.py#L209-L214))
  - Used inside `_open_loop_dispatch` as `p_reg_demand = activation * p_reg_committed`
    ([core.py:71](core/simulator/core.py#L71))
  - Used inside `RegulationController.compute` as `p_reg_demand = activation_signal * p_reg_committed`
    ([regulation.py:89](core/pi/regulation.py#L89))
- **EconomicMPC additionally sees `activation[k-1]` at every MPC tick**
  ([core.py:177](core/simulator/core.py#L177)), passed as
  `recent_activation` to `solve_setpoint` and used in the OU persistence
  forecast at [economic.py:165-168](core/mpc/economic.py#L165-L168).

**This means a "dumb" LP strategy with zero feedback control achieves
perfect delivery tracking** because the strategy layer is literally
multiplying the ground-truth activation sample by the committed P_reg
and handing it to the plant, which integrates it without any actuator
dynamics. There is no plant-internal regulation controller — *all*
activation tracking is done at the strategy layer, against perfect
information.

**What is NOT a cheat (verified by the audit, do not change):**
- `plant.get_state()` (true 5-state) is called only for initial logging
  and traces; no strategy consumes it
- `plant.get_measurement()` adds Gaussian noise (SOC ±0.01, T ±0.5°C,
  V ±1.0 V) — verified at [plant.py:618-620](core/physics/plant.py#L618-L620)
- EKF receives noisy measurements and produces lagged estimates;
  state_est is meaningfully different from ground truth
- MPC and PI consume `state_est` (correct), not `plant.get_state()`
  (would be a cheat)
- `p_reg_committed` is held constant hourly (correct commitment model)
- 5-scenario forecast with realized day held out (correct uncertainty
  model)
- 5-state plant ODE with Arrhenius thermal coupling and RC transients
  is realistic

**Secondary cheat (smaller, lives in the same fix):**
The plant has **no actuator lag, no rate limit, no first-order-hold** on
commanded power ([plant.py:516-602](core/physics/plant.py#L516-L602)).
Commanded power is integrated directly into SOC. Real inverter dynamics
are 10-50 ms; at the 4 s PI sample rate this barely matters but it is
physically incorrect.

**Architectural root cause.** On real hardware the layering is:
- **Strategy layer** (hours/minutes): commits to markets, schedules energy
- **Battery controller** (sub-second): tracks activation given the
  commitment, respects constraints, delivers power
- **Power electronics** (ms): translates commanded power to realized power

The current simulator collapses all three jobs into the strategy layer.
The fix is to push the "track activation given committed capacity" job
into the plant where it physically belongs. Then strategies cannot cheat
on information they never receive.

**Implications for empirical findings.** All previous strategy-comparison
results that involve delivery score are on a cheating baseline:
- The 100% delivery score across all strategies (all subsets) is an
  artifact of perfect feedforward, not a meritorious result
- The "$0.04/day PI vs no-PI" gap on MPC strategies is also on a
  cheating baseline — both with and without PI, all strategies are
  doing perfect feedforward in the activation channel
- The `tracking_mpc` 98.3% delivery in the stressed regime is the
  *only* delivery-side signal that survives the cheat, because
  tracking_mpc breaks for a different reason (its dead `P_reg`
  variable distorts dispatch enough to occasionally miss the target)
- **Profit numbers are still meaningful** for the energy arbitrage and
  capacity revenue axes (no cheat there), but profit comparisons that
  factor in delivery quality should be re-run after RF1.

**Treatment: Realism Fix 1 (RF1).** See
[docs/realism_fix_1_design.md](docs/realism_fix_1_design.md) (TBD next
session). Push activation tracking into `BatteryPlant.step()`. Strategy
layer outputs `(p_net_setpoint, p_reg_committed)`; plant internally
computes `p_total = p_net_setpoint + activation_k × p_reg_committed`,
applies clipping, integrates. Strategies never see `activation[k]`.
PI controller's activation-tracking code path is deleted (PI's job
shrinks to "SOC safety clamp + recovery bias"). MPC's `recent_activation`
hint is removed; MPC plans against the *expected* activation
(`expected_activation_frac`, already exists in EMSParams) and gains its
edge from the EKF state estimate + scenario picker (Wow Factor 1 step D),
not from peeking at the grid signal. Estimated scope: ~150 LOC across
4 files. **Sequence: RF1 lands before Wow Factor 1 resumes**, because
without RF1 the WF1 headline number cannot be defended on a
delivery-quality axis.

---

**EMS planner audit findings (2026-04-07).** End-to-end audit of
[core/planners/stochastic_ems.py](core/planners/stochastic_ems.py),
[core/planners/deterministic_lp.py](core/planners/deterministic_lp.py),
[core/simulator/strategy.py](core/simulator/strategy.py) (Plan dataclass),
and [core/markets/price_loader.py](core/markets/price_loader.py). The EMS
is genuinely a two-stage stochastic NLP, **not** "deterministic LP in
disguise" — but the way it hands its plan to MPC throws away most of the
stochastic structure. Findings:

- **What is correct.** Real two-stage SP with non-anticipativity at hour 0
  ([stochastic_ems.py:247-249](core/planners/stochastic_ems.py#L247-L249));
  per-scenario second-stage decisions for hours 1–24; capacity revenue,
  energy arbitrage, degradation, terminal SOC penalty, terminal SOH
  penalty, soft endurance constraint, soft SOC bounds, thermal/SOH
  bounds, power budget all in the formulation; full 3-state feedback
  (SOC, SOH, T); CasADi NLP via IPOPT, ~1.4s/solve, 24 solves/day;
  no information leak (realized prices held out cleanly via
  [price_loader.py:145-146](core/markets/price_loader.py#L145-L146));
  empirical $0.10–$0.63/day edge over `deterministic_lp` is real and
  attributable to the per-scenario hedged second-stage decisions.

- **EMS-A — Plan returns averages, throwing away the recourse channel.**
  [stochastic_ems.py:294-301](core/planners/stochastic_ems.py#L294-L301)
  returns probability-weighted averages of `P_chg / P_dis / P_reg / SOC /
  SOH / TEMP / VRC1 / VRC2` across scenarios. **The MPC and PI layers see
  a fictional smoothed plan that does not correspond to any specific
  scenario.** The whole point of stochastic optimization — "do A under s1,
  do B under s2" — is averaged away into a third action that is optimal
  for neither scenario. The MPC then anchors itself heavily to this
  averaged plan (per the MPC audit, F3) and inherits the averaging loss
  on top of the anchor loss. **This is the architectural mismatch that
  collapses the EMS+MPC hierarchy: EMS does real stochastic optimization,
  but immediately downgrades its own output to "deterministic average"
  before handoff.** Combined with MPC F3 (anchor weights pin MPC to the
  plan) and MPC F4 (no deliverability cost in MPC), the entire hierarchy
  is, *effectively*, deterministic at runtime — just expensive
  deterministic.

- **EMS-B — No delivery-failure penalty in EMS cost.** Same omission as
  the MPC audit (F1). EMS has capacity revenue ($+$) but no cost for
  failing to deliver committed P_reg. The endurance constraint
  ([stochastic_ems.py:192-205](core/planners/stochastic_ems.py#L192-L205))
  is the only safeguard, and it is *soft* (slack penalty 1e5). EMS will
  routinely commit P_reg = P_max because the only thing pulling it down
  is a constraint that does not bind in benign regimes.

- **EMS-C — `endurance_hours = 0.5` is very lenient.** The endurance
  constraint requires SOC headroom for **30 minutes** of full P_reg
  activation. On a 200 kWh / 100 kW battery this means committing
  P_reg = 100 kW only requires ~50 kWh of headroom = 25% of capacity.
  Combined with the soft formulation, **EMS commits ≈ P_max at every
  hour with no real bidding decision to make**. This is consistent with
  the empirical observation that all strategies bid identical P_reg,
  and with the diagnosis that capacity revenue is essentially constant
  across strategies. Real European FCR endurance is 0.25–1.0 h; tightening
  it would force EMS to make real per-hour bidding tradeoffs.

- **EMS Plan only commits hour 0.** The first-stage decision is just
  `P_chg[s][0], P_dis[s][0], P_reg[s][0]` — i.e. only the *current* hour.
  Hours 1–24 are contingent (per-scenario). EMS re-solves every hour
  with updated state, so the contingent plans only have to be roughly
  right for a few hours forward. This is the right design.

---

**Wow Factor 1 — Hierarchical scenario-aware recourse (proposed v5.5
upgrade, 2026-04-07).** This is the proposed next-step deliverable that
addresses both the MPC audit findings (F1-F4) and the EMS audit findings
(EMS-A, EMS-B, EMS-C) in a single architectural change. It also gives
the v5 pitch a real wow factor that maps to a real published academic
concept (multi-stage stochastic programming with policy decomposition,
SDDP family).

**Architectural concept.** EMS already does real stochastic optimization
(per the audit). The change is that EMS hands MPC the **per-scenario
contingent plans**, not the average. MPC then performs **scenario
selection / belief tracking**: at each tick, MPC compares the realized
state (from EKF) against where each EMS scenario expected the state to
be at that time, and picks (or weights) the closest scenario. MPC then
anchors itself to the chosen scenario's plan rather than the averaged
plan. Combined with a deliverability penalty in MPC's cost, this gives
MPC a structural job that is *distinct* from EMS — EMS plans, MPC
*identifies which plan we're actually living* and adapts dispatch to it.

**Distinction from stochastic MPC (v10).** Stochastic MPC means MPC
itself solves an N-scenario stochastic program at every tick. Wow Factor
1 means MPC stays a single-scenario deterministic NLP, but it *selects*
which EMS scenario to track based on belief over the EMS scenario set.
EMS does the stochastic reasoning; MPC does the recourse execution.
This is **strictly cheaper at runtime** than v10 (one NLP per tick
instead of N), **piggybacks on existing EMS work**, and is **compatible
with v10** — when v10 is later added, you get the gold-standard
two-level stochastic stack (outer EMS scenario tree + inner MPC scenario
tree).

**Theoretical lineage.** Multi-stage stochastic programming, scenario
tree decomposition, Stochastic Dual Dynamic Programming (Pereira, Pinto
1991), policy decomposition in hierarchical control. Belief tracking
draws from interacting multiple-model (IMM) filters and HMM scenario
tracking.

**Pitch positioning.** "Stochastic optimization at every layer,
deterministic-cost real-time dispatch, scenario-aware recourse via
belief tracking." Distinctive, defensible, novel-in-product even though
each piece has academic precedent, and runnable in real-time on
commodity hardware.

**Minimum viable implementation (~120 LOC, 3 files):**

1. **Plan dataclass + EMS return** (~30 LOC,
   [core/simulator/strategy.py](core/simulator/strategy.py),
   [stochastic_ems.py:294-301](core/planners/stochastic_ems.py#L294-L301)).
   Plan grows from `(p_net_hourly, p_reg_hourly, soc_ref_hourly, ...)`
   averages to `(scenarios: list[ScenarioPlan], probabilities: ndarray)`
   where each `ScenarioPlan` carries the per-scenario `P_chg / P_dis /
   P_reg / SOC` trajectory. The hour-0 commitment is identical across
   scenarios (non-anticipativity guarantees this).

2. **MPC scenario picker** (~50 LOC,
   [core/mpc/economic.py](core/mpc/economic.py),
   [core/mpc/adapters.py](core/mpc/adapters.py)). At each MPC tick:
   - Compute residual `r_s = ||state_est - scenario_s.state[current_hour]||`
     across scenarios (Mahalanobis or simple weighted L2).
   - Pick `s* = argmin_s r_s` (max-likelihood) OR maintain a softmax
     weight `w_s ∝ exp(-r_s² / 2σ²)` and use `Σ w_s × scenario_s.plan`
     as the anchor (mixture / Bayesian update).
   - Anchor MPC's `Q_soc_anchor` and `Q_terminal_econ` toward the chosen
     scenario's SOC trajectory rather than the averaged one.
   - Start simple: max-likelihood selection. Add mixture later if needed.

3. **MPC deliverability penalty** (~20 LOC,
   [core/mpc/economic.py](core/mpc/economic.py)). New cost term:
   `w_deliver × max(0, P_reg_committed - P_reg_deliverable_under_activation_forecast)²`
   where `P_reg_deliverable` is computed from the SOC trajectory and the
   activation forecast `recent_activation × OU_decay`. This is the
   missing F1+F4 cost term.

4. **MPC loosen anchor weights** (~2 LOC,
   [core/config/parameters.py](core/config/parameters.py)). `Q_soc_anchor: 10 → 1`,
   `Q_terminal_econ: 1000 → 100`. Once the deliverability penalty (#3)
   gives MPC a *real* reason to deviate from the EMS plan when threats
   appear, the anchor weights should be loose enough to allow it.

5. **EMS delivery-failure penalty** (~10 LOC,
   [stochastic_ems.py](core/planners/stochastic_ems.py)). Symmetric to
   #3: penalize SOC trajectories that cannot serve P_reg under expected
   activation. Forces EMS to make real per-hour bidding decisions
   instead of always committing P_max.

6. **Optionally: tighten `endurance_hours` 0.5 → 0.25** (~1 LOC,
   [core/config/parameters.py](core/config/parameters.py)). Closer to
   real ENTSO-E spec, makes the endurance constraint actually bind.

**Test plan after implementation.**
Re-run the big experiment (3 subsets × 8 strategies × 5 days). Expected
findings if the diagnosis is correct:
- `economic_mpc` gains $0.30–$1.50/day over `ems_pi` in volatile and
  stressed subsets (the gap that currently goes the wrong way).
- `economic_mpc` delivery score stays at 100% in stressed regime where
  `tracking_mpc` currently breaks at 98.3%.
- `economic_mpc` SOH degradation per day stays at the EMS+PI level
  (~0.0017 %/day) instead of `tracking_mpc`'s broken 0.0026 %/day.
- The wow-factor chart: "MPC value over EMS+PI vs disturbance intensity"
  shows a clean monotone curve. Calm ≈ tied. Volatile = small gap.
  Stressed = significant gap. This is the slide.

**Risk and unknowns.**
- Risk that the empirical gap doesn't open even with the fixes — meaning
  the regime is still too benign for the recourse channel to matter
  (Concern H still partially applies). Mitigation: include a more
  stressed regime (3× sigma, or smaller battery sizing) in the test
  matrix.
- Risk that the scenario-selection residual is degenerate — all
  scenarios produce similar SOC trajectories so `argmin` is noise.
  Mitigation: start with the simplest pick, then look at the residual
  histogram and decide if mixture weighting is needed.
- Risk that loosening anchor weights destabilizes MPC. Mitigation:
  the deliverability penalty replaces the anchor as the binding force,
  so it should be stable; tune incrementally.

**Pre-flight requirement.** Before implementing, write a focused
proposal document that walks through the 6 changes with exact function
signatures, what gets recorded in traces, and a step-by-step roll-out
order that allows reverting at each step. This is the largest
architectural change to the project so far and deserves a real
written design before code.

---

**Concern I — degradation cost calibration is non-physical (2026-04-07).**
With the current parameters (`alpha_deg = 2.6e-11`,
`degradation_cost = $50/SOH-lost`, [core/config/parameters.py](core/config/parameters.py)),
worst-case 24h continuous full-P_max cycling on a 200 kWh / 100 kW battery
costs **~$0.008/day** in degradation per the ledger formula at
[core/accounting/ledger.py:100-104](core/accounting/ledger.py#L100-L104).
Realistic BESS degradation costs are on the order of $50-200/MWh of
throughput → $10-40/day for the same cycling. The current model is
**3-4 orders of magnitude below realistic**. Practical effect: none of the
strategies face any meaningful tradeoff between cycling and wear, which
suppresses any differentiation that would come from degradation-aware
planning. Bug E (alpha_deg planner asymmetry, $0.006/day) is also dwarfed
by this calibration issue, making Bug E itself a footnote rather than a
blocker. *Treatment*: a separate calibration pass — either rescale
`degradation_cost` to a CAPEX-amortization basis (likely $20k-50k for full
SOH range), or document explicitly that the simulation is calibrated for
relative ranking only and absolute degradation $ should be ignored.
**This is also load-bearing for any v9 (degradation-aware MPC) pitch claim.**

**Architectural root cause**
The simulator's `run()` method is a 500+ line monolith with 6 strategy
branches inlined. Bugs A/B/C live in tangled if/else chains nobody can read in
one sitting. The plant's `step()` returns only `(state, measurement)` and
never reports the actually-applied power.
**Fix:** the major refactor planned in
[/home/user/.claude/plans/radiant-crafting-cosmos.md](../../.claude/plans/radiant-crafting-cosmos.md)
— linear simulator core, plant returns u_applied, single signed `P_net`,
strategies as composition recipes, modular `core/` + `strategies/` layout.

---

## Future work

### Upcoming versions (deferred until v5 refactor lands and audits pass)

- **v6** — Unscented Kalman Filter (replace EKF)
- **v7** — Joint state and parameter estimation (online R_internal, capacity, efficiency)
- **v8** — ACADOS NMPC (replace CasADi/IPOPT, RTI, control blocking)
- **v9** — Degradation-aware MPC (SOH in MPC state, profit-vs-degradation tradeoff)
- **v10** — Disturbance forecast uncertainty (scenario-based MPC, chance constraints)
- **v11** — Measurement and communication delays
- **v12** — Multi-battery system with central EMS coordinator
- **v13** — Grid-connected inverter model (id, iq, Vdc dynamics)
- **v14** — Market bidding optimization (day-ahead, reserve, intraday)

### Next session pickup (2026-04-07, end of investigation session)

The big experiment ran (5 days × 3 subsets × 8 strategies, 65 min wall),
results saved to [results/v5_big_experiment.json](results/v5_big_experiment.json)
and traces in [results/v5_big_traces/](results/v5_big_traces/). Headline
finding: **`ems_pi` beats both `tracking_mpc` and `economic_mpc` by
$0.20–$0.70/day across all regimes**, which violated the user's
intuition about MPC and triggered the MPC + EMS code audits. Both audits
are logged above in detail.

**Where to pick up next session:** the proposed deliverable is
**Wow Factor 1 — Hierarchical scenario-aware recourse** (logged above
with full implementation plan). The architectural diagnosis is:
- EMS does real two-stage stochastic optimization but throws away the
  scenario structure by averaging the plan before handoff (EMS-A).
- MPC has no delivery-feasibility cost (F1, F4) and is iron-rod-pinned
  to the (averaged) EMS plan via anchor penalties (F3).
- The result is that the EMS+MPC hierarchy is effectively deterministic
  at runtime, just expensive deterministic.

**The fix is a single architectural change** — propagate per-scenario
plans from EMS to MPC, add a scenario-selection / belief-tracking step
in MPC, add deliverability penalties to both EMS and MPC, loosen MPC's
anchor weights. ~120 LOC across 3 files.

**Pre-flight gate** before implementing Wow Factor 1: write a focused
written design doc that nails down (a) the new Plan dataclass shape,
(b) the exact MPC scenario-picker algorithm with pseudocode, (c) the
deliverability penalty formula and how it composes with existing terms,
(d) a roll-out order that allows reverting at each step. CLAUDE.md
rule #1 (PROPOSE BEFORE IMPLEMENTING) applies — this is the largest
architectural change to the project so far.

**What is already done in this session:**
- ✅ `pi_enabled` flag added to `Strategy` dataclass and to
  `economic_mpc` / `tracking_mpc` recipes — `economic_mpc_no_pi` and
  `tracking_mpc_no_pi` are now first-class strategies (the simulator
  routes them through `_open_loop_dispatch` when `pi_enabled=False`).
- ✅ `--big` mode added to [comparison/run_v5_comparison.py](comparison/run_v5_comparison.py)
  with day-set selection (calm / volatile / stressed-activation),
  non-numeric metrics (smoothness, constraint touches, thermal envelope,
  wall time), single shared worker pool, raw-trace persistence for
  representative strategies.
- ✅ `RegulationParams.sigma_mhz_mult` added to
  [core/config/parameters.py](core/config/parameters.py) and threaded
  through [core/markets/activation.py](core/markets/activation.py) so
  the activation generator's OU σ can be scaled per subset (used for
  the "stressed" subset at 2× nominal).
- ✅ `--strategies` CSV filter flag added to the standard comparison
  harness (used by phase 1 before the big mode existed).
- ✅ MPC pipeline audit logged (F1–F6 with citations).
- ✅ EMS planner audit logged (EMS-A, EMS-B, EMS-C with citations).
- ✅ Wow Factor 1 proposal logged with implementation plan.
- ✅ Bug E investigation: $0.006/day, footnote not blocker.
- ✅ Concerns H (regime too benign) and I (degradation cost calibration
  3-4 OOM off) logged.
- ✅ D1–D5 future-work catalog logged.

**Empirical key numbers from the big experiment:**

| Strategy | Calm | Volatile | Stressed |
|---|---|---|---|
| Rule-Based      |  0.09 |  3.75 |  0.22 |
| Det. LP         | 19.44 | 32.22 | 21.88 |
| EMS clamps      | 19.54 | 32.49 | 22.51 |
| **EMS+PI**      | **19.64** | **32.57** | **22.55** |
| Trk MPC+PI      | 19.33 | 32.05 | 20.79 |
| Trk MPC no-PI   | 19.26 | 32.09 | 20.74 |
| Econ MPC+PI     | 19.37 | 31.86 | 22.14 |
| Econ MPC no-PI  | 19.30 | 31.91 | 22.09 |

`ems_pi` wins everywhere; `tracking_mpc` is the only strategy that
breaks in stressed regime (280 P_max touches/day, 98.3% delivery,
$0.07 penalty cost, 42% faster SOH degradation). PI on/off counterfactual
shows ±$0.05–$0.07 deltas — PI is not the source of the MPC problem.
The MPC layer itself is the problem, per the audit.

### v5 follow-ups (after Wow Factor 1 ships)

- Add aFRR / mFRR revenue streams alongside FCR
- 84-day gate run with cleaned execution layer
- B2B pitch deck regeneration with audit-clean numbers
- Stress test sweep: tighter SOC bounds, smaller battery duration, intraday volatility

### Architectural / pitch-value upgrades (v5.5+)

These were surfaced during the 2026-04-07 strategy-comparison investigation
when the single-market (FCR + day-ahead) comparison failed to differentiate
LP from MPC. They are the most likely places where MPC's *capability*
actually converts into *profit*, because they break the symmetry the current
single-market regime imposes.

- **D1 — Multi-market revenue streams.** Currently we model only FCR
  capacity + delivery + day-ahead arbitrage. Missing and worth adding, in
  rough order of pitch value:
    - **aFRR** (automatic frequency restoration reserve): slower (~5 min),
      separate capacity + activation prices, much higher delivery (real
      energy, not symbolic), MWh-priced delivery typically €100-300/MWh in
      2024. A 100 kW battery doing aFRR for a few hours/day could double
      observed revenue. **Highest priority.**
    - **mFRR** (manual restoration reserve): even slower, manually called.
    - **Intraday continuous (XBID)**: short-term arbitrage on top of
      day-ahead, where MPC's recourse can demonstrably matter.
    - **Imbalance settlement**: exposure to system imbalance pricing.
    - **Behind-the-meter services** (peak shaving, demand-charge management):
      for C&I customers often the largest revenue stream.
    - **Black-start / reactive power**: niche but real.

- **D2 — Multi-market co-optimization.** With more than one revenue stream,
  the optimizer must split P_max across coupled markets at every hour
  (FCR vs aFRR vs energy arbitrage), respecting that they share a single
  battery. This is a stochastic, coupled, multi-stage problem that LP can
  only approximate by mean substitution while a stochastic EMS can solve
  natively. **This is the pitch slot where MPC vs LP becomes a real
  comparison rather than a tie.**

- **D3 — Day-ahead bidding vs real-time dispatch separation.** Today the
  LP/EMS planner makes both decisions simultaneously over the same horizon.
  In real markets they are temporally separated: capacity is bid one day
  ahead and locked in; dispatch happens in real time against realized
  prices and disturbances. A faithful experiment would have a day-ahead
  bidding stage (with imperfect forecast) followed by a real-time dispatch
  stage. **This is exactly where MPC's recourse value materializes** —
  locked-in bids + real-time execution against disturbance is the textbook
  case for receding-horizon control.

- **D4 — Forecast imperfection model.** Currently the forecast is "5 other
  days held out from the same quarter" — an unbiased estimator with no
  systematic bias and no serial correlation. Real forecasting has both.
  Swap in a published-RMSE noise model (or a damped persistence model
  calibrated to EPEX day-ahead forecasting errors) so LP and EMS face the
  forecast-error structure they would face in production. Predicted
  effect: LP's mean-substitution becomes brittle, EMS's stochastic
  formulation degrades gracefully.

- **D5 — Degradation-aware MPC (existing v9 slot).** Add SOH as an MPC
  state with explicit profit-vs-wear tradeoff in the cost. **Load-bearing
  on Concern I** — needs realistic degradation cost calibration first or
  the optimization has nothing to trade against. With realistic deg cost,
  this is a place where MPC structurally cannot be matched by LP because
  the convex non-monotone tradeoff doesn't linearize well.
