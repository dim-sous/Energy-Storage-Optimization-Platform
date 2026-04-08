# Wow Factor 1 — Hierarchical Scenario-Aware Recourse

**Status:** draft — pending user review (2026-04-15)
**Scope:** single-file design doc covering the next v5 deliverable. Intended
as the contract for Phase 2 implementation. Revisions tracked in git.

---

## Goal

Ship a **headline pitch number**:

> Mean net profit advantage of `economic_mpc` over `deterministic_lp`
> on real German Q1 2024 prices, reported per regime (calm / volatile /
> stressed), measured in $/day.

Currently this number is near-zero on calm/volatile and **negative** on
stressed (economic_mpc = $22.14 vs LP = $21.88, but tracking_mpc is
strictly worse). The audit (backlog.md: MPC F1-F6, EMS-A/B/C) found the
cause: the EMS+MPC hierarchy is effectively deterministic at runtime
because EMS averages its stochastic plans before handing them to MPC,
and MPC is iron-rod-pinned to that averaged plan with no
delivery-feasibility cost.

**Success threshold (user-approved):** any positive and *monotone*
advantage across calm → volatile → stressed. Prove the direction first,
tighten the gap later.

---

## Out of scope

- D1-D5 (multi-market, day-ahead/RT split, forecast noise, deg-aware MPC)
- Stochastic MPC (v10) — Wow Factor 1 is the *hierarchical recourse*
  version, not per-tick scenario-tree MPC
- Bug E (alpha_deg planner asymmetry, $0.006/day, dwarfed by Concern I)
- v6+ upgrades (UKF, joint state/param estimation, ACADOS)

---

## Six technical pieces

### 1. New `Plan` dataclass shape

**Current** ([core/simulator/strategy.py](../core/simulator/strategy.py)):
`Plan` holds averaged `p_net_hourly`, `p_reg_hourly`, `soc_ref_hourly`,
etc — each a single 1-D array of shape `(N_hours,)`.

**New:**
```python
@dataclass(frozen=True)
class ScenarioPlan:
    """One scenario's second-stage trajectory from the EMS solve."""
    p_chg:  np.ndarray   # (N_hours,)
    p_dis:  np.ndarray   # (N_hours,)
    p_reg:  np.ndarray   # (N_hours,)  — same across scenarios at hour 0 (non-anticipativity)
    soc:    np.ndarray   # (N_hours+1,)
    soh:    np.ndarray   # (N_hours+1,)
    temp:   np.ndarray   # (N_hours+1,)

@dataclass(frozen=True)
class Plan:
    start_step:    int
    scenarios:     tuple[ScenarioPlan, ...]   # length = n_scenarios
    probabilities: np.ndarray                 # shape (n_scenarios,), sums to 1
    # Averaged fields kept for backward compat with any caller that still
    # reads `p_net_hourly` etc. Computed in __post_init__ as E_s[scenario].
    p_net_hourly:    np.ndarray   # (N_hours,)
    p_reg_hourly:    np.ndarray   # (N_hours,)
    soc_ref_hourly:  np.ndarray   # (N_hours+1,)
```

**Why the backward-compat averages:** the `setpoint_at(k, steps_per_ems)`
and `soc_anchor_at(k, steps_per_ems)` methods currently return
single-scenario ZOH values. Keeping the averaged fields means
`_open_loop_dispatch`, `ems_pi`, `ems_clamps`, and `deterministic_lp`
(all of which just want a single hourly setpoint) keep working without
any change. **Only the MPC strategies read the per-scenario data.**

### 2. EMS return-dict change

**Current** ([core/planners/stochastic_ems.py:294-301](../core/planners/stochastic_ems.py#L294-L301)):
returns `P_chg_ref`, `P_dis_ref`, etc — each a probability-weighted
average via `sol.value(sum(prob[s] * var[s]) for s)`.

**New:** also return the per-scenario trajectories:
```python
return {
    # Per-scenario (new)
    "scenarios_p_chg":  np.stack([sol.value(P_chg[s])  for s in range(S)]),   # (S, N)
    "scenarios_p_dis":  np.stack([sol.value(P_dis[s])  for s in range(S)]),
    "scenarios_p_reg":  np.stack([sol.value(P_reg[s])  for s in range(S)]),
    "scenarios_soc":    np.stack([sol.value(SOC[s])    for s in range(S)]),   # (S, N+1)
    "scenarios_soh":    np.stack([sol.value(SOH[s])    for s in range(S)]),
    "scenarios_temp":   np.stack([sol.value(TEMP[s])   for s in range(S)]),
    "probabilities":    probabilities,
    # Averages (kept for backward compat)
    "P_chg_ref":        ...,   # as before
    ...
}
```

`Plan.from_planner_dict` builds `ScenarioPlan` tuple from `scenarios_*`
arrays and computes the averaged fields from them.

**DeterministicLP** ([core/planners/deterministic_lp.py](../core/planners/deterministic_lp.py))
returns a single "degenerate" scenario (the deterministic solution) with
probability 1.0. This keeps the new `Plan` shape uniform across all
planners — LP just has `n_scenarios=1`.

### 3. MPC scenario picker (the recourse channel)

**Placement:** inside `EconomicMPC.solve_setpoint` (and
`TrackingMPC.solve_setpoint` symmetrically). Runs once per MPC tick,
*before* the NLP solve, to pick which scenario's SOC trajectory to
anchor against.

**Algorithm (starting simple):**
```python
def _pick_scenario(plan: Plan, state_est: np.ndarray, sim_step: int, steps_per_ems: int) -> int:
    """Max-likelihood scenario selection via normalized state residual.

    state_est = [SOC, SOH, T, VRC1, VRC2]  — from EKF
    Returns index s* into plan.scenarios.
    """
    h = (sim_step - plan.start_step) // steps_per_ems   # current hour
    h = np.clip(h, 0, plan.scenarios[0].soc.shape[0] - 1)

    # Per-scenario residual: weighted L2 on (SOC, SOH, T). VRC ignored.
    # Weights chosen so that 1% SOC ≈ 1 unit, 0.01% SOH ≈ 1 unit, 1 degC ≈ 1 unit.
    w = np.array([100.0, 10000.0, 1.0])
    residuals = np.zeros(len(plan.scenarios))
    for s, scn in enumerate(plan.scenarios):
        diff = np.array([
            state_est[0] - scn.soc[h],
            state_est[1] - scn.soh[h],
            state_est[2] - scn.temp[h],
        ])
        residuals[s] = np.sum((w * diff) ** 2)

    return int(np.argmin(residuals))
```

**Starting simple = argmin (max-likelihood).** If residuals are
degenerate (all scenarios have near-identical SOC at hour h because the
disturbance hasn't spread them yet), we fall back to the averaged plan
— same as current behavior. No regression risk.

**Future enhancement (not in this doc):** softmax weighting with
`w_s ∝ exp(-residual_s² / 2σ²)` for Bayesian belief tracking. Added
later if argmin is too jumpy.

**Trace instrumentation:** record `picked_scenario[m]` per MPC step in
`traces.py`. Enables post-hoc analysis of how often the picker stays on
one scenario vs switches — direct evidence for whether the recourse
channel is doing real work.

### 4. MPC deliverability penalty

**The missing cost term** that gives MPC a *distinct* job from the EMS.

**Formula** (added to both `EconomicMPC` and `TrackingMPC` cost):
```
deliverability_penalty = w_deliver * sum_k max(0, P_reg_committed[k] - P_reg_deliverable[k])^2
```
where
```
P_reg_deliverable[k] = min(
    (SOC[k] - SOC_min) * E_nom * eta_d / tau,     # up-reg (discharge) headroom
    (SOC_max - SOC[k]) * E_nom / (eta_c * tau),   # down-reg (charge) headroom
) / max(|activation_forecast[k]|, eps)
```
and `tau = activation endurance window`, typically `0.25 h` (the
ENTSO-E FCR spec).

**Intuition:** if the SOC trajectory dips to where the committed P_reg
can no longer be delivered under the forecast activation magnitude,
pay a quadratic penalty proportional to the shortfall.

**Weight tuning:** start with `w_deliver = 100` ($/(kW²·h)) — heavy
enough that a 10 kW shortfall (10² × 100 = $10 000 × dt_h) dominates a
$5/hour arbitrage decision, so MPC will *always* prefer to protect
deliverability when a real threat exists. If it's too aggressive we can
reduce.

**Non-linearity handling:** `max(0, ...)²` is not smooth but it's
supported by IPOPT via the standard slack-variable reformulation:
introduce `s_k >= 0`, constrain `s_k >= P_reg_committed[k] - P_reg_deliverable[k]`,
add `w_deliver * sum(s_k^2)` to the cost. CasADi Opti handles this
idiomatically.

### 5. Anchor weight retune

Currently ([core/config/parameters.py](../core/config/parameters.py) in
`MPCParams`):
- `Q_soc_anchor = 10` (soft anchor per step)
- `Q_terminal_econ = 1000` (terminal pin)

**New:**
- `Q_soc_anchor = 1` (one order of magnitude softer)
- `Q_terminal_econ = 100` (one order of magnitude softer)

**Justification:** once the deliverability penalty (#4) is the binding
force that keeps MPC honest about FCR commitments, the anchor weights
should loosen enough for MPC to actually deviate from the EMS plan
when a scenario mismatch is detected (via the picker, #3) or when a
short-horizon opportunity appears that the EMS didn't plan for.

**If the retune causes instability**, fall back to `Q_soc_anchor = 3`,
`Q_terminal_econ = 300`. Document what we ended up with.

### 6. EMS symmetric changes

Per user decision: include these in the same change set, don't defer.

**6a. EMS delivery-failure penalty** (mirrors #4).
[stochastic_ems.py](../core/planners/stochastic_ems.py) objective: add
`w_deliver_ems * sum_{s,k} max(0, P_reg[s][k] - P_reg_deliverable[s][k])^2`,
probability-weighted across scenarios. Same formula as the MPC
deliverability penalty, just in the expected-value sense.

**6b. Tighten `endurance_hours` from 0.5 to 0.25.** Matches real ENTSO-E
FCR spec (15 minutes sustained). Currently the constraint is so lenient
that EMS commits P_reg = P_max at every hour with no bidding decision.
Tightening it forces EMS to make real per-hour tradeoffs.

**Expected effect:** EMS will bid strictly less than P_max in some hours
where SOC headroom is constrained. LP will follow the same logic.
Capacity revenue will drop modestly (~5-10% estimated), but total
profit should hold because penalty cost drops to zero where EMS
previously over-committed.

### 7. Concern I fix — realistic degradation cost

**Current:** `degradation_cost = $50 / unit SOH lost`
([parameters.py:227](../core/config/parameters.py#L227)). Makes
full-throttle 24h cycling cost ~$0.008/day, three to four orders of
magnitude below realistic.

**New:** `degradation_cost = $300_000 / unit SOH lost`.

**Derivation:** 200 kWh LFP BESS at ~$300/kWh installed CAPEX = $60k.
Real wear-to-replacement is typically specified as "80% SOH = EOL",
so replacement cost maps to a 20% SOH loss → $300k per full SOH unit.

**Effect on numbers** (estimated from the big experiment SOH/day
numbers):
- LP (SOH=0.00155%/day) new deg cost ≈ $4.65/day → LP profit drops
  $19.44 → ~$14.79 (calm)
- tracking_mpc (SOH=0.00183%/day) → ~$5.50/day → visible penalty for
  over-cycling
- economic_mpc (SOH=0.00154%/day) → ~$4.62/day
- **Revenue rankings unchanged** because all optimized strategies cycle
  at similar rates. **tracking_mpc's over-cycling now shows up in profit**,
  not just the SOH column — a good side-effect.
- Ratio of deg cost to revenue goes from ~0.02% to ~15-25%. Realistic.

This is a **one-line change** plus a comment update documenting the
CAPEX-amortization basis. Low risk, high pitch credibility (now we can
tell customers our model uses realistic degradation cost).

---

## Roll-out order (with revert points)

Each step is a separate commit. After each step, run a 1-day × 3 strategies
sanity check (~5 min) to catch regressions early. Full big-experiment
re-run only happens at the end.

| Step | Change | Files | Revert cost |
|---|---|---|---|
| **A** | Plan dataclass gains `scenarios` + `probabilities` fields (averages kept). EMS returns both. LP returns `n_scenarios=1` degenerate. | strategy.py, stochastic_ems.py, deterministic_lp.py | Revert commit — no behavior change yet |
| **B** | Fix F5 (unsafe solver fallback → use EMS plan on MPC failure) + F6 (delete dead soc_anchor) + F2 (remove dead P_reg variable from TrackingMPC) | core.py, tracking.py, economic.py | Safety fixes, cheap, riding along with this refactor |
| **C** | Concern I fix: rescale `degradation_cost` 50 → 300_000. Update comment. | parameters.py | 1-line revert |
| **D** | MPC scenario picker (argmin). No cost change, just anchor shift. Add `picked_scenario` trace. | economic.py, tracking.py, traces.py | Revert commit |
| **E** | Loosen anchor weights `Q_soc_anchor: 10 → 1`, `Q_terminal_econ: 1000 → 100` | parameters.py | 2-line revert |
| **F** | MPC deliverability penalty (slack var + quadratic cost). Start `w_deliver = 100`. | economic.py, tracking.py | Revert commit |
| **G** | EMS delivery-failure penalty + tighten `endurance_hours: 0.5 → 0.25` | stochastic_ems.py, parameters.py | Revert commit |
| **H** | Big experiment re-run (Phase 3) | (no code change) | N/A |

After step C, expect the headline number to **drop uniformly** because
all strategies now pay realistic degradation cost. That's fine — the
comparison is still pairwise and tracking_mpc will widen its gap.
After step D, expect small changes (the picker starts selecting
scenarios but the anchor still dominates).
After step E, expect MPC to start deviating from the EMS plan in
visible ways (TV(P_net) changes).
After step F, expect MPC to start *winning* on stressed subset and at
least tying on volatile.
After step G, expect EMS-using strategies (which is 5 of 8) to shift
their FCR bids, and delivery score to remain 100% where it currently
is. Capacity revenue drops slightly, penalty cost drops to zero.

---

## Success criterion (Phase 4 gate)

Run `--big --big-n 5` on the fixed hierarchy. Pass if:

1. `economic_mpc` mean profit > `deterministic_lp` mean profit, on
   **calm, volatile, AND stressed** (positive on all three).
2. The advantage is **monotone**: `advantage_stressed > advantage_volatile ≥ advantage_calm`.
3. `economic_mpc` delivery score stays at 100% in all three regimes.
4. `economic_mpc` does not regress on wall time more than 10% vs the
   pre-change run (currently ~108 s/day).

**If all four pass:** ship it. Phase 5 writes the pitch deck update.

**If 1 or 2 fails** but 3 and 4 pass: the regime is still too benign
(Concern H). Escalate to Case B — add a more stressed subset (3×
sigma, or tighter SOC bounds) to the experiment matrix and re-run the
stressed-only subset.

**If 3 fails:** the deliverability penalty weight is wrong. Retune
`w_deliver`, repeat step F sanity check.

**If 4 fails:** MPC solve time has blown up. Investigate — likely the
slack-variable reformulation or the picker is expensive. Profile and
fix.

---

## Open questions

- **Concern I exact value:** $300k is my estimate. Should we source it
  from a public LFP cost reference (BloombergNEF / Lazard LCOS) for
  pitch-defensibility? *Proposal: use $300k as a "typical 2024 LFP
  CAPEX" value and cite BNEF in the comment.*
- **Deliverability formula `tau` (the activation endurance window):**
  0.25 h matches FCR spec. EMS has its own `endurance_hours` param (0.25
  after step G). Should MPC share the same param or keep its own?
  *Proposal: share via `ep.endurance_hours`. One source of truth.*
- **Scenario picker weights `w = [100, 10000, 1]`:** I picked these by
  hand so that 1% SOC ≈ 0.0001 SOH ≈ 1 degC each contribute ~1. Probably
  overweights SOH (slow-moving) and underweights temperature
  (fast-moving). Acceptable for v1 of the picker. *Proposal: leave as
  is, revisit if picker behavior is weird in the traces.*

---

## Total budget

- **LOC:** ~200 across 7 files (up from 120 original estimate — the
  piggyback fixes F2/F5/F6 and Concern I add ~40, the symmetric EMS
  changes add ~40)
- **Files touched:** `core/simulator/strategy.py`, `core/simulator/core.py`,
  `core/simulator/traces.py`, `core/planners/stochastic_ems.py`,
  `core/planners/deterministic_lp.py`, `core/mpc/economic.py`,
  `core/mpc/tracking.py`, `core/config/parameters.py`
- **Reversibility:** every step is one commit. Any step can revert
  without touching the others except the sequencing (D depends on A,
  F depends on E, G depends on F).
- **Wall time to run the validation experiment:** ~65 min (same as the
  original big run)
