# Realism Fix 1 (RF1) — Move Activation Tracking Into the Plant

**Status:** approved 2026-04-15, ready to implement
**Predecessor:** [wow_factor_1_design.md](wow_factor_1_design.md) (paused
pending RF1)
**Audit reference:** backlog.md → "Simulator actuation realism audit (2026-04-15)"

---

## Goal

Eliminate the only systematic cheat in the v5 simulator: strategies
currently read the raw, ground-truth, sub-second `activation[k]` signal
with zero noise and zero latency. Every "delivery score" result in the
big experiment sits on top of this cheat.

After RF1:
- **The strategy layer never sees `activation[k]`.**
- **The plant tracks activation internally** given the committed FCR
  capacity, like a real BESS does.
- **PI's job shrinks** to "SOC safety clamp + recovery bias on the
  setpoint" — its activation-tracking code path is deleted (it was
  doing the same cheat as `_open_loop_dispatch`).
- **MPC loses its `recent_activation` hint.** It plans against the
  *expected* activation magnitude (the existing `expected_activation_frac`),
  not the most recent ground-truth sample.
- The headline pitch number from WF1 (econ_mpc vs LP advantage) becomes
  defensible because all strategies now face the same actuation problem.

---

## Out of scope

- All Wow Factor 1 work (paused; resumes after RF1 lands)
- Sensor noise / latency on the activation signal at the plant boundary
  — the plant has zero-latency access to the realized activation (this
  is OK because the plant *is* the BESS controller in this layering)
- A first-order actuator lag on commanded power (already minor; the 4 s
  PI cadence dwarfs any realistic inverter time constant)
- Communication delay between strategy and plant
- All D1-D5 architectural follow-ups

---

## What changes, in one paragraph

`BatteryPlant.step()` grows a new argument `activation_k: float`. Inside,
the plant computes `p_net_with_activation = p_net_setpoint + activation_k * p_reg_committed`,
clips to `±P_max`, applies the existing SOC headroom limiter, integrates.
It returns `(x_new, y_meas, u_applied, p_delivered)` where `p_delivered`
is the actually-delivered FCR power (`u_applied[0] - p_net_setpoint`,
signed). The simulator main loop generates `activation[k]` as before
and passes it to `plant.step()` instead of letting strategies see it.
`_open_loop_dispatch` becomes a 2-line passthrough. PI's `compute()`
loses the `activation_signal` argument and returns only the
SOC-safety-clamped setpoint. EconomicMPC loses `current_activation_p`
and falls back to `ep.expected_activation_frac` for OU pre-positioning.

---

## Seven pieces

### 1. `BatteryPlant.step()` signature change

**Current** ([core/physics/plant.py:516](../core/physics/plant.py#L516)):
```python
def step(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # u = [P_net_signed, P_reg_committed]
    # P_net is whatever the strategy commanded — already includes any
    # activation modulation the strategy chose to apply.
```

**New:**
```python
def step(
    self,
    u: np.ndarray,
    activation_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    u : [P_net_setpoint, P_reg_committed]   — strategy's commitment, NOT
                                              including activation modulation
    activation_k : float in [-1, 1]          — current FCR activation sample,
                                              the plant's own signal
    Returns
    -------
    x_new      : (5,)  true state
    y_meas     : (3,)  noisy measurement
    u_applied  : (2,)  [P_net_actual, P_reg_committed]   — what was billed
    p_delivered: float (signed)            — FCR power actually delivered
    """
    P_net_setpoint = float(u[0])
    P_reg = float(np.clip(u[1], 0.0, bp.P_max_kw))

    # Plant-internal activation tracking. The strategy never saw
    # activation_k; the plant did.
    P_net_with_activation = P_net_setpoint + activation_k * P_reg

    # Existing pipeline below this point.
    P_net = float(np.clip(P_net_with_activation, -bp.P_max_kw, bp.P_max_kw))
    # ... existing SOC headroom logic ...
    # ... existing RK4 integration ...

    p_delivered = float(P_net - P_net_setpoint)   # signed
    u_applied = np.array([P_net, P_reg])
    return self._x.copy(), y_meas, u_applied, p_delivered
```

**Why `p_delivered = P_net - P_net_setpoint`:** the plant clipped the
activation-modulated power; whatever made it through is the delivered
portion, by definition. If the clip ate some of the activation demand
(SOC at limit, P_max hit) the delivered amount falls short and the
ledger's penalty math fires.

`BatteryPack.step()` (the multi-cell variant) takes the same change.

### 2. Simulator main loop hand-off

**Current** ([core/simulator/core.py:206-225](../core/simulator/core.py#L206-L225)):
```python
activation_k = float(activation[k])
if strategy.pi is not None and strategy.pi_enabled:
    u_command, p_delivered = strategy.pi.compute(
        setpoint_pnet=..., p_reg_committed=...,
        activation_signal=activation_k,    # ← cheat lives here
        soc_current=state_est[0],
    )
else:
    u_command, p_delivered = _open_loop_dispatch(
        setpoint_pnet=..., p_reg_committed=...,
        activation=activation_k,           # ← and here
        soc_current=state_est[0],
        p_max_kw=bp.P_max_kw,
    )
x_new, y_meas, u_applied = plant.step(u_command)
```

**New:**
```python
activation_k = float(activation[k])
if strategy.pi is not None and strategy.pi_enabled:
    u_command = strategy.pi.compute(
        setpoint_pnet=...,
        p_reg_committed=...,
        soc_current=state_est[0],
    )
else:
    u_command = _open_loop_dispatch(
        setpoint_pnet=...,
        p_reg_committed=...,
    )
x_new, y_meas, u_applied, p_delivered = plant.step(u_command, activation_k)
```

The activation signal flows simulator → plant. Strategies are left out
entirely.

### 3. `_open_loop_dispatch` simplification

**Current** ([core/simulator/core.py:56-75](../core/simulator/core.py#L56-L75)):
13 lines, computes activation modulation and clipping.

**New:**
```python
def _open_loop_dispatch(
    setpoint_pnet: float,
    p_reg_committed: float,
) -> np.ndarray:
    """No SOC safety, no recovery bias, no activation. The plant
    handles activation tracking internally. Used by RULE_BASED,
    DETERMINISTIC_LP, and EMS_CLAMPS — these strategies command the
    EMS plan setpoint directly without any per-PI-step processing."""
    return np.array([setpoint_pnet, p_reg_committed])
```

3 lines. The fact that this becomes trivial is a *good* sign — it
reflects that "open loop" is supposed to mean "do nothing reactive,"
not "do perfect feedforward."

### 4. PI controller simplification

**Current** ([core/pi/regulation.py:52-141](../core/pi/regulation.py)):
PI takes `(setpoint_pnet, p_reg_committed, activation_signal, soc_current)`
and returns `(u_command, p_delivered)`. The activation_signal channel is
the cheat path.

**New signature:**
```python
def compute(
    self,
    setpoint_pnet: float,
    p_reg_committed: float,
    soc_current: float,
) -> np.ndarray:
    """
    SOC-safety-clamped setpoint. Returns [P_net_setpoint, P_reg_committed]
    with the SOC recovery bias applied to P_net_setpoint and the
    SOC safety clamps applied to P_reg_committed.

    No activation handling — that lives in the plant now.
    """
```

**What survives in PI:**
- The SOC recovery bias (`(SOC_terminal - soc) * recovery_gain` applied to
  the setpoint when activation is absent — but now we apply it
  *unconditionally* since PI no longer knows whether activation is
  active. This is a small change in semantics worth noting.)
- The SOC safety clamps that scale `p_reg_committed` linearly to zero
  near `SOC_min` / `SOC_max` — these still make sense, they protect
  against committing capacity the battery can't deliver.

**What dies in PI:**
- The `p_reg_demand = activation_signal * p_reg_committed` line.
- The `p_delivered` return value (plant computes it now).
- The `activation_signal` parameter and all conditionals that branch
  on it.

PI loses ~30 LOC. Simpler is better.

**Note on the recovery bias semantic change:** previously the bias only
fired when `|activation| < 1e-6`. After RF1, PI doesn't see activation,
so the bias either fires every step or never. Proposal: **fire every
step** but with a much smaller gain (current `recovery_gain = 0.05`,
new `recovery_gain = 0.005`) so the SOC drift back to `SOC_terminal`
over an hour is similar in magnitude to the current behavior. Tune in
sanity tests.

### 5. EconomicMPC's `recent_activation` removal

**Current** ([core/mpc/economic.py:138, 165-180](../core/mpc/economic.py#L138)):
EconomicMPC has a `current_activation_p` parameter, set per solve via
`opti.set_value(self._current_activation_p, recent_activation)`. It
forecasts the next hour's activation as `a_decay = exp(-k*dt/300)` × the
recent sample, then injects `a_forecast * p_reg_committed` into the SOC
dynamics for SOC pre-positioning.

**New:**
- `current_activation_p` parameter deleted (along with the
  `solve_setpoint(..., recent_activation=...)` argument).
- The OU forecast becomes constant: `a_forecast_k = ep.expected_activation_frac`
  (currently 0.04, the long-run E[|activation|] from the OU model). This
  matches what the EMS planner already uses and is honest because it's
  derived from the published CE frequency statistics, not from peeking
  at the realized signal.
- `EconomicMPCAdapter.solve_setpoint(...)` drops the `recent_activation`
  argument from its forwarding logic
  ([adapters.py:158-183](../core/mpc/adapters.py#L158-L183)).
- `TrackingMPC` already ignores `recent_activation`
  ([adapters.py:51](../core/mpc/adapters.py#L51)) so no change there
  beyond removing the dead parameter from its adapter signature.
- Simulator main loop no longer reads `activation[k-1]` at MPC tick
  ([core.py:177](../core/simulator/core.py#L177)).

**Cost:** EconomicMPC's claimed "activation-aware OU forecasting" pitch
differentiator is gone. **This is a feature, not a bug:** the audit
showed it was a cheat, and removing it forces MPC to find its real
edge somewhere else (which is exactly what Wow Factor 1's scenario
picker is for).

### 6. Ledger / accounting (no behavior change)

The ledger formula at [core/accounting/ledger.py:71-79](../core/accounting/ledger.py#L71-L79):
```python
p_demanded = np.abs(activation * p_reg_committed)
p_missed = np.maximum(0.0, p_demanded - np.abs(p_delivered))
penalty = penalty_mult * realized_r * p_missed * dt_pi_h
```

This is **already correct under RF1**. The ledger reads `activation`
(from traces) and `p_delivered` (from traces). After RF1, `p_delivered`
is what the plant returned, not what the dispatch layer commanded. The
penalty math fires whenever the plant clipped activation demand — exactly
what we want.

**No code change in the ledger.** The trace columns remain the same;
their *interpretation* improves.

### 7. Trace bookkeeping

`SimTraces.record_step` currently takes `p_delivered` from the
strategy/dispatch layer. Now it takes `p_delivered` from the plant
return. One-line change in the simulator main loop. The trace storage
in [core/simulator/traces.py](../core/simulator/traces.py) does not
change shape.

---

## Roll-out order (3 commits, each separately revertable)

| Step | Change | Files | Sanity test |
|---|---|---|---|
| **A** | Plant signature change: `step()` takes `activation_k`, returns `p_delivered`. Update `BatteryPlant`, `BatteryPack`, all callers in the simulator. Strategy-side code unchanged at this point — simulator just passes `0.0` activation to plant for now. | plant.py, core.py | 1-day LP run, expect identical numbers to current (activation=0 means no FCR delivery, but the structure is in place) |
| **B** | Move activation flow into the plant. Simulator passes `activation[k]` to `plant.step()`. PI / `_open_loop_dispatch` lose the activation argument. Strategies stop seeing activation. | core.py, regulation.py | 1-day LP+economic_mpc run, expect **delivery score to drop materially** for everyone — this is the moment of truth for whether the cheat fix works |
| **C** | Remove EconomicMPC's `recent_activation`. Replace with constant `ep.expected_activation_frac`. Drop the parameter from the adapter and the simulator main loop. Tune PI `recovery_gain: 0.05 → 0.005`. | economic.py, adapters.py, core.py, parameters.py | 1-day full 8-strategy comparison, eyeball delivery scores and profits |

After all 3: full big experiment re-run (Phase 3 of the original plan).

---

## Expected effects (hypotheses to verify after step B)

1. **Delivery score drops below 100% for every strategy** in calm and
   volatile regimes. The drop magnitude tells us how aggressive the
   activation signal is relative to the plant's headroom. If it stays
   at 100% even after RF1, the regime is just too benign and we need
   the stressed subset's higher sigma to bite.
2. **Profit drops slightly** for all strategies that previously earned
   delivery revenue. The delivery revenue was inflated by the cheat;
   now it reflects only what the plant actually delivered.
3. **Penalty cost becomes nonzero** in stressed regime for the strategies
   that don't pre-position SOC well. This is the failure mode MPC should
   protect against.
4. **`tracking_mpc`'s P_max touches in stressed regime probably go up
   even more** because it was already breaking and now the plant clips
   harder.
5. **Strategy ranking on profit changes very little** because the
   capacity revenue (94% of profit) is unaffected. The differentiation
   moves to delivery score and penalty cost.

If hypothesis 1 fails (delivery still 100% after RF1), the regime is
benign and Wow Factor 1's stressed subset becomes load-bearing for the
pitch. If hypothesis 1 passes, RF1 has done its job and Wow Factor 1
can resume.

---

## Open questions

- **Recovery bias semantics in PI:** propose firing it every step with
  `recovery_gain: 0.05 → 0.005`. Could also delete the recovery bias
  entirely — it was a workaround for activation idle periods that
  doesn't really make sense anymore. *Proposal: keep at smaller gain,
  delete only if it causes drift in sanity tests.*
- **Does the plant need a P_max budget check on the activation-modulated
  power?** Currently the plant clips `P_net_with_activation` to
  `±P_max`. This is correct (it's a physical limit) but means the
  battery silently fails to deliver excess activation demand without
  the strategy knowing. *Proposal: keep the clip, let the ledger
  surface the failure via penalty cost, no notification path.*
- **Activation-aware EMS planning** ([stochastic_ems.py:78](../core/planners/stochastic_ems.py#L78)
  uses `expected_activation_frac=0.04`): does the EMS planner need to
  know that activation now lives in the plant? *Proposal: no — the EMS
  is already activation-aware via `expected_activation_frac` and
  doesn't need a code change.*

---

## Total budget

- **LOC:** ~150 across 5 files (plant, core, regulation, economic, adapters)
- **Reversibility:** 3 commits, each revertable independently
- **Risk:** medium — touches the plant signature, which propagates to
  every strategy. Sanity-tested at each step.
- **Validation cost:** 3× single-day sanity runs (~5 min each) + 1
  full big-experiment re-run (~65 min)
- **Architectural impact:** *positive* — the plant now does what a
  real BESS controller does, the strategies do what real strategies
  do, and the layering matches reality.
