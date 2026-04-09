# rule_based — heuristic dispatch (NOT an OCP)

**Pitch-visible:** yes (strict lower bound)
**Composition:** `RuleBasedPlanner` (no MPC)

## What it does (plain words)

Sorts the day's forecast-mean energy prices, charges during the cheapest
hours, and discharges during the most expensive. No FCR commitment. No
optimization problem is solved at all — this is a fixed decision rule
applied to the price array. Strict lower bound for the comparison
harness; every other strategy should beat it by a wide margin.

## Decision rule (NOT an OCP)

This strategy does not solve an optimization problem. It executes a
closed-form heuristic over the forecast prices.

### Notation

| Symbol            | Meaning                                              | Units        |
|-------------------|------------------------------------------------------|--------------|
| `N`               | planning horizon                                     | hours (= 24) |
| `p_e[k]`          | probability-weighted mean of forecast energy prices  | \$/kWh       |
| `E_nom`           | nominal energy capacity                              | kWh          |
| `SOC_min`,`SOC_max` | SOC bounds                                         | —            |
| `P_max`           | max charge / discharge power                         | kW           |

### Algorithm

1. **Forecast collapse** — average the forecast scenarios:

    ```
    p_e[k] = Σ over s=1..S of  π[s] · p_e_scenario[s,k]      for k = 0..N−1
    ```

2. **Compute usable energy and dispatch power:**

    ```
    E_use = (SOC_max − SOC_min) · E_nom
    P     = 0.8 · P_max
    ```

3. **Number of active hours per direction:**

    ```
    n* = min( ⌈ E_use / P ⌉ ,  ⌊ N / 3 ⌋ )
    ```

4. **Sort hours by ascending forecast price:**

    ```
    σ = argsort(p_e)               # σ[0] is the cheapest hour
    ```

5. **Pick charge set C and discharge set D:**

    ```
    C = { σ[0], σ[1], ..., σ[n* − 1] }                # n* cheapest hours
    D = { σ[N − n*], σ[N − n* + 1], ..., σ[N − 1] }    # n* most expensive hours
    ```

6. **Profitability gate** — only commit if there is a positive spread:

    ```
    if  p_e[ max(D) ]  >  p_e[ max(C) ]:
        commit
    else:
        idle
    ```

    `max(D)` is the most expensive of the discharge hours;
    `max(C)` is the *most expensive of the cheap hours* (i.e., the
    least cheap charge hour). The gate fires only when the highest
    discharge price exceeds the worst charge price.

7. **Output schedule (when committed):**

    ```
    P_chg[k] = P      if k ∈ C, else 0
    P_dis[k] = P      if k ∈ D, else 0
    P_reg[k] = 0      for all k
    ```

### What this does NOT model

- **No optimization.** Greedy price-sorting only.
- **No state evolution.** SOC is not tracked by the rule itself.
- **No FCR participation.** `P_reg ≡ 0`.
- **No constraints.** SOC bounds are honoured only by the magic factor
  `0.8` and the `n*` heuristic; the plant clips anything that would
  violate physics.
- **No closed-loop feedback.** Single hourly setpoint, dispatched
  open-loop.
