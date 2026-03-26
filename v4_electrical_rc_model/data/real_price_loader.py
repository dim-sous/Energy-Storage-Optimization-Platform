"""Load real European electricity prices and build scenario bundles.

Reads historical day-ahead prices (from Energy-Charts / ENTSO-E data) and
FCR capacity prices (from SMARD / Bundesnetzagentur) to construct stochastic
scenario bundles compatible with the EMS interface.

Data sources
------------
  Energy: Energy-Charts (Fraunhofer ISE) — DE-LU day-ahead, EPEX SPOT.
  Regulation: SMARD (Bundesnetzagentur) — FCR total system cost (filter 4998),
    converted to per-MW capacity price by dividing by Germany's FCR
    requirement (~620 MW).

Units
-----
  Energy input CSV:  EUR/MWh  → converted to $/kWh
  Regulation input:  $/kW/h   (pre-converted from SMARD data)
"""

from __future__ import annotations

import pathlib

import numpy as np

# --------------------------------------------------------------------------
#  Conversion constants
# --------------------------------------------------------------------------
EUR_TO_USD = 1.08          # Approximate EUR→USD (Q1 2024 average)
MWH_TO_KWH = 1_000.0      # 1 MWh = 1 000 kWh


class RealPriceLoader:
    """Load real prices and create EMS-compatible scenario bundles.

    Parameters
    ----------
    energy_csv : path-like
        CSV with columns ``unix_timestamp, price_eur_per_mwh``.
    reg_csv : path-like or None
        CSV with columns ``unix_timestamp, reg_price_usd_per_kw_h``.
        If None, regulation prices are synthesised (less accurate).
    seed : int
        Random seed for scenario sampling.
    eur_to_usd : float
        EUR→USD conversion rate.
    """

    def __init__(
        self,
        energy_csv: str | pathlib.Path,
        reg_csv: str | pathlib.Path | None = None,
        seed: int = 42,
        eur_to_usd: float = EUR_TO_USD,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._eur_to_usd = eur_to_usd

        raw = np.genfromtxt(str(energy_csv), delimiter=",", skip_header=1)
        self._timestamps = raw[:, 0].astype(np.int64)
        self._prices_eur_mwh = raw[:, 1]

        # Convert to $/kWh (platform unit)
        self._prices_usd_kwh = (
            self._prices_eur_mwh * self._eur_to_usd / MWH_TO_KWH
        )

        # Load real regulation prices if available
        if reg_csv is not None:
            reg_raw = np.genfromtxt(str(reg_csv), delimiter=",", skip_header=1)
            self._reg_prices_usd_kw_h = reg_raw[:, 1]  # Already in $/kW/h
            self._has_real_reg = True
        else:
            self._reg_prices_usd_kw_h = None
            self._has_real_reg = False

        # Split into complete 24-hour days (use minimum of energy/reg lengths)
        n_energy_days = len(self._prices_usd_kwh) // 24
        if self._has_real_reg:
            n_reg_days = len(self._reg_prices_usd_kw_h) // 24
            n_full_days = min(n_energy_days, n_reg_days)
            usable_reg = n_full_days * 24
            self._daily_reg = self._reg_prices_usd_kw_h[:usable_reg].reshape(
                n_full_days, 24
            )
        else:
            n_full_days = n_energy_days
            self._daily_reg = None

        usable = n_full_days * 24
        self._daily_prices = self._prices_usd_kwh[:usable].reshape(
            n_full_days, 24
        )
        self.n_days = n_full_days

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def get_day(self, day_idx: int) -> np.ndarray:
        """Return energy prices for a single day [$/kWh], shape (24,)."""
        return self._daily_prices[day_idx].copy()

    def generate_scenarios_for_day(
        self,
        day_idx: int,
        n_hours: int = 48,
        n_scenarios: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a scenario bundle centred on a real day.

        Scenario 0 is the *actual* day's prices.  Scenarios 1–(N-1)
        are sampled from other historical days, providing genuine
        price-shape diversity (not synthetic perturbations).

        The 48-hour horizon is built by appending the *next* calendar
        day (when available) for the lookahead portion.

        Returns
        -------
        energy_scenarios : (n_scenarios, n_hours) [$/kWh]
        reg_scenarios    : (n_scenarios, n_hours) [$/kW/h]
        probabilities    : (n_scenarios,)
        """
        # --- Build 48-hour base from actual day + next day ----
        actual_e48, actual_r48 = self._build_48h(day_idx)

        # --- Other scenarios: sample different historical days ---
        other_days = [i for i in range(self.n_days) if i != day_idx]
        chosen = self._rng.choice(
            other_days, size=n_scenarios - 1, replace=False,
        )

        energy_scen = np.zeros((n_scenarios, n_hours))
        reg_scen = np.zeros((n_scenarios, n_hours))
        energy_scen[0, :n_hours] = actual_e48[:n_hours]
        reg_scen[0, :n_hours] = actual_r48[:n_hours]

        for s_idx, d_idx in enumerate(chosen, start=1):
            alt_e48, alt_r48 = self._build_48h(d_idx)
            energy_scen[s_idx, :n_hours] = alt_e48[:n_hours]
            reg_scen[s_idx, :n_hours] = alt_r48[:n_hours]

        # Clamp negative prices to a small positive floor (EMS assumes ≥ 0)
        energy_scen = np.maximum(energy_scen, 0.001)
        reg_scen = np.maximum(reg_scen, 0.0)

        # --- Probabilities: actual day gets highest weight ---
        probs = np.ones(n_scenarios)
        probs[0] = 2.0                   # Double-weight the actual day
        probs /= probs.sum()

        return energy_scen, reg_scen, probs

    def sample_day_indices(self, n: int) -> np.ndarray:
        """Sample *n* day indices without replacement for Monte Carlo."""
        return self._rng.choice(self.n_days, size=min(n, self.n_days), replace=False)

    @property
    def has_real_regulation(self) -> bool:
        """Whether real regulation prices were loaded."""
        return self._has_real_reg

    @property
    def price_stats(self) -> dict:
        """Summary statistics of loaded price data."""
        p = self._prices_eur_mwh[:self.n_days * 24]
        stats = {
            "n_hours": int(self.n_days * 24),
            "n_days": self.n_days,
            "mean_eur_mwh": float(np.mean(p)),
            "median_eur_mwh": float(np.median(p)),
            "min_eur_mwh": float(np.min(p)),
            "max_eur_mwh": float(np.max(p)),
            "std_eur_mwh": float(np.std(p)),
            "pct_negative": float(np.mean(p < 0) * 100),
            "reg_data": "real FCR (SMARD)" if self._has_real_reg else "synthesised",
        }
        if self._has_real_reg:
            r = self._reg_prices_usd_kw_h[:self.n_days * 24]
            # Convert back to EUR/MW/h for display
            r_eur = r * 1000 / self._eur_to_usd
            stats["reg_mean_eur_mw_h"] = float(np.mean(r_eur))
            stats["reg_median_eur_mw_h"] = float(np.median(r_eur))
        return stats

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _build_48h(
        self, day_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build 48-hour energy + regulation price windows."""
        e_day = self._daily_prices[day_idx]
        next_idx = day_idx + 1 if day_idx + 1 < self.n_days else day_idx
        e_next = self._daily_prices[next_idx]
        energy_48 = np.concatenate([e_day, e_next])

        if self._has_real_reg:
            r_day = self._daily_reg[day_idx]
            r_next = self._daily_reg[next_idx]
            reg_48 = np.concatenate([r_day, r_next])
        else:
            # Fallback: synthesise from energy (less accurate)
            noise = self._rng.normal(0.0, 0.002, 48)
            reg_48 = 0.4 * energy_48 + 0.006 + noise
            reg_48 = np.maximum(reg_48, 0.002)

        return energy_48, reg_48
