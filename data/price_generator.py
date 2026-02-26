"""Stochastic price scenario generator for energy and regulation markets.

Generates hourly price forecasts with multiple scenarios for the stochastic
EMS optimizer.  Also provides interpolation to MPC resolution (zero-order hold).

Units
-----
  - Energy price:      $/kWh
  - Regulation price:  $/kW/h  (capacity payment)
"""

from __future__ import annotations

import pathlib

import numpy as np


class PriceGenerator:
    """Generates energy and regulation price forecasts with stochastic scenarios.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    #  Base profiles
    # ------------------------------------------------------------------

    def generate_base_energy_prices(self, n_hours: int) -> np.ndarray:
        """Deterministic base energy price profile [$/kWh].

        Combines a base cost, sinusoidal daily cycle, and Gaussian
        morning / evening demand peaks.  Repeats the 24-hour pattern
        for horizons longer than one day.

        Parameters
        ----------
        n_hours : int

        Returns
        -------
        np.ndarray, shape (n_hours,)
        """
        t = np.arange(n_hours, dtype=np.float64)

        base = 0.050
        daily = 0.025 * np.sin(2.0 * np.pi * (t - 6.0) / 24.0)
        evening = 0.040 * np.exp(-0.5 * ((t % 24 - 18.0) / 2.0) ** 2)
        morning = 0.015 * np.exp(-0.5 * ((t % 24 - 8.0) / 1.5) ** 2)

        return np.maximum(base + daily + evening + morning, 0.005)

    def generate_regulation_prices(self, energy_prices: np.ndarray) -> np.ndarray:
        """Generate regulation market prices [$/kW/h].

        Regulation capacity prices are correlated with energy prices but
        typically lower, with their own noise component.

        Parameters
        ----------
        energy_prices : np.ndarray, shape (n_hours,)

        Returns
        -------
        np.ndarray, shape (n_hours,)
        """
        n = len(energy_prices)
        noise = self._rng.normal(0.0, 0.003, n)
        reg = 0.4 * energy_prices + 0.01 + noise
        return np.maximum(reg, 0.002)

    # ------------------------------------------------------------------
    #  Stochastic scenarios
    # ------------------------------------------------------------------

    def generate_scenarios(
        self,
        n_hours: int,
        n_scenarios: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate stochastic price scenarios.

        Parameters
        ----------
        n_hours : int
            Total hours to generate (must cover sim_hours + N_ems for lookahead).
        n_scenarios : int
            Number of scenarios (default 5).

        Returns
        -------
        energy_scenarios : np.ndarray, shape (n_scenarios, n_hours)
            Energy price scenarios [$/kWh].
        reg_scenarios : np.ndarray, shape (n_scenarios, n_hours)
            Regulation price scenarios [$/kW/h].
        probabilities : np.ndarray, shape (n_scenarios,)
            Scenario probabilities summing to 1.0.
        """
        base_energy = self.generate_base_energy_prices(n_hours)
        base_reg = self.generate_regulation_prices(base_energy)

        # Scenario probabilities
        if n_scenarios == 5:
            probs = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        else:
            probs = np.ones(n_scenarios) / n_scenarios

        # Perturbation amplitudes (fraction of base)
        perturbation_scales = self._scenario_perturbations(n_scenarios)

        energy_scen = np.zeros((n_scenarios, n_hours))
        reg_scen = np.zeros((n_scenarios, n_hours))

        for s in range(n_scenarios):
            # Time-correlated noise (exponential moving average with tau ~ 3 h)
            raw_noise = self._rng.normal(0.0, 1.0, n_hours)
            smoothed = self._smooth_noise(raw_noise, tau_hours=3.0)

            energy_pert = base_energy * (1.0 + perturbation_scales[s] * smoothed)
            energy_scen[s] = np.maximum(energy_pert, 0.005)

            raw_noise_reg = self._rng.normal(0.0, 1.0, n_hours)
            smoothed_reg = self._smooth_noise(raw_noise_reg, tau_hours=3.0)
            reg_pert = base_reg * (1.0 + perturbation_scales[s] * 0.8 * smoothed_reg)
            reg_scen[s] = np.maximum(reg_pert, 0.002)

        return energy_scen, reg_scen, probs

    # ------------------------------------------------------------------
    #  Interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_to_mpc(
        hourly: np.ndarray,
        dt_ems: float,
        dt_mpc: float,
    ) -> np.ndarray:
        """Zero-order hold interpolation of hourly values to MPC resolution.

        Parameters
        ----------
        hourly : np.ndarray, shape (N_hours,)
        dt_ems : float   [s]  (3600)
        dt_mpc : float   [s]  (60)

        Returns
        -------
        np.ndarray, shape (N_hours * ratio,)
        """
        ratio = int(round(dt_ems / dt_mpc))
        return np.repeat(hourly, ratio)

    # ------------------------------------------------------------------
    #  CSV loading (backward compatible)
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_csv(csv_path: str | pathlib.Path) -> np.ndarray:
        """Load energy prices from existing CSV.

        Parameters
        ----------
        csv_path : path-like
            CSV with columns ``hour, price_usd_per_kwh``.

        Returns
        -------
        np.ndarray   [$/kWh]
        """
        data = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1)
        return data[:, 1]

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scenario_perturbations(n_scenarios: int) -> list[float]:
        """Return fractional perturbation scale per scenario.

        Scenario 0: base (0 perturbation)
        Scenarios 1-2: +/- moderate (15 %)
        Scenarios 3-4: +/- high (30 %)
        """
        if n_scenarios == 5:
            return [0.0, 0.15, -0.15, 0.30, -0.30]
        # Symmetric spread for other counts
        scales = []
        for i in range(n_scenarios):
            frac = (i - (n_scenarios - 1) / 2) / max(n_scenarios - 1, 1)
            scales.append(frac * 0.30)
        return scales

    @staticmethod
    def _smooth_noise(raw: np.ndarray, tau_hours: float = 3.0) -> np.ndarray:
        """Apply exponential moving average to create time-correlated noise.

        The smoothed noise has zero mean and approximately unit variance.
        """
        alpha = 1.0 / (tau_hours + 1.0)
        smoothed = np.zeros_like(raw)
        smoothed[0] = raw[0]
        for i in range(1, len(raw)):
            smoothed[i] = alpha * raw[i] + (1.0 - alpha) * smoothed[i - 1]
        # Normalise to approximately unit variance
        std = smoothed.std()
        if std > 1e-12:
            smoothed /= std
        return smoothed
