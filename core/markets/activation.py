"""European FCR activation signal generator based on grid frequency.

Models the Continental European (CE) synchronous area frequency as an
Ornstein-Uhlenbeck (OU) process with deterministic hourly spikes (DFDs),
then converts frequency deviation to an FCR activation signal via the
standard linear droop characteristic.

Physical basis
--------------
  - CE grid frequency fluctuates around 50 Hz with std ~18 mHz.
  - Autocorrelation time (OU tau) is ~300 s (5 min).
  - Distribution is slightly heavy-tailed (kurtosis ~4.0).
  - Deterministic frequency deviations (DFDs) occur at the top of each
    hour due to generation schedule block transitions, with typical
    magnitude 40-80 mHz lasting ~3 min.

FCR droop response
------------------
  - Deadband: ±10 mHz (no activation inside).
  - Full activation at ±200 mHz.
  - Linear (proportional) between deadband and full activation.
  - The output signal is in [-1, +1], representing the fraction of
    committed capacity demanded.

References
----------
  Schäfer et al., "Non-Gaussian power grid frequency fluctuations
  characterized by Lévy-stable laws and superstatistics", Nature Energy
  3, 119–126 (2018).

  Kraljic, "Towards realistic statistical models of the grid
  frequency", arXiv:2104.09289 (2021).

Usage
-----
    gen = ActivationSignalGenerator(reg_params)
    signal = gen.generate(n_steps=21600)   # 24h at 4s
"""

from __future__ import annotations

import numpy as np

from core.config.parameters import RegulationParams

# FCR droop parameters (ENTSO-E standard for CE area)
_DEADBAND_MHZ = 10.0       # ±10 mHz deadband
_FULL_ACTIVATION_MHZ = 200.0  # Full activation at ±200 mHz

# OU process parameters (calibrated to CE frequency statistics)
_SIGMA_MHZ = 18.0          # Long-run std of frequency deviation [mHz]
_TAU_S = 300.0             # Correlation time [s]
_KURTOSIS_TARGET = 4.0     # Heavy-tailed (Gaussian = 3.0)

# Deterministic frequency deviation (DFD) parameters
_DFD_MAGNITUDE_MHZ = 60.0  # Typical DFD magnitude [mHz]
_DFD_STD_MHZ = 15.0        # Variability in DFD magnitude [mHz]
_DFD_DURATION_S = 180.0    # DFD duration [s]


class ActivationSignalGenerator:
    """European FCR activation signal from OU frequency model."""

    def __init__(
        self,
        reg_params: RegulationParams,
        dt: float = 4.0,
    ) -> None:
        self._rp = reg_params
        self._dt = dt
        self._rng = np.random.default_rng(reg_params.activation_seed)
        self._sigma_mhz = _SIGMA_MHZ * reg_params.sigma_mhz_mult

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate activation signal array of shape (n_steps,), values in [-1, +1].

        Steps:
          1. Simulate OU process for frequency deviation (mHz).
          2. Add deterministic hourly spikes (DFDs).
          3. Apply FCR droop to convert deviation → activation.
        """
        dt = self._dt
        rng = self._rng

        # --- OU process parameters ---
        # dx = -x/tau * dt + sigma * sqrt(2/tau) * dW
        alpha = dt / _TAU_S
        decay = np.exp(-alpha)
        # OU stationary variance = sigma^2, diffusion chosen accordingly
        noise_std = self._sigma_mhz * np.sqrt(1.0 - decay ** 2)

        # Heavy tails: use t-distribution instead of Gaussian.
        # t-distribution with df degrees of freedom has kurtosis 3 + 6/(df-4).
        # For kurtosis=4.0: 6/(df-4) = 1.0 → df = 10.
        df = 6.0 / (_KURTOSIS_TARGET - 3.0) + 4.0  # = 10.0

        # --- Simulate frequency deviation ---
        freq_dev = np.zeros(n_steps)
        x = 0.0  # Start at nominal frequency

        # Pre-generate t-distributed noise, scaled to unit variance
        # t(df) has variance df/(df-2), so scale to unit variance
        t_scale = np.sqrt((df - 2.0) / df)
        raw_noise = rng.standard_t(df, size=n_steps) * t_scale

        for i in range(n_steps):
            x = decay * x + noise_std * raw_noise[i]
            freq_dev[i] = x

        # --- Add deterministic frequency deviations at hour boundaries ---
        steps_per_hour = int(3600.0 / dt)
        dfd_steps = int(_DFD_DURATION_S / dt)

        for h_start in range(0, n_steps, steps_per_hour):
            # DFD occurs at the start of each hour (block transition)
            magnitude = rng.normal(_DFD_MAGNITUDE_MHZ, _DFD_STD_MHZ)
            sign = rng.choice([-1.0, 1.0])
            magnitude = abs(magnitude) * sign

            # Smooth ramp-up and decay (half-sine envelope)
            for j in range(min(dfd_steps, n_steps - h_start)):
                envelope = np.sin(np.pi * j / dfd_steps)
                freq_dev[h_start + j] += magnitude * envelope

        # --- Apply FCR droop characteristic ---
        signal = self._droop(freq_dev)

        return signal

    @staticmethod
    def _droop(freq_dev_mhz: np.ndarray) -> np.ndarray:
        """Convert frequency deviation [mHz] to activation signal [-1, +1].

        Linear droop between deadband and full activation threshold.
        """
        activation = np.zeros_like(freq_dev_mhz)

        # Positive deviation (frequency too high → down-regulation → charge)
        pos = freq_dev_mhz > _DEADBAND_MHZ
        activation[pos] = -np.clip(
            (freq_dev_mhz[pos] - _DEADBAND_MHZ)
            / (_FULL_ACTIVATION_MHZ - _DEADBAND_MHZ),
            0.0, 1.0,
        )

        # Negative deviation (frequency too low → up-regulation → discharge)
        neg = freq_dev_mhz < -_DEADBAND_MHZ
        activation[neg] = np.clip(
            (-freq_dev_mhz[neg] - _DEADBAND_MHZ)
            / (_FULL_ACTIVATION_MHZ - _DEADBAND_MHZ),
            0.0, 1.0,
        )

        return activation

    def reset(self, seed: int | None = None) -> None:
        """Reset RNG state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self._rp.activation_seed)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Kept for backward compatibility. Returns empty array."""
        return np.array([])
