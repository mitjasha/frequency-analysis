"""Reproducible reference signals for numerical verification."""
import numpy as np


def damped_sinusoid(*, sample_rate_hz: float, duration_s: float,
                    frequency_hz: float, damping_ratio: float,
                    amplitude: float = 1.0, phase_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Under-damped free response with specified damped frequency and zeta."""
    if not (sample_rate_hz > 0 and duration_s > 0 and 0 < frequency_hz < sample_rate_hz / 2
            and 0 <= damping_ratio < 1 and all(np.isfinite(x) for x in
            (sample_rate_hz, duration_s, frequency_hz, damping_ratio, amplitude, phase_rad))):
        raise ValueError("invalid signal parameters")
    t = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
    omega_d = 2 * np.pi * frequency_hz
    alpha = omega_d * damping_ratio / np.sqrt(1 - damping_ratio**2)
    return t, amplitude * np.exp(-alpha * t) * np.cos(omega_d * t + phase_rad)
