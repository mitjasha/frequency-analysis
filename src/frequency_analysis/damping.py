"""Time-domain damping estimation for a well-isolated decaying mode."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .spectral import _validate


@dataclass(frozen=True)
class DampingResult:
    damped_frequency_hz: float
    decay_rate_per_s: float
    damping_ratio: float
    undamped_frequency_hz: float
    fit_r_squared: float
    start_time_s: float
    end_time_s: float


def bandpass_filter(signal: np.ndarray, sampling_rate_hz: float,
                    lowcut_hz: float, highcut_hz: float, *, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filtering; invalid bands/short records raise errors."""
    data = _validate(signal, sampling_rate_hz)
    if not (0 < lowcut_hz < highcut_hz < sampling_rate_hz / 2):
        raise ValueError("cutoffs must satisfy 0 < low < high < Nyquist")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    sos = butter(order, (lowcut_hz, highcut_hz), btype="bandpass",
                 fs=sampling_rate_hz, output="sos")
    return sosfiltfilt(sos, data)


def analytic_envelope(signal: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return time (seconds) and Hilbert-magnitude envelope at matching indices."""
    data = _validate(signal, sampling_rate_hz)
    time = np.arange(data.size, dtype=float) / sampling_rate_hz
    return time, np.abs(hilbert(data))


def estimate_damping(
    signal: np.ndarray,
    sampling_rate_hz: float,
    damped_frequency_hz: float,
    *,
    fit_start_s: float,
    fit_end_s: float,
    band_hz: tuple[float, float] | None = None,
) -> DampingResult:
    """Fit ln(Hilbert envelope) = intercept - decay_rate * time.

    For an isolated underdamped free decay x(t)=A exp(-alpha t) sin(omega_d*t),
    omega_n=sqrt(omega_d**2+alpha**2), zeta=alpha/omega_n. The user must select
    a reliable free-decay interval away from impacts, noise floor, and Hilbert
    endpoint transients. R² is a diagnostic, not proof of a modal interpretation.
    """
    data = _validate(signal, sampling_rate_hz)
    if not (np.isfinite(damped_frequency_hz) and 0 < damped_frequency_hz < sampling_rate_hz / 2):
        raise ValueError("damped_frequency_hz must lie within (0, Nyquist)")
    duration = (data.size - 1) / sampling_rate_hz
    if not (np.isfinite(fit_start_s) and np.isfinite(fit_end_s)
            and 0 <= fit_start_s < fit_end_s <= duration):
        raise ValueError("fit interval must be within the recorded time")
    if band_hz is not None:
        data = bandpass_filter(data, sampling_rate_hz, *band_hz)
    t, envelope = analytic_envelope(data, sampling_rate_hz)
    mask = (t >= fit_start_s) & (t <= fit_end_s)
    if mask.sum() < 8:
        raise ValueError("fit interval must contain at least eight samples")
    selected = envelope[mask]
    if np.any(selected <= 0):
        raise ValueError("envelope contains zero values in fit interval")
    log_amplitude = np.log(selected)
    x = t[mask]
    slope, intercept = np.polyfit(x, log_amplitude, 1)
    prediction = slope * x + intercept
    sst = np.sum((log_amplitude - log_amplitude.mean()) ** 2)
    r2 = 1 - np.sum((log_amplitude - prediction) ** 2) / sst if sst > 0 else 0.0
    alpha = float(-slope)
    if alpha <= 0:
        raise ValueError("fitted envelope is not decaying")
    omega_n = float(np.hypot(2 * np.pi * damped_frequency_hz, alpha))
    return DampingResult(float(damped_frequency_hz), alpha, alpha / omega_n,
                         omega_n / (2 * np.pi), float(r2), float(fit_start_s), float(fit_end_s))
