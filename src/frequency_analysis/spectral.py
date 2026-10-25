"""One-sided spectra and Welch PSD for uniformly sampled real-valued signals."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import find_peaks, get_window, welch


@dataclass(frozen=True)
class Spectrum:
    frequency_hz: np.ndarray
    values: np.ndarray
    scaling: str
    units: str
    resolution_hz: float


@dataclass(frozen=True)
class Peak:
    frequency_hz: float
    value: float
    prominence: float


def _validate(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.ndim != 1 or len(data) < 4:
        raise ValueError("signal must be one-dimensional with at least four samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    return data


def fft_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    scaling: Literal["amplitude", "power", "psd"] = "amplitude",
    window: str = "hann",
    remove_mean: bool = True,
    signal_unit: str = "unit",
) -> Spectrum:
    """Compute a one-sided real FFT spectrum.

    Amplitude is peak sinusoidal amplitude after coherent-gain correction; a
    frequency-bin-centred tone is recovered exactly. For arbitrary frequency
    tones, windowing and leakage affect the peak value. Power is the mean-square
    contribution per FFT bin, equal to PSD times fs / N. Power and PSD use
    window energy, not coherent gain, and are suitable for noise analysis.
    """
    data = _validate(signal, sampling_rate_hz)
    if scaling not in ("amplitude", "power", "psd"):
        raise ValueError("scaling must be 'amplitude', 'power' or 'psd'")
    n = data.size
    w = get_window(window, n, fftbins=True)
    if remove_mean:
        data = data - np.mean(data)
    spectrum = np.fft.rfft(w * data)
    factor = np.full(spectrum.shape, 2.0)
    factor[0] = 1.0
    if n % 2 == 0:
        factor[-1] = 1.0
    df = sampling_rate_hz / n
    if scaling == "amplitude":
        values = np.abs(spectrum) * factor / np.sum(w)
        units = signal_unit
    else:
        psd = factor * np.abs(spectrum) ** 2 / (sampling_rate_hz * np.sum(w * w))
        values = psd if scaling == "psd" else psd * df
        units = f"{signal_unit}²/Hz" if scaling == "psd" else f"{signal_unit}²"
    return Spectrum(np.fft.rfftfreq(n, d=1 / sampling_rate_hz), values, scaling, units, df)


def welch_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    nperseg: int = 1024,
    overlap_fraction: float = 0.5,
    window: str = "hann",
    signal_unit: str = "unit",
) -> Spectrum:
    """Estimate a one-sided power spectral density with Welch averaging."""
    data = _validate(signal, sampling_rate_hz)
    if not isinstance(nperseg, int) or nperseg < 4:
        raise ValueError("nperseg must be an integer >= 4")
    if not (0 <= overlap_fraction < 1):
        raise ValueError("overlap_fraction must lie in [0, 1)")
    nperseg = min(nperseg, data.size)
    noverlap = int(nperseg * overlap_fraction)
    f, p = welch(data, fs=sampling_rate_hz, window=window, nperseg=nperseg,
                 noverlap=noverlap, detrend="constant", scaling="density")
    return Spectrum(f, p, "psd", f"{signal_unit}²/Hz", sampling_rate_hz / nperseg)


def detect_peaks(
    spectrum: Spectrum,
    *,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    min_prominence: float = 0.0,
    min_distance_hz: float = 0.0,
    limit: int | None = None,
) -> list[Peak]:
    """Find local maxima; return them in descending spectral magnitude."""
    f, y = spectrum.frequency_hz, spectrum.values
    nyquist = float(f[-1])
    upper = nyquist if max_frequency_hz is None else float(max_frequency_hz)
    if (not np.isfinite(min_frequency_hz) or min_frequency_hz < 0
            or not np.isfinite(upper) or upper <= min_frequency_hz
            or min_prominence < 0 or not np.isfinite(min_prominence)
            or min_distance_hz < 0 or not np.isfinite(min_distance_hz)):
        raise ValueError("invalid peak-search parameters")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    distance = max(1, int(np.ceil(min_distance_hz / spectrum.resolution_hz)))
    idx, props = find_peaks(y, prominence=min_prominence, distance=distance)
    selected = [(int(i), float(p)) for i, p in zip(idx, props["prominences"])
                if min_frequency_hz <= f[i] <= upper]
    selected.sort(key=lambda v: y[v[0]], reverse=True)
    if limit is not None:
        selected = selected[:limit]
    return [Peak(float(f[i]), float(y[i]), p) for i, p in selected]
