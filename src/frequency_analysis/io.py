"""Explicitly sampled multi-channel signal import."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


@dataclass(frozen=True)
class SignalRecord:
    time_s: np.ndarray
    samples: np.ndarray  # Shape (time, channels), dimensionless for normalized WAV.
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    unit: str = "unit"


def read_text(path: str | Path, *, time_column: str | None = None,
              sampling_rate_hz: float | None = None,
              channels: list[str] | None = None,
              delimiter: str | None = None, unit: str = "unit") -> SignalRecord:
    """Read headered CSV/TSV with time in seconds or an explicit sample rate.

    Time must be uniformly sampled within 0.1% relative interval tolerance.
    Columns must have unique names and all chosen samples must be finite.
    """
    if delimiter is None:
        # Automatic dialect detection on a single-column header is unreliable.
        suffix = Path(path).suffix.lower()
        delimiter = "\t" if suffix in (".tsv", ".txt") else "," if suffix == ".csv" else None
    df = pd.read_csv(path, sep=delimiter, engine="python" if delimiter is None else "c")
    if df.empty or len(df) < 4 or not df.columns.is_unique:
        raise ValueError("a valid header and >=4 rows with unique columns are required")
    if time_column is not None and time_column not in df.columns:
        raise ValueError(f"unknown time column: {time_column}")
    selected = list(channels) if channels is not None else [str(c) for c in df.columns if c != time_column]
    if not selected or len(set(selected)) != len(selected) or any(c not in df.columns for c in selected):
        raise ValueError("select at least one unique existing data column")
    if time_column in selected:
        raise ValueError("time column cannot also be a signal channel")
    data = df[selected].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("signal contains missing or nonfinite values")
    if time_column is not None:
        t = df[time_column].to_numpy(dtype=float)
        dt = np.diff(t)
        if not np.isfinite(t).all() or np.any(dt <= 0):
            raise ValueError("time must be finite and strictly increasing")
        typical = float(np.median(dt))
        if not np.allclose(dt, typical, rtol=1e-3, atol=1e-12):
            raise ValueError("irregular sampling is unsupported; resample explicitly")
        rate = 1 / typical
        if sampling_rate_hz is not None and not np.isclose(rate, sampling_rate_hz, rtol=1e-3):
            raise ValueError("specified sample rate disagrees with the time column")
    else:
        if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz is required without a time column")
        rate = float(sampling_rate_hz)
        t = np.arange(data.shape[0]) / rate
    return SignalRecord(t, data, float(rate), tuple(selected), unit)


def read_wav(path: str | Path) -> SignalRecord:
    """Import mono/stereo PCM or floating-point WAV; PCM normalized to [-1, 1]."""
    rate, raw = wavfile.read(path)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if info.min == 0:  # unsigned 8-bit PCM
            samples = (raw.astype(float) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            samples = raw.astype(float) / max(-int(info.min), int(info.max))
    else:
        samples = raw.astype(float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] < 4 or not np.isfinite(samples).all():
        raise ValueError("WAV contains invalid samples")
    t = np.arange(samples.shape[0], dtype=float) / rate
    return SignalRecord(t, samples, float(rate), tuple(f"ch{i+1}" for i in range(samples.shape[1])), "relative amplitude")
