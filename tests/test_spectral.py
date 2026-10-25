import numpy as np
import pytest

from frequency_analysis import detect_peaks, fft_spectrum, welch_spectrum


def test_bin_centered_amplitude_and_frequency():
    fs, n = 4096, 4096
    t = np.arange(n) / fs
    x = 2.4 * np.cos(2 * np.pi * 128 * t)
    s = fft_spectrum(x, fs, scaling="amplitude", window="hann", signal_unit="m/s²")
    assert s.resolution_hz == 1
    peaks = detect_peaks(s, min_frequency_hz=10, limit=1)
    assert peaks[0].frequency_hz == pytest.approx(128)
    assert peaks[0].value == pytest.approx(2.4, rel=1e-10)
    assert s.units == "m/s²"


def test_power_and_psd_integrate_to_mean_square():
    fs, n = 4096, 4096
    x = 2.0 * np.sin(2 * np.pi * 128 * np.arange(n) / fs)
    p = fft_spectrum(x, fs, scaling="power")
    density = fft_spectrum(x, fs, scaling="psd")
    assert p.values.sum() == pytest.approx(2, rel=1e-8)
    assert density.values.sum() * density.resolution_hz == pytest.approx(2, rel=1e-8)
    assert np.allclose(p.values, density.values * density.resolution_hz)


def test_welch_psd_white_noise_level():
    rng = np.random.default_rng(2026)
    fs = 2000.0
    data = rng.normal(scale=3.0, size=200_000)
    p = welch_spectrum(data, fs, nperseg=2000)
    assert p.units == "unit²/Hz"
    assert np.sum(p.values) * p.resolution_hz == pytest.approx(9.0, rel=0.08)


def test_peak_distance_and_frequency_range():
    fs, n = 2048, 2048
    x = np.sin(2*np.pi*90*np.arange(n)/fs) + .5*np.sin(2*np.pi*240*np.arange(n)/fs)
    s = fft_spectrum(x, fs)
    peaks = detect_peaks(s, min_frequency_hz=200, max_frequency_hz=300,
                         min_prominence=0.1)
    assert peaks[0].frequency_hz == pytest.approx(240)


@pytest.mark.parametrize("bad_fs", [-1, 0, float('nan')])
def test_reject_bad_rate(bad_fs):
    with pytest.raises(ValueError):
        fft_spectrum(np.arange(16), bad_fs)


def test_reject_multichannel_without_selection():
    with pytest.raises(ValueError):
        fft_spectrum(np.ones((100, 2)), 1000)


@pytest.mark.parametrize("limit", [1.5, True, 0, -1])
def test_reject_non_positive_integer_peak_limit(limit):
    spectrum = fft_spectrum(np.arange(16), 1000)
    with pytest.raises(ValueError, match="positive integer"):
        detect_peaks(spectrum, limit=limit)
