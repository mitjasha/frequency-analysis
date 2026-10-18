import numpy as np
import pytest

from frequency_analysis import (analytic_envelope, bandpass_filter,
                                damped_sinusoid, estimate_damping)


def test_envelope_samples_align_with_time():
    x = np.sin(2*np.pi*25*np.arange(2000)/1000)
    t, env = analytic_envelope(x, 1000)
    assert len(env) == len(t) == len(x)
    assert t[500] == pytest.approx(0.5)
    assert np.median(env[100:-100]) == pytest.approx(1, abs=0.02)


def test_recover_known_damping_ratio():
    fs = 4000.0
    _, x = damped_sinusoid(sample_rate_hz=fs, duration_s=2,
                           frequency_hz=50, damping_ratio=0.01)
    fit = estimate_damping(x, fs, 50, fit_start_s=0.15, fit_end_s=1.3)
    assert fit.damping_ratio == pytest.approx(.01, rel=.04)
    assert fit.fit_r_squared > .99


def test_zero_phase_filter_reject_invalid_band():
    with pytest.raises(ValueError):
        bandpass_filter(np.ones(500), 1000, 250, 600)


def test_reject_nondecaying_response():
    fs = 1000
    t = np.arange(3000)/fs
    x = np.exp(0.3*t)*np.cos(2*np.pi*30*t)
    with pytest.raises(ValueError, match="not decaying"):
        estimate_damping(x, fs, 30, fit_start_s=.2, fit_end_s=2)


def test_reject_bad_fit_interval():
    _, x = damped_sinusoid(sample_rate_hz=1000, duration_s=1,
                           frequency_hz=50, damping_ratio=.02)
    with pytest.raises(ValueError):
        estimate_damping(x, 1000, 50, fit_start_s=.5, fit_end_s=2)
