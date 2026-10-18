# Assumptions and numerical validation

This document distinguishes analytical checks from unavailable historical measurements.

## Signal handling

- All imported text data must be sampled uniformly (up to 0.1% timing deviation); time is in seconds. If time is missing, the caller supplies the sampling rate.
- Three-axis measurements may be imported, but analysis functions require explicit selection of **one channel**.
- WAV integer PCM is normalized to relative full-scale amplitude. Physical acceleration units cannot be inferred from WAV without a sensor calibration.
- FFT/PSD functions operate on real, finite, uniformly sampled arrays with at least four points.

## Spectra and units

Let `X = rfft(window * signal)` and `N` be sample count. Interior one-sided bins are multiplied by 2 in amplitude, and one-sided **power** uses doubled squared magnitude; DC and the even-length Nyquist bin are not doubled.

- `amplitude`: `abs(X) * one_sided_factor / sum(window)` (peak-amplitude units). For a bin-centred tone, peak amplitude is recovered; it is **not** RMS amplitude. Off-bin leakage, window shape and transient signals can prevent peak equality.
- `psd`: `abs(X)**2 * one_sided_factor / (fs * sum(window**2))` (signal_unit²/Hz).
- `power`: `psd * fs/N` (signal_unit² per FFT bin). Its sum approximates the signal mean square under the window-power convention; the PSD integral provides the same quantity.
- For PSD of noise, prefer Welch averaging. `nperseg`, overlap and window parameters affect the estimator's variance, resolution and bias.

A logarithmic plot does not convert linear amplitudes or PSD automatically to decibels. Conversion and the reference level must be chosen explicitly, e.g. `10*log10(PSD/ref_psd)` for power, or `20*log10(A/ref_amplitude)` for amplitude.

## Damping

For an isolated linear underdamped component:

`x(t) = A exp(-alpha*t) cos(2*pi*f_d*t + phase)`

A least-squares fit of `ln(abs(hilbert(x)))` to time gives `alpha > 0`, then

`omega_n = sqrt((2*pi*f_d)**2 + alpha**2)` and `zeta = alpha/omega_n`.

The envelope near record endpoints can be biased by the Hilbert transform. The fit window must avoid forcing, filter transients, beating from nearby modes and noise floor. `fit_r_squared` only scores an exponential envelope fit, not a proof of a single physical vibration mode. A multi-degree-of-freedom system may require FRFs or appropriate operational modal analysis before reporting modal parameters.

## Automated checks

The tests cover exactly bin-centred tonal frequencies and FFT amplitude, spectral power consistency, Welch PSD integration of seeded white noise, three-channel TSV import, PCM WAV normalization, envelope/time alignment, synthetic free-decay damping recovery, and rejection of invalid parameters.

**Historical data:** The original notebooks refer to external local text/WAV files which are not available in this repository. Their saved figures and numbers cannot currently be independently reproduced; we do not claim validation against those measurements.
