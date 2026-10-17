# Frequency Analysis

**Scientific signal processing and vibration frequency analysis**

Frequency Analysis is an evolving open-source Python toolkit for reproducible analysis of measured vibration signals. The project began with experimental work on three-axis recordings and acoustic signals. Its original Jupyter notebooks are preserved in [`research/legacy`](research/legacy) without rewriting historical results.

Version **0.1.0** establishes a testable signal-processing core. Planned research directions include experimental modal identification, damping estimation, multichannel measurements, and experiment-to-FEM correlation.

## Implemented in v0.1

- Headered CSV/TSV and mono/multichannel WAV import, explicit sampling rates and channel selection.
- Real one-sided FFT with separate **peak-amplitude**, **power-per-bin**, and **power spectral density** normalizations.
- Welch PSD, configurable peak detection and zero-phase Butterworth bandpass filtering.
- Sample-aligned Hilbert envelope and exponential free-decay damping estimation with fit R².
- Deterministic damped synthetic reference signals, numerical verification tests, and CI.

**Scope:** This release identifies spectral peaks and damping under explicitly stated assumptions. A peak is **not automatically an experimentally identified structural mode**. Modal shapes, force-response functions, operational modal analysis, FEM correlation, and a GUI are planned, **not yet implemented**.

## Install

Python 3.10+:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/analyze_synthetic.py
```

## Quick start

```python
from frequency_analysis import read_text, fft_spectrum, detect_peaks

record = read_text("measurement.tsv", time_column="time_s", channels=["accel_z"], unit="m/s²")
spectrum = fft_spectrum(record.samples[:, 0], record.sampling_rate_hz,
                        scaling="amplitude", signal_unit=record.unit)
peaks = detect_peaks(spectrum, min_frequency_hz=5, min_prominence=0.01, limit=5)
for peak in peaks:
    print(f"{peak.frequency_hz:.2f} Hz: {peak.value:.5g} {spectrum.units}")
```

For a text file without a time column, supply `sampling_rate_hz=...` explicitly. A standalone PSD:

```python
from frequency_analysis import welch_spectrum
psd = welch_spectrum(record.samples[:, 0], record.sampling_rate_hz, nperseg=2048)
```

## Damping analysis

```python
from frequency_analysis import estimate_damping
result = estimate_damping(record.samples[:, 0], record.sampling_rate_hz,
                          damped_frequency_hz=50,
                          fit_start_s=0.15, fit_end_s=1.3)
print(result.damping_ratio, result.fit_r_squared)
```

The selected interval **must represent isolated, approximately linear single-mode free decay**, after initial forcing and before the noise floor. For multi-mode signals, first isolate the band if justified, examine the envelope and reassess fit quality. See [validation and limitations](docs/validation.md).

## Project layout

- `src/frequency_analysis/` — reusable analysis library
- `tests/` — synthetic reference tests
- `examples/` — reproducible calculations
- `research/legacy/` — original 2020/2021 research notebooks, unchanged
- `docs/` — assumptions, verification results, and research roadmap

## Roadmap

- **0.2** — desktop GUI (English/Russian), interactive spectral and time-domain views, exports.
- **0.3** — FRF/coherence with force inputs, modal identification and uncertainty evaluation.
- **0.4** — experimental mode shape mapping and FEM mode comparison (including MAC where applicable).
- **1.0** — validated end-to-end engineering workflows with experimental references.

Contributions and reproducible measurement cases are welcome. The license has not yet been selected; until clarified, repository availability alone does not grant redistribution rights.
