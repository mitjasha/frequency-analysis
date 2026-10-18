"""Generate a known vibration and compare identified values to its inputs."""
from frequency_analysis import damped_sinusoid, detect_peaks, estimate_damping, fft_spectrum

fs = 4000
time, signal = damped_sinusoid(sample_rate_hz=fs, duration_s=2.0,
                               frequency_hz=50.0, damping_ratio=0.01)
peaks = detect_peaks(fft_spectrum(signal, fs), min_frequency_hz=10, limit=1)
fit = estimate_damping(signal, fs, 50.0, fit_start_s=0.15, fit_end_s=1.3)
print(f"Main spectral peak: {peaks[0].frequency_hz:.2f} Hz")
print(f"Known damping ratio: 1.000%")
print(f"Estimated damping ratio: {fit.damping_ratio*100:.3f}% (fit R²={fit.fit_r_squared:.4f})")
