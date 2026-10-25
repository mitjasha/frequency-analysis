"""Frequency Analysis: reproducible vibration signal processing."""
from .spectral import Peak, Spectrum, detect_peaks, fft_spectrum, welch_spectrum
from .damping import DampingResult, analytic_envelope, bandpass_filter, estimate_damping
from .io import SignalRecord, read_text, read_wav
from .synthetic import damped_sinusoid

__all__ = ["Peak", "Spectrum", "detect_peaks", "fft_spectrum", "welch_spectrum",
           "DampingResult", "analytic_envelope", "bandpass_filter", "estimate_damping",
           "SignalRecord", "read_text", "read_wav", "damped_sinusoid"]
