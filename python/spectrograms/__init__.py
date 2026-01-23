"""
Spectrograms - Fast spectrogram computation library powered by Rust

This library provides efficient computation of spectrograms and related
audio features using Rust's performance with Python's ease of use.

Supports:
- Linear, Mel, ERB, and CQT spectrograms
- Power, Magnitude, and Decibel scaling
- Plan-based computation for batch processing
- Streaming/frame-by-frame processing
- MFCC and Chromagram features

Example:
    >>> import numpy as np
    >>> import spectrograms as sg
    >>>
    >>> # Generate a test signal
    >>> sr = 16000
    >>> t = np.linspace(0, 1, sr)
    >>> samples = np.sin(2 * np.pi * 440 * t)
    >>>
    >>> # Create parameters
    >>> stft = sg.StftParams(n_fft=512, hop_size=256, window="hanning")
    >>> params = sg.SpectrogramParams(stft, sample_rate=sr)
    >>>
    >>> # Compute spectrogram
    >>> spec = sg.compute_linear_power_spectrogram(samples, params)
    >>> print(f"Shape: {spec.shape}")
"""

# Import everything that's actually exported from the Rust module
from ._spectrograms import *

# For backwards compatibility, alias CQT functions
try:
    compute_cqt = compute_cqt_power_spectrogram
except NameError:
    pass

__all__ = [
    # Exceptions
    "SpectrogramError",
    "InvalidInputError",
    "DimensionMismatchError",
    "FFTBackendError",
    "InternalError",
    # Parameters
    "StftParams",
    "LogParams",
    "SpectrogramParams",
    "MelParams",
    "ErbParams",
    "CqtParams",
    "ChromaParams",
    "MfccParams",
    # Results
    "Spectrogram",
    # Planner
    "SpectrogramPlanner",
    "LinearPowerPlan",
    "LinearMagnitudePlan",
    "LinearDbPlan",
    "MelPowerPlan",
    "MelMagnitudePlan",
    "MelDbPlan",
    "ErbPowerPlan",
    "ErbMagnitudePlan",
    "ErbDbPlan",
    # Functions
    "compute_linear_power_spectrogram",
    "compute_linear_magnitude_spectrogram",
    "compute_linear_db_spectrogram",
    "compute_mel_power_spectrogram",
    "compute_mel_magnitude_spectrogram",
    "compute_mel_db_spectrogram",
    "compute_erb_power_spectrogram",
    "compute_erb_magnitude_spectrogram",
    "compute_erb_db_spectrogram",
    "compute_cqt",
    "compute_chromagram",
    "compute_mfcc",
    "compute_stft",
    # Version
    "__version__",
]
