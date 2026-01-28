"""
Spectrograms - FFT-based computations for audio and image processing

This library provides efficient computation of FFT-based operations for both
audio (1D) and image (2D) data using Rust's performance with Python's ease of use.

Audio Processing:
- Linear, Mel, ERB, and CQT spectrograms
- Power, Magnitude, and Decibel scaling
- Plan-based computation for batch processing
- Streaming/frame-by-frame processing
- MFCC and Chromagram features

Image Processing:
- 2D FFT and inverse FFT
- Convolution via FFT (faster for large kernels)
- Spatial filtering (low-pass, high-pass, band-pass)
- Edge detection and sharpening

Audio Example:
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

Image Example:
    >>> import numpy as np
    >>> import spectrograms as sg
    >>>
    >>> # Create a 256x256 image
    >>> image = np.random.randn(256, 256)
    >>>
    >>> # Compute 2D FFT
    >>> spectrum = sg.fft2d(image)
    >>> print(f"Spectrum shape: {spectrum.shape}")  # (256, 129)
    >>>
    >>> # Apply Gaussian blur
    >>> kernel = sg.gaussian_kernel_2d(9, 2.0)
    >>> blurred = sg.convolve_fft(image, kernel)
"""

# Import everything that's actually exported from the Rust module
from ._spectrograms import *

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
    "WindowType",
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
    # Audio Functions
    "compute_linear_power_spectrogram",
    "compute_linear_magnitude_spectrogram",
    "compute_linear_db_spectrogram",
    "compute_mel_power_spectrogram",
    "compute_mel_magnitude_spectrogram",
    "compute_mel_db_spectrogram",
    "compute_erb_power_spectrogram",
    "compute_erb_magnitude_spectrogram",
    "compute_erb_db_spectrogram",
    "compute_chromagram",
    "compute_mfcc",
    "compute_stft",
    # 2D FFT Functions
    "fft2d",
    "ifft2d",
    "power_spectrum_2d",
    "magnitude_spectrum_2d",
    "fftshift",
    "ifftshift",
    # Image Processing Functions
    "gaussian_kernel_2d",
    "convolve_fft",
    "lowpass_filter",
    "highpass_filter",
    "bandpass_filter",
    "detect_edges_fft",
    "sharpen_fft",
    # 2D FFT Planner
    "Fft2dPlanner",
    # Version
    "__version__",
]

# For backwards compatibility, alias CQT functions if available
try:
    compute_cqt = compute_cqt_power_spectrogram
    __all__.append("compute_cqt")
except NameError:
    # CQT not available in this build
    pass
