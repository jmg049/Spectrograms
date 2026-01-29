#!/usr/bin/env python3
"""
Custom Window Functions Example
================================

This example demonstrates how to use custom window functions with the spectrograms library.
You can use windows from NumPy, SciPy, or create your own custom designs.
"""

import numpy as np
import spectrograms as sg
from scipy.signal.windows import tukey

# Generate a test signal: 440 Hz tone
sample_rate = 16000
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
signal = np.sin(2 * np.pi * 440 * t)

n_fft = 512
hop_size = 256

print("=" * 70)
print("Custom Window Functions Example")
print("=" * 70)

# Example 1: Using NumPy windows
print("\n1. Using NumPy Blackman Window")
print("-" * 70)

numpy_window = sg.WindowType.custom(np.blackman(n_fft))
stft = sg.StftParams(n_fft=n_fft, hop_size=hop_size, window=numpy_window)
params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
spec = sg.compute_linear_power_spectrogram(signal, params)
print(f"Window: {numpy_window}")

print(f"Spectrogram shape: {spec.data.shape}")

# Example 2: Using SciPy windows (if available)
print("\n2. Using SciPy Tukey Window")
print("-" * 70)

scipy_window = sg.WindowType.custom(tukey(n_fft, alpha=0.5))
stft = sg.StftParams(n_fft=n_fft, hop_size=hop_size, window=scipy_window)
params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
spec = sg.compute_linear_power_spectrogram(signal, params)

print(f"Window: {scipy_window}")
print(f"Spectrogram shape: {spec.data.shape}")


# Example 3: Creating a custom window design
print("\n3. Creating a Custom-Designed Window")
print("-" * 70)

# Create a window that's a product of Hanning and a Gaussian
hann = sg.WindowType.make_hanning(n_fft)
gaussian = np.exp(-0.5 * ((np.arange(n_fft) - n_fft / 2) / (n_fft / 8)) ** 2)
custom_design = hann * gaussian
custom_window = sg.WindowType.custom(custom_design)
stft = sg.StftParams(n_fft=n_fft, hop_size=hop_size, window=custom_window)
params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
spec = sg.compute_linear_power_spectrogram(signal, params)

print(f"Window: {custom_window}")
print(f"Spectrogram shape: {spec.data.shape}")

# Example 4: Using normalization
print("\n4. Using Window Normalization")
print("-" * 70)

# Create a window normalized to unit sum
window_coeffs = sg.WindowType.make_hamming(n_fft)
print(f"Original sum: {window_coeffs.sum():.6f}")

window_sum_norm = sg.WindowType.custom(window_coeffs.copy(), normalize="sum")
print(f"Window with sum normalization: {window_sum_norm}")

window_peak_norm = sg.WindowType.custom(window_coeffs.copy(), normalize="peak")
print(f"Window with peak normalization: {window_peak_norm}")

window_energy_norm = sg.WindowType.custom(window_coeffs.copy(), normalize="energy")
print(f"Window with energy normalization: {window_energy_norm}")

# Example 5: Comparing different windows
print("\n5. Comparing Different Windows")
print("-" * 70)

windows = {
    "Rectangular": sg.WindowType.rectangular,
    "Hanning": sg.WindowType.hanning,
    "Hamming": sg.WindowType.hamming,
    "Blackman": sg.WindowType.blackman,
    "Kaiser (β=5)": sg.WindowType.kaiser(5.0),
    "Custom NumPy Blackman": sg.WindowType.custom(np.blackman(n_fft)),
}

for name, window in windows.items():
    stft = sg.StftParams(n_fft=n_fft, hop_size=hop_size, window=window)
    params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
    spec = sg.compute_linear_power_spectrogram(signal, params)
    print(f"{name:25s} -> Shape: {spec.data.shape}")

# Example 6: Error handling
print("\n6. Error Handling Examples")
print("-" * 70)

# Size mismatch detected early
try:
    wrong_size = sg.WindowType.custom(np.blackman(256))
    stft = sg.StftParams(n_fft=512, hop_size=256, window=wrong_size)
except Exception as e:
    print(f"Size mismatch caught: {e}")

# Empty array
try:
    empty_window = sg.WindowType.custom(np.array([]))
except Exception as e:
    print(f"Empty array caught: {e}")

# NaN values
try:
    nan_window = sg.WindowType.custom(np.array([1.0, np.nan, 1.0]))
except Exception as e:
    print(f"NaN values caught: {e}")

# Invalid normalization
try:
    invalid_norm = sg.WindowType.custom(np.hamming(512), normalize="invalid")
except Exception as e:
    print(f"Invalid normalization caught: {e}")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
