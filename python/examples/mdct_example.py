"""MDCT (Modified Discrete Cosine Transform) example."""

import numpy as np
import spectrograms as sg

# -------------------------------------------------------------------
# Generate a test signal
# -------------------------------------------------------------------
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Mix of two tones
signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

# -------------------------------------------------------------------
# MDCT with sine window (perfect reconstruction at 50% hop)
# -------------------------------------------------------------------
params = sg.MdctParams.sine_window(window_size=1024)
print(f"Window size: {params.window_size}")
print(f"Hop size:    {params.hop_size}")
print(f"Coefficients per frame: {params.n_coefficients}")

# Forward MDCT
coefficients = sg.mdct(signal, params)
print(f"\nMDCT output shape: {coefficients.shape}  (n_coefficients x n_frames)")

# Inverse MDCT — perfect reconstruction with sine window + 50% hop
reconstructed = sg.imdct(coefficients, params, original_length=len(signal))
print(f"Reconstructed length: {len(reconstructed)}")

# Verify reconstruction error (interior samples, away from boundary)
margin = params.window_size
interior = slice(margin, len(signal) - margin)
max_err = np.max(np.abs(reconstructed[interior] - signal[interior]))
print(f"Max reconstruction error (interior): {max_err:.2e}  (should be ~1e-14)")

# -------------------------------------------------------------------
# MDCT with custom parameters (e.g. larger window for higher frequency resolution)
# -------------------------------------------------------------------
params_large = sg.MdctParams.sine_window(window_size=2048)
coefs_large = sg.mdct(signal, params_large)
print(f"\nLarge-window MDCT shape: {coefs_large.shape}")

# -------------------------------------------------------------------
# Using a non-sine window (no perfect reconstruction guarantee)
# -------------------------------------------------------------------
params_hann = sg.MdctParams(window_size=1024, hop_size=512, window=sg.WindowType.hanning)
coefs_hann = sg.mdct(signal, params_hann)
print(f"Hanning-window MDCT shape: {coefs_hann.shape}")
rec_hann = sg.imdct(coefs_hann, params_hann, original_length=len(signal))
err_hann = np.max(np.abs(rec_hann[interior] - signal[interior]))
print(f"Hanning reconstruction error (interior): {err_hann:.4f}  (non-PR window)")
