#!/usr/bin/env python3
"""
Basic Linear Spectrogram Example

Demonstrates computing a simple linear-frequency power spectrogram.
"""

import numpy as np
import spectrograms as sg


def main():
    print("=" * 60)
    print("Basic Linear Spectrogram Example")
    print("=" * 60)

    # Generate a test signal: 1 second of 440 Hz (A4 note)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
    samples = np.sin(2 * np.pi * frequency * t)

    print(f"\nGenerated test signal:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Samples: {len(samples)}")

    # Configure STFT parameters
    stft = sg.StftParams(
        n_fft=512,  # FFT size
        hop_size=256,  # Hop between frames
        window="hanning",  # Window function
        centre=True,  # Center frames with padding
    )

    params = sg.SpectrogramParams(stft, sample_rate=sample_rate)

    print(f"\nSTFT parameters:")
    print(f"  FFT size: {stft.n_fft}")
    print(f"  Hop size: {stft.hop_size}")
    print(f"  Window: {stft.window}")
    print(f"  Centered: {stft.centre}")

    # Compute power spectrogram
    print("\nComputing linear power spectrogram...")
    spec = sg.compute_linear_power_spectrogram(samples, params)

    print(f"\nSpectrogram result:")
    print(f"  Shape: {spec.shape}")
    print(f"  Frequency bins: {spec.n_bins}")
    print(f"  Time frames: {spec.n_frames}")
    print(
        f"  Frequency range: {spec.frequency_range()[0]:.1f} - {spec.frequency_range()[1]:.1f} Hz"
    )
    print(f"  Duration: {spec.duration():.3f} s")
    print(f"  Data type: {spec.data.dtype}")
    print(f"  Data shape: {spec.data.shape}")

    # Show some frequency values
    print(f"\nFirst 5 frequency bins:")
    for i in range(min(5, len(spec.frequencies))):
        print(f"  Bin {i}: {spec.frequencies[i]:.2f} Hz")

    # Show some time values
    print(f"\nFirst 5 time frames:")
    for i in range(min(5, len(spec.times))):
        print(f"  Frame {i}: {spec.times[i]:.4f} s")

    # Find the peak frequency (should be close to 440 Hz)
    # Average power across time
    avg_power = np.mean(spec.data, axis=1)
    peak_bin = np.argmax(avg_power)
    peak_freq = spec.frequencies[peak_bin]

    print(f"\nPeak frequency detected:")
    print(f"  Bin {peak_bin}: {peak_freq:.2f} Hz")
    print(f"  Expected: {frequency:.2f} Hz")
    print(f"  Error: {abs(peak_freq - frequency):.2f} Hz")

    print("\nBasic linear spectrogram example completed!")


if __name__ == "__main__":
    main()
