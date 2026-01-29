#!/usr/bin/env python3
"""
Window Function Comparison Example

Demonstrates the effect of different window functions on spectrograms.
"""

import numpy as np
import spectrograms as sg


def main():
    print("=" * 60)
    print("Window Function Comparison")
    print("=" * 60)

    # Generate a test signal with multiple frequencies
    sample_rate = 16000
    duration = 0.5

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)

    # Mix of three frequencies: 440 Hz (A4), 554 Hz (C#5), 659 Hz (E5) - A major chord
    signal = (
        np.sin(2 * np.pi * 440 * t)
        + 0.8 * np.sin(2 * np.pi * 554 * t)
        + 0.6 * np.sin(2 * np.pi * 659 * t)
    )

    print(f"\nTest signal:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Frequencies: 440 Hz, 554 Hz, 659 Hz (A major chord)")

    # Window functions to compare
    windows = [
        ("hanning", "Hann Window"),
        ("hamming", "Hamming Window"),
        ("blackman", "Blackman Window"),
        ("bartlett", "Bartlett (Triangular) Window"),
        ("rectangular", "Rectangular Window (no windowing)"),
        ("kaiser=5.0", "Kaiser Window (β=5.0)"),
        ("kaiser=8.6", "Kaiser Window (β=8.6)"),
        ("gaussian=0.4", "Gaussian Window (σ=0.4)"),
    ]

    # Common STFT parameters
    n_fft = 512
    hop_size = 256

    print(f"\nSTFT configuration:")
    print(f"  FFT size: {n_fft}")
    print(f"  Hop size: {hop_size}")

    results = []

    # Compute spectrogram with each window
    for window_name, window_desc in windows:
        print(f"\n" + "-" * 60)
        print(f"{window_desc}")
        print("-" * 60)

        stft = sg.StftParams(
            n_fft=n_fft, hop_size=hop_size, window=window_name, centre=True
        )

        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)

        # Compute power spectrogram
        spec = sg.compute_linear_power_spectrogram(signal, params)

        print(f"Window: {stft.window}")
        print(f"Shape: {spec.shape}")

        # Find the three peaks
        avg_power = np.mean(spec.data, axis=1)
        top_bins = np.argsort(avg_power)[-3:][::-1]  # Top 3 bins, descending

        print(f"\nTop 3 frequency peaks:")
        for i, bin_idx in enumerate(top_bins):
            freq = spec.frequencies[bin_idx]
            power = avg_power[bin_idx]
            print(f"  {i + 1}. Bin {bin_idx:3d}: {freq:7.2f} Hz (power: {power:.2e})")

        # Calculate spectral leakage (energy outside main peaks)
        main_peak_energy = np.sum(avg_power[top_bins])
        total_energy = np.sum(avg_power)
        leakage_ratio = (total_energy - main_peak_energy) / total_energy * 100

        print(f"\nSpectral leakage: {leakage_ratio:.2f}%")

        results.append(
            {
                "window": window_desc,
                "window_name": window_name,
                "peaks": [(spec.frequencies[i], avg_power[i]) for i in top_bins],
                "leakage": leakage_ratio,
                "max_power": np.max(avg_power),
            }
        )

    # ========================================================================
    # Summary comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary: Window Function Characteristics")
    print("=" * 60)

    print("\n{:<35} {:>12} {:>12}".format("Window", "Leakage", "Peak Power"))
    print("-" * 60)
    for result in results:
        print(
            "{:<35} {:>11.2f}% {:>12.2e}".format(
                result["window"], result["leakage"], result["max_power"]
            )
        )

    print("\n" + "=" * 60)
    print("Window Function Guidelines")
    print("=" * 60)

    guidelines = [
        ("Hann (hanning)", "Good general-purpose window, smooth sidelobes"),
        ("Hamming", "Better frequency resolution than Hann, slightly higher sidelobes"),
        ("Blackman", "Excellent sidelobe suppression, wider main lobe"),
        ("Bartlett", "Simple triangular window, moderate performance"),
        ("Rectangular", "No windowing, highest leakage, best time localization"),
        ("Kaiser", "Adjustable trade-off between main lobe width and sidelobe level"),
        ("Gaussian", "Smooth window, good time-frequency localization"),
    ]

    for window, description in guidelines:
        print(f"\n{window}:")
        print(f"  {description}")

    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    print("\nFor speech analysis:")
    print("  • Hann window (default)")
    print("  • Good balance of frequency and time resolution")

    print("\nFor music analysis:")
    print("  • Blackman window")
    print("  • Better frequency discrimination for harmonic content")

    print("\nFor transient detection:")
    print("  • Gaussian with small σ")
    print("  • Better time localization")

    print("\nFor spectral analysis:")
    print("  • Kaiser window with β=8.6")
    print("  • Adjustable sidelobe suppression")

    print("\nWindow function comparison completed!")


if __name__ == "__main__":
    main()
