#!/usr/bin/env python3
"""
MFCC (Mel-Frequency Cepstral Coefficients) Example

Demonstrates computing MFCCs, which are widely used features in speech recognition
and music analysis.
"""

import numpy as np
import spectrograms as sg


def generate_speech_like_signal(sample_rate, duration):
    """Generate a signal with formant-like structure (simulating speech)."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)

    # Fundamental frequency (pitch) - varies like speech prosody
    f0 = 120 + 30 * np.sin(2 * np.pi * 2 * t)  # 120-150 Hz

    # Three formants (resonant frequencies typical of vowels)
    f1 = 700  # First formant
    f2 = 1220  # Second formant
    f3 = 2600  # Third formant

    # Generate signal with formants
    signal = (
        np.sin(2 * np.pi * f0 * t)  # Fundamental
        + 0.7 * np.sin(2 * np.pi * f1 * t)  # F1
        + 0.5 * np.sin(2 * np.pi * f2 * t)  # F2
        + 0.3 * np.sin(2 * np.pi * f3 * t)  # F3
    )

    # Add some noise (breathiness)
    signal += 0.05 * np.random.randn(len(t))

    # Apply amplitude envelope (like speech intensity variation)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    signal *= envelope

    return signal


def main():
    print("=" * 60)
    print("MFCC (Mel-Frequency Cepstral Coefficients) Example")
    print("=" * 60)

    # Generate a speech-like test signal
    sample_rate = 16000
    duration = 1.0

    signal = generate_speech_like_signal(sample_rate, duration)

    print(f"\nGenerated speech-like signal:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Samples: {len(signal)}")

    # Configure STFT parameters (typical for speech)
    stft = sg.StftParams(
        n_fft=512,  # 32ms frames at 16kHz
        hop_size=160,  # 10ms hop (typical for speech)
        window="hanning",
        centre=True,
    )

    print(f"\nSTFT parameters:")
    print(f"  FFT size: {stft.n_fft} ({stft.n_fft / sample_rate * 1000:.1f} ms)")
    print(f"  Hop size: {stft.hop_size} ({stft.hop_size / sample_rate * 1000:.1f} ms)")
    print(f"  Window: {stft.window}")

    # ========================================================================
    # Standard MFCCs (13 coefficients - typical for speech recognition)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Standard MFCCs (13 coefficients)")
    print("=" * 60)

    mfcc_params = sg.MfccParams(n_mfcc=13)
    n_mels = 40  # Standard number of mel bands for MFCCs

    print(f"\nMFCC parameters:")
    print(f"  Number of coefficients: {mfcc_params.n_mfcc}")
    print(f"  Number of mel bands: {n_mels}")

    print("\nComputing MFCCs...")
    mfccs = sg.compute_mfcc(signal, stft, sample_rate, n_mels, mfcc_params)

    print(f"\nMFCCs computed:")
    print(f"  Shape: {mfccs.shape} (n_mfcc x n_frames)")
    print(f"  Number of frames: {mfccs.shape[1]}")
    print(f"  Frame rate: {mfccs.shape[1] / duration:.1f} frames/second")

    # Show statistics for each coefficient
    print(f"\nMFCC statistics:")
    print(f"{'Coeff':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("-" * 60)

    for i in range(mfccs.shape[0]):
        mean = np.mean(mfccs[i, :])
        std = np.std(mfccs[i, :])
        min_val = np.min(mfccs[i, :])
        max_val = np.max(mfccs[i, :])
        print(
            f"  C{i:2d}   {mean:10.3f}  {std:10.3f}  {min_val:10.3f}  {max_val:10.3f}"
        )

    # ========================================================================
    # Using the speech standard preset
    # ========================================================================
    print("\n" + "=" * 60)
    print("Using Speech Standard Preset")
    print("=" * 60)

    standard_mfcc_params = sg.MfccParams.speech_standard()
    print(f"\nSpeech standard parameters:")
    print(f"  Number of coefficients: {standard_mfcc_params.n_mfcc}")

    mfccs_standard = sg.compute_mfcc(
        signal, stft, sample_rate, n_mels, standard_mfcc_params
    )
    print(f"\nStandard MFCCs computed: {mfccs_standard.shape}")

    # Verify they're the same (both use 13 coefficients)
    print(f"\nVerifying results match:")
    print(f"  Shapes match: {mfccs.shape == mfccs_standard.shape}")
    print(f"  Values match: {np.allclose(mfccs, mfccs_standard)}")

    # ========================================================================
    # Different numbers of coefficients
    # ========================================================================
    print("\n" + "=" * 60)
    print("Comparing Different Numbers of Coefficients")
    print("=" * 60)

    coefficient_counts = [13, 20, 26]

    for n_coeff in coefficient_counts:
        params = sg.MfccParams(n_mfcc=n_coeff)
        result = sg.compute_mfcc(signal, stft, sample_rate, n_mels, params)
        print(f"\n{n_coeff} coefficients:")
        print(f"  Shape: {result.shape}")
        print(f"  Data range: [{np.min(result):.2f}, {np.max(result):.2f}]")

    # ========================================================================
    # Understanding MFCC coefficients
    # ========================================================================
    print("\n" + "=" * 60)
    print("Understanding MFCC Coefficients")
    print("=" * 60)

    print("\nMFCC coefficient interpretation:")
    print("  C0:  Energy/loudness (often excluded or replaced with log energy)")
    print("  C1:  Spectral slope (balance between low and high frequencies)")
    print("  C2:  Spectral shape (formant structure)")
    print("  C3+: Fine spectral details")

    # Show typical coefficient ranges
    print("\nTypical coefficient behavior:")
    c0_range = np.max(mfccs[0, :]) - np.min(mfccs[0, :])
    c1_range = np.max(mfccs[1, :]) - np.min(mfccs[1, :])
    c2_range = np.max(mfccs[2, :]) - np.min(mfccs[2, :])

    print(f"  C0 variation: {c0_range:.2f} (largest - captures overall energy)")
    print(f"  C1 variation: {c1_range:.2f} (captures spectral tilt)")
    print(f"  C2 variation: {c2_range:.2f} (captures spectral shape)")

    # ========================================================================
    # Frame-by-frame analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("Frame-by-Frame Analysis")
    print("=" * 60)

    # Show MFCCs for first few frames
    n_show = min(5, mfccs.shape[1])

    print(f"\nFirst {n_show} frames:")
    for frame_idx in range(n_show):
        print(
            f"\nFrame {frame_idx} (t = {frame_idx * stft.hop_size / sample_rate:.3f}s):"
        )
        print(
            f"  MFCCs: [{', '.join(f'{mfccs[i, frame_idx]:6.2f}' for i in range(min(5, mfccs.shape[0])))}...]"
        )

    # ========================================================================
    # Feature normalization (common in ML applications)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Feature Normalization (for ML)")
    print("=" * 60)

    # Mean-variance normalization (per coefficient across time)
    mfccs_normalized = np.zeros_like(mfccs)
    for i in range(mfccs.shape[0]):
        mean = np.mean(mfccs[i, :])
        std = np.std(mfccs[i, :])
        if std > 0:
            mfccs_normalized[i, :] = (mfccs[i, :] - mean) / std

    print("\nNormalized MFCCs (zero mean, unit variance per coefficient):")
    print(
        f"  Mean of means: {np.mean([np.mean(mfccs_normalized[i, :]) for i in range(mfccs_normalized.shape[0])]):.6f}"
    )
    print(
        f"  Mean of stds:  {np.mean([np.std(mfccs_normalized[i, :]) for i in range(mfccs_normalized.shape[0])]):.6f}"
    )

    # ========================================================================
    # Application examples
    # ========================================================================
    print("\n" + "=" * 60)
    print("Common Applications")
    print("=" * 60)

    print("\n1. Speech Recognition:")
    print("   • Use 13 MFCCs + deltas + delta-deltas (39 features)")
    print("   • 10ms frame rate (hop_size=160 at 16kHz)")
    print("   • Feed into HMM or DNN acoustic model")

    print("\n2. Speaker Recognition:")
    print("   • Use 13-20 MFCCs")
    print("   • Compute statistics (mean, covariance) over utterances")
    print("   • Model speaker characteristics with GMM or i-vectors")

    print("\n3. Music Analysis:")
    print("   • Use 13-20 MFCCs with higher sample rate")
    print("   • Useful for genre classification, similarity")
    print("   • Captures timbral characteristics")

    print("\n4. Audio Fingerprinting:")
    print("   • Use compact MFCC representation")
    print("   • Fast similarity matching")
    print("   • Robust to noise and compression")

    print("\nMFCC example completed!")


if __name__ == "__main__":
    main()
