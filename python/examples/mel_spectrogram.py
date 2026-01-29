#!/usr/bin/env python3
"""
Mel Spectrogram Example

Demonstrates computing mel-scale spectrograms with different amplitude scales.
"""

import numpy as np
import spectrograms as sg


def generate_chirp(sample_rate, duration, f_start, f_end):
    """Generate a linear frequency chirp from f_start to f_end."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
    # Instantaneous frequency increases linearly
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.sin(phase)


def main():
    print("=" * 60)
    print("Mel Spectrogram Example")
    print("=" * 60)

    # Generate a chirp signal (frequency sweep from 200 Hz to 4000 Hz)
    sample_rate = 16000
    duration = 2.0
    f_start = 200.0
    f_end = 4000.0

    samples = generate_chirp(sample_rate, duration, f_start, f_end)

    print("\nGenerated chirp signal:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Frequency sweep: {f_start} Hz → {f_end} Hz")
    print(f"  Samples: {len(samples)}")

    # Configure parameters
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    params = sg.SpectrogramParams(stft, sample_rate=sample_rate)

    # Mel filterbank parameters
    mel_params = sg.MelParams(
        n_mels=80,  # Number of mel bands
        f_min=0.0,  # Minimum frequency
        f_max=8000.0,  # Maximum frequency (Nyquist)
    )

    print("\nMel filterbank parameters:")
    print(f"  Number of mel bands: {mel_params.n_mels}")
    print(f"  Frequency range: {mel_params.f_min} - {mel_params.f_max} Hz")

    # 1. Compute mel power spectrogram
    print("\n" + "-" * 60)
    print("1. Mel Power Spectrogram")
    print("-" * 60)

    mel_power = sg.compute_mel_power_spectrogram(samples, params, mel_params)

    print(f"Shape: {mel_power.shape}")
    print(f"Data range: {np.min(mel_power.data):.2e} to {np.max(mel_power.data):.2e}")
    print(f"Mel frequency range: {mel_power.frequency_range()}")

    # 2. Compute mel magnitude spectrogram
    print("\n" + "-" * 60)
    print("2. Mel Magnitude Spectrogram")
    print("-" * 60)

    mel_mag = sg.compute_mel_magnitude_spectrogram(samples, params, mel_params)

    print(f"Shape: {mel_mag.shape}")
    print(f"Data range: {np.min(mel_mag.data):.2e} to {np.max(mel_mag.data):.2e}")

    # 3. Compute mel dB spectrogram (most common for visualization)
    print("\n" + "-" * 60)
    print("3. Mel dB Spectrogram")
    print("-" * 60)

    db_params = sg.LogParams(floor_db=-80.0)  # Clip values below -80 dB
    print(f"dB floor: {db_params.floor_db} dB")

    mel_db = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)

    print(f"Shape: {mel_db.shape}")
    print(f"Data range: {np.min(mel_db.data):.2f} to {np.max(mel_db.data):.2f} dB")
    print(f"Duration: {mel_db.duration():.3f} s")

    # Show mel frequency scale (first 10 bands)
    print("\nFirst 10 mel frequency bands:")
    for i in range(min(10, len(mel_db.frequencies))):
        print(f"  Band {i:2d}: {mel_db.frequencies[i]:7.2f} Hz")

    # Compare different scales
    print("\n" + "=" * 60)
    print("Comparison of Amplitude Scales")
    print("=" * 60)

    # Get values at a specific time/frequency point (middle of spectrogram)
    mid_time = mel_power.n_frames // 2
    mid_freq = mel_power.n_bins // 2

    power_val = mel_power.data[mid_freq, mid_time]
    mag_val = mel_mag.data[mid_freq, mid_time]
    db_val = mel_db.data[mid_freq, mid_time]

    print(f"\nValues at bin {mid_freq}, frame {mid_time}:")
    print(f"  Power:     {power_val:.6e}")
    print(f"  Magnitude: {mag_val:.6e}")
    print(f"  Decibels:  {db_val:.2f} dB")

    # Verify relationship: magnitude = sqrt(power)
    print("\nVerifying relationships:")
    print(f"  sqrt(power) = {np.sqrt(power_val):.6e}")
    print(f"  magnitude   = {mag_val:.6e}")
    print(f"  Match: {np.allclose(np.sqrt(power_val), mag_val)}")

    print("\nMel spectrogram example completed!")


if __name__ == "__main__":
    main()
