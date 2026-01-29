"""
Spectro-Temporal Modulation Transfer Function (STMTF) Example

This example demonstrates computing a 2D FFT on a spectrogram to obtain
the spectro-temporal modulation transfer function - a key analysis technique
in auditory neuroscience.

The STMTF reveals energy distribution across:
- Spectral modulation (vertical): how rapidly the spectrum changes
- Temporal modulation (horizontal): how rapidly amplitude changes over time
"""

import numpy as np
import spectrograms as sp
import matplotlib.pyplot as plt


def main():
    print("=== Spectro-Temporal Modulation Transfer Function (STMTF) ===\n")

    # Signal parameters
    sample_rate = 16000.0
    duration = 2.0
    n_samples = int(sample_rate * duration)

    # Create amplitude-modulated tone
    carrier_freq = 1000.0  # 1 kHz carrier
    mod_freq = 10.0  # 10 Hz amplitude modulation

    print("Signal parameters:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Carrier frequency: {carrier_freq} Hz")
    print(f"  Modulation frequency: {mod_freq} Hz")
    print()

    # Generate signal
    t = np.arange(n_samples) / sample_rate
    am = 1.0 + 0.5 * np.cos(2 * np.pi * mod_freq * t)
    signal = am * np.sin(2 * np.pi * carrier_freq * t)

    # Compute mel spectrogram
    print("Computing mel spectrogram...")
    stft = sp.StftParams(
        n_fft=512, hop_size=128, window=sp.WindowType.hanning, centre=True
    )
    params = sp.SpectrogramParams(stft=stft, sample_rate=sample_rate)
    mel = sp.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)

    spectrogram = sp.compute_mel_power_spectrogram(signal, params, mel)

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Frequency range: {spectrogram.frequency_range()}")
    print(f"Duration: {spectrogram.duration():.2f} s\n")

    # Compute STMTF using spectrograms' built-in 2D FFT
    print("Computing STMTF via 2D FFT...")

    # Use the library's fft2d functions - spectrogram is passed directly
    stmtf_magnitude = sp.magnitude_spectrum_2d(spectrogram)

    # Shift zero-frequency to center using the library's fftshift
    stmtf_centered = sp.fftshift(stmtf_magnitude)

    print(f"STMTF shape: {stmtf_centered.shape}\n")

    # Calculate modulation frequencies
    freq_bins, time_frames = spectrogram.shape
    hop_size = params.stft.hop_size
    frame_period = hop_size / sample_rate

    spectral_mod_freqs = sp.fftshift_1d(sp.fftfreq(freq_bins, 1.0))
    temporal_mod_freqs = sp.fftshift_1d(sp.fftfreq(time_frames, frame_period))

    print("Modulation frequency ranges:")
    print(
        f"  Spectral: {spectral_mod_freqs[0]:.2f} to {spectral_mod_freqs[-1]:.2f} cycles/bin"
    )
    print(f"  Temporal: {temporal_mod_freqs[0]:.2f} to {temporal_mod_freqs[-1]:.2f} Hz")
    print()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Original spectrogram
    im1 = axes[0].imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, duration, 0, 8000],
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Mel Spectrogram (Power)")
    plt.colorbar(im1, ax=axes[0], label="Power")

    # Plot 2: STMTF
    im2 = axes[1].imshow(
        np.log10(stmtf_centered + 1e-10),  # Log scale for visualization
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[
            temporal_mod_freqs[0],
            temporal_mod_freqs[-1],
            spectral_mod_freqs[0],
            spectral_mod_freqs[-1],
        ],
    )
    axes[1].set_xlabel("Temporal Modulation (Hz)")
    axes[1].set_ylabel("Spectral Modulation (cycles/bin)")
    axes[1].set_title("Spectro-Temporal Modulation Transfer Function")
    axes[1].axvline(
        mod_freq, color="cyan", linestyle="--", alpha=0.7, label=f"{mod_freq} Hz AM"
    )
    axes[1].axvline(-mod_freq, color="cyan", linestyle="--", alpha=0.7)
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label="Log Magnitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
