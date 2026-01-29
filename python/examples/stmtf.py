"""
Spectro-Temporal Modulation Transfer Function (STMTF)

Expected peaks:
    Temporal modulation  ≈ ±6 Hz
    Spectral modulation  ≈ ±0.125 cycles / bin
"""

import numpy as np
import spectrograms as sp
import matplotlib.pyplot as plt


def main():
    # -----------------------------
    # Signal parameters
    # -----------------------------
    sample_rate = 16_000.0
    duration = 3.0
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # -----------------------------
    # Travelling ripple parameters
    # -----------------------------
    fm = 6.0  # temporal modulation (Hz)
    cycles_across_band = 8.0  # spectral cycles across tone bank
    alpha = 0.9  # modulation depth

    n_tones = 64
    f_min, f_max = 300.0, 5_000.0
    freqs = np.geomspace(f_min, f_max, n_tones)

    print("Injected modulation:")
    print(f"  Temporal:  ±{fm:.2f} Hz")
    print(f"  Spectral:  ±{cycles_across_band / n_tones:.3f} cycles/bin\n")

    # -----------------------------
    # Generate signal
    # -----------------------------
    signal = np.zeros_like(t)

    for k, fk in enumerate(freqs):
        phase_ramp = 2 * np.pi * cycles_across_band * (k / n_tones)
        env = 1.0 + alpha * np.cos(2 * np.pi * fm * t + phase_ramp)
        signal += env * np.sin(2 * np.pi * fk * t)

    signal /= np.max(np.abs(signal)) + 1e-12

    # -----------------------------
    # Linear spectrogram (NOT mel)
    # -----------------------------
    stft = sp.StftParams(
        n_fft=512,
        hop_size=128,
        window=sp.WindowType.hanning,
        centre=True,
    )
    params = sp.SpectrogramParams(stft=stft, sample_rate=sample_rate)

    print("Computing linear spectrogram...")
    spectrogram = sp.compute_linear_power_spectrogram(signal, params)

    print("Spectrogram shape:", spectrogram.shape)
    print("Duration:", spectrogram.duration(), "s\n")

    # -----------------------------
    # Remove DC + normalise
    # -----------------------------
    spec = np.ascontiguousarray(spectrogram.T)
    spec -= spec.mean()
    spec /= spec.std() + 1e-12
    spec -= spec.mean(axis=1, keepdims=True)  # remove per-frequency DC
    spec -= spec.mean(axis=0, keepdims=True)  # remove per-time DC
    # -----------------------------
    # STMTF
    # -----------------------------
    print("Computing STMTF...")
    stmtf_mag = sp.magnitude_spectrum_2d(spec)
    stmtf = sp.fftshift(stmtf_mag)

    # -----------------------------
    # Modulation axes
    # -----------------------------
    n_freq_bins, n_time_frames = spec.shape

    hop = params.stft.hop_size
    frame_period = hop / sample_rate

    spectral_mod = sp.fftshift_1d(
        sp.fftfreq(n_freq_bins, d=1.0)  # cycles per bin
    )
    temporal_mod = sp.fftshift_1d(
        sp.fftfreq(n_time_frames, d=frame_period)  # Hz
    )

    # -----------------------------
    # Locate strongest non-DC peak
    # -----------------------------
    h, w = stmtf.shape
    mask = np.ones_like(stmtf, dtype=bool)
    mask[h // 2 - 2 : h // 2 + 3, w // 2 - 2 : w // 2 + 3] = False

    flat_idx = np.argmax(stmtf[mask])
    coords = np.argwhere(mask)[flat_idx]
    i, j = coords

    print("Measured peak:")
    print(f"  Spectral modulation: {spectral_mod[i]:.4f} cycles/bin")
    print(f"  Temporal modulation: {temporal_mod[j]:.4f} Hz\n")

    # -----------------------------
    # Visualisation
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spectrogram
    im1 = axes[0].imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[0].set_title("Linear Spectrogram (normalised)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Frequency bin")
    plt.colorbar(im1, ax=axes[0])

    # STMTF (log + clipped dynamic range)
    img = np.log10(stmtf + 1e-12)
    vmax = np.percentile(img, 99.5)
    vmin = vmax - 6.0

    tmin, tmax = temporal_mod.min(), temporal_mod.max()
    smin, smax = spectral_mod.min(), spectral_mod.max()

    im2 = axes[1].imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        extent=[tmin, tmax, smin, smax],
    )

    axes[1].set_title("STMTF (log magnitude)")
    axes[1].set_xlabel("Temporal modulation (Hz)")
    axes[1].set_ylabel("Spectral modulation (cycles/bin)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
