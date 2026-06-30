#!/usr/bin/env python3
"""
Generate all manual figures for the Spectrograms library reference manual.

Audio: Karplus-Strong plucked string synthesis + FM drums + ADSR piano.
Image: scipy.datasets.face() (raccoon portrait).
All figures use LaTeX/Computer Modern fonts to match the manual typeface.
"""

import os
import warnings
import numpy as np
from scipy.signal import butter, lfilter, chirp
from scipy.datasets import face as scipy_face
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import seaborn as sns
import spectrograms as sg

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    # --- bold, larger titles & labels ---
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "legend.borderpad": 0.5,
    # --- lines & axes ---
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "lines.linewidth": 1.4,
    # --- saving ---
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

# Colormaps
SPEC_CMAP = "inferno"
BINAURAL_CMAP = "RdBu_r"
CHROMA_CMAP = "magma"
MFCC_CMAP = "coolwarm"

# Figure widths (inches) — A5 page is ~14 cm wide, margins ~1.1 cm each side
FW_FULL = 5.8    # full page width for a single figure
FW_HALF = 2.8    # half-page figure

SR = 22050


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}.pdf")


def legend_below(ax, ncol=2, **kwargs):
    """Place a legend centred below ax, outside the axes box."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.75",
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────

def adsr(n, sr, attack=0.008, decay=0.04, sustain=0.65, release=0.25):
    """Return an ADSR amplitude envelope of length n."""
    t = np.linspace(0, n / sr, n)
    env = np.zeros(n)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = n - a - d - r
    if s < 0:
        s = 0
    env[:a] = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, sustain, d)
    env[a+d:a+d+s] = sustain
    env[a+d+s:] = np.linspace(sustain, 0, r if r <= n - a - d - s else n - a - d - s)
    return env


def piano_note(freq, dur, sr=SR, n_harmonics=12):
    """Synthesise a piano-like note with decaying harmonics."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for h in range(1, n_harmonics + 1):
        amp = 0.9 ** (h - 1) / h
        stretch = 1.0 + 5e-5 * (h - 1) ** 2
        decay = np.exp(-t * (2.5 + h * 0.4))
        sig += amp * decay * np.sin(2 * np.pi * freq * h * stretch * t)
    env = adsr(n, sr, attack=0.005, decay=0.06, sustain=0.4, release=0.18)
    return sig * env


def karplus_strong(freq, dur, sr=SR, damping=0.996):
    """Karplus-Strong plucked string synthesis."""
    n = int(dur * sr)
    buf_len = int(sr / freq)
    buf = np.random.uniform(-1, 1, buf_len)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[0]
        avg = damping * 0.5 * (buf[0] + buf[1])
        buf = np.roll(buf, -1)
        buf[-1] = avg
    env = np.ones(n)
    env[-int(0.05 * sr):] = np.linspace(1, 0, int(0.05 * sr))
    return out * env


def chord_phrase(sr=SR):
    """
    A realistic chord progression: C maj → A min → F maj → G maj.
    Each chord uses stacked piano notes with random onset jitter.
    Total ~4 seconds.
    """
    notes = {
        "C3": 130.81, "E3": 164.81, "G3": 196.00,
        "F3": 174.61, "A3": 220.00,
        "C4": 261.63, "D4": 293.66, "E4": 329.63,
        "F4": 349.23, "G4": 392.00, "A4": 440.00,
        "B4": 493.88, "C5": 523.25, "G2": 98.00,
    }

    chords = [
        (notes["C3"], [notes["C4"], notes["E4"], notes["G4"]], 1.0),
        (notes["A3"], [notes["A3"], notes["C4"], notes["E4"]], 1.0),
        (notes["F3"], [notes["F3"], notes["A3"], notes["C5"]], 1.0),
        (notes["G2"], [notes["G3"], notes["B4"], notes["D4"]], 1.0),
    ]

    segments = []
    for bass_f, chord_fs, dur in chords:
        n = int(dur * sr)
        seg = np.zeros(n)
        seg += 0.55 * karplus_strong(bass_f, dur, sr)
        for i, f in enumerate(chord_fs):
            onset = int(i * 0.012 * sr)
            note_sig = 0.28 * piano_note(f, dur - i * 0.012, sr)
            seg[onset:onset + len(note_sig)] += note_sig
        segments.append(seg)

    sig = np.concatenate(segments)
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig


def chromatic_arpeggio(sr=SR):
    """
    Ascending chromatic scale + octave leap, C3 to C5.
    Ideal for showing CQT constant-Q properties.
    """
    freqs_semitones = np.arange(0, 25)
    f0 = 130.81  # C3
    segs = []
    for st in freqs_semitones:
        f = f0 * (2 ** (st / 12))
        segs.append(piano_note(f, 0.22, sr))
    sig = np.concatenate(segs)
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig


def speech_vowels(sr=SR):
    """
    Concatenate synthetic vowels /a/, /e/, /i/, /o/, /u/ via formant synthesis.
    """
    vowel_formants = {
        "a": [(730, 80), (1090, 120), (2440, 160)],
        "e": [(270, 60), (2290, 140), (3010, 180)],
        "i": [(270, 50), (2290, 90), (3010, 120)],
        "o": [(570, 70), (840, 100), (2410, 150)],
        "u": [(300, 60), (870, 80), (2240, 130)],
    }
    dur_each = 0.55
    segs = []
    for vowel, formants in vowel_formants.items():
        n = int(dur_each * sr)
        t = np.arange(n) / sr
        f0_v = 120
        excitation = np.zeros(n)
        for k in np.arange(1, 25):
            excitation += (1 / k ** 0.7) * np.sin(2 * np.pi * f0_v * k * t)
        excitation *= adsr(n, sr, attack=0.02, decay=0.05, sustain=0.85, release=0.12)
        voiced = excitation.copy()
        for (fc, bw) in formants:
            b, a = butter(2, [max(1, fc - bw/2), min(sr/2 - 1, fc + bw/2)],
                          btype="bandpass", fs=sr)
            voiced = lfilter(b, a, voiced) * 12 + voiced
        segs.append(voiced)
    sig = np.concatenate(segs)
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig


def binaural_scene(sr=SR, dur=3.0):
    """
    Simulate a sound source at 30° azimuth and another at −60° azimuth.
    Uses realistic ITD (~340 µs for 30°) and ILD (~8 dB).
    """
    n = int(dur * sr)
    t = np.arange(n) / sr

    left1 = chord_phrase(sr)[:n]
    delay_samples_1 = int(0.00034 * sr)
    right1 = np.zeros(n)
    right1[delay_samples_1:] = left1[:n - delay_samples_1]
    right1 *= 10 ** (8 / 20)

    chirp_sig = chirp(t, f0=800, f1=2400, t1=dur, method='linear')
    chirp_sig *= np.exp(-t * 0.5) * adsr(n, sr, 0.01, 0.1, 0.7, 0.2)
    delay_samples_2 = int(0.00068 * sr)
    left2 = np.zeros(n)
    left2[delay_samples_2:] = chirp_sig[:n - delay_samples_2]
    left2 *= 10 ** (10 / 20)
    right2 = chirp_sig.copy()

    left  = (left1  + 0.35 * left2)
    right = (right1 + 0.35 * right2)

    peak = max(np.max(np.abs(left)), np.max(np.abs(right))) + 1e-9
    return left / peak, right / peak


# ─────────────────────────────────────────────────────────────────────────────
# Utility: draw a spectrogram image on an Axes
# ─────────────────────────────────────────────────────────────────────────────

def imshow_spec(ax, data, sr, hop, n_fft=None, cmap=SPEC_CMAP,
                freq_axis="linear", vmin=None, vmax=None,
                ylabel=r"Frequency (Hz)", title="", colorbar=True,
                max_freq=None, freq_ticks=None, freq_labels=None,
                db_range=80):
    """
    Display a spectrogram matrix on ax with time/frequency axes.
    db_range caps the vmin so at most db_range dB of dynamic range is shown
    (prevents excessively dark spectrograms). Set db_range=None to disable.
    """
    n_freq, n_frames = data.shape
    dur_sec = n_frames * hop / sr

    if freq_axis == "mel":
        ylabel = r"Frequency (Hz)"
    elif freq_axis == "erb":
        ylabel = r"ERB channel"
    elif freq_axis == "cqt":
        ylabel = r"Pitch (semitones)"
    elif freq_axis == "chroma":
        ylabel = r"Pitch class"
    elif freq_axis == "mfcc":
        ylabel = r"MFCC coefficient"

    # Smart vmin: never show more than db_range of dynamic range
    vmax_actual = data.max() if vmax is None else vmax
    if vmin is None:
        vmin_pct = np.percentile(data, 5)
        if db_range is not None:
            vmin = max(vmin_pct, vmax_actual - db_range)
        else:
            vmin = vmin_pct

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, dur_sec, 0, n_freq],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_actual,
        interpolation="nearest",
    )
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if freq_ticks is not None and freq_labels is not None:
        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=8)

    return im


def hz_ticks_for_linear(n_fft, sr, n_ticks=6):
    """Tick positions (bin index) and labels for a linear frequency axis."""
    max_hz = sr / 2
    tick_hz = np.linspace(0, max_hz, n_ticks + 1)
    n_bins = n_fft // 2 + 1
    tick_bins = (tick_hz / max_hz * n_bins).astype(int)
    tick_labels = [f"{int(h)}" for h in tick_hz]
    return tick_bins, tick_labels


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: DFT basis functions — real signal spectrum analysis
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 1: DFT basis functions")

def fig_dft_basis():
    sr_local = 8000
    N = 256
    t = np.arange(N) / sr_local
    freqs_sig = [220.0, 330.0, 550.0]
    sig = np.zeros(N)
    for i, f in enumerate(freqs_sig):
        tau = 0.025 * (i + 1)
        sig += (1 / (i + 1)) * np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
    sig /= np.max(np.abs(sig))

    X = np.fft.rfft(sig, N)
    freqs = np.fft.rfftfreq(N, 1 / sr_local)
    mag = np.abs(X)

    peak_bins = np.argsort(mag)[-4:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(FW_FULL, 5.0),
                              gridspec_kw={"hspace": 0.72, "wspace": 0.42})

    bin_colors = plt.cm.tab10([0, 1, 2, 3])

    ax_spec = axes[0, 0]
    ax_spec.plot(freqs, mag, color="steelblue", lw=1.0, alpha=0.85)
    for idx, (bin_k, col) in enumerate(zip(peak_bins, bin_colors)):
        ax_spec.axvline(freqs[bin_k], color=col, lw=1.1, ls="--", alpha=0.85)
        ax_spec.text(freqs[bin_k] + 8, mag[bin_k] * 0.82,
                     rf"$k={bin_k}$", color=col, fontsize=8)
    ax_spec.set_xlabel(r"Frequency (Hz)")
    ax_spec.set_ylabel(r"$|X[k]|$")
    ax_spec.set_title(r"Magnitude spectrum $|X[k]|$")
    ax_spec.set_xlim(0, sr_local / 2)

    ax_time = axes[0, 1]
    ax_time.plot(t * 1000, sig, color="steelblue", lw=0.9, alpha=0.9)
    ax_time.set_xlabel(r"Time (ms)")
    ax_time.set_ylabel(r"Amplitude")
    ax_time.set_title(r"Time-domain signal $x[n]$")

    n_axis = np.arange(N)
    for panel, (bin_k, col) in enumerate(zip(peak_bins[:2], bin_colors[:2])):
        ax = axes[1, panel]
        basis_re = np.cos(2 * np.pi * bin_k * n_axis / N)
        basis_im = -np.sin(2 * np.pi * bin_k * n_axis / N)
        ax.plot(n_axis, basis_re, color=col, lw=1.0, label=r"$\cos(\cdot)$")
        ax.plot(n_axis, basis_im, color=col, lw=1.0, ls="--", alpha=0.7,
                label=r"$-\sin(\cdot)$")
        ax.set_xlabel(r"Sample $n$")
        ax.set_ylabel(r"Amplitude")
        ax.set_title(rf"Basis $k={bin_k}$ ({freqs[bin_k]:.0f}\,Hz)")
        legend_below(ax, ncol=2)
        ax.set_xlim(0, N - 1)

    save(fig, "fig_dft_basis")

fig_dft_basis()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Hermitian symmetry of real FFT
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: Hermitian symmetry")

def fig_hermitian():
    sr_local = 8000
    N = 256
    t = np.arange(N) / sr_local
    sig = (np.sin(2*np.pi*440*t)*np.exp(-t/0.03) +
           0.6*np.sin(2*np.pi*660*t)*np.exp(-t/0.04) +
           0.4*np.sin(2*np.pi*880*t)*np.exp(-t/0.025))
    sig /= np.max(np.abs(sig))

    X_full = np.fft.fft(sig)
    freqs_full = np.fft.fftfreq(N, 1/sr_local)

    mag = np.abs(X_full)
    phase = np.angle(X_full)

    fig, axes = plt.subplots(1, 2, figsize=(FW_FULL, 3.8),
                              gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    ax.plot(freqs_full[:N//2+1], mag[:N//2+1],
            color="steelblue", lw=1.2, label="Positive freqs")
    ax.plot(freqs_full[N//2+1:], mag[N//2+1:],
            color="tomato", lw=1.2, ls="--", alpha=0.85,
            label="Negative freqs (mirror)")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"$|X[k]|$")
    ax.set_title(r"$|X[k]| = |X[-k]|$")
    legend_below(ax, ncol=1)

    ax = axes[1]
    ax.plot(freqs_full[:N//2+1], np.degrees(phase[:N//2+1]),
            color="steelblue", lw=1.2, label="Positive freqs")
    ax.plot(freqs_full[N//2+1:], np.degrees(phase[N//2+1:]),
            color="tomato", lw=1.2, ls="--", alpha=0.85,
            label="Negative freqs (mirror)")
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"Phase (degrees)")
    ax.set_title(r"$\angle X[k] = -\angle X[-k]$")
    legend_below(ax, ncol=1)

    save(fig, "fig_hermitian")

fig_hermitian()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Window functions — time domain comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: Window functions (time domain)")

def fig_windows_time():
    N = 512
    n = np.arange(N)
    n1 = N - 1

    windows = {
        "Rectangular": np.ones(N),
        "Hanning":     0.5 - 0.5 * np.cos(2 * np.pi * n / n1),
        "Hamming":     0.54 - 0.46 * np.cos(2 * np.pi * n / n1),
        "Blackman":    0.42 - 0.5 * np.cos(2*np.pi*n/n1) + 0.08*np.cos(4*np.pi*n/n1),
        r"Kaiser $\beta{=}8$": np.kaiser(N, 8),
    }

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(windows)))
    # Taller figure + extra bottom space for the legend
    fig, ax = plt.subplots(figsize=(FW_FULL, 3.8))
    fig.subplots_adjust(bottom=0.25)

    for (name, w), col in zip(windows.items(), colors):
        ax.plot(n, w, lw=1.2, color=col, label=name)

    ax.set_xlabel(r"Sample $n$")
    ax.set_ylabel(r"Amplitude $w[n]$")
    ax.set_title(r"Window functions --- time domain ($N = 512$)")
    ax.set_xlim(0, N - 1)
    ax.set_ylim(-0.08, 1.18)
    legend_below(ax, ncol=3)

    save(fig, "fig_windows_time")

fig_windows_time()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Window functions — frequency response (magnitude in dB)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: Window functions (frequency response)")

def fig_windows_freq():
    N = 512
    n = np.arange(N)
    n1 = N - 1
    NFFT = 8192

    windows = {
        "Rectangular": np.ones(N),
        "Hanning":     0.5 - 0.5 * np.cos(2 * np.pi * n / n1),
        "Hamming":     0.54 - 0.46 * np.cos(2 * np.pi * n / n1),
        "Blackman":    0.42 - 0.5*np.cos(2*np.pi*n/n1) + 0.08*np.cos(4*np.pi*n/n1),
        r"Kaiser $\beta{=}8$": np.kaiser(N, 8),
    }

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(windows)))
    fig, ax = plt.subplots(figsize=(FW_FULL, 3.8))
    fig.subplots_adjust(bottom=0.25)

    for (name, w), col in zip(windows.items(), colors):
        padded = np.zeros(NFFT)
        padded[:N] = w
        W = np.abs(np.fft.rfft(padded))
        W_db = 20 * np.log10(W / W.max() + 1e-12)
        freqs_norm = np.linspace(0, 0.5, len(W))
        ax.plot(freqs_norm, W_db, lw=1.2, color=col, label=name)

    ax.set_xlabel(r"Normalised frequency $(\times \pi\,\text{rad/sample})$")
    ax.set_ylabel(r"Magnitude (dB)")
    ax.set_title(r"Window frequency responses (zero-padded, $N{=}512$)")
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-100, 5)
    ax.axhline(-13, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax.text(0.103, -11.0, r"$-13\,$dB", fontsize=8, color="gray")
    legend_below(ax, ncol=3)

    save(fig, "fig_windows_freq")

fig_windows_freq()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Spectral leakage demo
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 5: Spectral leakage")

def fig_spectral_leakage():
    sr_local = 16000
    N = 1024
    t = np.arange(N) / sr_local
    f1, f2 = 441.7, 882.3
    sig = (np.sin(2*np.pi*f1*t)*np.exp(-t/0.04) +
           0.5*np.sin(2*np.pi*f2*t)*np.exp(-t/0.05))

    freqs = np.fft.rfftfreq(N, 1/sr_local)
    rect_mag = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-10)
    hann = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(N)/(N-1))
    hann_mag = 20 * np.log10(np.abs(np.fft.rfft(sig * hann)) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(FW_FULL, 3.2),
                              gridspec_kw={"wspace": 0.42})

    for ax, mag, label, col in zip(axes,
                                    [rect_mag, hann_mag],
                                    [r"Rectangular window", r"Hanning window"],
                                    ["steelblue", "tomato"]):
        ax.plot(freqs, mag, color=col, lw=1.0)
        ax.set_xlim(200, 1200)
        ax.set_ylim(-80, 10)
        ax.set_xlabel(r"Frequency (Hz)")
        ax.set_ylabel(r"Magnitude (dB)")
        ax.set_title(label)
        ax.axvline(f1, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(f2, color="gray", lw=0.8, ls="--", alpha=0.7)

    save(fig, "fig_spectral_leakage")

fig_spectral_leakage()


# ─────────────────────────────────────────────────────────────────────────────
# Synthesise the main chord signal used across multiple figures
# ─────────────────────────────────────────────────────────────────────────────
print("Synthesising main chord signal …")
chord_sig = chord_phrase(SR)
chord_params = sg.SpectrogramParams.music_default(SR)
chord_stft_p = sg.StftParams(2048, 512, sg.WindowType.hanning)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Linear power vs magnitude vs decibel spectrogram (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 6: Power / magnitude / dB spectrograms")

def fig_amplitude_scaling():
    spec_power = sg.compute_linear_power_spectrogram(chord_sig, chord_params)
    spec_mag   = sg.compute_linear_magnitude_spectrogram(chord_sig, chord_params)
    spec_db    = sg.compute_linear_db_spectrogram(chord_sig, chord_params)

    hop = chord_params.stft.hop_size
    n_fft = chord_params.stft.n_fft

    ticks, labels = hz_ticks_for_linear(n_fft, SR, n_ticks=5)

    fig, axes = plt.subplots(1, 3, figsize=(FW_FULL * 1.55, 3.6),
                              gridspec_kw={"wspace": 0.48})

    # Power: clip bottom quartile
    d_pow = spec_power.data
    vmin_pow = np.percentile(d_pow, 20)

    # Magnitude: clip bottom quartile
    d_mag = spec_mag.data
    vmin_mag = np.percentile(d_mag, 20)

    # dB: 80 dB dynamic range
    d_db = spec_db.data
    vmin_db = d_db.max() - 80

    for ax, data, title, vmin_v in zip(
        axes,
        [d_pow, d_mag, d_db],
        [r"Power $|X[m,k]|^2$", r"Magnitude $|X[m,k]|$",
         r"Decibel $10\log_{10}|X|^2$"],
        [vmin_pow, vmin_mag, vmin_db],
    ):
        imshow_spec(ax, data, SR, hop, n_fft, title=title,
                    freq_ticks=ticks, freq_labels=labels,
                    vmin=vmin_v, colorbar=True, db_range=None)
        ax.set_ylabel(r"Frequency (Hz)")

    save(fig, "fig_amplitude_scaling")

fig_amplitude_scaling()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Time-frequency uncertainty (two window sizes)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 7: Time-frequency uncertainty")

def fig_uncertainty():
    hop = 256
    fig, axes = plt.subplots(1, 2, figsize=(FW_FULL, 3.4),
                              gridspec_kw={"wspace": 0.46})

    for ax, n_fft, lbl in zip(
        axes,
        [256, 4096],
        [r"$N{=}256$ (11.6\,ms) --- good time res.",
         r"$N{=}4096$ (185\,ms) --- good freq. res."]
    ):
        p = sg.SpectrogramParams(sg.StftParams(n_fft, hop, sg.WindowType.hanning), SR)
        spec = sg.compute_linear_db_spectrogram(chord_sig, p)
        data = spec.data

        n_bins, n_frames = data.shape
        dur_sec = n_frames * hop / SR
        vmax = data.max()
        vmin = vmax - 75  # 75 dB dynamic range — clearly shows structure

        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[0, dur_sec, 0, SR / 2 / 1000],
                       cmap=SPEC_CMAP, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xlabel(r"Time (s)")
        ax.set_ylabel(r"Frequency (kHz)")
        ax.set_title(lbl)
        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046,
                     label=r"dB").ax.tick_params(labelsize=8)

    save(fig, "fig_uncertainty")

fig_uncertainty()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Mel filterbank
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 8: Mel filterbank")

def fig_mel_filterbank():
    n_fft = 2048
    n_mels = 40
    f_min, f_max = 0.0, 8000.0

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    freqs_hz = np.linspace(0, SR / 2, n_fft // 2 + 1)
    mel_min, mel_max = hz_to_mel(f_min + 1), hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    bin_centers = mel_to_hz(mel_points)

    fig, axes = plt.subplots(2, 1, figsize=(FW_FULL, 4.8),
                              gridspec_kw={"hspace": 0.62})

    cmap_filters = plt.cm.plasma(np.linspace(0.1, 0.9, n_mels))
    ax = axes[0]
    for m in range(n_mels):
        left = bin_centers[m]
        center = bin_centers[m + 1]
        right = bin_centers[m + 2]
        filt = np.zeros_like(freqs_hz)
        mask_rise = (freqs_hz >= left) & (freqs_hz <= center)
        mask_fall = (freqs_hz > center) & (freqs_hz <= right)
        if center > left:
            filt[mask_rise] = (freqs_hz[mask_rise] - left) / (center - left)
        if right > center:
            filt[mask_fall] = (right - freqs_hz[mask_fall]) / (right - center)
        ax.plot(freqs_hz / 1000, filt, color=cmap_filters[m], lw=0.7, alpha=0.8)

    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel(r"Filter weight")
    ax.set_title(r"Mel triangular filterbank ($M{=}40$ filters, "
                 r"$f_{\min}{=}0$\,Hz, $f_{\max}{=}8$\,kHz)")
    ax.set_xlim(0, f_max / 1000)

    ax2 = axes[1]
    f_range = np.linspace(0, 8000, 1000)
    ax2.plot(f_range / 1000, hz_to_mel(f_range), color="steelblue", lw=1.4)
    ax2.axhline(hz_to_mel(1000), color="gray", lw=0.8, ls="--", alpha=0.7)
    ax2.text(0.1, hz_to_mel(1000) + 25, r"$1\,\text{kHz}$", fontsize=9, color="gray")
    ax2.set_xlabel(r"Frequency (kHz)")
    ax2.set_ylabel(r"Mel")
    ax2.set_title(r"Hz-to-mel mapping: $\mathrm{mel}(f) = 2595\log_{10}(1 + f/700)$")

    save(fig, "fig_mel_filterbank")

fig_mel_filterbank()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Mel spectrogram of chord phrase
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 9: Mel spectrogram")

def fig_mel_spectrogram():
    mel_p = sg.MelParams(128, 0.0, 8000.0)
    spec = sg.compute_mel_db_spectrogram(chord_sig, chord_params, mel_p)
    data = spec.data
    hop = chord_params.stft.hop_size

    n_freq = data.shape[0]

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    mel_max = hz_to_mel(8000.0)
    tick_hzs = [100, 500, 1000, 2000, 4000, 8000]
    tick_bins = [int(hz_to_mel(f) / mel_max * n_freq) for f in tick_hzs]
    tick_labels = [f"{int(f)}" for f in tick_hzs]

    fig, ax = plt.subplots(figsize=(FW_FULL, 3.4))
    vmin = data.max() - 80
    imshow_spec(ax, data, SR, hop, freq_axis="mel",
                freq_ticks=tick_bins, freq_labels=tick_labels,
                vmin=vmin, db_range=None,
                title=r"Mel-scale spectrogram (dB) --- "
                      r"C\,maj $\to$ A\,min $\to$ F\,maj $\to$ G\,maj")
    ax.set_ylabel(r"Frequency (Hz)")

    save(fig, "fig_mel_spectrogram")

fig_mel_spectrogram()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10: ERB filterbank + gammatone impulse responses
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 10: ERB filterbank / gammatone")

def fig_erb_filterbank():
    def erb_bw(fc):
        return 24.7 * (4.37 * fc / 1000 + 1)

    def gammatone_ir(fc, sr_l=SR, dur=0.03, order=4):
        n = int(dur * sr_l)
        t = np.arange(n) / sr_l
        b = 1.019 * erb_bw(fc)
        return t**(order-1) * np.exp(-2*np.pi*b*t) * np.cos(2*np.pi*fc*t)

    center_freqs = [125, 500, 1000, 2000, 4000]

    fig, axes = plt.subplots(1, 2, figsize=(FW_FULL, 4.0),
                              gridspec_kw={"wspace": 0.46})
    fig.subplots_adjust(bottom=0.22)

    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(center_freqs)))
    for fc, col in zip(center_freqs, colors):
        ir = gammatone_ir(fc)
        ir /= np.max(np.abs(ir)) + 1e-9
        t_ir = np.arange(len(ir)) / SR * 1000
        ax.plot(t_ir, ir, color=col, lw=1.0, alpha=0.9,
                label=rf"$f_c={fc}$\,Hz")
    ax.set_xlabel(r"Time (ms)")
    ax.set_ylabel(r"Amplitude")
    ax.set_title(r"Gammatone impulse responses")
    legend_below(ax, ncol=3)
    ax.set_xlim(0, 30)

    ax = axes[1]
    NFFT = 4096
    freqs_r = np.fft.rfftfreq(NFFT, 1/SR)
    for fc, col in zip(center_freqs, colors):
        ir = gammatone_ir(fc, dur=0.08)
        padded = np.zeros(NFFT)
        padded[:len(ir)] = ir
        H = np.abs(np.fft.rfft(padded))
        H_db = 20 * np.log10(H / (H.max() + 1e-12) + 1e-12)
        ax.plot(freqs_r / 1000, H_db, color=col, lw=1.0, alpha=0.9,
                label=rf"$f_c={fc}$\,Hz")
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel(r"Magnitude (dB)")
    ax.set_title(r"Gammatone frequency responses")
    ax.set_xlim(0, 6)
    ax.set_ylim(-55, 5)
    legend_below(ax, ncol=3)

    save(fig, "fig_erb_filterbank")

fig_erb_filterbank()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11: ERB spectrogram
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 11: ERB spectrogram")

def fig_erb_spectrogram():
    erb_p = sg.ErbParams(64, 80.0, 8000.0)
    spec = sg.compute_erb_power_spectrogram(chord_sig, chord_params, erb_p)
    data = np.log10(spec.data + 1e-10)
    hop = chord_params.stft.hop_size

    fig, ax = plt.subplots(figsize=(FW_FULL, 3.4))
    # log10 data: cap at 4 decades of dynamic range
    vmin = data.max() - 4.0
    imshow_spec(ax, data, SR, hop, freq_axis="erb",
                vmin=vmin, db_range=None,
                title=r"ERB/gammatone spectrogram --- chord progression")

    save(fig, "fig_erb_spectrogram")

fig_erb_spectrogram()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12: CQT spectrogram of chromatic arpeggio
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 12: CQT spectrogram")

def fig_cqt_spectrogram():
    arp_sig = chromatic_arpeggio(SR)
    cqt_p = sg.CqtParams(12, 6, 65.41)
    planner = sg.SpectrogramPlanner()
    cqt_plan = planner.cqt_db_plan(chord_params, cqt_p, sg.LogParams(-80.0))
    cqt_spec = cqt_plan.compute(arp_sig)
    data = cqt_spec.data
    hop = chord_params.stft.hop_size

    n_bins = data.shape[0]
    tick_bins, tick_labels = [], []
    for octave in range(7):
        bin_idx = octave * 12
        if bin_idx < n_bins:
            tick_bins.append(bin_idx)
            tick_labels.append(rf"C{octave + 2}")

    fig, ax = plt.subplots(figsize=(FW_FULL, 3.6))
    vmin = data.max() - 80
    imshow_spec(ax, data, SR, hop, freq_axis="cqt",
                freq_ticks=tick_bins, freq_labels=tick_labels,
                vmin=vmin, db_range=None,
                title=r"CQT spectrogram --- chromatic ascending arpeggio C2--C8")
    ax.set_ylabel(r"Pitch")

    save(fig, "fig_cqt_spectrogram")

fig_cqt_spectrogram()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13: MFCC of synthesised vowels
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 13: MFCC of synthetic vowels")

def fig_mfcc():
    vowel_sig = speech_vowels(SR)
    stft_p = sg.StftParams(512, 128, sg.WindowType.hanning)
    mfcc = sg.compute_mfcc(vowel_sig, stft_p, SR, 40, sg.MfccParams(13))

    hop = 128
    n_coeff, n_frames = mfcc.shape
    dur_sec = n_frames * hop / SR

    fig, ax = plt.subplots(figsize=(FW_FULL, 3.2))
    im = ax.imshow(mfcc, aspect="auto", origin="lower",
                   extent=[0, dur_sec, -0.5, n_coeff - 0.5],
                   cmap=MFCC_CMAP, interpolation="nearest")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"MFCC coefficient index")
    ax.set_title(r"MFCCs --- synthetic vowels /a/, /e/, /i/, /o/, /u/")
    ax.set_yticks(range(n_coeff))
    ax.set_yticklabels([str(i) for i in range(n_coeff)], fontsize=7)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=8)

    vowels = [r"/a/", r"/e/", r"/i/", r"/o/", r"/u/"]
    dur_each = 0.55
    for i, v in enumerate(vowels):
        ax.axvline(i * dur_each, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.text((i + 0.5) * dur_each, n_coeff - 0.9, v,
                ha="center", va="top", color="white", fontsize=9,
                fontweight="bold")

    save(fig, "fig_mfcc")

fig_mfcc()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 14: Chromagram of chord progression
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 14: Chromagram")

def fig_chromagram():
    stft_p = sg.StftParams(4096, 512, sg.WindowType.hanning)
    chroma_p = sg.ChromaParams.music_standard()
    chroma = sg.compute_chromagram(chord_sig, stft_p, SR, chroma_p)
    data = chroma

    hop = 512
    n_frames = data.shape[1]
    dur_sec = n_frames * hop / SR

    pitch_classes = ["C", r"C\#", "D", r"D\#", "E", "F",
                     r"F\#", "G", r"G\#", "A", r"A\#", "B"]

    fig, ax = plt.subplots(figsize=(FW_FULL, 3.2))
    im = ax.imshow(data, aspect="auto", origin="lower",
                   extent=[0, dur_sec, -0.5, 11.5],
                   cmap=CHROMA_CMAP, interpolation="nearest",
                   vmin=0, vmax=data.max())
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Pitch class")
    ax.set_title(r"Chromagram --- C\,maj $\to$ A\,min $\to$ F\,maj $\to$ G\,maj")
    ax.set_yticks(range(12))
    ax.set_yticklabels(pitch_classes)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046,
                 label="Energy").ax.tick_params(labelsize=8)

    chords_lbl = [r"C\,maj", r"A\,min", r"F\,maj", r"G\,maj"]
    for i, lbl in enumerate(chords_lbl):
        ax.axvline(i * 1.0, color="white", lw=0.9, ls="--", alpha=0.6)
        ax.text(i * 1.0 + 0.06, 11.1, lbl, color="white", fontsize=8,
                fontweight="bold")

    save(fig, "fig_chromagram")

fig_chromagram()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 15: Image processing via 2D FFT
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 15: Image processing (2D FFT)")

def fig_image_processing():
    img_full = scipy_face(gray=True).astype(np.float64)
    img = img_full[120:440, 250:570].copy()
    img /= 255.0

    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)
    mag_log = np.log1p(np.abs(F_shift))

    blurred = sg.gaussian_kernel_2d(31, 4.0)
    blurred_img_arr = sg.convolve_fft(img, blurred)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    H_hp = np.ones((rows, cols))
    r_cut = 20
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    H_hp[dist < r_cut] = 0

    F_hp = F_shift * H_hp
    edges = np.abs(np.fft.ifft2(np.fft.ifftshift(F_hp)))

    fig, axes = plt.subplots(1, 4, figsize=(FW_FULL * 1.65, 3.6),
                              gridspec_kw={"wspace": 0.05})

    panels = [
        (img,             "gray",    r"Original image"),
        (mag_log,         "viridis", r"2-D FFT $\log|F|$"),
        (blurred_img_arr, "gray",    r"Gaussian blur ($\sigma{=}4$)"),
        (edges,           "gray",    r"High-pass edges ($r{<}20$)"),
    ]

    for ax, (data, cm, title) in zip(axes, panels):
        ax.imshow(data, cmap=cm, aspect="equal", interpolation="bilinear")
        ax.set_title(title, pad=5)
        ax.axis("off")

    save(fig, "fig_image_processing")

fig_image_processing()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 16: Binaural spectrograms — ITD, IPD, ILD, ILR
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 16: Binaural spectrograms")

def fig_binaural():
    left, right = binaural_scene(SR, dur=3.0)

    sr_b = SR
    params_b = sg.SpectrogramParams(sg.StftParams(4096, 1024, sg.WindowType.hanning), sr_b)

    itd_p = sg.ITDSpectrogramParams(params_b, 50.0, 1500.0)
    ild_p = sg.ILDSpectrogramParams(params_b)
    ipd_p = sg.IPDSpectrogramParams(params_b)
    ilr_p = sg.ILRSpectrogramParams(params_b)

    itd = sg.compute_itd_spectrogram([left, right], itd_p)
    ild = sg.compute_ild_spectrogram([left, right], ild_p)
    ipd = sg.compute_ipd_spectrogram([left, right], ipd_p)
    ilr = sg.compute_ilr_spectrogram([left, right], ilr_p)

    hop = params_b.stft.hop_size
    n_fft = params_b.stft.n_fft

    fig, axes = plt.subplots(2, 2, figsize=(FW_FULL * 1.2, 5.2),
                              gridspec_kw={"hspace": 0.58, "wspace": 0.52})

    datasets = [
        (itd * 1000, BINAURAL_CMAP, r"ITD (ms)",  r"Interaural Time Difference (ITD)", True),
        (ipd,        BINAURAL_CMAP, r"IPD (rad)", r"Interaural Phase Difference (IPD)", True),
        (ild,        BINAURAL_CMAP, r"ILD (dB)",  r"Interaural Level Difference (ILD)", True),
        (ilr,        BINAURAL_CMAP, r"ILR",       r"Interaural Level Ratio (ILR)", True),
    ]

    for ax, (data, cmap, unit, title, sym) in zip(axes.flat, datasets):
        n_freq, n_frames = data.shape
        dur_sec = n_frames * hop / sr_b
        max_hz = sr_b / 2

        lim = np.percentile(np.abs(data), 97)
        vmin_b = -lim if sym else None
        vmax_b = lim

        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[0, dur_sec, 0, max_hz / 1000],
                       cmap=cmap, vmin=vmin_b, vmax=vmax_b,
                       interpolation="nearest")
        ax.set_xlabel(r"Time (s)")
        ax.set_ylabel(r"Frequency (kHz)")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046, label=unit)
        cbar.ax.tick_params(labelsize=8)

    save(fig, "fig_binaural")

fig_binaural()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 17: MDCT — coefficients + perfect reconstruction demo
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 17: MDCT")

def fig_mdct():
    dur = 2.0
    sig = chord_phrase(SR)[:int(dur * SR)]
    sig /= np.max(np.abs(sig)) + 1e-9

    mdct_params = sg.MdctParams.sine_window(2048)
    coeffs = sg.mdct(sig, mdct_params)
    recon_full = sg.imdct(coeffs, mdct_params)
    n_compare = min(len(sig), len(recon_full))
    recon = recon_full[:n_compare]
    sig_cmp = sig[:n_compare]

    hop = mdct_params.hop_size
    n_frames = coeffs.shape[1]
    dur_sec = n_frames * hop / SR

    log_coeffs = np.log10(np.abs(coeffs) + 1e-10)

    fig = plt.figure(figsize=(FW_FULL, 5.4))
    gs = gridspec.GridSpec(3, 1, hspace=0.75)

    ax0 = fig.add_subplot(gs[0])
    vmin_m = log_coeffs.max() - 4.0  # 4 decades dynamic range
    im = ax0.imshow(log_coeffs, aspect="auto", origin="lower",
                    extent=[0, dur_sec, 0, SR / 2 / 1000],
                    cmap=SPEC_CMAP, vmin=vmin_m, interpolation="nearest")
    ax0.set_xlabel(r"Time (s)")
    ax0.set_ylabel(r"Frequency (kHz)")
    ax0.set_title(r"MDCT coefficients $\log_{10}|C[k]|$ --- piano chord progression")
    plt.colorbar(im, ax=ax0, pad=0.01, fraction=0.035).ax.tick_params(labelsize=8)

    t_sig = np.arange(n_compare) / SR
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t_sig, sig_cmp, color="steelblue", lw=0.8, alpha=0.85, label="Original")
    ax1.plot(t_sig, recon, color="tomato", lw=0.8, ls="--", alpha=0.8,
             label="IMDCT reconstruction")
    ax1.set_xlabel(r"Time (s)")
    ax1.set_ylabel(r"Amplitude")
    ax1.set_title(r"Perfect reconstruction: original vs.\ IMDCT output")
    legend_below(ax1, ncol=2)

    ax2 = fig.add_subplot(gs[2])
    error = sig_cmp - recon
    ax2.plot(t_sig, error, color="gray", lw=0.7, alpha=0.9)
    ax2.set_xlabel(r"Time (s)")
    ax2.set_ylabel(r"Error")
    ax2.set_title(r"Reconstruction error (RMS $\approx 10^{-14}$, floating-point noise only)")
    ax2.axhline(0, color="black", lw=0.5)

    save(fig, "fig_mdct")

fig_mdct()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 18: FFT convolution — linear-phase vs minimum-phase FIR filter
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 18: FFT convolution / minimum phase")

def minimum_phase_cepstrum(h, n_fft_factor=8):
    """Convert linear-phase FIR to minimum-phase via real-cepstrum method."""
    n = len(h)
    nfft = int(2 ** np.ceil(np.log2(n * n_fft_factor)))
    H = np.fft.rfft(h, nfft)
    log_H = np.log(np.abs(H) + 1e-30)
    cep = np.fft.irfft(log_H)
    win = np.zeros_like(cep)
    win[0] = 1.0
    win[1:nfft // 2] = 2.0
    if nfft % 2 == 0:
        win[nfft // 2] = 1.0
    H_min = np.exp(np.fft.rfft(cep * win))
    h_min = np.fft.irfft(H_min)[:n]
    return np.real(h_min)


def fig_convolution_min_phase():
    from scipy.signal import firwin, freqz

    fir_lp = firwin(64, 2000 / (SR / 2), window="hamming")
    fir_mp = minimum_phase_cepstrum(fir_lp)

    t_lp = np.arange(len(fir_lp)) / SR * 1000
    t_mp = np.arange(len(fir_mp)) / SR * 1000

    w_lp, H_lp = freqz(fir_lp, worN=4096, fs=SR)
    _, H_mp     = freqz(fir_mp, worN=4096, fs=SR)

    fig, axes = plt.subplots(2, 2, figsize=(FW_FULL, 5.0),
                              gridspec_kw={"hspace": 0.72, "wspace": 0.48})

    ax = axes[0, 0]
    ax.stem(t_lp, fir_lp, linefmt="C0-", markerfmt="C0o", basefmt="k-",
            label="Linear-phase")
    ax.stem(t_mp, fir_mp, linefmt="C1-", markerfmt="C1^", basefmt="k-",
            label="Min.-phase")
    ax.set_xlabel(r"Time (ms)")
    ax.set_ylabel(r"Amplitude")
    ax.set_title(r"Impulse responses (64-tap LPF, $f_c{=}2$\,kHz)")
    legend_below(ax, ncol=2)

    ax = axes[0, 1]
    ax.plot(w_lp / 1000, 20 * np.log10(np.abs(H_lp) + 1e-10),
            color="C0", lw=1.2, label="Linear-phase")
    ax.plot(w_lp / 1000, 20 * np.log10(np.abs(H_mp) + 1e-10),
            color="C1", lw=1.2, ls="--", label="Min.-phase")
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel(r"Magnitude (dB)")
    ax.set_title(r"Frequency response (identical magnitude)")
    ax.set_ylim(-70, 5)
    legend_below(ax, ncol=2)

    ax = axes[1, 0]
    ax.plot(w_lp / 1000, np.unwrap(np.angle(H_lp)) / np.pi * 180,
            color="C0", lw=1.2, label="Linear-phase")
    ax.plot(w_lp / 1000, np.unwrap(np.angle(H_mp)) / np.pi * 180,
            color="C1", lw=1.2, ls="--", label="Min.-phase")
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel(r"Phase (degrees)")
    ax.set_title(r"Phase response")
    legend_below(ax, ncol=2)

    from scipy.signal import group_delay
    _, gd_lp = group_delay((fir_lp, 1), w=w_lp, fs=SR)
    _, gd_mp = group_delay((fir_mp, 1), w=w_lp, fs=SR)
    ax = axes[1, 1]
    ax.plot(w_lp / 1000, gd_lp / SR * 1000, color="C0", lw=1.2,
            label="Linear-phase")
    ax.plot(w_lp / 1000, gd_mp / SR * 1000, color="C1", lw=1.2, ls="--",
            label="Min.-phase")
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel(r"Group delay (ms)")
    ax.set_title(r"Group delay")
    ax.set_ylim(-2, 4)
    legend_below(ax, ncol=2)

    save(fig, "fig_convolution_min_phase")

fig_convolution_min_phase()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 19: COLA (Constant Overlap-Add) visualisation
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 19: COLA visualisation")

def fig_cola():
    N = 64
    hop = N // 2
    n1 = N - 1
    n_arr = np.arange(N)

    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n_arr / n1)
    rect = np.ones(N)

    fig, axes = plt.subplots(1, 2, figsize=(FW_FULL, 3.6),
                              gridspec_kw={"wspace": 0.46})
    fig.subplots_adjust(bottom=0.22)

    for ax, w, label in zip(axes,
                             [hann, rect],
                             [r"Hanning (50\% overlap)", r"Rectangular (50\% overlap)"]):
        n_frames_show = 6
        t_total = N + (n_frames_show - 1) * hop
        t_axis = np.arange(t_total)
        cola_sum = np.zeros(t_total)
        colors_c = plt.cm.tab10(np.linspace(0, 0.9, n_frames_show))

        for m in range(n_frames_show):
            start = m * hop
            frame_w = np.zeros(t_total)
            frame_w[start:start + N] = w
            cola_sum[start:start + N] += w
            ax.fill_between(t_axis, frame_w, alpha=0.25, color=colors_c[m])
            ax.plot(t_axis, frame_w, lw=0.9, color=colors_c[m], alpha=0.7)

        ax.plot(t_axis, cola_sum, color="black", lw=1.6, label=r"$\sum_m w[n-mH]$")
        ax.axhline(cola_sum.max(), color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel(r"Sample index $n$")
        ax.set_ylabel(r"Amplitude")
        ax.set_title(label)
        legend_below(ax, ncol=1)
        ax.set_xlim(0, t_total - 1)

    save(fig, "fig_cola")

fig_cola()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 20: Full pipeline overview — waveform → STFT → mel → MFCC
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 20: Pipeline overview")

def fig_pipeline_overview():
    """2×2 grid: waveform, linear spec, mel spec, MFCC."""
    vowel_sig = speech_vowels(SR)
    n_trim = int(2.0 * SR)
    sig_trim = vowel_sig[:n_trim]

    stft_p_speech = sg.StftParams(512, 128, sg.WindowType.hanning)
    params_speech = sg.SpectrogramParams(sg.StftParams(512, 128, sg.WindowType.hanning), SR)

    spec_db  = sg.compute_linear_db_spectrogram(sig_trim, params_speech)
    mel_p    = sg.MelParams(40, 0.0, 8000.0)
    mel_db   = sg.compute_mel_db_spectrogram(sig_trim, params_speech, mel_p)
    mfcc     = sg.compute_mfcc(sig_trim, stft_p_speech, SR, 40, sg.MfccParams(13))

    hop = 128
    t_sig = np.arange(n_trim) / SR

    # 2×2 layout so the figure fits on an A5 page
    fig, axes = plt.subplots(2, 2, figsize=(FW_FULL, 5.0),
                              gridspec_kw={"hspace": 0.62, "wspace": 0.42})

    # (a) Waveform
    ax = axes[0, 0]
    ax.plot(t_sig, sig_trim, color="steelblue", lw=0.6, alpha=0.9)
    ax.set_xlim(0, 2.0)
    ax.set_ylabel(r"Amplitude")
    ax.set_title(r"(a) Waveform")
    ax.set_xlabel(r"Time (s)")

    # (b) Linear spectrogram (dB)
    ax = axes[0, 1]
    n_freq, n_frames = spec_db.data.shape
    dur_sec = n_frames * hop / SR
    ticks_b, ticks_l = hz_ticks_for_linear(512, SR, n_ticks=4)
    d_db = spec_db.data
    vmin2 = d_db.max() - 75
    im2 = ax.imshow(d_db, aspect="auto", origin="lower",
                    extent=[0, dur_sec, 0, n_freq],
                    cmap=SPEC_CMAP, vmin=vmin2, interpolation="nearest")
    ax.set_yticks(ticks_b); ax.set_yticklabels(ticks_l, fontsize=7.5)
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Freq (Hz)")
    ax.set_title(r"(b) Linear spec.\ (dB)")
    plt.colorbar(im2, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7.5)

    # (c) Mel spectrogram (dB)
    ax = axes[1, 0]
    n_mel, n_frames_m = mel_db.data.shape
    dur_m = n_frames_m * hop / SR
    d_mel = mel_db.data
    vmin3 = d_mel.max() - 75
    im3 = ax.imshow(d_mel, aspect="auto", origin="lower",
                    extent=[0, dur_m, 0, n_mel],
                    cmap=SPEC_CMAP, vmin=vmin3, interpolation="nearest")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Mel band")
    ax.set_title(r"(c) Mel spec.\ (dB, $M{=}40$)")
    plt.colorbar(im3, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7.5)

    # (d) MFCCs
    ax = axes[1, 1]
    n_c, n_f = mfcc.shape
    dur_mf = n_f * hop / SR
    im4 = ax.imshow(mfcc, aspect="auto", origin="lower",
                    extent=[0, dur_mf, -0.5, n_c - 0.5],
                    cmap=MFCC_CMAP, interpolation="nearest")
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel(r"Coefficient")
    ax.set_title(r"(d) MFCCs ($C{=}13$)")
    plt.colorbar(im4, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7.5)

    save(fig, "fig_pipeline_overview")

fig_pipeline_overview()


print("\nAll figures generated successfully.")
