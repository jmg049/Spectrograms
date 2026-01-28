import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fftpack import dct


def stft(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    centre: bool = True,
):
    signal = np.asarray(signal, dtype=np.float64)

    if centre:
        pad = n_fft // 2
        signal = np.pad(signal, (pad, pad), mode="constant")

    frames = sliding_window_view(signal, n_fft)[::hop_length]
    frames = frames * window[None, :]

    spectrum = np.fft.rfft(frames, axis=-1)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    times = np.arange(frames.shape[0]) * hop_length / sample_rate

    if centre:
        times -= (n_fft / 2) / sample_rate

    return spectrum.T, freqs, times


def hann_window(n_fft: int) -> np.ndarray:
    n = np.arange(n_fft)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (n_fft - 1))


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    return np.abs(stft_matrix) ** 2


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    return np.abs(stft_matrix)


def db_spectrogram(power: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, eps))


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
):
    n_freqs = n_fft // 2 + 1
    df = sample_rate / n_fft

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points / df).astype(int)

    fb = np.zeros((n_mels, n_freqs))

    for m in range(n_mels):
        left, centre, right = bin_points[m : m + 3]

        for k in range(left, centre):
            fb[m, k] = (k - left) / (centre - left)

        for k in range(centre, right):
            fb[m, k] = (right - k) / (right - centre)

    return fb


def mel_spectrogram(power: np.ndarray, fb: np.ndarray) -> np.ndarray:
    return fb @ power


def log_frequency_matrix(
    sample_rate: int,
    n_fft: int,
    n_bins: int,
    f_min: float,
    f_max: float,
):
    n_freqs = n_fft // 2 + 1
    df = sample_rate / n_fft

    log_f = np.linspace(np.log(f_min), np.log(f_max), n_bins)
    freqs = np.exp(log_f)

    M = np.zeros((n_bins, n_freqs))

    for i, f in enumerate(freqs):
        exact = f / df
        lo = int(np.floor(exact))
        hi = int(np.ceil(exact))

        if lo == hi and lo < n_freqs:
            M[i, lo] = 1.0
        elif hi < n_freqs:
            alpha = exact - lo
            M[i, lo] = 1.0 - alpha
            M[i, hi] = alpha

    return M


def logfreq_spectrogram(power: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M @ power


def erb(fc):
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def erb_to_hz(erb_value):
    return (erb_value / 24.7 - 1.0) * 1000.0 / 4.37


def erb_centers(f_min, f_max, n_filters):
    erb_min = erb(f_min)
    erb_max = erb(f_max)
    return erb_to_hz(np.linspace(erb_min, erb_max, n_filters))


def gammatone_response(freqs, fc, order=4):
    b = 1.019 * erb(fc)
    return 1.0 / (1.0 + 1j * (freqs - fc) / b) ** order


def erb_spectrogram(
    stft_matrix: np.ndarray,
    freqs: np.ndarray,
    centre_freqs: np.ndarray,
):
    out = np.zeros((len(centre_freqs), stft_matrix.shape[1]))

    for i, fc in enumerate(centre_freqs):
        G = gammatone_response(freqs, fc)
        Y = G[:, None] * stft_matrix
        out[i] = np.sum(np.abs(Y) ** 2, axis=0)

    return out


def cqt(
    signal: np.ndarray,
    sample_rate: int,
    freqs: np.ndarray,
    Q: float,
    window_fn,
):
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    out = np.zeros((len(freqs), N), dtype=np.complex128)

    for k, fk in enumerate(freqs):
        Nk = int(np.ceil(Q * sample_rate / fk))
        window = window_fn(Nk)

        t = np.arange(Nk) / sample_rate
        kernel = window * np.exp(2j * np.pi * fk * t)

        for n in range(N - Nk):
            out[k, n] = np.dot(signal[n : n + Nk], kernel.conj())

    return out


def mfcc(
    power: np.ndarray,
    mel_fb: np.ndarray,
    n_mfcc: int = 13,
    eps: float = 1e-10,
):
    mel = mel_fb @ power
    log_mel = np.log(mel + eps)
    coeffs = dct(log_mel, type=2, axis=0, norm=None)
    return coeffs[:n_mfcc]


def chroma(
    power: np.ndarray,
    freqs: np.ndarray,
    f_ref: float = 440.0,
):
    chroma = np.zeros((12, power.shape[1]))

    for i, f in enumerate(freqs):
        if f <= 0:
            continue

        note = 12 * np.log2(f / f_ref)
        pitch_class = int(np.round(note)) % 12
        chroma[pitch_class] += power[i]

    chroma /= np.maximum(chroma.sum(axis=0, keepdims=True), 1e-12)
    return chroma
