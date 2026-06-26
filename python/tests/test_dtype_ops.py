"""Tests for the native ``dtype`` parameter on the standalone generic ops.

Covers the 1D FFT/STFT family, the 2D FFT family, image ops, MDCT, and the
binaural spectrograms. For each family we compute at ``dtype="float32"`` and
``dtype="float64"`` and assert the returned array dtype matches and that the
two precisions agree within tolerance.
"""

import numpy as np
import pytest
import spectrograms as sg


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(2048).astype(np.float64)


@pytest.fixture
def params():
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    return sg.SpectrogramParams(stft, sample_rate=16000.0)


# ---------------------------------------------------------------------------
# 1D transforms
# ---------------------------------------------------------------------------


def test_stft_dtype(signal, params):
    s32 = sg.compute_stft(signal, params, dtype="float32")
    s64 = sg.compute_stft(signal, params, dtype="float64")
    # The result is a rich StftResult object: complex data + real-precision dtype.
    assert s32.dtype == "float32"
    assert s64.dtype == "float64"
    assert s32.data.dtype == np.complex64
    assert s64.data.dtype == np.complex128
    assert s32.data.shape == s64.data.shape
    np.testing.assert_allclose(np.asarray(s32), np.asarray(s64), rtol=1e-3, atol=1e-3)


def test_stft_default_is_float64(signal, params):
    result = sg.compute_stft(signal, params)
    assert result.dtype == "float64"
    assert result.data.dtype == np.complex128


def test_fft_dtype(signal):
    f32 = sg.compute_fft(signal, 2048, dtype="float32")
    f64 = sg.compute_fft(signal, 2048, dtype="float64")
    assert f32.dtype == np.complex64
    assert f64.dtype == np.complex128
    np.testing.assert_allclose(f32, f64, rtol=1e-3, atol=1e-2)


def test_rfft_dtype(signal):
    r32 = sg.compute_rfft(signal, 2048, dtype="float32")
    r64 = sg.compute_rfft(signal, 2048, dtype="float64")
    assert r32.dtype == np.float32
    assert r64.dtype == np.float64
    np.testing.assert_allclose(r32, r64, rtol=1e-3, atol=1e-2)


def test_power_and_magnitude_spectrum_dtype(signal):
    p32 = sg.compute_power_spectrum(signal, 2048, dtype="float32")
    p64 = sg.compute_power_spectrum(signal, 2048, dtype="float64")
    assert p32.dtype == np.float32
    assert p64.dtype == np.float64
    np.testing.assert_allclose(p32, p64, rtol=1e-3, atol=1e-1)

    m32 = sg.compute_magnitude_spectrum(signal, 2048, dtype="float32")
    m64 = sg.compute_magnitude_spectrum(signal, 2048, dtype="float64")
    assert m32.dtype == np.float32
    assert m64.dtype == np.float64
    np.testing.assert_allclose(m32, m64, rtol=1e-3, atol=1e-2)


def test_irfft_roundtrip_dtype(signal):
    spectrum = sg.compute_fft(signal, 2048, dtype="float64")
    out32 = sg.compute_irfft(spectrum, 2048, dtype="float32")
    out64 = sg.compute_irfft(spectrum, 2048, dtype="float64")
    assert out32.dtype == np.float32
    assert out64.dtype == np.float64
    np.testing.assert_allclose(out64, signal, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(out32, signal, rtol=1e-3, atol=1e-3)


def test_istft_roundtrip_dtype(signal, params):
    stft = sg.compute_stft(signal, params, dtype="float64").data
    rec32 = sg.compute_istft(
        stft, 512, 256, sg.WindowType.hanning, True, dtype="float32"
    )
    rec64 = sg.compute_istft(
        stft, 512, 256, sg.WindowType.hanning, True, dtype="float64"
    )
    assert rec32.dtype == np.float32
    assert rec64.dtype == np.float64


# ---------------------------------------------------------------------------
# 2D FFT family
# ---------------------------------------------------------------------------


@pytest.fixture
def image():
    rng = np.random.default_rng(1)
    return rng.standard_normal((32, 32)).astype(np.float64)


def test_fft2d_dtype(image):
    f32 = sg.fft2d(image, dtype="float32")
    f64 = sg.fft2d(image, dtype="float64")
    assert f32.dtype == np.complex64
    assert f64.dtype == np.complex128
    assert f32.shape == f64.shape
    np.testing.assert_allclose(f32, f64, rtol=1e-3, atol=1e-3)


def test_ifft2d_roundtrip_dtype(image):
    spectrum = sg.fft2d(image, dtype="float64")
    r32 = sg.ifft2d(spectrum, 32, dtype="float32")
    r64 = sg.ifft2d(spectrum, 32, dtype="float64")
    assert r32.dtype == np.float32
    assert r64.dtype == np.float64
    np.testing.assert_allclose(r64, image, rtol=1e-6, atol=1e-9)


def test_power_magnitude_2d_dtype(image):
    for fn in (sg.power_spectrum_2d, sg.magnitude_spectrum_2d):
        a32 = fn(image, dtype="float32")
        a64 = fn(image, dtype="float64")
        assert a32.dtype == np.float32
        assert a64.dtype == np.float64
        np.testing.assert_allclose(a32, a64, rtol=1e-3, atol=1e-2)


def test_fftshift_dtype(image):
    s32 = sg.fftshift(image, dtype="float32")
    s64 = sg.fftshift(image, dtype="float64")
    assert s32.dtype == np.float32
    assert s64.dtype == np.float64
    np.testing.assert_allclose(s32.astype(np.float64), s64)


def test_fftshift_1d_dtype():
    arr = np.arange(8, dtype=np.float64)
    s32 = sg.fftshift_1d(arr, dtype="float32")
    s64 = sg.fftshift_1d(arr, dtype="float64")
    assert s32.dtype == np.float32
    assert s64.dtype == np.float64
    np.testing.assert_allclose(s32.astype(np.float64), s64)


# ---------------------------------------------------------------------------
# Image ops
# ---------------------------------------------------------------------------


def test_convolve_fft_dtype(image):
    kernel = sg.gaussian_kernel_2d(5, 1.0)
    c32 = sg.convolve_fft(image, kernel, dtype="float32")
    c64 = sg.convolve_fft(image, kernel, dtype="float64")
    assert c32.dtype == np.float32
    assert c64.dtype == np.float64
    np.testing.assert_allclose(c32, c64, rtol=1e-3, atol=1e-3)


def test_lowpass_filter_dtype(image):
    l32 = sg.lowpass_filter(image, 0.3, dtype="float32")
    l64 = sg.lowpass_filter(image, 0.3, dtype="float64")
    assert l32.dtype == np.float32
    assert l64.dtype == np.float64
    np.testing.assert_allclose(l32, l64, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# MDCT
# ---------------------------------------------------------------------------


def test_mdct_imdct_dtype(signal):
    mp = sg.MdctParams.sine_window(512)
    c32 = sg.mdct(signal, mp, dtype="float32")
    c64 = sg.mdct(signal, mp, dtype="float64")
    assert c32.dtype == np.float32
    assert c64.dtype == np.float64
    assert c32.shape == c64.shape
    np.testing.assert_allclose(c32, c64, rtol=1e-3, atol=1e-3)

    r32 = sg.imdct(c64, mp, dtype="float32")
    r64 = sg.imdct(c64, mp, dtype="float64")
    assert r32.dtype == np.float32
    assert r64.dtype == np.float64


# ---------------------------------------------------------------------------
# Binaural
# ---------------------------------------------------------------------------


@pytest.fixture
def stereo():
    rng = np.random.default_rng(2)
    n = 16000
    left = rng.standard_normal(n).astype(np.float64)
    right = np.roll(left, 5) + 0.01 * rng.standard_normal(n)
    return [left, right]


@pytest.fixture
def itd_params():
    stft = sg.StftParams(
        n_fft=4096, hop_size=1024, window=sg.WindowType.hanning, centre=True
    )
    sp = sg.SpectrogramParams(stft, sample_rate=44100.0)
    return sg.ITDSpectrogramParams(sp)


def test_itd_spectrogram_dtype(stereo, itd_params):
    a32 = sg.compute_itd_spectrogram(stereo, itd_params, dtype="float32")
    a64 = sg.compute_itd_spectrogram(stereo, itd_params, dtype="float64")
    # compute_*_spectrogram now returns a rich result object, not a bare array
    assert a32.data.dtype == np.float32
    assert a64.data.dtype == np.float64
    assert a32.shape == a64.shape
    np.testing.assert_allclose(np.asarray(a32), np.asarray(a64), rtol=1e-2, atol=1e-3)


def test_itd_spectrogram_diff_dtype(stereo, itd_params):
    rng = np.random.default_rng(3)
    test = [stereo[0], np.roll(stereo[1], 2) + 0.01 * rng.standard_normal(16000)]
    arr32, deg32, itd32 = sg.compute_itd_spectrogram_diff(
        stereo, test, itd_params, dtype="float32"
    )
    arr64, deg64, itd64 = sg.compute_itd_spectrogram_diff(
        stereo, test, itd_params, dtype="float64"
    )
    assert arr32.dtype == np.float32
    assert arr64.dtype == np.float64
    # scalar tuple elements are always Python floats regardless of dtype
    assert isinstance(deg32, float) and isinstance(itd32, float)
    assert isinstance(deg64, float) and isinstance(itd64, float)


def test_invalid_dtype_raises(signal, params):
    with pytest.raises(ValueError):
        sg.compute_fft(signal, 2048, dtype="float16")
