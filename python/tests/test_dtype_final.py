"""Tests for the ``dtype`` parameter on the final batch of generic entry points:
``compute_chromagram``, ``compute_mfcc``, ``gaussian_kernel_2d``, ``fftfreq``,
``rfftfreq`` and the ``Fft2dPlanner`` class.
"""

import numpy as np
import pytest
import spectrograms as sg


@pytest.fixture
def signal():
    """A deterministic mono test signal."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(16000).astype(np.float64)


@pytest.fixture
def stft():
    return sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )


# ---------------------------------------------------------------------------
# compute_chromagram
# ---------------------------------------------------------------------------


def test_compute_chromagram_dtype(signal, stft):
    chroma_params = sg.ChromaParams()
    c32 = sg.compute_chromagram(signal, stft, 16000.0, chroma_params, dtype="float32")
    c64 = sg.compute_chromagram(signal, stft, 16000.0, chroma_params, dtype="float64")
    # compute_chromagram now returns a rich Chromagram object.
    assert c32.dtype == "float32"
    assert c64.dtype == "float64"
    assert c32.data.dtype == np.float32
    assert c64.data.dtype == np.float64
    assert c32.data.shape == c64.data.shape
    np.testing.assert_allclose(np.asarray(c32), np.asarray(c64), rtol=1e-3, atol=1e-3)


def test_compute_chromagram_default_is_float64(signal, stft):
    chroma_params = sg.ChromaParams()
    c = sg.compute_chromagram(signal, stft, 16000.0, chroma_params)
    assert c.dtype == "float64"
    assert c.data.dtype == np.float64


def test_compute_chromagram_invalid_dtype(signal, stft):
    chroma_params = sg.ChromaParams()
    with pytest.raises(ValueError):
        sg.compute_chromagram(signal, stft, 16000.0, chroma_params, dtype="float16")


# ---------------------------------------------------------------------------
# compute_mfcc
# ---------------------------------------------------------------------------


def test_compute_mfcc_dtype(signal, stft):
    mfcc_params = sg.MfccParams(n_mfcc=13)
    m32 = sg.compute_mfcc(signal, stft, 16000.0, 40, mfcc_params, dtype="float32")
    m64 = sg.compute_mfcc(signal, stft, 16000.0, 40, mfcc_params, dtype="float64")
    # compute_mfcc now returns a rich Mfcc object.
    assert m32.dtype == "float32"
    assert m64.dtype == "float64"
    assert m32.data.dtype == np.float32
    assert m64.data.dtype == np.float64
    assert m32.data.shape == m64.data.shape
    np.testing.assert_allclose(np.asarray(m32), np.asarray(m64), rtol=1e-2, atol=1e-2)


def test_compute_mfcc_default_is_float64(signal, stft):
    mfcc_params = sg.MfccParams(n_mfcc=13)
    m = sg.compute_mfcc(signal, stft, 16000.0, 40, mfcc_params)
    assert m.dtype == "float64"
    assert m.data.dtype == np.float64


def test_compute_mfcc_invalid_dtype(signal, stft):
    mfcc_params = sg.MfccParams(n_mfcc=13)
    with pytest.raises(ValueError):
        sg.compute_mfcc(signal, stft, 16000.0, 40, mfcc_params, dtype="nope")


# ---------------------------------------------------------------------------
# gaussian_kernel_2d
# ---------------------------------------------------------------------------


def test_gaussian_kernel_2d_dtype():
    k32 = sg.gaussian_kernel_2d(5, 1.0, dtype="float32")
    k64 = sg.gaussian_kernel_2d(5, 1.0, dtype="float64")
    assert k32.dtype == np.float32
    assert k64.dtype == np.float64
    assert k32.shape == (5, 5)
    np.testing.assert_allclose(k32, k64, rtol=1e-5, atol=1e-6)
    assert abs(float(k64.sum()) - 1.0) < 1e-10


def test_gaussian_kernel_2d_default_is_float64():
    assert sg.gaussian_kernel_2d(5, 1.0).dtype == np.float64


def test_gaussian_kernel_2d_invalid_dtype():
    with pytest.raises(ValueError):
        sg.gaussian_kernel_2d(5, 1.0, dtype="float128")


# ---------------------------------------------------------------------------
# fftfreq / rfftfreq
# ---------------------------------------------------------------------------


def test_fftfreq_dtype():
    f32 = sg.fftfreq(8, 1.0, dtype="float32")
    f64 = sg.fftfreq(8, 1.0, dtype="float64")
    assert f32.dtype == np.float32
    assert f64.dtype == np.float64
    assert f32.shape == (8,)
    np.testing.assert_allclose(f32, f64, rtol=1e-5, atol=1e-6)


def test_rfftfreq_dtype():
    f32 = sg.rfftfreq(8, 1.0, dtype="float32")
    f64 = sg.rfftfreq(8, 1.0, dtype="float64")
    assert f32.dtype == np.float32
    assert f64.dtype == np.float64
    assert f32.shape == (5,)
    np.testing.assert_allclose(f32, f64, rtol=1e-5, atol=1e-6)


def test_fftfreq_default_is_float64():
    assert sg.fftfreq(8, 1.0).dtype == np.float64
    assert sg.rfftfreq(8, 1.0).dtype == np.float64


def test_fftfreq_invalid_dtype():
    with pytest.raises(ValueError):
        sg.fftfreq(8, 1.0, dtype="bad")
    with pytest.raises(ValueError):
        sg.rfftfreq(8, 1.0, dtype="bad")


# ---------------------------------------------------------------------------
# Fft2dPlanner
# ---------------------------------------------------------------------------


@pytest.fixture
def image():
    rng = np.random.default_rng(1)
    return rng.standard_normal((32, 32)).astype(np.float64)


def test_planner_dtype_getter():
    assert sg.Fft2dPlanner().dtype == "float64"
    assert sg.Fft2dPlanner(dtype="float32").dtype == "float32"
    assert sg.Fft2dPlanner(dtype="float64").dtype == "float64"


def test_planner_invalid_dtype():
    with pytest.raises(ValueError):
        sg.Fft2dPlanner(dtype="float16")


def test_planner_fft2d_dtype(image):
    p32 = sg.Fft2dPlanner(dtype="float32")
    p64 = sg.Fft2dPlanner(dtype="float64")
    s32 = p32.fft2d(image)
    s64 = p64.fft2d(image)
    assert s32.dtype == np.complex64
    assert s64.dtype == np.complex128
    assert s32.shape == s64.shape
    np.testing.assert_allclose(s32, s64, rtol=1e-3, atol=1e-3)


def test_planner_roundtrip_dtype(image):
    for dtype, real_t in (("float32", np.float32), ("float64", np.float64)):
        p = sg.Fft2dPlanner(dtype=dtype)
        spectrum = p.fft2d(image)
        recon = p.ifft2d(spectrum, image.shape[1])
        assert recon.dtype == real_t
        np.testing.assert_allclose(recon, image, rtol=1e-2, atol=1e-2)


def test_planner_power_and_magnitude_dtype(image):
    for dtype, real_t in (("float32", np.float32), ("float64", np.float64)):
        p = sg.Fft2dPlanner(dtype=dtype)
        pw = p.power_spectrum_2d(image)
        mag = p.magnitude_spectrum_2d(image)
        assert pw.dtype == real_t
        assert mag.dtype == real_t
