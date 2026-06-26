"""Tests for the rich binaural result classes returned by the compute_* functions.

The ``compute_itd/ipd/ild/ilr_spectrogram`` functions return dual-dtype result
objects (``ItdSpectrogram`` / ``IpdSpectrogram`` / ``IldSpectrogram`` /
``IlrSpectrogram``) exposing ``.data`` / ``.dtype`` / metadata / ``.histogram(...)``
and the array protocols, rather than a bare NumPy array.
"""

import numpy as np
import pytest
import spectrograms as sg

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def stereo():
    rng = np.random.default_rng(0)
    n = 16000
    left = rng.standard_normal(n)
    right = np.roll(left, 5) + 0.01 * rng.standard_normal(n)
    return [left, right]


@pytest.fixture
def spec_params():
    stft = sg.StftParams(
        n_fft=4096, hop_size=1024, window=sg.WindowType.hanning, centre=True
    )
    return sg.SpectrogramParams(stft, sample_rate=44100.0)


def _cases(spec_params):
    return [
        (
            sg.compute_itd_spectrogram,
            sg.ITDSpectrogramParams(spec_params),
            sg.ItdSpectrogram,
        ),
        (
            sg.compute_ipd_spectrogram,
            sg.IPDSpectrogramParams(spec_params),
            sg.IpdSpectrogram,
        ),
        (
            sg.compute_ild_spectrogram,
            sg.ILDSpectrogramParams(spec_params),
            sg.IldSpectrogram,
        ),
        (
            sg.compute_ilr_spectrogram,
            sg.ILRSpectrogramParams(spec_params),
            sg.IlrSpectrogram,
        ),
    ]


@pytest.mark.parametrize("dtype,np_dtype", [("float32", np.float32), ("float64", np.float64)])
def test_returns_class_with_dtype(stereo, spec_params, dtype, np_dtype):
    for fn, params, cls in _cases(spec_params):
        result = fn(stereo, params, dtype=dtype)
        assert isinstance(result, cls)
        assert result.dtype == dtype
        assert result.data.dtype == np_dtype
        assert result.data.ndim == 2
        assert result.shape == (result.n_bins, result.n_frames)
        assert result.data.shape == result.shape


def test_metadata_getters(stereo, spec_params):
    for fn, params, _cls in _cases(spec_params):
        result = fn(stereo, params, dtype="float64")
        freqs = result.frequencies
        times = result.times
        assert len(freqs) == result.n_bins
        assert len(times) == result.n_frames
        fmin, fmax = result.frequency_range()
        assert fmin == pytest.approx(freqs[0])
        assert fmax == pytest.approx(freqs[-1])
        assert result.duration() == pytest.approx(times[-1] - times[0])
        # params round-trips to the correct param class
        assert result.params is not None


def test_histogram_basic(stereo, spec_params):
    for fn, params, _cls in _cases(spec_params):
        result = fn(stereo, params, dtype="float64")
        hist = result.histogram()
        assert isinstance(hist, np.ndarray)
        assert hist.dtype == np.float64
        assert hist.ndim == 2
        # default num_bins is 400
        assert hist.shape == (400, result.n_frames)


def test_histogram_respects_num_bins(stereo, spec_params):
    for fn, params, _cls in _cases(spec_params):
        result = fn(stereo, params, dtype="float64")
        hist = result.histogram(num_bins=64)
        assert hist.shape == (64, result.n_frames)


def test_histogram_dtype_independent(stereo, spec_params):
    # histogram returns float64 counts regardless of the data dtype
    for fn, params, _cls in _cases(spec_params):
        r32 = fn(stereo, params, dtype="float32")
        r64 = fn(stereo, params, dtype="float64")
        assert r32.histogram(num_bins=32).dtype == np.float64
        assert r64.histogram(num_bins=32).dtype == np.float64


def test_array_protocol(stereo, spec_params):
    for fn, params, _cls in _cases(spec_params):
        result = fn(stereo, params, dtype="float32")
        arr = np.asarray(result)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, result.data)
        # __array__ with explicit dtype upcasts
        arr64 = np.asarray(result, dtype=np.float64)
        assert arr64.dtype == np.float64


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@pytest.mark.parametrize("dtype,torch_dtype", [("float32", "float32"), ("float64", "float64")])
def test_torch_dlpack_roundtrip(stereo, spec_params, dtype, torch_dtype):
    for fn, params, _cls in _cases(spec_params):
        result = fn(stereo, params, dtype=dtype)
        tensor = torch.from_dlpack(result)
        assert str(tensor.dtype) == f"torch.{torch_dtype}"
        assert tuple(tensor.shape) == result.shape
        np.testing.assert_array_equal(tensor.numpy(), result.data)
