"""Tests for the rich result classes returned by ``compute_stft``,
``compute_chromagram`` and ``compute_mfcc``.

Each of these functions now returns a dedicated result *object* (``StftResult``,
``Chromagram``, ``Mfcc``) that carries the data plus metadata and stays
array-compatible via ``__array__`` / ``__dlpack__``, mirroring the existing
``Spectrogram`` class. These tests pin the new contract: type, ``.data`` /
``.dtype`` semantics, ``np.asarray`` interop and DLPack round-trips through
torch at both precisions.
"""

import numpy as np
import pytest
import spectrograms as sg

torch = pytest.importorskip("torch")


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(16000).astype(np.float64)


@pytest.fixture
def stft():
    return sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )


@pytest.fixture
def params(stft):
    return sg.SpectrogramParams(stft, sample_rate=16000.0)


# ---------------------------------------------------------------------------
# compute_stft -> StftResult
# ---------------------------------------------------------------------------


def test_compute_stft_returns_stftresult(signal, params):
    result = sg.compute_stft(signal, params)
    assert isinstance(result, sg.StftResult)


@pytest.mark.parametrize(
    "dtype, real_str, cplx",
    [("float32", "float32", np.complex64), ("float64", "float64", np.complex128)],
)
def test_stft_dtype_follows_request(signal, params, dtype, real_str, cplx):
    result = sg.compute_stft(signal, params, dtype=dtype)
    assert result.dtype == real_str
    assert result.data.dtype == cplx
    # metadata is present and consistent
    assert result.data.shape == result.shape
    assert result.n_bins == result.data.shape[0]
    assert result.n_frames == result.data.shape[1]
    assert len(result.frequencies) == result.n_bins
    # norm() is the real magnitude at the matching precision
    norm = result.norm()
    assert norm.dtype == np.dtype(real_str)
    np.testing.assert_allclose(norm, np.abs(result.data))


def test_stft_asarray(signal, params):
    result = sg.compute_stft(signal, params, dtype="float32")
    arr = np.asarray(result)
    assert arr.dtype == np.complex64
    np.testing.assert_allclose(arr, result.data)


@pytest.mark.parametrize(
    "dtype, cplx", [("float32", np.complex64), ("float64", np.complex128)]
)
def test_stft_dlpack_torch(signal, params, dtype, cplx):
    result = sg.compute_stft(signal, params, dtype=dtype)
    tensor = torch.from_dlpack(result)
    assert tuple(tensor.shape) == result.shape
    np.testing.assert_allclose(tensor.numpy(), result.data)


# ---------------------------------------------------------------------------
# compute_chromagram -> Chromagram
# ---------------------------------------------------------------------------


def test_compute_chromagram_returns_chromagram(signal, stft):
    result = sg.compute_chromagram(signal, stft, 16000.0, sg.ChromaParams())
    assert isinstance(result, sg.Chromagram)
    assert result.n_bins == 12
    assert list(result.labels)[0] == "C"


@pytest.mark.parametrize(
    "dtype, real_t", [("float32", np.float32), ("float64", np.float64)]
)
def test_chromagram_dtype_follows_request(signal, stft, dtype, real_t):
    result = sg.compute_chromagram(signal, stft, 16000.0, sg.ChromaParams(), dtype=dtype)
    assert result.dtype == dtype
    assert result.data.dtype == real_t
    assert result.data.shape == result.shape
    arr = np.asarray(result)
    assert arr.dtype == real_t


@pytest.mark.parametrize(
    "dtype, real_t", [("float32", np.float32), ("float64", np.float64)]
)
def test_chromagram_dlpack_torch(signal, stft, dtype, real_t):
    result = sg.compute_chromagram(signal, stft, 16000.0, sg.ChromaParams(), dtype=dtype)
    tensor = torch.from_dlpack(result)
    assert tuple(tensor.shape) == result.shape
    np.testing.assert_allclose(tensor.numpy(), result.data)


# ---------------------------------------------------------------------------
# compute_mfcc -> Mfcc
# ---------------------------------------------------------------------------


def test_compute_mfcc_returns_mfcc(signal, stft):
    result = sg.compute_mfcc(signal, stft, 16000.0, 40, sg.MfccParams(n_mfcc=13))
    assert isinstance(result, sg.Mfcc)
    assert result.n_bins == 13


@pytest.mark.parametrize(
    "dtype, real_t", [("float32", np.float32), ("float64", np.float64)]
)
def test_mfcc_dtype_follows_request(signal, stft, dtype, real_t):
    result = sg.compute_mfcc(
        signal, stft, 16000.0, 40, sg.MfccParams(n_mfcc=13), dtype=dtype
    )
    assert result.dtype == dtype
    assert result.data.dtype == real_t
    assert result.data.shape == result.shape
    arr = np.asarray(result)
    assert arr.dtype == real_t


@pytest.mark.parametrize(
    "dtype, real_t", [("float32", np.float32), ("float64", np.float64)]
)
def test_mfcc_dlpack_torch(signal, stft, dtype, real_t):
    result = sg.compute_mfcc(
        signal, stft, 16000.0, 40, sg.MfccParams(n_mfcc=13), dtype=dtype
    )
    tensor = torch.from_dlpack(result)
    assert tuple(tensor.shape) == result.shape
    np.testing.assert_allclose(tensor.numpy(), result.data)
