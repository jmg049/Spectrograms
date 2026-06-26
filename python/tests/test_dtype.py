"""Tests for the native f32/f64 ``dtype`` parameter on the compute functions."""

import numpy as np
import pytest
import spectrograms as sg

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def signal():
    """A deterministic mono test signal."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(16000).astype(np.float64)


@pytest.fixture
def params():
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    return sg.SpectrogramParams(stft, sample_rate=16000.0)


def _mel(signal, params, dtype):
    mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
    return sg.compute_mel_power_spectrogram(signal, params, mel, dtype=dtype)


def _linear(signal, params, dtype):
    return sg.compute_linear_power_spectrogram(signal, params, dtype=dtype)


def _cqt(signal, params, dtype):
    cqt = sg.CqtParams(bins_per_octave=12, n_octaves=6, f_min=32.7)
    return sg.compute_cqt_power_spectrogram(signal, params, cqt, dtype=dtype)


COMPUTERS = {"mel": _mel, "linear": _linear, "cqt": _cqt}


class TestDtypeDefault:
    def test_explicit_float64(self, signal, params):
        spec = _mel(signal, params, "float64")
        assert spec.dtype == "float64"

    def test_omitted_dtype_is_float64(self, signal, params):
        mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
        spec = sg.compute_mel_power_spectrogram(signal, params, mel)
        assert spec.dtype == "float64"
        assert spec.data.dtype == np.float64


@pytest.mark.parametrize("name", list(COMPUTERS))
class TestDtypeRoundtrip:
    def test_float32_dtype_reported(self, name, signal, params):
        spec = COMPUTERS[name](signal, params, "float32")
        assert spec.dtype == "float32"
        assert spec.data.dtype == np.float32

    def test_float64_dtype_reported(self, name, signal, params):
        spec = COMPUTERS[name](signal, params, "float64")
        assert spec.dtype == "float64"
        assert spec.data.dtype == np.float64

    def test_shapes_match(self, name, signal, params):
        s32 = COMPUTERS[name](signal, params, "float32")
        s64 = COMPUTERS[name](signal, params, "float64")
        assert s32.shape == s64.shape
        assert s32.data.shape == s64.data.shape

    def test_values_close(self, name, signal, params):
        s32 = COMPUTERS[name](signal, params, "float32")
        s64 = COMPUTERS[name](signal, params, "float64")
        # f32 computed natively should be close to the f64 reference.
        assert np.allclose(
            s32.data.astype(np.float64), s64.data, rtol=1e-2, atol=1e-3
        )


class TestDtypeAliases:
    def test_f32_alias(self, signal, params):
        assert _mel(signal, params, "f32").dtype == "float32"

    def test_f64_alias(self, signal, params):
        assert _mel(signal, params, "f64").dtype == "float64"

    def test_invalid_dtype_raises(self, signal, params):
        with pytest.raises(ValueError):
            _mel(signal, params, "int32")


class TestDtypeInputCoercion:
    def test_float32_input_accepted(self, signal, params):
        # Passing f32 samples should still work for both output dtypes.
        s = signal.astype(np.float32)
        assert _linear(s, params, "float32").dtype == "float32"
        assert _linear(s, params, "float64").dtype == "float64"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestDtypeDLPack:
    def test_dlpack_float32(self, signal, params):
        spec = _mel(signal, params, "float32")
        t = torch.from_dlpack(spec)
        assert t.dtype == torch.float32
        assert tuple(t.shape) == spec.shape

    def test_dlpack_float64(self, signal, params):
        spec = _mel(signal, params, "float64")
        t = torch.from_dlpack(spec)
        assert t.dtype == torch.float64
        assert tuple(t.shape) == spec.shape
