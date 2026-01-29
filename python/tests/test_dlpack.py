"""Tests for DLPack protocol implementation."""

import numpy as np
import pytest
import spectrograms as sg

# Try to import PyTorch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import JAX
try:
    import jax
    import jax.dlpack

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.fixture
def sample_spectrogram():
    """Create a sample spectrogram for testing."""
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    spec = sg.compute_linear_power_spectrogram(samples, params)
    return spec


@pytest.fixture
def sample_chromagram():
    """Create a sample chromagram for testing."""
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    chroma_params = sg.ChromaParams()
    chroma = sg.compute_chromagram(samples, stft, 16000.0, chroma_params)
    return chroma


class TestDLPackDevice:
    """Test __dlpack_device__ method."""

    def test_spectrogram_device(self, sample_spectrogram):
        """Test that spectrogram returns correct device info."""
        device_type, device_id = sample_spectrogram.__dlpack_device__()
        assert device_type == 1  # kDLCPU
        assert device_id == 0

    def test_chromagram_device(self, sample_chromagram):
        """Test that chromagram returns correct device info."""
        device_type, device_id = sample_chromagram.__dlpack_device__()
        assert device_type == 1  # kDLCPU
        assert device_id == 0


class TestDLPackBasic:
    """Test basic __dlpack__ functionality."""

    def test_spectrogram_dlpack_returns_capsule(self, sample_spectrogram):
        """Test that __dlpack__ returns a PyCapsule."""
        capsule = sample_spectrogram.__dlpack__()
        assert capsule is not None
        # PyCapsule objects have a specific type name
        assert "PyCapsule" in str(type(capsule))

    def test_chromagram_dlpack_returns_capsule(self, sample_chromagram):
        """Test that __dlpack__ returns a PyCapsule."""
        capsule = sample_chromagram.__dlpack__()
        assert capsule is not None
        assert "PyCapsule" in str(type(capsule))

    def test_dlpack_with_none_parameters(self, sample_spectrogram):
        """Test that __dlpack__ accepts None for all optional parameters."""
        capsule = sample_spectrogram.__dlpack__(
            stream=None, max_version=None, dl_device=None, copy=None
        )
        assert capsule is not None


class TestDLPackParameterValidation:
    """Test parameter validation in __dlpack__."""

    def test_stream_must_be_none(self, sample_spectrogram):
        """Test that non-None stream raises BufferError."""
        with pytest.raises(BufferError, match="stream must be None"):
            sample_spectrogram.__dlpack__(stream=0)

    def test_invalid_version(self, sample_spectrogram):
        """Test that unsupported version raises BufferError."""
        with pytest.raises(BufferError, match="Unsupported DLPack version"):
            sample_spectrogram.__dlpack__(max_version=(0, 9))

    def test_valid_version(self, sample_spectrogram):
        """Test that valid versions are accepted."""
        # Version 1.0 should work
        capsule = sample_spectrogram.__dlpack__(max_version=(1, 0))
        assert capsule is not None

        # Version 1.1 should work
        capsule = sample_spectrogram.__dlpack__(max_version=(1, 1))
        assert capsule is not None

        # Higher versions should work (backward compatible)
        capsule = sample_spectrogram.__dlpack__(max_version=(2, 0))
        assert capsule is not None

    def test_invalid_device_type(self, sample_spectrogram):
        """Test that non-CPU device types raise BufferError."""
        with pytest.raises(BufferError, match="Only CPU device"):
            sample_spectrogram.__dlpack__(dl_device=(2, 0))  # CUDA

    def test_invalid_device_id(self, sample_spectrogram):
        """Test that non-zero device ID raises BufferError."""
        with pytest.raises(BufferError, match="Only CPU device"):
            sample_spectrogram.__dlpack__(dl_device=(1, 1))  # CPU device 1

    def test_valid_device(self, sample_spectrogram):
        """Test that CPU device (1, 0) is accepted."""
        capsule = sample_spectrogram.__dlpack__(dl_device=(1, 0))
        assert capsule is not None


class TestDLPackCopy:
    """Test copy parameter behavior."""

    def test_copy_false(self, sample_spectrogram):
        """Test that copy=False returns a capsule."""
        capsule = sample_spectrogram.__dlpack__(copy=False)
        assert capsule is not None

    def test_copy_true(self, sample_spectrogram):
        """Test that copy=True returns a capsule."""
        capsule = sample_spectrogram.__dlpack__(copy=True)
        assert capsule is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchInterop:
    """Test interoperability with PyTorch."""

    def test_spectrogram_to_torch(self, sample_spectrogram):
        """Test  conversion to PyTorch tensor."""
        tensor = torch.from_dlpack(sample_spectrogram)  # type: ignore

        # Verify shape matches
        assert tensor.shape == sample_spectrogram.shape

        # Verify dtype is float64
        assert tensor.dtype == torch.float64  # type: ignore

        # Verify device is CPU
        assert tensor.device.type == "cpu"

    def test_chromagram_to_torch(self, sample_chromagram):
        """Test conversion of chromagram to PyTorch tensor."""
        tensor = torch.from_dlpack(sample_chromagram)  # type: ignore

        # Verify shape matches (12 pitch classes x n_frames)
        assert tensor.shape[0] == 12
        assert (
            tensor.shape[1] == sample_chromagram.shape[1]
        )  # Chromagrams are numpy arrays

        # Verify dtype is float64
        assert tensor.dtype == torch.float64  # type: ignore

        # Verify device is CPU
        assert tensor.device.type == "cpu"

    def test_zero_copy_verification(self, sample_spectrogram):
        """Test that DLPack provides true  access."""
        # Get original value
        original_value = sample_spectrogram.data[0, 0]

        # Convert to torch
        tensor = torch.from_dlpack(sample_spectrogram)  # type: ignore

        # Modify the tensor
        tensor[0, 0] = 999.0

        # With Python-allocated data from the start, this is TRUE !
        # The spectrogram data should reflect the change
        assert sample_spectrogram.data[0, 0] == 999.0
        assert sample_spectrogram.data[0, 0] != original_value
        assert tensor[0, 0].item() == 999.0

    def test_data_values_match(self, sample_spectrogram):
        """Test that data values match exactly between NumPy and PyTorch."""
        tensor = torch.from_dlpack(sample_spectrogram)  # type: ignore
        np_data = sample_spectrogram.data

        # Convert tensor to numpy for comparison
        tensor_np = tensor.numpy()

        # Should be exactly equal (same memory)
        assert np.array_equal(np_data, tensor_np)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJAXInterop:
    """Test interoperability with JAX."""

    def test_spectrogram_to_jax(self, sample_spectrogram):
        """Test conversion to JAX array."""
        jax_array = jax.dlpack.from_dlpack(sample_spectrogram)  # type: ignore

        # Verify shape matches
        assert jax_array.shape == sample_spectrogram.shape

        # JAX may convert to float32 by default, accept either float32 or float64
        assert jax_array.dtype in (np.float32, np.float64)

    def test_chromagram_to_jax(self, sample_chromagram):
        """Test conversion of chromagram to JAX array."""
        jax_array = jax.dlpack.from_dlpack(sample_chromagram)  # type: ignore

        # Verify shape matches
        assert jax_array.shape[0] == 12
        assert (
            jax_array.shape[1] == sample_chromagram.shape[1]
        )  # Chromagrams are numpy arrays

        # JAX may convert to float32 by default, accept either
        assert jax_array.dtype in (np.float32, np.float64)

    def test_data_values_match_jax(self, sample_spectrogram):
        """Test that data values match between NumPy and JAX."""
        jax_array = jax.dlpack.from_dlpack(sample_spectrogram)  # type: ignore
        np_data = sample_spectrogram.data

        # Convert JAX array to numpy for comparison
        jax_np = np.array(jax_array)

        # Should be very close (JAX may make a copy)
        assert np.allclose(np_data, jax_np)


class TestDifferentSpectrogramTypes:
    """Test DLPack with different spectrogram types."""

    def test_mel_spectrogram(self):
        """Test DLPack with Mel spectrogram."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
        spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

        # Should have __dlpack__ methods
        assert hasattr(spec, "__dlpack__")
        assert hasattr(spec, "__dlpack_device__")

        # Get device
        device_type, device_id = spec.__dlpack_device__()
        assert device_type == 1
        assert device_id == 0

        # Create capsule
        capsule = spec.__dlpack__()
        assert capsule is not None

    def test_cqt_spectrogram(self):
        """Test DLPack with CQT spectrogram."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        cqt_params = sg.CqtParams(bins_per_octave=12, n_octaves=7, f_min=32.7)
        spec = sg.compute_cqt_power_spectrogram(samples, params, cqt_params)

        capsule = spec.__dlpack__()
        assert capsule is not None

    def test_decibel_spectrogram(self):
        """Test DLPack with decibel-scaled spectrogram."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_db_spectrogram(samples, params)

        capsule = spec.__dlpack__()
        assert capsule is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_different_types_to_torch(self):
        """Test that different spectrogram types all work with PyTorch."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)

        # Test different types
        mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
        cqt_params = sg.CqtParams(bins_per_octave=12, n_octaves=7, f_min=32.7)
        specs = [
            sg.compute_linear_power_spectrogram(samples, params),
            sg.compute_mel_power_spectrogram(samples, params, mel_params),
            sg.compute_cqt_power_spectrogram(samples, params, cqt_params),
            sg.compute_linear_db_spectrogram(samples, params),
        ]

        for spec in specs:
            tensor = torch.from_dlpack(spec)  # type: ignore
            assert tensor.shape == spec.shape
            assert tensor.dtype == torch.float64  # type: ignore


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_spectrogram(self):
        """Test with a very small spectrogram."""
        samples = np.random.randn(1024).astype(np.float64)
        stft = sg.StftParams(n_fft=128, hop_size=64, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_power_spectrogram(samples, params)

        capsule = spec.__dlpack__()
        assert capsule is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_small_spectrogram_to_torch(self):
        """Test converting a small spectrogram to PyTorch."""
        samples = np.random.randn(1024).astype(np.float64)
        stft = sg.StftParams(n_fft=128, hop_size=64, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_power_spectrogram(samples, params)

        tensor = torch.from_dlpack(spec)  # type: ignore
        assert tensor.shape == spec.shape

    def test_multiple_conversions(self, sample_spectrogram):
        """Test that we can create multiple capsules from the same spectrogram."""
        capsule1 = sample_spectrogram.__dlpack__()
        capsule2 = sample_spectrogram.__dlpack__()

        assert capsule1 is not None
        assert capsule2 is not None
        # They should be different capsule objects
        assert capsule1 is not capsule2
