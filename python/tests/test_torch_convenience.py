"""Tests for PyTorch convenience layer."""

import numpy as np
import pytest
import spectrograms as sg

# Try to import PyTorch
try:
    import torch
    import spectrograms.torch  # Adds convenience methods

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


@pytest.fixture
def sample_spectrogram():
    """Create a sample spectrogram for testing."""
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    spec = sg.compute_linear_power_spectrogram(samples, params)
    return spec


@pytest.fixture
def sample_chromagram():
    """Create a sample chromagram for testing."""
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    chroma_params = sg.ChromaParams()
    chroma = sg.compute_chromagram(samples, stft, 16000.0, chroma_params)
    return chroma


class TestBasicConversion:
    """Test basic conversion to PyTorch tensors."""

    def test_to_torch_simple(self, sample_spectrogram):
        """Test simple conversion without metadata."""
        tensor = sample_spectrogram.to_torch()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == sample_spectrogram.shape
        assert tensor.dtype == torch.float64
        assert tensor.device.type == "cpu"

    def test_to_torch_chromagram(self, sample_chromagram):
        """Test chromagram conversion using DLPack directly."""
        # Chromagrams are returned as numpy arrays with DLPack support
        tensor = torch.from_dlpack(sample_chromagram)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 12  # 12 pitch classes
        assert tensor.dtype == torch.float64

    def test_data_copy(self, sample_spectrogram):
        """Test that to_torch() provides  access via DLPack."""
        tensor = sample_spectrogram.to_torch()

        # Modify tensor
        original_spec = sample_spectrogram.data[0, 0]
        original_tensor = tensor[0, 0].item()
        assert original_spec == original_tensor  # Data matches initially

        tensor[0, 0] = 999.0

        # With Python-allocated data from the start, to_torch() is !
        # The spectrogram data should reflect the change
        assert sample_spectrogram.data[0, 0] == 999.0
        assert sample_spectrogram.data[0, 0] != original_spec


class TestMetadataPreservation:
    """Test metadata preservation functionality."""

    def test_with_metadata(self, sample_spectrogram):
        """Test conversion with metadata preservation."""
        result = sample_spectrogram.to_torch(with_metadata=True)

        assert isinstance(result, sg.torch.TorchSpectrogram)
        assert isinstance(result.tensor, torch.Tensor)
        assert result.tensor.shape == sample_spectrogram.shape

        # Check metadata
        assert result.frequencies is not None
        assert len(result.frequencies) == sample_spectrogram.n_bins
        assert result.times is not None
        assert len(result.times) == sample_spectrogram.n_frames
        assert result.params is not None
        assert result.shape == sample_spectrogram.shape

    def test_chromagram_batching(self, sample_chromagram):
        """Test that chromagrams can be batched using DLPack."""
        # Chromagrams don't have to_torch() but can be batched
        chromagrams = [sample_chromagram, sample_chromagram]
        batch = sg.torch.batch(chromagrams)

        assert batch.shape[0] == 2
        assert batch.shape[1] == 12  # pitch classes

    def test_metadata_device_transfer(self, sample_spectrogram):
        """Test that metadata is preserved when moving devices."""
        result = sample_spectrogram.to_torch(with_metadata=True)

        # Move to CPU (already there, but test the method)
        cpu_result = result.cpu()
        assert cpu_result.frequencies is not None
        assert np.array_equal(cpu_result.frequencies, result.frequencies)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDeviceHandling:
    """Test device handling (GPU support)."""

    def test_to_cuda(self, sample_spectrogram):
        """Test conversion to CUDA device."""
        tensor = sample_spectrogram.to_torch(device="cuda")

        assert tensor.device.type == "cuda"
        assert tensor.shape == sample_spectrogram.shape

    def test_cuda_with_metadata(self, sample_spectrogram):
        """Test metadata preservation with CUDA transfer."""
        result = sample_spectrogram.to_torch(device="cuda", with_metadata=True)

        assert result.tensor.device.type == "cuda"
        assert result.frequencies is not None
        assert result.times is not None

    def test_device_transfer(self, sample_spectrogram):
        """Test moving between devices."""
        result = sample_spectrogram.to_torch(device="cuda", with_metadata=True)

        # Move to CPU
        cpu_result = result.cpu()
        assert cpu_result.tensor.device.type == "cpu"
        assert torch.allclose(cpu_result.tensor, result.tensor.cpu())

        # Move back to CUDA
        cuda_result = cpu_result.cuda()
        assert cuda_result.tensor.device.type == "cuda"


class TestDtypeConversion:
    """Test dtype conversion."""

    def test_float32_conversion(self, sample_spectrogram):
        """Test conversion to float32."""
        tensor = sample_spectrogram.to_torch(dtype=torch.float32)

        assert tensor.dtype == torch.float32
        assert tensor.shape == sample_spectrogram.shape

    def test_float16_conversion(self, sample_spectrogram):
        """Test conversion to float16 (half precision)."""
        tensor = sample_spectrogram.to_torch(dtype=torch.float16)

        assert tensor.dtype == torch.float16


class TestBatching:
    """Test batching utilities."""

    def test_batch_same_shape(self):
        """Test batching spectrograms with same shape."""
        # Create multiple spectrograms with same shape
        specs = []
        for _ in range(4):
            samples = np.random.randn(16000).astype(np.float64)
            stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
            params = sg.SpectrogramParams(stft, sample_rate=16000.0)
            spec = sg.compute_linear_power_spectrogram(samples, params)
            specs.append(spec)

        batch = sg.torch.batch(specs)

        assert batch.shape[0] == 4  # batch size
        assert batch.shape[1:] == specs[0].shape
        assert batch.dtype == torch.float64

    def test_batch_different_shapes_error(self):
        """Test that batching different shapes raises error without padding."""
        # Create spectrograms with different n_frames
        specs = []
        for length in [8000, 16000]:
            samples = np.random.randn(length).astype(np.float64)
            stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
            params = sg.SpectrogramParams(stft, sample_rate=16000.0)
            spec = sg.compute_linear_power_spectrogram(samples, params)
            specs.append(spec)

        with pytest.raises(ValueError, match="same shape"):
            sg.torch.batch(specs, pad=False)

    def test_batch_with_padding(self):
        """Test batching with automatic padding."""
        # Create spectrograms with different n_frames
        specs = []
        for length in [8000, 16000]:
            samples = np.random.randn(length).astype(np.float64)
            stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
            params = sg.SpectrogramParams(stft, sample_rate=16000.0)
            spec = sg.compute_linear_power_spectrogram(samples, params)
            specs.append(spec)

        batch = sg.torch.batch(specs, pad=True)

        assert batch.shape[0] == 2  # batch size
        # Should be padded to max n_frames
        assert batch.shape[2] == max(s.n_frames for s in specs)

    def test_batch_with_metadata(self):
        """Test batching with metadata preservation."""
        specs = []
        for _ in range(3):
            samples = np.random.randn(16000).astype(np.float64)
            stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
            params = sg.SpectrogramParams(stft, sample_rate=16000.0)
            spec = sg.compute_linear_power_spectrogram(samples, params)
            specs.append(spec)

        batch, metadata = sg.torch.batch_with_metadata(specs)

        assert batch.shape[0] == 3
        assert len(metadata) == 3
        assert all("frequencies" in m for m in metadata)
        assert all("times" in m for m in metadata)
        assert all("params" in m for m in metadata)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_cuda(self):
        """Test batching directly to CUDA."""
        specs = []
        for _ in range(3):
            samples = np.random.randn(16000).astype(np.float64)
            stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
            params = sg.SpectrogramParams(stft, sample_rate=16000.0)
            spec = sg.compute_linear_power_spectrogram(samples, params)
            specs.append(spec)

        batch = sg.torch.batch(specs, device="cuda")

        assert batch.device.type == "cuda"
        assert batch.shape[0] == 3


class TestDifferentSpectrogramTypes:
    """Test convenience methods with different spectrogram types."""

    def test_mel_spectrogram(self):
        """Test Mel spectrogram conversion."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
        spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

        tensor = spec.to_torch()
        assert tensor.shape[0] == 128  # n_mels

        result = spec.to_torch(with_metadata=True)
        assert len(result.frequencies) == 128

    def test_cqt_spectrogram(self):
        """Test CQT spectrogram conversion."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        cqt_params = sg.CqtParams(bins_per_octave=12, n_octaves=7, f_min=32.7)
        spec = sg.compute_cqt_power_spectrogram(samples, params, cqt_params)

        tensor = spec.to_torch()
        assert tensor.shape[0] == 84  # 12 * 7

    def test_decibel_spectrogram(self):
        """Test decibel spectrogram conversion."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_db_spectrogram(samples, params)

        result = spec.to_torch(with_metadata=True)
        assert result.db_range is not None  # Should have db_range


class TestEdgeCases:
    """Test edge cases."""

    def test_small_spectrogram(self):
        """Test with very small spectrogram."""
        samples = np.random.randn(1024).astype(np.float64)
        stft = sg.StftParams(n_fft=128, hop_size=64, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_power_spectrogram(samples, params)

        tensor = spec.to_torch()
        assert tensor.shape == spec.shape

    def test_batch_single_spectrogram(self):
        """Test batching with single spectrogram."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        spec = sg.compute_linear_power_spectrogram(samples, params)

        batch = sg.torch.batch([spec])
        assert batch.shape[0] == 1
        assert batch.shape[1:] == spec.shape

    def test_batch_empty_error(self):
        """Test that batching empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            sg.torch.batch([])


class TestCommonPatterns:
    """Test common usage patterns."""

    def test_neural_network_input_pipeline(self):
        """Test typical neural network input preparation."""
        # Simulate batch of audio samples
        signals = [np.random.randn(16000).astype(np.float64) for _ in range(4)]

        # Compute spectrograms
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)

        specs = [
            sg.compute_mel_power_spectrogram(s, params, mel_params) for s in signals
        ]

        # Batch and prepare for neural network
        batch = sg.torch.batch(specs, dtype=torch.float32)

        # Apply log scaling (common for audio ML)
        log_batch = torch.log(batch + 1e-10)

        # Normalize
        mean = log_batch.mean()
        std = log_batch.std()
        normalized = (log_batch - mean) / std

        # Add channel dimension for CNN
        cnn_input = normalized.unsqueeze(1)

        assert cnn_input.shape == (4, 1, 128, specs[0].n_frames)
        assert cnn_input.dtype == torch.float32

    def test_metadata_for_visualization(self):
        """Test using metadata for visualization/analysis."""
        samples = np.random.randn(16000).astype(np.float64)
        stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
        params = sg.SpectrogramParams(stft, sample_rate=16000.0)
        mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
        spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

        result = spec.to_torch(with_metadata=True)

        # Can use metadata for plotting axes, analysis, etc.
        # Mel frequencies start near f_min but not exactly at it due to mel scale transformation
        assert result.frequencies[0] >= 0.0
        assert result.frequencies[0] < 50.0  # Should be close to f_min=0.0
        assert result.frequencies[-1] == pytest.approx(
            8000.0, abs=200.0
        )  # Mel scale transformation
        assert result.times[0] >= 0
        assert result.times[-1] > 0
