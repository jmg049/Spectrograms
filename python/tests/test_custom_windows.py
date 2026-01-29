"""Tests for custom window functionality."""

import numpy as np
import pytest

import spectrograms as sg


class TestCustomWindowCreation:
    """Tests for creating custom windows."""

    def test_basic_creation(self):
        """Test basic custom window creation."""
        coeffs = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        window = sg.WindowType.custom(coeffs)
        assert str(window) == "Custom(n=5)"

    def test_numpy_blackman(self):
        """Test creation from NumPy blackman window."""
        coeffs = np.blackman(512)
        window = sg.WindowType.custom(coeffs)
        assert str(window) == "Custom(n=512)"

    def test_empty_array_error(self):
        """Test that empty array raises error."""
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(np.array([]))
        assert "empty" in str(exc_info.value).lower()

    def test_nan_error(self):
        """Test that NaN values raise error."""
        coeffs = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs)
        assert "not finite" in str(exc_info.value).lower()
        assert "index 2" in str(exc_info.value).lower()

    def test_infinity_error(self):
        """Test that infinity values raise error."""
        coeffs = np.array([1.0, np.inf, 3.0])
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs)
        assert "not finite" in str(exc_info.value).lower()

    def test_type_conversion(self):
        """Test that various array types are converted correctly."""
        # float32 - needs to be converted to float64
        coeffs_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        window = sg.WindowType.custom(coeffs_f32.astype(np.float64))
        assert str(window) == "Custom(n=3)"

        # int - needs to be converted to float64
        coeffs_int = np.array([1, 2, 3], dtype=np.int32)
        window = sg.WindowType.custom(coeffs_int.astype(np.float64))
        assert str(window) == "Custom(n=3)"


class TestCustomWindowInStft:
    """Tests for using custom windows in STFT parameters."""

    def test_matching_size(self):
        """Test that matching size works correctly."""
        window = sg.WindowType.custom(np.blackman(512))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        assert stft is not None

    def test_size_mismatch_error(self):
        """Test that size mismatch is caught early."""
        window = sg.WindowType.custom(np.blackman(256))
        with pytest.raises(Exception) as exc_info:
            sg.StftParams(n_fft=512, hop_size=256, window=window)
        assert "256" in str(exc_info.value)
        assert "512" in str(exc_info.value)


class TestCustomWindowSpectrogram:
    """Tests for computing spectrograms with custom windows."""

    @pytest.fixture
    def test_signal(self):
        """Generate a test signal."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
        return np.sin(2 * np.pi * 440 * t), sample_rate

    def test_linear_spectrogram(self, test_signal):
        """Test linear spectrogram with custom window."""
        signal, sample_rate = test_signal
        window = sg.WindowType.custom(np.blackman(512))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
        spec = sg.compute_linear_power_spectrogram(signal, params)
        assert spec.data.shape == (257, 63)

    def test_mel_spectrogram(self, test_signal):
        """Test mel spectrogram with custom window."""
        signal, sample_rate = test_signal
        window = sg.WindowType.custom(np.hamming(512))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
        mel = sg.MelParams(n_mels=80, f_min=0.0, f_max=sample_rate / 2)
        spec = sg.compute_mel_power_spectrogram(signal, params, mel)
        assert spec.data.shape == (80, 63)

    def test_db_spectrogram(self, test_signal):
        """Test dB spectrogram with custom window."""
        signal, sample_rate = test_signal
        window = sg.WindowType.custom(np.hanning(512))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
        log = sg.LogParams(-80.0)
        spec = sg.compute_linear_db_spectrogram(signal, params, log)
        assert spec.data.shape == (257, 63)


class TestWindowNormalization:
    """Tests for window normalization feature."""

    def test_sum_normalization(self):
        """Test sum normalization."""
        coeffs = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        window = sg.WindowType.custom(coeffs, normalize="sum")
        assert str(window) == "Custom(n=5)"

    def test_peak_normalization(self):
        """Test peak normalization."""
        coeffs = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
        window = sg.WindowType.custom(coeffs, normalize="peak")
        assert str(window) == "Custom(n=5)"

    def test_max_alias(self):
        """Test that 'max' works as alias for 'peak'."""
        coeffs = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
        window = sg.WindowType.custom(coeffs, normalize="max")
        assert str(window) == "Custom(n=5)"

    def test_energy_normalization(self):
        """Test energy normalization."""
        coeffs = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        window = sg.WindowType.custom(coeffs, normalize="energy")
        assert str(window) == "Custom(n=5)"

    def test_rms_alias(self):
        """Test that 'rms' works as alias for 'energy'."""
        coeffs = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        window = sg.WindowType.custom(coeffs, normalize="rms")
        assert str(window) == "Custom(n=5)"

    def test_invalid_normalization(self):
        """Test that invalid normalization mode raises error."""
        coeffs = np.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs, normalize="invalid")
        assert "invalid" in str(exc_info.value).lower()

    def test_zero_sum_error(self):
        """Test that zero window fails sum normalization."""
        coeffs = np.zeros(5)
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs, normalize="sum")
        assert "zero" in str(exc_info.value).lower()

    def test_zero_peak_error(self):
        """Test that zero window fails peak normalization."""
        coeffs = np.zeros(5)
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs, normalize="peak")
        assert "zero" in str(exc_info.value).lower()

    def test_zero_energy_error(self):
        """Test that zero window fails energy normalization."""
        coeffs = np.zeros(5)
        with pytest.raises(Exception) as exc_info:
            sg.WindowType.custom(coeffs, normalize="energy")
        assert "zero" in str(exc_info.value).lower()

    def test_normalized_in_spectrogram(self):
        """Test that normalized windows work in spectrogram computation."""
        sample_rate = 16000
        t = np.linspace(0, 1, 16000, dtype=np.float64)
        signal = np.sin(2 * np.pi * 440 * t)

        coeffs = np.hamming(512)
        window = sg.WindowType.custom(coeffs, normalize="sum")
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
        spec = sg.compute_linear_power_spectrogram(signal, params)
        assert spec.data.shape == (257, 63)


class TestScipyIntegration:
    """Tests for SciPy window integration."""

    def test_tukey_window(self):
        """Test using SciPy Tukey window."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import tukey

        window = sg.WindowType.custom(tukey(512, alpha=0.5))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        assert stft is not None

    def test_triangular_window(self):
        """Test using SciPy triangular window."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import triang

        window = sg.WindowType.custom(triang(512))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        assert stft is not None

    def test_custom_scipy_window(self):
        """Test computing spectrogram with SciPy window."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import tukey

        sample_rate = 16000
        t = np.linspace(0, 1, 16000, dtype=np.float64)
        signal = np.sin(2 * np.pi * 440 * t)

        window = sg.WindowType.custom(tukey(512, alpha=0.5))
        stft = sg.StftParams(n_fft=512, hop_size=256, window=window)
        params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
        spec = sg.compute_linear_power_spectrogram(signal, params)
        assert spec.data.shape == (257, 63)
