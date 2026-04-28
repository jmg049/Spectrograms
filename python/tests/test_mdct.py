"""Tests for MDCT/IMDCT implementation."""
import numpy as np
import pytest
import spectrograms as sg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sine_wave(n=4096, freq=440.0, sr=44100.0):
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# MdctParams construction
# ---------------------------------------------------------------------------

class TestMdctParams:
    def test_default_construction(self):
        params = sg.MdctParams(1024, 512, sg.WindowType.hanning)
        assert params.window_size == 1024
        assert params.hop_size == 512
        assert params.n_coefficients == 512

    def test_sine_window(self):
        params = sg.MdctParams.sine_window(1024)
        assert params.window_size == 1024
        assert params.hop_size == 512  # 50% hop
        assert params.n_coefficients == 512

    def test_odd_window_size_raises(self):
        with pytest.raises((ValueError, Exception)):
            sg.MdctParams(1025, 512, sg.WindowType.hanning)

    def test_odd_window_size_sine_raises(self):
        with pytest.raises((ValueError, Exception)):
            sg.MdctParams.sine_window(1025)

    def test_too_small_window_raises(self):
        with pytest.raises((ValueError, Exception)):
            sg.MdctParams(2, 1, sg.WindowType.hanning)

    def test_zero_hop_raises(self):
        with pytest.raises((ValueError, Exception)):
            sg.MdctParams(1024, 0, sg.WindowType.hanning)

    def test_zero_window_raises(self):
        with pytest.raises((ValueError, Exception)):
            sg.MdctParams(0, 512, sg.WindowType.hanning)

    def test_repr(self):
        params = sg.MdctParams.sine_window(1024)
        r = repr(params)
        assert "MdctParams" in r
        assert "1024" in r

    def test_various_window_types(self):
        for wt in [sg.WindowType.hanning, sg.WindowType.hamming, sg.WindowType.blackman]:
            params = sg.MdctParams(512, 256, wt)
            assert params.window_size == 512


# ---------------------------------------------------------------------------
# compute_mdct shape
# ---------------------------------------------------------------------------

class TestMdctShape:
    def test_basic_shape(self):
        params = sg.MdctParams.sine_window(1024)
        samples = sine_wave(n=8192)
        coefs = sg.mdct(samples, params)
        assert coefs.ndim == 2
        assert coefs.shape[0] == 512  # N = window_size // 2

    def test_expected_n_frames(self):
        window_size = 1024
        hop_size = 512
        n_samples = 8192
        params = sg.MdctParams.sine_window(window_size)
        samples = np.random.randn(n_samples)
        coefs = sg.mdct(samples, params)
        expected_frames = (n_samples - window_size) // hop_size + 1
        assert coefs.shape[1] == expected_frames

    def test_exact_one_frame(self):
        params = sg.MdctParams.sine_window(1024)
        samples = np.random.randn(1024)
        coefs = sg.mdct(samples, params)
        assert coefs.shape == (512, 1)

    def test_dtype_float64(self):
        params = sg.MdctParams.sine_window(512)
        samples = np.random.randn(4096)
        coefs = sg.mdct(samples, params)
        assert coefs.dtype == np.float64

    def test_signal_too_short_raises(self):
        params = sg.MdctParams.sine_window(1024)
        samples = np.random.randn(512)  # shorter than window_size
        with pytest.raises((ValueError, Exception)):
            sg.mdct(samples, params)

    def test_empty_signal_raises(self):
        params = sg.MdctParams.sine_window(1024)
        with pytest.raises((ValueError, Exception)):
            sg.mdct(np.array([]), params)

    def test_float32_input_accepted(self):
        params = sg.MdctParams.sine_window(512)
        samples = np.random.randn(4096).astype(np.float32)
        coefs = sg.mdct(samples, params)
        assert coefs.shape[0] == 256

    def test_different_hop_sizes(self):
        n_samples = 8192
        window_size = 1024
        for hop in [256, 512, 1024]:
            params = sg.MdctParams(window_size, hop, sg.WindowType.hanning)
            samples = np.random.randn(n_samples)
            coefs = sg.mdct(samples, params)
            expected = (n_samples - window_size) // hop + 1
            assert coefs.shape == (512, expected)


# ---------------------------------------------------------------------------
# Perfect reconstruction test
# ---------------------------------------------------------------------------

class TestPerfectReconstruction:
    def test_pr_sine_signal(self):
        """Sine window + 50% hop = perfect reconstruction (up to boundary effects)."""
        params = sg.MdctParams.sine_window(1024)
        n = 8192
        x = sine_wave(n=n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params, original_length=n)
        # Perfect reconstruction holds in the interior (away from boundaries)
        margin = 1024
        np.testing.assert_allclose(x_rec[margin:-margin], x[margin:-margin], atol=1e-10)

    def test_pr_random_signal(self):
        """Random signal, sine window, 50% hop."""
        params = sg.MdctParams.sine_window(512)
        np.random.seed(42)
        n = 4096
        x = np.random.randn(n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params, original_length=n)
        margin = 512
        np.testing.assert_allclose(x_rec[margin:-margin], x[margin:-margin], atol=1e-10)

    def test_pr_small_window(self):
        params = sg.MdctParams.sine_window(16)
        n = 256
        x = np.random.randn(n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params, original_length=n)
        margin = 16
        np.testing.assert_allclose(x_rec[margin:-margin], x[margin:-margin], atol=1e-9)

    def test_pr_original_length_truncates(self):
        params = sg.MdctParams.sine_window(1024)
        n = 8192
        x = np.random.randn(n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params, original_length=n)
        assert len(x_rec) == n

    def test_pr_without_original_length(self):
        params = sg.MdctParams.sine_window(1024)
        n = 8192
        x = np.random.randn(n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params)
        assert x_rec.ndim == 1
        assert len(x_rec) >= n


# ---------------------------------------------------------------------------
# Single-frame correctness vs scipy DCT-IV
# ---------------------------------------------------------------------------

class TestSingleFrameCorrectness:
    def test_compare_with_direct_formula(self):
        """Compare one MDCT frame against direct formula reference."""
        n_coefs = 8  # N
        window_size = 2 * n_coefs  # 2N = 16
        N = n_coefs

        params = sg.MdctParams(window_size, n_coefs, sg.WindowType.rectangular)

        np.random.seed(7)
        x = np.random.randn(window_size)

        # Direct MDCT formula: C[k] = sum_n x[n]*w[n]*cos(pi*(2n+1+N)*(2k+1)/(4N))
        # rectangular window w[n]=1
        ref = np.array([
            sum(x[n] * np.cos(np.pi * (2*n+1+N) * (2*k+1) / (4*N))
                for n in range(window_size))
            for k in range(N)
        ])

        coefs = sg.mdct(x, params)
        np.testing.assert_allclose(coefs[:, 0], ref, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_imdct
# ---------------------------------------------------------------------------

class TestImdct:
    def test_output_shape(self):
        params = sg.MdctParams.sine_window(1024)
        n = 8192
        x = np.random.randn(n)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params)
        assert x_rec.ndim == 1

    def test_output_dtype(self):
        params = sg.MdctParams.sine_window(512)
        x = np.random.randn(4096)
        coefs = sg.mdct(x, params)
        x_rec = sg.imdct(coefs, params)
        assert x_rec.dtype == np.float64

    def test_wrong_n_rows_raises(self):
        params = sg.MdctParams.sine_window(1024)
        bad_coefs = np.zeros((300, 5))  # 300 != 512
        with pytest.raises((ValueError, Exception)):
            sg.imdct(bad_coefs, params)

    def test_empty_frames(self):
        params = sg.MdctParams.sine_window(1024)
        coefs = np.zeros((512, 0))
        x_rec = sg.imdct(coefs, params)
        assert len(x_rec) == 0
