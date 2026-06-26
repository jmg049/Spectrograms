"""Tests for the native f32/f64 ``dtype`` parameter on the planner plan classes.

The standalone compute functions already accept ``dtype="float32"|"float64"``.
These tests cover the same parameter on ``SpectrogramPlanner``'s plan-builder
methods: the requested precision is baked into the plan at creation time and is
reflected by ``plan.dtype``, ``plan.compute(...).data.dtype``, and the dtype of
``plan.compute_frame(...)``.
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
def params():
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    return sg.SpectrogramParams(stft, sample_rate=16000.0)


@pytest.fixture
def planner():
    return sg.SpectrogramPlanner()


def _mel_plan(planner, params, dtype):
    mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
    return planner.mel_power_plan(params, mel, dtype=dtype)


def _cqt_plan(planner, params, dtype):
    cqt = sg.CqtParams(bins_per_octave=12, n_octaves=6, f_min=32.7)
    return planner.cqt_power_plan(params, cqt, dtype=dtype)


BUILDERS = {"mel": _mel_plan, "cqt": _cqt_plan}

NP_DTYPE = {"float32": np.float32, "float64": np.float64}


class TestPlannerDtypeDefault:
    def test_explicit_float64(self, planner, params):
        plan = _mel_plan(planner, params, "float64")
        assert plan.dtype == "float64"

    def test_omitted_dtype_is_float64(self, planner, params, signal):
        # No dtype argument -> backward-compatible float64 default.
        mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
        plan = planner.mel_power_plan(params, mel)
        assert plan.dtype == "float64"

        spec = plan.compute(signal)
        assert spec.dtype == "float64"
        assert spec.data.dtype == np.float64

    def test_omitted_dtype_is_float64_cqt(self, planner, params, signal):
        cqt = sg.CqtParams(bins_per_octave=12, n_octaves=6, f_min=32.7)
        plan = planner.cqt_power_plan(params, cqt)
        assert plan.dtype == "float64"
        assert plan.compute(signal).data.dtype == np.float64

    def test_invalid_dtype_raises(self, planner, params):
        mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
        with pytest.raises(ValueError):
            planner.mel_power_plan(params, mel, dtype="float16")


@pytest.mark.parametrize("name", list(BUILDERS))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestPlannerDtypeRoundtrip:
    def test_plan_dtype_property(self, planner, params, name, dtype):
        plan = BUILDERS[name](planner, params, dtype)
        assert plan.dtype == dtype

    def test_compute_data_dtype(self, planner, params, signal, name, dtype):
        plan = BUILDERS[name](planner, params, dtype)
        spec = plan.compute(signal)
        assert spec.dtype == dtype
        assert spec.data.dtype == NP_DTYPE[dtype]

    def test_compute_frame_dtype(self, planner, params, signal, name, dtype):
        plan = BUILDERS[name](planner, params, dtype)
        frame = plan.compute_frame(signal, 0)
        assert frame.dtype == NP_DTYPE[dtype]
        assert frame.ndim == 1


@pytest.mark.parametrize("name", list(BUILDERS))
class TestPlannerPrecisionAgreement:
    def test_f32_close_to_f64(self, planner, params, signal, name):
        f64 = BUILDERS[name](planner, params, "float64").compute(signal).data
        f32 = BUILDERS[name](planner, params, "float32").compute(signal).data
        assert f64.shape == f32.shape
        # f32 results should be close to the f64 reference within single
        # precision tolerance, scaled to the magnitude of the data.
        scale = float(np.max(np.abs(f64))) or 1.0
        np.testing.assert_allclose(
            f32.astype(np.float64), f64, rtol=1e-4, atol=1e-4 * scale
        )

    def test_compute_frame_close(self, planner, params, signal, name):
        f64 = BUILDERS[name](planner, params, "float64").compute_frame(signal, 0)
        f32 = BUILDERS[name](planner, params, "float32").compute_frame(signal, 0)
        scale = float(np.max(np.abs(f64))) or 1.0
        np.testing.assert_allclose(
            f32.astype(np.float64), f64, rtol=1e-4, atol=1e-4 * scale
        )


def test_default_dtype_matches_standalone(planner, params, signal):
    """A default planner plan should match the standalone compute function."""
    mel = sg.MelParams(n_mels=64, f_min=0.0, f_max=8000.0)
    plan_spec = planner.mel_power_plan(params, mel).compute(signal)
    fn_spec = sg.compute_mel_power_spectrogram(signal, params, mel)
    np.testing.assert_allclose(plan_spec.data, fn_spec.data)
