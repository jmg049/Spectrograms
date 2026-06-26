//! Planner and plan classes for efficient batch processing.

use pyo3::prelude::*;

use crate::{
    AmpScaleSpec, Cqt, Decibels, Gammatone, LinearHz, LogHz, Magnitude, Mel, Power,
    SpectrogramPlan, SpectrogramPlanner,
};

use super::dtype::{array1_to_py, parse_dtype, real_1d_vec, Dtype};
use super::params::{
    PyCqtParams, PyErbParams, PyLogHzParams, PyLogParams, PyMelParams, PySpectrogramParams,
};
use super::spectrogram::PySpectrogram;
use ndarray::Array1;
use non_empty_slice::NonEmptySlice;
use std::num::NonZeroUsize;

/// A precision-erased spectrogram plan.
///
/// A [`SpectrogramPlan`] owns precision-specific state (`StftPlan<T>` and a
/// `Workspace<T>`), so the requested `dtype` is baked into the plan at creation
/// time. This enum lets the Python plan classes hold either an `f32`- or
/// `f64`-precision plan behind a single type while remaining generic over the
/// frequency (`F`) and amplitude (`A`) scales.
pub(crate) enum DualPlan<F, A>
where
    A: AmpScaleSpec + 'static,
    F: Copy + Clone + 'static,
{
    F32(SpectrogramPlan<F, A, f32>),
    F64(SpectrogramPlan<F, A, f64>),
}

fn empty_samples_err() -> PyErr {
    pyo3::exceptions::PyValueError::new_err("Input samples cannot be empty")
}

impl<F, A> DualPlan<F, A>
where
    A: AmpScaleSpec + 'static,
    F: Copy + Clone + 'static,
{
    /// The NumPy dtype name this plan computes in (`"float32"` / `"float64"`).
    fn dtype(&self) -> &'static str {
        match self {
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
        }
    }

    /// Compute a full spectrogram, coercing `samples` to the plan's precision.
    fn compute(&mut self, py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<PySpectrogram> {
        match self {
            Self::F32(plan) => {
                let samples = real_1d_vec::<f32>(py, samples)?;
                let samples = NonEmptySlice::new(&samples).ok_or_else(empty_samples_err)?;
                let spec = plan.compute(samples)?;
                Ok(PySpectrogram::from_spectrogram(py, spec))
            }
            Self::F64(plan) => {
                let samples = real_1d_vec::<f64>(py, samples)?;
                let samples = NonEmptySlice::new(&samples).ok_or_else(empty_samples_err)?;
                let spec = plan.compute(samples)?;
                Ok(PySpectrogram::from_spectrogram(py, spec))
            }
        }
    }

    /// Compute a single frame, returning a 1D NumPy array in the plan's precision.
    fn compute_frame(
        &mut self,
        py: Python<'_>,
        samples: &Bound<'_, PyAny>,
        frame_idx: usize,
    ) -> PyResult<Py<PyAny>> {
        match self {
            Self::F32(plan) => {
                let samples = real_1d_vec::<f32>(py, samples)?;
                let samples = NonEmptySlice::new(&samples).ok_or_else(empty_samples_err)?;
                let frame = plan.compute_frame(samples, frame_idx)?;
                Ok(array1_to_py(py, Array1::from(frame.to_vec())))
            }
            Self::F64(plan) => {
                let samples = real_1d_vec::<f64>(py, samples)?;
                let samples = NonEmptySlice::new(&samples).ok_or_else(empty_samples_err)?;
                let frame = plan.compute_frame(samples, frame_idx)?;
                Ok(array1_to_py(py, Array1::from(frame.to_vec())))
            }
        }
    }

    /// The output `(n_bins, n_frames)` shape for a given signal length.
    fn output_shape(&self, signal_length: NonZeroUsize) -> PyResult<(NonZeroUsize, NonZeroUsize)> {
        match self {
            Self::F32(plan) => Ok(plan.output_shape(signal_length)?),
            Self::F64(plan) => Ok(plan.output_shape(signal_length)?),
        }
    }
}

/// Spectrogram planner for creating reusable computation plans.
///
/// Creating a plan is more expensive than a single computation, but plans can be
/// reused for multiple signals with the same parameters, providing significant
/// performance benefits for batch processing.
#[pyclass(name = "SpectrogramPlanner", skip_from_py_object)]
pub struct PySpectrogramPlanner {
    inner: SpectrogramPlanner,
}

#[pymethods]
impl PySpectrogramPlanner {
    /// Create a new spectrogram planner.
    #[new]
    const fn new() -> Self {
        Self {
            inner: SpectrogramPlanner::new(),
        }
    }

    /// Create a plan for computing linear power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LinearPowerPlan`
    ///     Plan for computing linear power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, dtype: str = \"float64\") -> LinearPowerPlan")]
    fn linear_power_plan(
        &self,
        params: &PySpectrogramParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLinearPowerPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.linear_plan::<Power, f32>(&params.inner, None)?),
            Dtype::F64 => DualPlan::F64(self.inner.linear_plan::<Power, f64>(&params.inner, None)?),
        };
        Ok(PyLinearPowerPlan { inner })
    }

    /// Create a plan for computing linear magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LinearMagnitudePlan`
    ///     Plan for computing linear magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, dtype: str = \"float64\") -> LinearMagnitudePlan")]
    fn linear_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLinearMagnitudePlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => {
                DualPlan::F32(self.inner.linear_plan::<Magnitude, f32>(&params.inner, None)?)
            }
            Dtype::F64 => {
                DualPlan::F64(self.inner.linear_plan::<Magnitude, f64>(&params.inner, None)?)
            }
        };
        Ok(PyLinearMagnitudePlan { inner })
    }

    /// Create a plan for computing linear decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LinearDbPlan`
    ///     Plan for computing linear decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", db_params: "LogParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, db_params: LogParams, dtype: str = \"float64\") -> LinearDbPlan")]
    fn linear_db_plan(
        &self,
        params: &PySpectrogramParams,
        db_params: PyLogParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLinearDbPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.linear_plan::<Decibels, f32>(
                &params.inner,
                Some(&db_params.inner),
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.linear_plan::<Decibels, f64>(
                &params.inner,
                Some(&db_params.inner),
            )?),
        };
        Ok(PyLinearDbPlan { inner })
    }

    /// Create a plan for computing mel power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `mel_params` : `MelParams`
    ///     Mel-scale filterbank parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `MelPowerPlan`
    ///     Plan for computing mel power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, mel_params: MelParams, dtype: str = \"float64\") -> MelPowerPlan")]
    fn mel_power_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
        dtype: Option<&str>,
    ) -> PyResult<PyMelPowerPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.mel_plan::<Power, f32>(
                &params.inner,
                &mel_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.mel_plan::<Power, f64>(
                &params.inner,
                &mel_params.inner,
                None,
            )?),
        };
        Ok(PyMelPowerPlan { inner })
    }

    /// Create a plan for computing mel magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `mel_params` : `MelParams`
    ///     Mel-scale filterbank parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `MelMagnitudePlan`
    ///     Plan for computing mel magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, mel_params: MelParams, dtype: str = \"float64\") -> MelMagnitudePlan")]
    fn mel_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
        dtype: Option<&str>,
    ) -> PyResult<PyMelMagnitudePlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.mel_plan::<Magnitude, f32>(
                &params.inner,
                &mel_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.mel_plan::<Magnitude, f64>(
                &params.inner,
                &mel_params.inner,
                None,
            )?),
        };
        Ok(PyMelMagnitudePlan { inner })
    }

    /// Create a plan for computing mel decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `mel_params` : `MelParams`
    ///     Mel-scale filterbank parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `MelDbPlan`
    ///     Plan for computing mel decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams", db_params: "LogParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, mel_params: MelParams, db_params: LogParams, dtype: str = \"float64\") -> MelDbPlan")]
    fn mel_db_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
        db_params: PyLogParams,
        dtype: Option<&str>,
    ) -> PyResult<PyMelDbPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.mel_plan::<Decibels, f32>(
                &params.inner,
                &mel_params.inner,
                Some(&db_params.inner),
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.mel_plan::<Decibels, f64>(
                &params.inner,
                &mel_params.inner,
                Some(&db_params.inner),
            )?),
        };
        Ok(PyMelDbPlan { inner })
    }

    /// Create a plan for computing ERB power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `erb_params` : `ErbParams`
    ///     ERB-scale filterbank parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `ErbPowerPlan`
    ///     Plan for computing ERB power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, erb_params: ErbParams, dtype: str = \"float64\") -> ErbPowerPlan")]
    fn erb_power_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
        dtype: Option<&str>,
    ) -> PyResult<PyErbPowerPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.erb_plan::<Power, f32>(
                &params.inner,
                &erb_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.erb_plan::<Power, f64>(
                &params.inner,
                &erb_params.inner,
                None,
            )?),
        };
        Ok(PyErbPowerPlan { inner })
    }

    /// Create a plan for computing ERB magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `erb_params` : `ErbParams`
    ///     ERB-scale filterbank parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `ErbMagnitudePlan`
    ///     Plan for computing ERB magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, erb_params: ErbParams, dtype: str = \"float64\") -> ErbMagnitudePlan")]
    fn erb_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
        dtype: Option<&str>,
    ) -> PyResult<PyErbMagnitudePlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.erb_plan::<Magnitude, f32>(
                &params.inner,
                &erb_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.erb_plan::<Magnitude, f64>(
                &params.inner,
                &erb_params.inner,
                None,
            )?),
        };
        Ok(PyErbMagnitudePlan { inner })
    }

    /// Create a plan for computing ERB decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `erb_params` : `ErbParams`
    ///     ERB-scale filterbank parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `ErbDbPlan`
    ///     Plan for computing ERB decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams", db_params: "LogParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, erb_params: ErbParams, db_params: LogParams, dtype: str = \"float64\") -> ErbDbPlan")]
    fn erb_db_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
        db_params: PyLogParams,
        dtype: Option<&str>,
    ) -> PyResult<PyErbDbPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.erb_plan::<Decibels, f32>(
                &params.inner,
                &erb_params.inner,
                Some(&db_params.inner),
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.erb_plan::<Decibels, f64>(
                &params.inner,
                &erb_params.inner,
                Some(&db_params.inner),
            )?),
        };
        Ok(PyErbDbPlan { inner })
    }

    /// Create a plan for computing logarithmic Hz power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `loghz_params` : `LogHzParams`
    ///     Logarithmic Hz scale parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LogHzPowerPlan`
    ///     Plan for computing logarithmic Hz power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams, dtype: str = \"float64\") -> LogHzPowerPlan")]
    fn loghz_power_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLogHzPowerPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.log_hz_plan::<Power, f32>(
                &params.inner,
                &loghz_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.log_hz_plan::<Power, f64>(
                &params.inner,
                &loghz_params.inner,
                None,
            )?),
        };
        Ok(PyLogHzPowerPlan { inner })
    }

    /// Create a plan for computing logarithmic Hz magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `loghz_params` : `LogHzParams`
    ///     Logarithmic Hz scale parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LogHzMagnitudePlan`
    ///     Plan for computing logarithmic Hz magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams, dtype: str = \"float64\") -> LogHzMagnitudePlan")]
    fn loghz_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLogHzMagnitudePlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.log_hz_plan::<Magnitude, f32>(
                &params.inner,
                &loghz_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.log_hz_plan::<Magnitude, f64>(
                &params.inner,
                &loghz_params.inner,
                None,
            )?),
        };
        Ok(PyLogHzMagnitudePlan { inner })
    }

    /// Create a plan for computing logarithmic Hz decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `loghz_params` : `LogHzParams`
    ///     Logarithmic Hz scale parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `LogHzDbPlan`
    ///     Plan for computing logarithmic Hz decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams", db_params: "LogParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams, db_params: LogParams, dtype: str = \"float64\") -> LogHzDbPlan")]
    fn loghz_db_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
        db_params: PyLogParams,
        dtype: Option<&str>,
    ) -> PyResult<PyLogHzDbPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.log_hz_plan::<Decibels, f32>(
                &params.inner,
                &loghz_params.inner,
                Some(&db_params.inner),
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.log_hz_plan::<Decibels, f64>(
                &params.inner,
                &loghz_params.inner,
                Some(&db_params.inner),
            )?),
        };
        Ok(PyLogHzDbPlan { inner })
    }

    /// Create a plan for computing CQT power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `cqt_params` : `CqtParams`
    ///     Constant-Q Transform parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `CqtPowerPlan`
    ///     Plan for computing CQT power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams, dtype: str = \"float64\") -> CqtPowerPlan")]
    fn cqt_power_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
        dtype: Option<&str>,
    ) -> PyResult<PyCqtPowerPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.cqt_plan::<Power, f32>(
                &params.inner,
                &cqt_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.cqt_plan::<Power, f64>(
                &params.inner,
                &cqt_params.inner,
                None,
            )?),
        };
        Ok(PyCqtPowerPlan { inner })
    }

    /// Create a plan for computing CQT magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `cqt_params` : `CqtParams`
    ///     Constant-Q Transform parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `CqtMagnitudePlan`
    ///     Plan for computing CQT magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams, dtype: str = \"float64\") -> CqtMagnitudePlan")]
    fn cqt_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
        dtype: Option<&str>,
    ) -> PyResult<PyCqtMagnitudePlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.cqt_plan::<Magnitude, f32>(
                &params.inner,
                &cqt_params.inner,
                None,
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.cqt_plan::<Magnitude, f64>(
                &params.inner,
                &cqt_params.inner,
                None,
            )?),
        };
        Ok(PyCqtMagnitudePlan { inner })
    }

    /// Create a plan for computing CQT decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `cqt_params` : `CqtParams`
    ///     Constant-Q Transform parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    /// dtype : str, optional
    ///     Output precision, ``"float32"`` or ``"float64"`` (default).
    ///
    /// Returns
    /// -------
    /// `CqtDbPlan`
    ///     Plan for computing CQT decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams", db_params: "LogParams", dtype: "str" = None), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams, db_params: LogParams, dtype: str = \"float64\") -> CqtDbPlan")]
    fn cqt_db_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
        db_params: PyLogParams,
        dtype: Option<&str>,
    ) -> PyResult<PyCqtDbPlan> {
        let inner = match parse_dtype(dtype)? {
            Dtype::F32 => DualPlan::F32(self.inner.cqt_plan::<Decibels, f32>(
                &params.inner,
                &cqt_params.inner,
                Some(&db_params.inner),
            )?),
            Dtype::F64 => DualPlan::F64(self.inner.cqt_plan::<Decibels, f64>(
                &params.inner,
                &cqt_params.inner,
                Some(&db_params.inner),
            )?),
        };
        Ok(PyCqtDbPlan { inner })
    }
}

// Macro to reduce boilerplate for plan classes
macro_rules! impl_plan {
    ($py_name:ident, $py_name_str:literal, $rust_freq:ty, $rust_amp:ty, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        #[pyclass(name = $py_name_str, unsendable)]
        pub struct $py_name {
            inner: DualPlan<$rust_freq, $rust_amp>,
        }

        #[pymethods]
        impl $py_name {
            /// The NumPy dtype the plan computes in (``"float32"`` / ``"float64"``).
            #[getter]
            fn dtype(&self) -> &'static str {
                self.inner.dtype()
            }

            /// Compute a spectrogram from audio samples.
            ///
            /// Parameters
            /// ----------
            /// samples : numpy.typing.NDArray
            ///     Audio samples as a 1D array (coerced to the plan's precision)
            ///
            /// Returns
            /// -------
            /// Spectrogram
            ///     Computed spectrogram result (``.data`` follows the plan's ``dtype``)
            #[pyo3(signature = (samples: "numpy.typing.NDArray"), text_signature = "(samples: numpy.typing.NDArray) -> Spectrogram")]
            fn compute(
                &mut self,
                py: Python,
                samples: &Bound<'_, PyAny>,
            ) -> PyResult<PySpectrogram> {
                self.inner.compute(py, samples)
            }

            /// Compute a single frame of the spectrogram.
            ///
            /// Parameters
            /// ----------
            /// `samples` : numpy.typing.NDArray
            ///     Audio samples as a 1D array (coerced to the plan's precision)
            /// `frame_idx` : int
            ///     Frame index to compute
            ///
            /// Returns
            /// -------
            /// numpy.typing.NDArray
            ///     1D array containing the frame data (in the plan's precision)
            #[pyo3(signature = (samples: "numpy.typing.NDArray", frame_idx: "int"), text_signature = "(samples: numpy.typing.NDArray, frame_idx: int) -> numpy.typing.NDArray")]
            fn compute_frame(
                &mut self,
                py: Python,
                samples: &Bound<'_, PyAny>,
                frame_idx: usize,
            ) -> PyResult<Py<PyAny>> {
                self.inner.compute_frame(py, samples, frame_idx)
            }

            /// Get the output shape for a given signal length.
            ///
            /// Parameters
            /// ----------
            /// `signal_length` : int
            ///     Length of the input signal
            ///
            /// Returns
            /// -------
            /// tuple[int, int]
            ///     Tuple of (`n_bins`, `n_frames`)
            #[pyo3(signature = (signal_length: "int"), text_signature = "(signal_length: int) -> tuple[int, int]")]
            fn output_shape(
                &self,
                signal_length: NonZeroUsize,
            ) -> PyResult<(NonZeroUsize, NonZeroUsize)> {
                self.inner.output_shape(signal_length)
            }
        }
    };
}

// Linear frequency plans
impl_plan!(
    PyLinearPowerPlan,
    "LinearPowerPlan",
    LinearHz,
    Power,
    LinearPower,
    "Plan for computing linear power spectrograms."
);
impl_plan!(
    PyLinearMagnitudePlan,
    "LinearMagnitudePlan",
    LinearHz,
    Magnitude,
    LinearMagnitude,
    "Plan for computing linear magnitude spectrograms."
);
impl_plan!(
    PyLinearDbPlan,
    "LinearDbPlan",
    LinearHz,
    Decibels,
    LinearDb,
    "Plan for computing linear decibel spectrograms."
);

// Mel frequency plans
impl_plan!(
    PyMelPowerPlan,
    "MelPowerPlan",
    Mel,
    Power,
    MelPower,
    "Plan for computing mel power spectrograms."
);
impl_plan!(
    PyMelMagnitudePlan,
    "MelMagnitudePlan",
    Mel,
    Magnitude,
    MelMagnitude,
    "Plan for computing mel magnitude spectrograms."
);
impl_plan!(
    PyMelDbPlan,
    "MelDbPlan",
    Mel,
    Decibels,
    MelDb,
    "Plan for computing mel decibel spectrograms."
);

// ERB frequency plans
impl_plan!(
    PyErbPowerPlan,
    "ErbPowerPlan",
    Gammatone,
    Power,
    GammatonePower,
    "Plan for computing ERB power spectrograms."
);
impl_plan!(
    PyErbMagnitudePlan,
    "ErbMagnitudePlan",
    Gammatone,
    Magnitude,
    GammatoneMagnitude,
    "Plan for computing ERB magnitude spectrograms."
);
impl_plan!(
    PyErbDbPlan,
    "ErbDbPlan",
    Gammatone,
    Decibels,
    GammatoneDb,
    "Plan for computing ERB decibel spectrograms."
);

// LogHz frequency plans
impl_plan!(
    PyLogHzPowerPlan,
    "LogHzPowerPlan",
    LogHz,
    Power,
    LogHzPower,
    "Plan for computing logarithmic Hz power spectrograms."
);
impl_plan!(
    PyLogHzMagnitudePlan,
    "LogHzMagnitudePlan",
    LogHz,
    Magnitude,
    LogHzMagnitude,
    "Plan for computing logarithmic Hz magnitude spectrograms."
);
impl_plan!(
    PyLogHzDbPlan,
    "LogHzDbPlan",
    LogHz,
    Decibels,
    LogHzDb,
    "Plan for computing logarithmic Hz decibel spectrograms."
);

// CQT plans
impl_plan!(
    PyCqtPowerPlan,
    "CqtPowerPlan",
    Cqt,
    Power,
    CqtPower,
    "Plan for computing CQT power spectrograms."
);
impl_plan!(
    PyCqtMagnitudePlan,
    "CqtMagnitudePlan",
    Cqt,
    Magnitude,
    CqtMagnitude,
    "Plan for computing CQT magnitude spectrograms."
);
impl_plan!(
    PyCqtDbPlan,
    "CqtDbPlan",
    Cqt,
    Decibels,
    CqtDb,
    "Plan for computing CQT decibel spectrograms."
);

/// Register the planner and all plan classes with the Python module.
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectrogramPlanner>()?;

    // Linear plans
    m.add_class::<PyLinearPowerPlan>()?;
    m.add_class::<PyLinearMagnitudePlan>()?;
    m.add_class::<PyLinearDbPlan>()?;

    // Mel plans
    m.add_class::<PyMelPowerPlan>()?;
    m.add_class::<PyMelMagnitudePlan>()?;
    m.add_class::<PyMelDbPlan>()?;

    // ERB plans
    m.add_class::<PyErbPowerPlan>()?;
    m.add_class::<PyErbMagnitudePlan>()?;
    m.add_class::<PyErbDbPlan>()?;

    // LogHz plans
    m.add_class::<PyLogHzPowerPlan>()?;
    m.add_class::<PyLogHzMagnitudePlan>()?;
    m.add_class::<PyLogHzDbPlan>()?;

    // CQT plans
    m.add_class::<PyCqtPowerPlan>()?;
    m.add_class::<PyCqtMagnitudePlan>()?;
    m.add_class::<PyCqtDbPlan>()?;

    Ok(())
}
