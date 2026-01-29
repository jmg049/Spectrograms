//! Planner and plan classes for efficient batch processing.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{
    Cqt, Decibels, Gammatone, LinearHz, LogHz, Magnitude, Mel, Power, SpectrogramPlan,
    SpectrogramPlanner,
};

use super::params::{
    PyCqtParams, PyErbParams, PyLogHzParams, PyLogParams, PyMelParams, PySpectrogramParams,
};
use super::spectrogram::PySpectrogram;
use non_empty_slice::NonEmptySlice;
use std::num::NonZeroUsize;
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
    ///
    /// Returns
    /// -------
    /// `LinearPowerPlan`
    ///     Plan for computing linear power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams"), text_signature = "(params: SpectrogramParams) -> LinearPowerPlan")]
    fn linear_power_plan(&self, params: &PySpectrogramParams) -> PyResult<PyLinearPowerPlan> {
        let plan = self.inner.linear_plan::<Power>(&params.inner, None)?;
        Ok(PyLinearPowerPlan { inner: plan })
    }

    /// Create a plan for computing linear magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    ///
    /// Returns
    /// -------
    /// `LinearMagnitudePlan`
    ///     Plan for computing linear magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams"), text_signature = "(params: SpectrogramParams) -> LinearMagnitudePlan")]
    fn linear_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
    ) -> PyResult<PyLinearMagnitudePlan> {
        let plan = self.inner.linear_plan::<Magnitude>(&params.inner, None)?;
        Ok(PyLinearMagnitudePlan { inner: plan })
    }

    /// Create a plan for computing linear decibel spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `db_params` : `LogParams`
    ///     Decibel conversion parameters
    ///
    /// Returns
    /// -------
    /// `LinearDbPlan`
    ///     Plan for computing linear decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", db_params: "LogParams"), text_signature = "(params: SpectrogramParams, db_params: LogParams) -> LinearDbPlan")]
    fn linear_db_plan(
        &self,
        params: &PySpectrogramParams,
        db_params: PyLogParams,
    ) -> PyResult<PyLinearDbPlan> {
        let plan = self
            .inner
            .linear_plan::<Decibels>(&params.inner, Some(&db_params.inner))?;
        Ok(PyLinearDbPlan { inner: plan })
    }

    /// Create a plan for computing mel power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `mel_params` : `MelParams`
    ///     Mel-scale filterbank parameters
    ///
    /// Returns
    /// -------
    /// `MelPowerPlan`
    ///     Plan for computing mel power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams"), text_signature = "(params: SpectrogramParams, mel_params: MelParams) -> MelPowerPlan")]
    fn mel_power_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
    ) -> PyResult<PyMelPowerPlan> {
        let plan = self
            .inner
            .mel_plan::<Power>(&params.inner, &mel_params.inner, None)?;
        Ok(PyMelPowerPlan { inner: plan })
    }

    /// Create a plan for computing mel magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `mel_params` : `MelParams`
    ///     Mel-scale filterbank parameters
    ///
    /// Returns
    /// -------
    /// `MelMagnitudePlan`
    ///     Plan for computing mel magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams"), text_signature = "(params: SpectrogramParams, mel_params: MelParams) -> MelMagnitudePlan")]
    fn mel_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
    ) -> PyResult<PyMelMagnitudePlan> {
        let plan = self
            .inner
            .mel_plan::<Magnitude>(&params.inner, &mel_params.inner, None)?;
        Ok(PyMelMagnitudePlan { inner: plan })
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
    ///
    /// Returns
    /// -------
    /// `MelDbPlan`
    ///     Plan for computing mel decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", mel_params: "MelParams", db_params: "LogParams"), text_signature = "(params: SpectrogramParams, mel_params: MelParams, db_params: LogParams) -> MelDbPlan")]
    fn mel_db_plan(
        &self,
        params: &PySpectrogramParams,
        mel_params: &PyMelParams,
        db_params: PyLogParams,
    ) -> PyResult<PyMelDbPlan> {
        let plan = self.inner.mel_plan::<Decibels>(
            &params.inner,
            &mel_params.inner,
            Some(&db_params.inner),
        )?;
        Ok(PyMelDbPlan { inner: plan })
    }

    /// Create a plan for computing ERB power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `erb_params` : `ErbParams`
    ///     ERB-scale filterbank parameters
    ///
    /// Returns
    /// -------
    /// `ErbPowerPlan`
    ///     Plan for computing ERB power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams"), text_signature = "(params: SpectrogramParams, erb_params: ErbParams) -> ErbPowerPlan")]
    fn erb_power_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
    ) -> PyResult<PyErbPowerPlan> {
        let plan = self
            .inner
            .erb_plan::<Power>(&params.inner, &erb_params.inner, None)?;
        Ok(PyErbPowerPlan { inner: plan })
    }

    /// Create a plan for computing ERB magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `erb_params` : `ErbParams`
    ///     ERB-scale filterbank parameters
    ///
    /// Returns
    /// -------
    /// `ErbMagnitudePlan`
    ///     Plan for computing ERB magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams"), text_signature = "(params: SpectrogramParams, erb_params: ErbParams) -> ErbMagnitudePlan")]
    fn erb_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
    ) -> PyResult<PyErbMagnitudePlan> {
        let plan = self
            .inner
            .erb_plan::<Magnitude>(&params.inner, &erb_params.inner, None)?;
        Ok(PyErbMagnitudePlan { inner: plan })
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
    ///
    /// Returns
    /// -------
    /// `ErbDbPlan`
    ///     Plan for computing ERB decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", erb_params: "ErbParams", db_params: "LogParams"), text_signature = "(params: SpectrogramParams, erb_params: ErbParams, db_params: LogParams) -> ErbDbPlan")]
    fn erb_db_plan(
        &self,
        params: &PySpectrogramParams,
        erb_params: &PyErbParams,
        db_params: PyLogParams,
    ) -> PyResult<PyErbDbPlan> {
        let plan = self.inner.erb_plan::<Decibels>(
            &params.inner,
            &erb_params.inner,
            Some(&db_params.inner),
        )?;
        Ok(PyErbDbPlan { inner: plan })
    }

    /// Create a plan for computing logarithmic Hz power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `loghz_params` : `LogHzParams`
    ///     Logarithmic Hz scale parameters
    ///
    /// Returns
    /// -------
    /// `LogHzPowerPlan`
    ///     Plan for computing logarithmic Hz power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams"), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams) -> LogHzPowerPlan")]
    fn loghz_power_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
    ) -> PyResult<PyLogHzPowerPlan> {
        let plan = self
            .inner
            .log_hz_plan::<Power>(&params.inner, &loghz_params.inner, None)?;
        Ok(PyLogHzPowerPlan { inner: plan })
    }

    /// Create a plan for computing logarithmic Hz magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `loghz_params` : `LogHzParams`
    ///     Logarithmic Hz scale parameters
    ///
    /// Returns
    /// -------
    /// `LogHzMagnitudePlan`
    ///     Plan for computing logarithmic Hz magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams"), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams) -> LogHzMagnitudePlan")]
    fn loghz_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
    ) -> PyResult<PyLogHzMagnitudePlan> {
        let plan = self
            .inner
            .log_hz_plan::<Magnitude>(&params.inner, &loghz_params.inner, None)?;
        Ok(PyLogHzMagnitudePlan { inner: plan })
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
    ///
    /// Returns
    /// -------
    /// `LogHzDbPlan`
    ///     Plan for computing logarithmic Hz decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", loghz_params: "LogHzParams", db_params: "LogParams"), text_signature = "(params: SpectrogramParams, loghz_params: LogHzParams, db_params: LogParams) -> LogHzDbPlan")]
    fn loghz_db_plan(
        &self,
        params: &PySpectrogramParams,
        loghz_params: &PyLogHzParams,
        db_params: PyLogParams,
    ) -> PyResult<PyLogHzDbPlan> {
        let plan = self.inner.log_hz_plan::<Decibels>(
            &params.inner,
            &loghz_params.inner,
            Some(&db_params.inner),
        )?;
        Ok(PyLogHzDbPlan { inner: plan })
    }

    /// Create a plan for computing CQT power spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `cqt_params` : `CqtParams`
    ///     Constant-Q Transform parameters
    ///
    /// Returns
    /// -------
    /// `CqtPowerPlan`
    ///     Plan for computing CQT power spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams"), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams) -> CqtPowerPlan")]
    fn cqt_power_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
    ) -> PyResult<PyCqtPowerPlan> {
        let plan = self
            .inner
            .cqt_plan::<Power>(&params.inner, &cqt_params.inner, None)?;
        Ok(PyCqtPowerPlan { inner: plan })
    }

    /// Create a plan for computing CQT magnitude spectrograms.
    ///
    /// Parameters
    /// ----------
    /// params : `SpectrogramParams`
    ///     Spectrogram parameters
    /// `cqt_params` : `CqtParams`
    ///     Constant-Q Transform parameters
    ///
    /// Returns
    /// -------
    /// `CqtMagnitudePlan`
    ///     Plan for computing CQT magnitude spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams"), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams) -> CqtMagnitudePlan")]
    fn cqt_magnitude_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
    ) -> PyResult<PyCqtMagnitudePlan> {
        let plan = self
            .inner
            .cqt_plan::<Magnitude>(&params.inner, &cqt_params.inner, None)?;
        Ok(PyCqtMagnitudePlan { inner: plan })
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
    ///
    /// Returns
    /// -------
    /// `CqtDbPlan`
    ///     Plan for computing CQT decibel spectrograms
    #[pyo3(signature = (params: "SpectrogramParams", cqt_params: "CqtParams", db_params: "LogParams"), text_signature = "(params: SpectrogramParams, cqt_params: CqtParams, db_params: LogParams) -> CqtDbPlan")]
    fn cqt_db_plan(
        &self,
        params: &PySpectrogramParams,
        cqt_params: &PyCqtParams,
        db_params: PyLogParams,
    ) -> PyResult<PyCqtDbPlan> {
        let plan = self.inner.cqt_plan::<Decibels>(
            &params.inner,
            &cqt_params.inner,
            Some(&db_params.inner),
        )?;
        Ok(PyCqtDbPlan { inner: plan })
    }
}

// Macro to reduce boilerplate for plan classes
macro_rules! impl_plan {
    ($py_name:ident, $py_name_str:literal, $rust_freq:ty, $rust_amp:ty, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        #[pyclass(name = $py_name_str, unsendable)]
        pub struct $py_name {
            inner: SpectrogramPlan<$rust_freq, $rust_amp>,
        }

        #[pymethods]
        impl $py_name {
            /// Compute a spectrogram from audio samples.
            ///
            /// Parameters
            /// ----------
            /// samples : numpy.typing.NDArray[numpy.float64]
            ///     Audio samples as a 1D array
            ///
            /// Returns
            /// -------
            /// Spectrogram
            ///     Computed spectrogram result
            #[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(samples: numpy.typing.NDArray[numpy.float64]) -> Spectrogram")]
            fn compute(
                &mut self,
                py: Python,
                samples: PyReadonlyArray1<f64>,
            ) -> PyResult<PySpectrogram> {
                let samples = samples.as_slice()?;
                let samples = NonEmptySlice::new(samples).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Input samples cannot be empty")
                })?;
                let spec = self.inner.compute(samples)?;
                Ok(PySpectrogram::from_spectrogram(py, spec))
            }

            /// Compute a single frame of the spectrogram.
            ///
            /// Parameters
            /// ----------
            /// `samples` : numpy.typing.NDArray[numpy.float64]
            ///     Audio samples as a 1D array
            /// `frame_idx` : int
            ///     Frame index to compute
            ///
            /// Returns
            /// -------
            /// numpy.typing.NDArray[numpy.float64]
            ///     1D array containing the frame data
            #[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float64]",frame_idx: "int"), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], frame_idx: int) -> numpy.typing.NDArray[numpy.float64]")]
            fn compute_frame<'py>(
                &mut self,
                py: Python<'py>,
                samples: PyReadonlyArray1<f64>,
                frame_idx: usize,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {

                let samples = samples.as_slice()?;
                let samples = NonEmptySlice::new(samples).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Input samples cannot be empty")
                })?;
                let frame = self.inner.compute_frame(samples, frame_idx)?;
                Ok(PyArray1::from_vec(py, frame.to_vec()))
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
            fn output_shape(&self, signal_length: NonZeroUsize ) -> PyResult<(NonZeroUsize , NonZeroUsize )> {
                Ok(self.inner.output_shape(signal_length)?)
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
