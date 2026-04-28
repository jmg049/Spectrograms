//! Python bindings for binaural audio spectrograms.
//!
//! Credit to @barrydn for the original implementation of all the spectrograms in this file.
//! Taken from https://github.com/QxLabIreland/Binaspect/

use std::num::NonZeroUsize;

use non_empty_slice::NonEmptySlice;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::binaural::{
    ILDSpectrogramParams, ILRSpectrogramParams, IPDSpectrogramParams, ITDSpectrogramParams,
    compute_ild_spectrogram, compute_ilr_spectrogram, compute_ilr_spectrogram_diff,
    compute_ipd_spectrogram, compute_itd_spectrogram, compute_itd_spectrogram_diff,
};
use crate::{StftPlan, python::PySpectrogramParams};

/// Parameters for computing the Interaural Time Difference (ITD) spectrogram.
///
/// ITD represents the time difference between when a sound reaches the left and right ears,
/// which is a primary cue for sound localization, especially at low frequencies.
///
/// The ITD spectrogram captures how this time difference varies across frequency and time,
/// providing insight into the spatial properties of binaural audio signals.
#[pyclass(name = "ITDSpectrogramParams", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyITDSpectrogramParams {
    pub(crate) inner: ITDSpectrogramParams,
}

#[pymethods]
impl PyITDSpectrogramParams {
    /// Create new ITD spectrogram parameters.
    ///
    /// Parameters
    /// ----------
    /// spectrogram_params : SpectrogramParams
    ///     Base spectrogram parameters (FFT size, hop length, window, etc.)
    /// start_freq : float, optional
    ///     Lower frequency bound in Hz for ITD analysis (default: 50.0)
    /// end_freq : float, optional
    ///     Upper frequency bound in Hz for ITD analysis (default: 620.0)
    /// magphase_power : int, optional
    ///     Power to raise magnitude-phase product to (default: 1)
    ///
    /// Returns
    /// -------
    /// ITDSpectrogramParams
    ///     Configured ITD spectrogram parameters
    #[new]
    #[pyo3(signature = (spectrogram_params: "SpectrogramParams", start_freq: "float" = 50.0, end_freq: "float" = 620.0, magphase_power: "Optional[int]" = 1), text_signature = "(spectrogram_params: SpectrogramParams, start_freq: float = 50.0, end_freq: float = 620.0, magphase_power: Optional[int] = 1) -> ITDSpectrogramParams")]
    fn new(
        spectrogram_params: PySpectrogramParams,
        start_freq: Option<f64>,
        end_freq: Option<f64>,
        magphase_power: Option<usize>,
    ) -> Self {
        let inner = ITDSpectrogramParams {
            spectrogram_params: spectrogram_params.into(),
            start_freq: start_freq.unwrap_or(50.0),
            end_freq: end_freq.unwrap_or(620.0),
            magphase_power: magphase_power
                .and_then(NonZeroUsize::new)
                .unwrap_or_else(|| crate::nzu!(1)),
        };
        Self { inner }
    }

    /// Get the base spectrogram parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     Base spectrogram configuration
    #[getter]
    fn spectrogram_params(&self) -> PySpectrogramParams {
        PySpectrogramParams::from(self.inner.spectrogram_params.clone())
    }

    /// Get the lower frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Lower frequency bound in Hz
    #[getter]
    const fn start_freq(&self) -> f64 {
        self.inner.start_freq
    }

    /// Get the upper frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Upper frequency bound in Hz
    #[getter]
    const fn end_freq(&self) -> f64 {
        self.inner.end_freq
    }

    /// Get the magnitude-phase power parameter.
    ///
    /// Returns
    /// -------
    /// int
    ///     Power to raise magnitude-phase product to
    #[getter]
    const fn magphase_power(&self) -> NonZeroUsize {
        self.inner.magphase_power
    }
}

impl From<ITDSpectrogramParams> for PyITDSpectrogramParams {
    #[inline]
    fn from(inner: ITDSpectrogramParams) -> Self {
        Self { inner }
    }
}

impl From<PyITDSpectrogramParams> for ITDSpectrogramParams {
    #[inline]
    fn from(val: PyITDSpectrogramParams) -> Self {
        val.inner
    }
}

/// Compute the Interaural Time Difference (ITD) spectrogram for a stereo audio signal.
///
/// Parameters
/// ----------
/// audio : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel]
/// params : ITDSpectrogramParams
///     ITD spectrogram parameters
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     2D array containing the ITD spectrogram
///
/// Raises
/// ------
/// RuntimeError
///     If STFT plan creation or ITD computation fails
/// ValueError
///     If audio arrays are not contiguous or not of type float64
#[pyfunction(name = "compute_itd_spectrogram")]
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "ITDSpectrogramParams"), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ITDSpectrogramParams) -> numpy.typing.NDArray[numpy.float64]")]
fn py_compute_itd_spectrogram<'py>(
    py: Python<'py>,
    audio: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyITDSpectrogramParams,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut plan: StftPlan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    let left_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let right_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right audio array must be contiguous and of type float64.",
            )
        })?)
    };
    let audio_slices = [left_slice, right_slice];

    let itd_spectrogram =
        compute_itd_spectrogram(audio_slices, &params.inner, &mut plan).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to compute ITD spectrogram: {e}"
            ))
        })?;

    let py_array = PyArray2::from_owned_array(py, itd_spectrogram.data);
    Ok(py_array)
}

/// Parameters for computing the Interaural Phase Difference (IPD) spectrogram.
///
/// IPD represents the phase difference between the left and right ear signals,
/// which provides spatial cues, particularly at higher frequencies where ITD becomes ambiguous.
#[pyclass(name = "IPDSpectrogramParams", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyIPDSpectrogramParams {
    pub(crate) inner: IPDSpectrogramParams,
}

#[pymethods]
impl PyIPDSpectrogramParams {
    /// Create new IPD spectrogram parameters.
    ///
    /// Parameters
    /// ----------
    /// spectrogram_params : SpectrogramParams
    ///     Base spectrogram parameters (FFT size, hop length, window, etc.)
    /// start_freq : float, optional
    ///     Lower frequency bound in Hz for IPD analysis (default: 50.0)
    /// end_freq : float, optional
    ///     Upper frequency bound in Hz for IPD analysis (default: 620.0)
    /// wrapped : bool, optional
    ///     If True, return wrapped phase difference in [-π, π], otherwise unwrapped (default: False)
    ///
    /// Returns
    /// -------
    /// IPDSpectrogramParams
    ///     Configured IPD spectrogram parameters
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameters are invalid
    #[new]
    #[pyo3(signature = (spectrogram_params, start_freq = 50.0, end_freq = 620.0, wrapped = false), text_signature = "(spectrogram_params: SpectrogramParams, start_freq: float = 50.0, end_freq: float = 620.0, wrapped: bool = False) -> IPDSpectrogramParams")]
    fn new(
        spectrogram_params: PySpectrogramParams,
        start_freq: Option<f64>,
        end_freq: Option<f64>,
        wrapped: Option<bool>,
    ) -> PyResult<Self> {
        let inner = IPDSpectrogramParams::new(
            spectrogram_params.into(),
            start_freq.unwrap_or(50.0),
            end_freq.unwrap_or(620.0),
            wrapped.unwrap_or(false),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Get the base spectrogram parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     Base spectrogram configuration
    #[getter]
    fn spectrogram_params(&self) -> PySpectrogramParams {
        PySpectrogramParams::from(self.inner.spectrogram_params.clone())
    }

    /// Get the lower frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Lower frequency bound in Hz
    #[getter]
    const fn start_freq(&self) -> f64 {
        self.inner.start_freq
    }

    /// Get the upper frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Upper frequency bound in Hz
    #[getter]
    const fn end_freq(&self) -> f64 {
        self.inner.end_freq
    }

    /// Get whether phase difference is wrapped.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if wrapped to [-π, π], False if unwrapped
    #[getter]
    const fn wrapped(&self) -> bool {
        self.inner.wrapped
    }
}

/// Compute the Interaural Phase Difference (IPD) spectrogram for a stereo audio signal.
///
/// Parameters
/// ----------
/// audio : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel]
/// params : IPDSpectrogramParams
///     IPD spectrogram parameters
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     2D array containing the IPD spectrogram
///
/// Raises
/// ------
/// RuntimeError
///     If STFT plan creation or IPD computation fails
/// ValueError
///     If audio arrays are not contiguous or not of type float64
#[pyfunction(name = "compute_ipd_spectrogram")]
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "IPDSpectrogramParams"), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: IPDSpectrogramParams) -> numpy.typing.NDArray[numpy.float64]")]
fn py_compute_ipd_spectrogram<'py>(
    py: Python<'py>,
    audio: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyIPDSpectrogramParams,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut plan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    let left_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let right_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let ipd_spectrogram =
        compute_ipd_spectrogram([left_slice, right_slice], &params.inner, &mut plan).map_err(
            |e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute IPD spectrogram: {e}"
                ))
            },
        )?;

    Ok(PyArray2::from_owned_array(py, ipd_spectrogram.data))
}

/// Parameters for computing the Interaural Level Difference (ILD) spectrogram.
///
/// ILD represents the difference in sound intensity between the left and right ears,
/// which is an important cue for sound localization, especially at higher frequencies.
#[pyclass(name = "ILDSpectrogramParams", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyILDSpectrogramParams {
    pub(crate) inner: ILDSpectrogramParams,
}

#[pymethods]
impl PyILDSpectrogramParams {
    /// Create new ILD spectrogram parameters.
    ///
    /// Parameters
    /// ----------
    /// spectrogram_params : SpectrogramParams
    ///     Base spectrogram parameters (FFT size, hop length, window, etc.)
    /// start_freq : float, optional
    ///     Lower frequency bound in Hz for ILD analysis (default: 1700.0)
    /// end_freq : float, optional
    ///     Upper frequency bound in Hz for ILD analysis (default: 4600.0)
    ///
    /// Returns
    /// -------
    /// ILDSpectrogramParams
    ///     Configured ILD spectrogram parameters
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameters are invalid
    #[new]
    #[pyo3(signature = (spectrogram_params: "SpectrogramParams", start_freq: "float" = 1700.0, end_freq: "float" = 4600.0), text_signature = "(spectrogram_params: SpectrogramParams, start_freq: float = 1700.0, end_freq: float = 4600.0) -> ILDSpectrogramParams")]
    fn new(
        spectrogram_params: PySpectrogramParams,
        start_freq: Option<f64>,
        end_freq: Option<f64>,
    ) -> PyResult<Self> {
        let inner = ILDSpectrogramParams::new(
            spectrogram_params.into(),
            start_freq.unwrap_or(1700.0),
            end_freq.unwrap_or(4600.0),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Get the base spectrogram parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     Base spectrogram configuration
    #[getter]
    fn spectrogram_params(&self) -> PySpectrogramParams {
        PySpectrogramParams::from(self.inner.spectrogram_params.clone())
    }

    /// Get the lower frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Lower frequency bound in Hz
    #[getter]
    const fn start_freq(&self) -> f64 {
        self.inner.start_freq
    }

    /// Get the upper frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Upper frequency bound in Hz
    #[getter]
    const fn end_freq(&self) -> f64 {
        self.inner.end_freq
    }
}

/// Compute the Interaural Level Difference (ILD) spectrogram for a stereo audio signal.
///
/// Parameters
/// ----------
/// audio : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel]
/// params : ILDSpectrogramParams
///     ILD spectrogram parameters
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     2D array containing the ILD spectrogram
///
/// Raises
/// ------
/// RuntimeError
///     If STFT plan creation or ILD computation fails
/// ValueError
///     If audio arrays are not contiguous or not of type float64
#[pyfunction(name = "compute_ild_spectrogram")]
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "ILDSpectrogramParams"), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ILDSpectrogramParams) -> numpy.typing.NDArray[numpy.float64]")]
fn py_compute_ild_spectrogram<'py>(
    py: Python<'py>,
    audio: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyILDSpectrogramParams,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut plan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    let left_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let right_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let ild_spectrogram =
        compute_ild_spectrogram([left_slice, right_slice], &params.inner, &mut plan).map_err(
            |e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute ILD spectrogram: {e}"
                ))
            },
        )?;

    Ok(PyArray2::from_owned_array(py, ild_spectrogram.data))
}

/// Parameters for computing the Interaural Level Ratio (ILR) spectrogram.
///
/// ILR represents the ratio of sound levels between the left and right ears,
/// providing a normalized measure of level differences for spatial analysis.
#[pyclass(name = "ILRSpectrogramParams", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyILRSpectrogramParams {
    pub(crate) inner: ILRSpectrogramParams,
}

#[pymethods]
impl PyILRSpectrogramParams {
    /// Create new ILR spectrogram parameters.
    ///
    /// Parameters
    /// ----------
    /// spectrogram_params : SpectrogramParams
    ///     Base spectrogram parameters (FFT size, hop length, window, etc.)
    /// start_freq : float, optional
    ///     Lower frequency bound in Hz for ILR analysis (default: 1700.0)
    /// end_freq : float, optional
    ///     Upper frequency bound in Hz for ILR analysis (default: 4600.0)
    ///
    /// Returns
    /// -------
    /// ILRSpectrogramParams
    ///     Configured ILR spectrogram parameters
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If parameters are invalid
    #[new]
    #[pyo3(signature = (spectrogram_params: "SpectrogramParams", start_freq: "float" = 1700.0, end_freq: "float" = 4600.0), text_signature = "(spectrogram_params: SpectrogramParams, start_freq: float = 1700.0, end_freq: float = 4600.0) -> ILRSpectrogramParams")]
    fn new(
        spectrogram_params: PySpectrogramParams,
        start_freq: Option<f64>,
        end_freq: Option<f64>,
    ) -> PyResult<Self> {
        let inner = ILRSpectrogramParams::new(
            spectrogram_params.into(),
            start_freq.unwrap_or(1700.0),
            end_freq.unwrap_or(4600.0),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Get the base spectrogram parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     Base spectrogram configuration
    #[getter]
    fn spectrogram_params(&self) -> PySpectrogramParams {
        PySpectrogramParams::from(self.inner.spectrogram_params.clone())
    }

    /// Get the lower frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Lower frequency bound in Hz
    #[getter]
    const fn start_freq(&self) -> f64 {
        self.inner.start_freq
    }

    /// Get the upper frequency bound.
    ///
    /// Returns
    /// -------
    /// float
    ///     Upper frequency bound in Hz
    #[getter]
    const fn end_freq(&self) -> f64 {
        self.inner.end_freq
    }
}

/// Compute the Interaural Level Ratio (ILR) spectrogram for a stereo audio signal.
///
/// Parameters
/// ----------
/// audio : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel]
/// params : ILRSpectrogramParams
///     ILR spectrogram parameters
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     2D array containing the ILR spectrogram
///
/// Raises
/// ------
/// RuntimeError
///     If STFT plan creation or ILR computation fails
/// ValueError
///     If audio arrays are not contiguous or not of type float64
#[pyfunction(name = "compute_ilr_spectrogram")]
#[pyo3(signature = (audio, params), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ILRSpectrogramParams) -> numpy.typing.NDArray[numpy.float64]")]
fn py_compute_ilr_spectrogram<'py>(
    py: Python<'py>,
    audio: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyILRSpectrogramParams,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut plan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    let left_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let right_slice = unsafe {
        NonEmptySlice::new_unchecked(audio[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right audio array must be contiguous and of type float64.",
            )
        })?)
    };

    let ilr_spectrogram =
        compute_ilr_spectrogram([left_slice, right_slice], &params.inner, &mut plan).map_err(
            |e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute ILR spectrogram: {e}"
                ))
            },
        )?;

    Ok(PyArray2::from_owned_array(py, ilr_spectrogram.data))
}

/// Compute the difference between two ITD spectrograms.
///
/// Parameters
/// ----------
/// reference : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel] for the reference signal
/// test : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel] for the test signal
/// params : ITDSpectrogramParams
///     ITD spectrogram parameters
///
/// Returns
/// -------
/// tuple[numpy.typing.NDArray[numpy.float64], float, float]
///     Tuple of (itd_time_diff, mean_diff_degrees, mean_diff_itd)
#[pyfunction(name = "compute_itd_spectrogram_diff")]
#[pyo3(signature = (reference, test, params), text_signature = "(reference: list[numpy.typing.NDArray[numpy.float64]], test: list[numpy.typing.NDArray[numpy.float64]], params: ITDSpectrogramParams) -> tuple[numpy.typing.NDArray[numpy.float64], float, float]")]
fn py_compute_itd_spectrogram_diff<'py>(
    py: Python<'py>,
    reference: [Bound<'py, PyArray1<f64>>; 2],
    test: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyITDSpectrogramParams,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let mut plan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    let left_ref = unsafe {
        NonEmptySlice::new_unchecked(reference[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left reference array must be contiguous float64.",
            )
        })?)
    };
    let right_ref = unsafe {
        NonEmptySlice::new_unchecked(reference[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right reference array must be contiguous float64.",
            )
        })?)
    };
    let left_test = unsafe {
        NonEmptySlice::new_unchecked(test[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left test array must be contiguous float64.",
            )
        })?)
    };
    let right_test = unsafe {
        NonEmptySlice::new_unchecked(test[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right test array must be contiguous float64.",
            )
        })?)
    };

    let (time_diff, mean_deg, mean_itd) = compute_itd_spectrogram_diff(
        [left_ref, right_ref],
        [left_test, right_test],
        &params.inner,
        &mut plan,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to compute ITD diff: {e}"
        ))
    })?;

    Ok((
        PyArray1::from_owned_array(py, time_diff),
        mean_deg,
        mean_itd,
    ))
}

/// Compute the difference between two ILR spectrograms.
///
/// Parameters
/// ----------
/// reference : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel] for the reference signal
/// test : list[numpy.typing.NDArray[numpy.float64]]
///     List containing two 1D arrays [left_channel, right_channel] for the test signal
/// params : ILRSpectrogramParams
///     ILR spectrogram parameters
///
/// Returns
/// -------
/// tuple[numpy.typing.NDArray[numpy.float64], float]
///     Tuple of (ilr_time_diff, mean_diff)
#[pyfunction(name = "compute_ilr_spectrogram_diff")]
#[pyo3(signature = (reference, test, params), text_signature = "(reference: list[numpy.typing.NDArray[numpy.float64]], test: list[numpy.typing.NDArray[numpy.float64]], params: ILRSpectrogramParams) -> tuple[numpy.typing.NDArray[numpy.float64], float]")]
fn py_compute_ilr_spectrogram_diff<'py>(
    py: Python<'py>,
    reference: [Bound<'py, PyArray1<f64>>; 2],
    test: [Bound<'py, PyArray1<f64>>; 2],
    params: &'py PyILRSpectrogramParams,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let mut plan = StftPlan::new(&params.inner.spectrogram_params).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create STFT plan: {e}"
        ))
    })?;

    // todo - change to using PyReadonlyArrays
    let left_ref = unsafe {
        NonEmptySlice::new_unchecked(reference[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left reference array must be contiguous float64.",
            )
        })?)
    };
    let right_ref = unsafe {
        NonEmptySlice::new_unchecked(reference[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right reference array must be contiguous float64.",
            )
        })?)
    };
    let left_test = unsafe {
        NonEmptySlice::new_unchecked(test[0].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Left test array must be contiguous float64.",
            )
        })?)
    };
    let right_test = unsafe {
        NonEmptySlice::new_unchecked(test[1].as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Right test array must be contiguous float64.",
            )
        })?)
    };

    let (time_diff, mean_diff) = compute_ilr_spectrogram_diff(
        [left_ref, right_ref],
        [left_test, right_test],
        &params.inner,
        &mut plan,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to compute ILR diff: {e}"
        ))
    })?;

    Ok((PyArray1::from_owned_array(py, time_diff), mean_diff))
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ITD
    m.add_function(wrap_pyfunction!(py_compute_itd_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_itd_spectrogram_diff, m)?)?;
    m.add_class::<PyITDSpectrogramParams>()?;

    // IPD
    m.add_function(wrap_pyfunction!(py_compute_ipd_spectrogram, m)?)?;
    m.add_class::<PyIPDSpectrogramParams>()?;

    // ILD
    m.add_function(wrap_pyfunction!(py_compute_ild_spectrogram, m)?)?;
    m.add_class::<PyILDSpectrogramParams>()?;

    // ILR
    m.add_function(wrap_pyfunction!(py_compute_ilr_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ilr_spectrogram_diff, m)?)?;
    m.add_class::<PyILRSpectrogramParams>()?;

    Ok(())
}
