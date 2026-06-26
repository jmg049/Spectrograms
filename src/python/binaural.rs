//! Python bindings for binaural audio spectrograms.
//!
//! Credit to @barrydn for the original implementation of all the spectrograms in this file.
//! Taken from https://github.com/QxLabIreland/Binaspect/

use std::num::NonZeroUsize;

use non_empty_slice::NonEmptySlice;
use pyo3::prelude::*;

use super::dtype::{Dtype, array1_to_py, array2_to_py, parse_dtype, real_1d_vec};
use super::spectrogram::{PyArrayData, PyScalar, real_dlpack};
use crate::binaural::{
    ILDSpectrogramParams, ILRSpectrogramParams, IPDSpectrogramParams, ITDSpectrogramParams,
    IldSpectrogram, IlrSpectrogram, IpdSpectrogram, ItdSpectrogram, compute_ild_spectrogram,
    compute_ilr_spectrogram, compute_ilr_spectrogram_diff, compute_ipd_spectrogram,
    compute_itd_spectrogram, compute_itd_spectrogram_diff,
};
use crate::{StftPlan, python::PySpectrogramParams};

/// Read a two-channel audio argument into owned `Vec<T>` per channel at the
/// requested precision (forcing the numpy dtype via `ascontiguousarray`).
fn read_stereo<T: PyScalar>(
    py: Python<'_>,
    audio: &[Bound<'_, PyAny>; 2],
) -> PyResult<(Vec<T>, Vec<T>)> {
    let left = real_1d_vec::<T>(py, &audio[0])?;
    let right = real_1d_vec::<T>(py, &audio[1])?;
    Ok((left, right))
}

/// Wrap two owned channel buffers as a pair of non-empty slice references.
fn as_stereo_slices<'a, T>(
    left: &'a [T],
    right: &'a [T],
) -> PyResult<(&'a NonEmptySlice<T>, &'a NonEmptySlice<T>)> {
    let left = NonEmptySlice::new(left).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("left audio array must not be empty")
    })?;
    let right = NonEmptySlice::new(right).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("right audio array must not be empty")
    })?;
    Ok((left, right))
}

/// Generate a dual-dtype Python result class wrapping one of the binaural
/// spectrogram result structs (`ItdSpectrogram` / `IpdSpectrogram` / ...).
///
/// The full typed Rust result is retained (in an `f32`/`f64` enum) so that the
/// `histogram(...)` method — which needs the native data — stays reachable while
/// `data` / `__array__` / `__dlpack__` still report the dtype-correct array.
/// Metadata (frequencies, times, params) is `f64` and identical across variants.
///
/// The per-type `histogram` signature differs (range keyword + optional
/// `exponent`), so it is supplied by the caller as a trailing method item.
macro_rules! binaural_result_class {
    (
        $enum:ident,
        $py_struct:ident,
        $py_name:literal,
        $rust:ident,
        $py_params:ident,
        $hist:item
    ) => {
        /// Owned binaural result in the precision requested at compute time.
        pub(crate) enum $enum {
            F32($rust<f32>),
            F64($rust<f64>),
        }

        impl From<$rust<f32>> for $enum {
            #[inline]
            fn from(s: $rust<f32>) -> Self {
                Self::F32(s)
            }
        }

        impl From<$rust<f64>> for $enum {
            #[inline]
            fn from(s: $rust<f64>) -> Self {
                Self::F64(s)
            }
        }

        #[doc = concat!($py_name, " computation result.")]
        ///
        /// Carries the spectrogram matrix as a native-precision NumPy array
        /// (`float32` or `float64`, see `dtype`) along with the frequency and
        /// time axes and the parameters used to compute it. The object is
        /// array-compatible via `__array__` / `__dlpack__`, and exposes the
        /// `histogram(...)` method over the underlying values.
        #[pyclass(name = $py_name, skip_from_py_object)]
        pub struct $py_struct {
            inner: $enum,
        }

        impl $py_struct {
            /// Build the Python result from a computed Rust result, retaining the
            /// native-precision data (no copy of metadata).
            pub(crate) fn from_result<T>(_py: Python<'_>, result: $rust<T>) -> Self
            where
                $rust<T>: Into<$enum>,
            {
                Self {
                    inner: result.into(),
                }
            }

            /// Move a fresh, dtype-correct copy of the data onto the Python heap.
            fn data_any<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
                match &self.inner {
                    $enum::F32(s) => {
                        numpy::PyArray2::from_owned_array(py, s.data.clone()).into_any()
                    }
                    $enum::F64(s) => {
                        numpy::PyArray2::from_owned_array(py, s.data.clone()).into_any()
                    }
                }
            }

            /// Wrap a fresh, dtype-correct copy of the data for DLPack export.
            fn array_data(&self, py: Python<'_>) -> PyArrayData {
                match &self.inner {
                    $enum::F32(s) => <f32 as PyScalar>::into_array_data(py, s.data.clone()),
                    $enum::F64(s) => <f64 as PyScalar>::into_array_data(py, s.data.clone()),
                }
            }
        }

        #[pymethods]
        impl $py_struct {
            /// Spectrogram matrix as a NumPy array with shape (`n_bins`, `n_frames`).
            #[getter]
            fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
                self.data_any(py)
            }

            /// NumPy dtype name of the stored data (`"float32"` / `"float64"`).
            #[getter]
            fn dtype(&self) -> &'static str {
                match &self.inner {
                    $enum::F32(_) => "float32",
                    $enum::F64(_) => "float64",
                }
            }

            /// Number of frequency bins (rows).
            #[getter]
            fn n_bins(&self) -> usize {
                match &self.inner {
                    $enum::F32(s) => s.n_bins().get(),
                    $enum::F64(s) => s.n_bins().get(),
                }
            }

            /// Number of time frames (columns).
            #[getter]
            fn n_frames(&self) -> usize {
                match &self.inner {
                    $enum::F32(s) => s.n_frames().get(),
                    $enum::F64(s) => s.n_frames().get(),
                }
            }

            /// Shape of the matrix as (`n_bins`, `n_frames`).
            #[getter]
            fn shape(&self) -> (usize, usize) {
                (self.n_bins(), self.n_frames())
            }

            /// Frequency axis values in Hz.
            #[getter]
            fn frequencies(&self) -> Vec<f64> {
                match &self.inner {
                    $enum::F32(s) => s.frequencies().to_vec(),
                    $enum::F64(s) => s.frequencies().to_vec(),
                }
            }

            /// Time axis values in seconds.
            #[getter]
            fn times(&self) -> Vec<f64> {
                match &self.inner {
                    $enum::F32(s) => s.times().to_vec(),
                    $enum::F64(s) => s.times().to_vec(),
                }
            }

            /// Frequency range as (`f_min`, `f_max`) in Hz.
            fn frequency_range(&self) -> (f64, f64) {
                match &self.inner {
                    $enum::F32(s) => s.frequency_range(),
                    $enum::F64(s) => s.frequency_range(),
                }
            }

            /// Total duration spanned by the time axis in seconds.
            fn duration(&self) -> f64 {
                match &self.inner {
                    $enum::F32(s) => s.duration(),
                    $enum::F64(s) => s.duration(),
                }
            }

            /// The parameters used to compute this spectrogram.
            #[getter]
            fn params(&self) -> $py_params {
                match &self.inner {
                    $enum::F32(s) => $py_params {
                        inner: s.params().clone(),
                    },
                    $enum::F64(s) => $py_params {
                        inner: s.params().clone(),
                    },
                }
            }

            #[pyo3(signature = (dtype=None))]
            fn __array__<'py>(
                &self,
                py: Python<'py>,
                dtype: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let arr = self.data_any(py);
                if let Some(dt) = dtype {
                    arr.call_method1("astype", (dt,))
                } else {
                    Ok(arr)
                }
            }

            /// Return the device type and device ID for DLPack protocol.
            #[staticmethod]
            const fn __dlpack_device__() -> (i32, i32) {
                (1, 0) // (kDLCPU, device_id=0)
            }

            /// Export the data as a DLPack capsule for tensor exchange.
            #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
            fn __dlpack__<'py>(
                &self,
                py: Python<'py>,
                stream: Option<&Bound<'py, PyAny>>,
                max_version: Option<(u32, u32)>,
                dl_device: Option<(i32, i32)>,
                copy: Option<bool>,
            ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
                let data = self.array_data(py);
                real_dlpack(py, &data, stream, max_version, dl_device, copy)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(shape=({}, {}), dtype={})",
                    $py_name,
                    self.n_bins(),
                    self.n_frames(),
                    self.dtype(),
                )
            }

            $hist
        }
    };
}

binaural_result_class!(
    ItdInner,
    PyItdSpectrogram,
    "ItdSpectrogram",
    ItdSpectrogram,
    PyITDSpectrogramParams,
    /// Compute a 2D histogram of ITD values over time.
    ///
    /// Parameters
    /// ----------
    /// num_bins : int, optional
    ///     Number of histogram bins (default: 400).
    /// delay_range : tuple[float, float], optional
    ///     Range of delays in seconds as (min, max) (default: (-0.00088, 0.00088)).
    /// energy_weighted : bool, optional
    ///     If True, weight by energy (default: False).
    /// normalize : bool, optional
    ///     If True, normalize each time frame to sum to 1 (default: False).
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D float64 array with shape (num_bins, n_frames).
    #[pyo3(signature = (num_bins=None, delay_range=None, energy_weighted=false, normalize=false))]
    fn histogram<'py>(
        &self,
        py: Python<'py>,
        num_bins: Option<usize>,
        delay_range: Option<(f64, f64)>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Bound<'py, PyAny> {
        let num_bins = num_bins.and_then(NonZeroUsize::new);
        let hist = match &self.inner {
            ItdInner::F32(s) => s.histogram(num_bins, delay_range, energy_weighted, normalize),
            ItdInner::F64(s) => s.histogram(num_bins, delay_range, energy_weighted, normalize),
        };
        array2_to_py(py, hist).into_bound(py)
    }
);

binaural_result_class!(
    IpdInner,
    PyIpdSpectrogram,
    "IpdSpectrogram",
    IpdSpectrogram,
    PyIPDSpectrogramParams,
    /// Compute a 2D histogram of IPD values over time.
    ///
    /// Parameters
    /// ----------
    /// num_bins : int, optional
    ///     Number of histogram bins (default: 400).
    /// phase_range : tuple[float, float], optional
    ///     Range of phase in radians as (min, max) (default: (-pi, pi)).
    /// energy_weighted : bool, optional
    ///     If True, weight by energy (default: False).
    /// normalize : bool, optional
    ///     If True, normalize each time frame to sum to 1 (default: False).
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D float64 array with shape (num_bins, n_frames).
    #[pyo3(signature = (num_bins=None, phase_range=None, energy_weighted=false, normalize=false))]
    fn histogram<'py>(
        &self,
        py: Python<'py>,
        num_bins: Option<usize>,
        phase_range: Option<(f64, f64)>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Bound<'py, PyAny> {
        let num_bins = num_bins.and_then(NonZeroUsize::new);
        let hist = match &self.inner {
            IpdInner::F32(s) => s.histogram(num_bins, phase_range, energy_weighted, normalize),
            IpdInner::F64(s) => s.histogram(num_bins, phase_range, energy_weighted, normalize),
        };
        array2_to_py(py, hist).into_bound(py)
    }
);

binaural_result_class!(
    IldInner,
    PyIldSpectrogram,
    "IldSpectrogram",
    IldSpectrogram,
    PyILDSpectrogramParams,
    /// Compute a 2D histogram of ILD values over time.
    ///
    /// Parameters
    /// ----------
    /// num_bins : int, optional
    ///     Number of histogram bins (default: 400).
    /// db_range : tuple[float, float], optional
    ///     Range of dB values as (min, max) (default: (-24, 24)).
    /// exponent : int, optional
    ///     Power to raise histogram values to, to enhance peaks (default: 3).
    /// energy_weighted : bool, optional
    ///     If True, weight by energy (default: False).
    /// normalize : bool, optional
    ///     If True, normalize each time frame to sum to 1 (default: False).
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D float64 array with shape (num_bins, n_frames).
    #[pyo3(signature = (num_bins=None, db_range=None, exponent=None, energy_weighted=false, normalize=false))]
    fn histogram<'py>(
        &self,
        py: Python<'py>,
        num_bins: Option<usize>,
        db_range: Option<(f64, f64)>,
        exponent: Option<i32>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Bound<'py, PyAny> {
        let num_bins = num_bins.and_then(NonZeroUsize::new);
        let hist = match &self.inner {
            IldInner::F32(s) => {
                s.histogram(num_bins, db_range, exponent, energy_weighted, normalize)
            }
            IldInner::F64(s) => {
                s.histogram(num_bins, db_range, exponent, energy_weighted, normalize)
            }
        };
        array2_to_py(py, hist).into_bound(py)
    }
);

binaural_result_class!(
    IlrInner,
    PyIlrSpectrogram,
    "IlrSpectrogram",
    IlrSpectrogram,
    PyILRSpectrogramParams,
    /// Compute a 2D histogram of ILR values over time.
    ///
    /// Parameters
    /// ----------
    /// num_bins : int, optional
    ///     Number of histogram bins (default: 400).
    /// ratio_range : tuple[float, float], optional
    ///     Range of ratio values as (min, max) (default: (-1, 1)).
    /// exponent : int, optional
    ///     Power to raise histogram values to, to enhance peaks (default: 3).
    /// energy_weighted : bool, optional
    ///     If True, weight by energy (default: False).
    /// normalize : bool, optional
    ///     If True, normalize each time frame to sum to 1 (default: False).
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D float64 array with shape (num_bins, n_frames).
    #[pyo3(signature = (num_bins=None, ratio_range=None, exponent=None, energy_weighted=false, normalize=false))]
    fn histogram<'py>(
        &self,
        py: Python<'py>,
        num_bins: Option<usize>,
        ratio_range: Option<(f64, f64)>,
        exponent: Option<i32>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Bound<'py, PyAny> {
        let num_bins = num_bins.and_then(NonZeroUsize::new);
        let hist = match &self.inner {
            IlrInner::F32(s) => {
                s.histogram(num_bins, ratio_range, exponent, energy_weighted, normalize)
            }
            IlrInner::F64(s) => {
                s.histogram(num_bins, ratio_range, exponent, energy_weighted, normalize)
            }
        };
        array2_to_py(py, hist).into_bound(py)
    }
);

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
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "ITDSpectrogramParams", dtype: "str" = None), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ITDSpectrogramParams, dtype: str = \"float64\") -> ItdSpectrogram")]
fn py_compute_itd_spectrogram(
    py: Python<'_>,
    audio: [Bound<'_, PyAny>; 2],
    params: &PyITDSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<PyItdSpectrogram> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        audio: &[Bound<'_, PyAny>; 2],
        params: &PyITDSpectrogramParams,
    ) -> PyResult<PyItdSpectrogram>
    where
        ItdSpectrogram<T>: Into<ItdInner>,
    {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (left, right) = read_stereo::<T>(py, audio)?;
        let (left_s, right_s) = as_stereo_slices(&left, &right)?;
        let itd =
            compute_itd_spectrogram([left_s, right_s], &params.inner, &mut plan).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute ITD spectrogram: {e}"
                ))
            })?;
        Ok(PyItdSpectrogram::from_result(py, itd))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &audio, params),
        Dtype::F64 => run::<f64>(py, &audio, params),
    }
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
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "IPDSpectrogramParams", dtype: "str" = None), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: IPDSpectrogramParams, dtype: str = \"float64\") -> IpdSpectrogram")]
fn py_compute_ipd_spectrogram(
    py: Python<'_>,
    audio: [Bound<'_, PyAny>; 2],
    params: &PyIPDSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<PyIpdSpectrogram> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        audio: &[Bound<'_, PyAny>; 2],
        params: &PyIPDSpectrogramParams,
    ) -> PyResult<PyIpdSpectrogram>
    where
        IpdSpectrogram<T>: Into<IpdInner>,
    {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (left, right) = read_stereo::<T>(py, audio)?;
        let (left_s, right_s) = as_stereo_slices(&left, &right)?;
        let ipd =
            compute_ipd_spectrogram([left_s, right_s], &params.inner, &mut plan).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute IPD spectrogram: {e}"
                ))
            })?;
        Ok(PyIpdSpectrogram::from_result(py, ipd))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &audio, params),
        Dtype::F64 => run::<f64>(py, &audio, params),
    }
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
#[pyo3(signature = (audio: "list[numpy.typing.NDArray[numpy.float64]]", params: "ILDSpectrogramParams", dtype: "str" = None), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ILDSpectrogramParams, dtype: str = \"float64\") -> IldSpectrogram")]
fn py_compute_ild_spectrogram(
    py: Python<'_>,
    audio: [Bound<'_, PyAny>; 2],
    params: &PyILDSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<PyIldSpectrogram> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        audio: &[Bound<'_, PyAny>; 2],
        params: &PyILDSpectrogramParams,
    ) -> PyResult<PyIldSpectrogram>
    where
        IldSpectrogram<T>: Into<IldInner>,
    {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (left, right) = read_stereo::<T>(py, audio)?;
        let (left_s, right_s) = as_stereo_slices(&left, &right)?;
        let ild =
            compute_ild_spectrogram([left_s, right_s], &params.inner, &mut plan).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute ILD spectrogram: {e}"
                ))
            })?;
        Ok(PyIldSpectrogram::from_result(py, ild))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &audio, params),
        Dtype::F64 => run::<f64>(py, &audio, params),
    }
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
#[pyo3(signature = (audio, params, dtype=None), text_signature = "(audio: list[numpy.typing.NDArray[numpy.float64]], params: ILRSpectrogramParams, dtype: str = \"float64\") -> IlrSpectrogram")]
fn py_compute_ilr_spectrogram(
    py: Python<'_>,
    audio: [Bound<'_, PyAny>; 2],
    params: &PyILRSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<PyIlrSpectrogram> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        audio: &[Bound<'_, PyAny>; 2],
        params: &PyILRSpectrogramParams,
    ) -> PyResult<PyIlrSpectrogram>
    where
        IlrSpectrogram<T>: Into<IlrInner>,
    {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (left, right) = read_stereo::<T>(py, audio)?;
        let (left_s, right_s) = as_stereo_slices(&left, &right)?;
        let ilr =
            compute_ilr_spectrogram([left_s, right_s], &params.inner, &mut plan).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to compute ILR spectrogram: {e}"
                ))
            })?;
        Ok(PyIlrSpectrogram::from_result(py, ilr))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &audio, params),
        Dtype::F64 => run::<f64>(py, &audio, params),
    }
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
#[pyo3(signature = (reference, test, params, dtype=None), text_signature = "(reference: list[numpy.typing.NDArray[numpy.float64]], test: list[numpy.typing.NDArray[numpy.float64]], params: ITDSpectrogramParams, dtype: str = \"float64\") -> tuple[numpy.typing.NDArray[numpy.float64], float, float]")]
fn py_compute_itd_spectrogram_diff(
    py: Python<'_>,
    reference: [Bound<'_, PyAny>; 2],
    test: [Bound<'_, PyAny>; 2],
    params: &PyITDSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<(Py<PyAny>, f64, f64)> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        reference: &[Bound<'_, PyAny>; 2],
        test: &[Bound<'_, PyAny>; 2],
        params: &PyITDSpectrogramParams,
    ) -> PyResult<(Py<PyAny>, f64, f64)> {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (lr, rr) = read_stereo::<T>(py, reference)?;
        let (lt, rt) = read_stereo::<T>(py, test)?;
        let (lr_s, rr_s) = as_stereo_slices(&lr, &rr)?;
        let (lt_s, rt_s) = as_stereo_slices(&lt, &rt)?;

        let (time_diff, mean_deg, mean_itd) =
            compute_itd_spectrogram_diff([lr_s, rr_s], [lt_s, rt_s], &params.inner, &mut plan)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to compute ITD diff: {e}"
                    ))
                })?;

        Ok((
            array1_to_py(py, time_diff),
            mean_deg.to_f64().unwrap_or(f64::NAN),
            mean_itd.to_f64().unwrap_or(f64::NAN),
        ))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &reference, &test, params),
        Dtype::F64 => run::<f64>(py, &reference, &test, params),
    }
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
#[pyo3(signature = (reference, test, params, dtype=None), text_signature = "(reference: list[numpy.typing.NDArray[numpy.float64]], test: list[numpy.typing.NDArray[numpy.float64]], params: ILRSpectrogramParams, dtype: str = \"float64\") -> tuple[numpy.typing.NDArray[numpy.float64], float]")]
fn py_compute_ilr_spectrogram_diff(
    py: Python<'_>,
    reference: [Bound<'_, PyAny>; 2],
    test: [Bound<'_, PyAny>; 2],
    params: &PyILRSpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<(Py<PyAny>, f64)> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        reference: &[Bound<'_, PyAny>; 2],
        test: &[Bound<'_, PyAny>; 2],
        params: &PyILRSpectrogramParams,
    ) -> PyResult<(Py<PyAny>, f64)> {
        let mut plan = StftPlan::<T>::new(&params.inner.spectrogram_params).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create STFT plan: {e}"
            ))
        })?;
        let (lr, rr) = read_stereo::<T>(py, reference)?;
        let (lt, rt) = read_stereo::<T>(py, test)?;
        let (lr_s, rr_s) = as_stereo_slices(&lr, &rr)?;
        let (lt_s, rt_s) = as_stereo_slices(&lt, &rt)?;

        let (time_diff, mean_diff) =
            compute_ilr_spectrogram_diff([lr_s, rr_s], [lt_s, rt_s], &params.inner, &mut plan)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to compute ILR diff: {e}"
                    ))
                })?;

        Ok((
            array1_to_py(py, time_diff),
            mean_diff.to_f64().unwrap_or(f64::NAN),
        ))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, &reference, &test, params),
        Dtype::F64 => run::<f64>(py, &reference, &test, params),
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ITD
    m.add_function(wrap_pyfunction!(py_compute_itd_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_itd_spectrogram_diff, m)?)?;
    m.add_class::<PyITDSpectrogramParams>()?;
    m.add_class::<PyItdSpectrogram>()?;

    // IPD
    m.add_function(wrap_pyfunction!(py_compute_ipd_spectrogram, m)?)?;
    m.add_class::<PyIPDSpectrogramParams>()?;
    m.add_class::<PyIpdSpectrogram>()?;

    // ILD
    m.add_function(wrap_pyfunction!(py_compute_ild_spectrogram, m)?)?;
    m.add_class::<PyILDSpectrogramParams>()?;
    m.add_class::<PyIldSpectrogram>()?;

    // ILR
    m.add_function(wrap_pyfunction!(py_compute_ilr_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ilr_spectrogram_diff, m)?)?;
    m.add_class::<PyILRSpectrogramParams>()?;
    m.add_class::<PyIlrSpectrogram>()?;

    Ok(())
}
