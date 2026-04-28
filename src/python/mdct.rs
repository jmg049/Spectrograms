//! Python bindings for MDCT/IMDCT.

use std::num::NonZeroUsize;

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

use super::params::PyWindowType;
use non_empty_slice::NonEmptySlice;

use crate::mdct::{MdctParams, imdct, mdct};

/// Parameters for MDCT computation.
///
/// The Modified Discrete Cosine Transform (MDCT) is a lapped orthogonal transform
/// used in audio codecs (MP3, AAC, Vorbis, Opus).
#[pyclass(name = "MdctParams", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMdctParams {
    pub(crate) inner: MdctParams,
}

#[pymethods]
impl PyMdctParams {
    /// Create MDCT parameters.
    ///
    /// Parameters
    /// ----------
    /// window_size : int
    ///     Total window size (2N). Must be even and >= 4.
    /// hop_size : int
    ///     Hop size between frames.
    /// window : WindowType
    ///     Window function.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If window_size is odd, less than 4, or hop_size is zero.
    #[new]
    #[pyo3(signature = (window_size: "int", hop_size: "int", window: "WindowType"), text_signature = "(window_size: int, hop_size: int, window: WindowType) -> MdctParams")]
    fn new(window_size: usize, hop_size: usize, window: PyWindowType) -> PyResult<Self> {
        let ws = NonZeroUsize::new(window_size)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("window_size must be > 0"))?;
        let hs = NonZeroUsize::new(hop_size)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("hop_size must be > 0"))?;
        let inner = MdctParams::new(ws, hs, window.into_inner())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Create parameters with a sine window and 50% hop for perfect reconstruction.
    ///
    /// Parameters
    /// ----------
    /// window_size : int
    ///     Total window size (2N). Must be even and >= 4.
    ///
    /// Returns
    /// -------
    /// MdctParams
    ///     Parameters configured for perfect reconstruction.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If window_size is odd or less than 4.
    #[classmethod]
    #[pyo3(signature = (window_size: "int"), text_signature = "(window_size: int) -> MdctParams")]
    fn sine_window(_cls: &Bound<'_, PyType>, window_size: usize) -> PyResult<Self> {
        let ws = NonZeroUsize::new(window_size)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("window_size must be > 0"))?;
        let inner = MdctParams::sine_window(ws)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Total window size (= 2N).
    #[getter]
    const fn window_size(&self) -> usize {
        self.inner.window_size.get()
    }

    /// Hop size between consecutive frames.
    #[getter]
    const fn hop_size(&self) -> usize {
        self.inner.hop_size.get()
    }

    /// Number of MDCT coefficients per frame (= window_size // 2).
    #[getter]
    const fn n_coefficients(&self) -> usize {
        self.inner.n_coefficients()
    }

    fn __repr__(&self) -> String {
        format!(
            "MdctParams(window_size={}, hop_size={}, n_coefficients={})",
            self.inner.window_size.get(),
            self.inner.hop_size.get(),
            self.inner.n_coefficients()
        )
    }
}

/// Compute the MDCT of an audio signal.
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     1D array of real audio samples. Length must be >= window_size.
/// params : MdctParams
///     MDCT parameters.
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     2D array of shape (N, n_frames) where N = window_size // 2.
///
/// Raises
/// ------
/// ValueError
///     If samples is too short or parameters are invalid.
#[pyfunction(name = "mdct")]
#[pyo3(signature = (samples, params), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: MdctParams) -> numpy.typing.NDArray[numpy.float64]")]
pub fn py_compute_mdct<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    params: &PyMdctParams,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (samples, "float64"))?;
    let arr = arr.cast::<numpy::PyArray1<f64>>()?;
    let ro = arr.try_readonly()?;
    let slice = ro.as_slice()?;

    let samples_ne = NonEmptySlice::new(slice)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("samples must not be empty"))?;

    let result = py
        .detach(|| mdct(samples_ne, &params.inner))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Compute the IMDCT (inverse MDCT) from MDCT coefficients.
///
/// Parameters
/// ----------
/// coefficients : numpy.typing.NDArray[numpy.float64]
///     2D array of shape (N, n_frames) as returned by compute_mdct.
/// params : MdctParams
///     MDCT parameters (must match those used for analysis).
/// original_length : int, optional
///     If provided, output is truncated to this length.
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     1D array of reconstructed audio samples.
///
/// Raises
/// ------
/// ValueError
///     If coefficients shape doesn't match params.
#[pyfunction(name = "imdct")]
#[pyo3(signature = (coefficients, params, original_length=None), text_signature = "(coefficients: numpy.typing.NDArray[numpy.float64], params: MdctParams, original_length: int | None = None) -> numpy.typing.NDArray[numpy.float64]")]
pub fn py_compute_imdct<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray2<f64>,
    params: &PyMdctParams,
    original_length: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coeffs_arr: Array2<f64> = coefficients.as_array().to_owned();
    let result = py
        .detach(|| imdct(&coeffs_arr, &params.inner, original_length))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(PyArray1::from_vec(py, result))
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMdctParams>()?;
    m.add_function(wrap_pyfunction!(py_compute_mdct, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_imdct, m)?)?;
    Ok(())
}
