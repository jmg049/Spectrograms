//! Python bindings for MDCT/IMDCT.

use std::num::NonZeroUsize;

use pyo3::prelude::*;
use pyo3::types::PyType;

use super::dtype::{Dtype, array2_to_py, parse_dtype, real_2d_owned, vec1_to_py, with_real_1d};
use super::params::PyWindowType;
use super::spectrogram::PyScalar;

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
/// dtype : str, optional
///     Output precision: "float64" (default) or "float32".
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
#[pyo3(signature = (samples, params, dtype=None), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: MdctParams, dtype: str = \"float64\") -> numpy.typing.NDArray[numpy.float64]")]
pub fn py_compute_mdct(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    params: &PyMdctParams,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        samples: &Bound<'_, PyAny>,
        params: &PyMdctParams,
    ) -> PyResult<Py<PyAny>> {
        with_real_1d::<T, _, _>(py, samples, |s| {
            let result = py
                .detach(|| mdct::<T>(s, &params.inner))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
            Ok(array2_to_py(py, result))
        })
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, samples, params),
        Dtype::F64 => run::<f64>(py, samples, params),
    }
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
/// dtype : str, optional
///     Output precision: "float64" (default) or "float32".
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
#[pyo3(signature = (coefficients, params, original_length=None, dtype=None), text_signature = "(coefficients: numpy.typing.NDArray[numpy.float64], params: MdctParams, original_length: int | None = None, dtype: str = \"float64\") -> numpy.typing.NDArray[numpy.float64]")]
pub fn py_compute_imdct(
    py: Python<'_>,
    coefficients: &Bound<'_, PyAny>,
    params: &PyMdctParams,
    original_length: Option<usize>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        coefficients: &Bound<'_, PyAny>,
        params: &PyMdctParams,
        original_length: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let coeffs = real_2d_owned::<T>(py, coefficients)?;
        let result = py
            .detach(|| imdct::<T>(&coeffs, &params.inner, original_length))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(vec1_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, coefficients, params, original_length),
        Dtype::F64 => run::<f64>(py, coefficients, params, original_length),
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMdctParams>()?;
    m.add_function(wrap_pyfunction!(py_compute_mdct, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_imdct, m)?)?;
    Ok(())
}
