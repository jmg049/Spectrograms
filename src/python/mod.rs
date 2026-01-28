//! Python bindings for the spectrograms library.
//!
//! This module provides PyO3-based Python bindings that expose the full
//! functionality of the spectrograms library to Python users.

use std::num::NonZeroUsize;

use numpy::PyArray2;
use pyo3::prelude::*;

mod error;
mod fft2d;
mod functions;
mod params;
mod planner;
mod spectrogram;

pub use error::*;

use crate::{Chromagram, python::params::PyChromaParams};

/// Chromagram representation with 12 pitch classes.
///
/// Can act as an numpy array via the `__array__` protocol.
#[pyclass(name = "Chromagram")]
#[derive(Debug, Clone)]
pub struct PyChromagram {
    pub(crate) inner: Chromagram,
}

impl From<Chromagram> for PyChromagram {
    #[inline]
    fn from(inner: Chromagram) -> Self {
        Self { inner }
    }
}

impl From<PyChromagram> for Chromagram {
    #[inline]
    fn from(val: PyChromagram) -> Self {
        val.inner
    }
}

#[pymethods]
impl PyChromagram {
    #[getter]
    fn n_frames(&self) -> NonZeroUsize {
        self.inner.n_frames()
    }

    #[getter]
    fn n_bins(&self) -> NonZeroUsize {
        self.inner.n_bins()
    }

    #[getter]
    fn params(&self) -> PyChromaParams {
        PyChromaParams::from(*self.inner.params())
    }

    #[classattr]
    const fn labels() -> [&'static str; 12] {
        Chromagram::labels()
    }

    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let arr = PyArray2::from_array(py, &self.inner.data);
        if let Some(dtype) = dtype {
            let casted: Bound<'py, PyAny> = arr.call_method1("astype", (dtype,))?;
            Ok(casted.unbind())
        } else {
            Ok(arr.into_any().unbind())
        }
    }
}

/// Register the Python module with `PyO3`.
///
/// This function is called from the `_spectrograms` module definition in lib.rs.
pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception types
    m.add("SpectrogramError", py.get_type::<PySpectrogramError>())?;
    m.add("InvalidInputError", py.get_type::<PyInvalidInputError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type::<PyDimensionMismatchError>(),
    )?;
    m.add("FFTBackendError", py.get_type::<PyFFTBackendError>())?;
    m.add("InternalError", py.get_type::<PyInternalError>())?;

    // Register parameter classes
    params::register(py, m)?;

    // Register spectrogram result class
    spectrogram::register(py, m)?;

    // Register planner and plan classes
    planner::register(py, m)?;

    // Register convenience functions
    functions::register(py, m)?;

    // Register 2D FFT functions and image operations
    fft2d::register(py, m)?;

    // Register FFT plan cache management functions
    #[cfg(feature = "realfft")]
    {
        m.add_function(wrap_pyfunction!(clear_fft_plan_cache, m)?)?;
        m.add_function(wrap_pyfunction!(fft_plan_cache_info, m)?)?;
    }

    Ok(())
}

/// Clear all cached FFT plans to free memory.
///
/// FFT plans are cached globally for performance. This function clears the cache,
/// which can be useful for:
/// - Memory management in long-running applications
/// - Benchmarking (to measure cold vs warm performance)
/// - Testing
///
/// Plans will be automatically recreated on next use.
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> sg.clear_fft_plan_cache()
#[pyfunction]
#[cfg(feature = "realfft")]
fn clear_fft_plan_cache() {
    crate::fft_backend::clear_plan_cache();
}

/// Get information about the FFT plan cache.
///
/// Returns a tuple (forward_plans, inverse_plans) indicating the number of
/// cached forward and inverse FFT plans.
///
/// This is useful for monitoring memory usage and cache effectiveness.
///
/// Returns
/// -------
/// tuple[int, int]
///     (number_of_forward_plans, number_of_inverse_plans)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> signal = np.random.randn(16000)
/// >>> _ = sg.compute_fft(signal)  # Creates a plan
/// >>> forward, inverse = sg.fft_plan_cache_info()
/// >>> print(f"Cached: {forward} forward, {inverse} inverse plans")
#[pyfunction]
#[cfg(feature = "realfft")]
fn fft_plan_cache_info() -> (usize, usize) {
    crate::fft_backend::cache_stats()
}
