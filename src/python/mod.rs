//! Python bindings for the spectrograms library.
//!
//! This module provides PyO3-based Python bindings that expose the full
//! functionality of the spectrograms library to Python users.

use pyo3::prelude::*;

mod binaural;
mod dlpack;
mod dtype;
mod error;
mod fft2d;
mod functions;
mod mdct;
mod mfcc;
mod params;
mod planner;
mod spectrogram;

use crate::{ChromaParams, Chromagram};
use spectrogram::{PyArrayData, PyScalar, real_dlpack};

pub use error::*;
pub use mfcc::PyMfcc;
pub use params::*;

/// Chromagram (pitch class profile) result.
///
/// Carries the chroma feature matrix as a native-precision NumPy array
/// (`float32` or `float64`, see :attr:`dtype`) plus the parameters used to
/// compute it. The data lives on the Python heap so it can be shared zero-copy
/// with array libraries via ``__array__`` / ``__dlpack__``.
#[pyclass(name = "Chromagram", skip_from_py_object)]
pub struct PyChromagram {
    data: PyArrayData,
    params: ChromaParams,
    n_bins: usize,
    n_frames: usize,
}

impl PyChromagram {
    /// Build a `PyChromagram` from a computed Rust [`Chromagram`], moving its
    /// data onto the Python heap (no copy).
    pub(crate) fn from_chromagram<T: PyScalar>(py: Python<'_>, chroma: Chromagram<T>) -> Self {
        let n_bins = chroma.data.nrows();
        let n_frames = chroma.data.ncols();
        let params = *chroma.params();
        let data = T::into_array_data(py, chroma.data);
        Self {
            data,
            params,
            n_bins,
            n_frames,
        }
    }
}

#[pymethods]
impl PyChromagram {
    /// Chroma feature matrix as a NumPy array with shape (`n_bins`, `n_frames`).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.data.bind(py)
    }

    /// NumPy dtype name of the stored data (`"float32"` / `"float64"`).
    #[getter]
    const fn dtype(&self) -> &'static str {
        self.data.dtype()
    }

    #[getter]
    const fn n_frames(&self) -> usize {
        self.n_frames
    }

    #[getter]
    const fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Shape of the chroma matrix as (`n_bins`, `n_frames`).
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        (self.n_bins, self.n_frames)
    }

    #[getter]
    fn params(&self) -> PyChromaParams {
        PyChromaParams::from(self.params)
    }

    #[classattr]
    fn labels() -> [&'static str; 12] {
        Chromagram::<f64>::labels()
    }

    #[pyo3(signature = (dtype=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.data.bind(py);
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

    /// Export the chromagram data as a DLPack capsule for tensor exchange.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
        real_dlpack(py, &self.data, stream, max_version, dl_device, copy)
    }

    fn __repr__(&self) -> String {
        format!(
            "Chromagram(shape=({}, {}), dtype={})",
            self.n_bins,
            self.n_frames,
            self.dtype()
        )
    }
}

/// Register the Python module
#[inline]
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

    // Register the chromagram and MFCC result classes
    m.add_class::<PyChromagram>()?;
    m.add_class::<PyMfcc>()?;

    // Register planner and plan classes
    planner::register(py, m)?;

    // Register convenience functions
    functions::register(py, m)?;

    // Register 2D FFT functions and image operations
    fft2d::register(py, m)?;

    dlpack::register(py, m)?;

    binaural::register(py, m)?;

    mdct::register(py, m)?;

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
