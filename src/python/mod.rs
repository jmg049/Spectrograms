//! Python bindings for the spectrograms library.
//!
//! This module provides PyO3-based Python bindings that expose the full
//! functionality of the spectrograms library to Python users.

use std::num::NonZeroUsize;

use numpy::PyArray2;
use pyo3::prelude::*;

mod dlpack;
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
#[pyclass(name = "Chromagram", skip_from_py_object)]
#[derive(Debug)]
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

    /// Return the device type and device ID for DLPack protocol.
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    ///     A tuple of (device_type, device_id). Always returns (1, 0) for CPU.
    ///
    /// Notes
    /// -----
    /// This method is part of the DLPack protocol for tensor exchange.
    /// Device type 1 indicates CPU. This library only supports CPU tensors.
    #[staticmethod]
    const fn __dlpack_device__() -> (i32, i32) {
        (1, 0) // (kDLCPU, device_id=0)
    }

    /// Export the chromagram data as a DLPack capsule for tensor exchange.
    ///
    /// This method implements the DLPack protocol, enabling efficient data sharing with
    /// deep learning frameworks like PyTorch, JAX, and TensorFlow without copying data.
    ///
    /// Parameters
    /// ----------
    /// stream : int, optional
    ///     Must be None for CPU tensors. Provided for protocol compatibility.
    /// max_version : tuple[int, int], optional
    ///     Maximum DLPack version supported by the consumer. Must be >= (1, 0).
    /// dl_device : tuple[int, int], optional
    ///     Target device (device_type, device_id). If specified, must be (1, 0) for CPU.
    /// copy : bool, optional
    ///     If True, create a copy of the data. If False or None (default), return
    ///     a view when possible.
    ///
    /// Returns
    /// -------
    /// PyCapsule
    ///     A DLPack capsule named "dltensor" containing the tensor data.
    ///
    /// Raises
    /// ------
    /// BufferError
    ///     If stream is not None, if the requested device is not CPU, or if the
    ///     requested DLPack version is not supported.
    ///
    /// Examples
    /// --------
    /// >>> import spectrograms as sg
    /// >>> import torch
    /// >>> import numpy as np
    /// >>>
    /// >>> samples = np.random.randn(16000)
    /// >>> stft = sg.StftParams(n_fft=512, hop_size=256, window= sg.WindowType.hanning)
    /// >>> params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    /// >>> spec = sg.compute_cqt_power_spectrogram(samples, params)
    /// >>> chroma = sg.compute_chromagram(spec)
    /// >>>
    /// >>> # conversion to PyTorch
    /// >>> tensor = torch.from_dlpack(chroma)
    /// >>> print(tensor.shape, tensor.dtype)
    ///
    /// Notes
    /// -----
    /// The DLPack protocol enables data exchange between Python array libraries.
    /// The returned capsule can be consumed by frameworks supporting DLPack (PyTorch, JAX,
    /// TensorFlow, etc.) using their respective `from_dlpack()` functions.
    ///
    /// The data remains owned by the Python array until all consumers release it.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
        use crate::python::dlpack::{DLPACK_FLAG_BITMASK_IS_COPIED, create_dlpack_capsule};

        // Validate: stream must be None for CPU
        if stream.is_some() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "stream must be None for CPU tensors",
            ));
        }

        // Validate: version must be >= 1.0
        if let Some((major, minor)) = max_version {
            if major < 1 {
                return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                    "Unsupported DLPack version: {major}.{minor}"
                )));
            }
        }

        // Validate: only CPU device supported
        if let Some((dev_type, dev_id)) = dl_device {
            if dev_type != 1 || dev_id != 0 {
                return Err(pyo3::exceptions::PyBufferError::new_err(
                    "Only CPU device (1, 0) is supported",
                ));
            }
        }

        // Handle copy parameter
        let mut flags = 0u64;
        if copy == Some(true) {
            flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
        }

        // Get the data and create array
        let arr = PyArray2::from_array(py, &self.inner.data);

        create_dlpack_capsule(py, &arr, flags)
    }
}

/// Register the Python module
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

    dlpack::register(py, m)?;

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
