//! Python spectrogram result class.

use numpy::PyArray2;
use pyo3::prelude::*;

use crate::{AmpScaleSpec, Spectrogram, SpectrogramParams};

use super::params::PySpectrogramParams;

/// Spectrogram computation result.
///
/// Contains the spectrogram data as a `NumPy` array along with frequency and time axes and the parameters used to create it.
///
#[pyclass(name = "Spectrogram", skip_from_py_object)]
pub struct PySpectrogram {
    py_data: Py<PyArray2<f64>>,
    // Extracted metadata (no longer storing full Spectrogram to avoid duplication)
    frequencies: Vec<f64>,
    times: Vec<f64>,
    params: SpectrogramParams,
    db_range: Option<(f64, f64)>,
}

impl PySpectrogram {
    /// Create a PySpectrogram from computed Rust spectrogram.
    /// Extracts metadata and transfers data ownership to Python (!).
    pub(crate) fn from_spectrogram<FreqScale, AmpScale>(
        py: Python<'_>,
        spec: Spectrogram<FreqScale, AmpScale>,
    ) -> Self
    where
        FreqScale: Copy + Clone + 'static,
        AmpScale: AmpScaleSpec + 'static,
    {
        // Extract metadata before consuming spectrogram
        let frequencies = spec.frequencies().to_vec();
        let times = spec.times().to_vec();
        let params = spec.params().clone();
        let db_range = spec.db_range();

        // Transfer ownership of Array2 to Python (NO COPY!)
        let array = spec.into_data();
        let py_data = PyArray2::from_owned_array(py, array).unbind();

        Self {
            py_data,
            frequencies,
            times,
            params,
            db_range,
        }
    }
}

#[pymethods]
impl PySpectrogram {
    /// Get the spectrogram data as a `NumPy` array.
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D `NumPy` array with shape (`n_bins`, `n_frames`)
    #[getter]
    fn data<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // Return the Python-allocated array (no copy!)
        self.py_data.bind(py).clone()
    }

    /// Get the frequency axis values.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     List of frequency values (Hz or scale-specific units)
    #[getter]
    fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Get the time axis values.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     List of time values in seconds
    #[getter]
    fn times(&self) -> &[f64] {
        &self.times
    }

    /// Get the number of frequency bins.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of frequency bins
    #[getter]
    const fn n_bins(&self) -> usize {
        self.frequencies.len()
    }

    /// Get the number of time frames.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of time frames
    #[getter]
    const fn n_frames(&self) -> usize {
        self.times.len()
    }

    /// Get the shape of the spectrogram.
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    ///     Tuple of (`n_bins`, `n_frames`)
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        (self.n_bins(), self.n_frames())
    }

    /// Get the frequency range.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     Tuple of (`f_min`, `f_max`) in Hz or scale-specific units
    fn frequency_range(&self) -> (f64, f64) {
        if self.frequencies.is_empty() {
            (0.0, 0.0)
        } else {
            (
                self.frequencies[0],
                self.frequencies[self.frequencies.len() - 1],
            )
        }
    }

    /// Get the total duration.
    ///
    /// Returns
    /// -------
    /// float
    ///     Duration in seconds
    fn duration(&self) -> f64 {
        if self.times.is_empty() {
            0.0
        } else {
            self.times[self.times.len() - 1]
        }
    }

    /// Get the decibel range if applicable.
    ///
    /// Returns
    /// -------
    /// tuple[float, float] or None
    ///     Tuple of (`min_db`, `max_db`) for decibel-scaled spectrograms, None otherwise
    const fn db_range(&self) -> Option<(f64, f64)> {
        self.db_range
    }

    /// Get the computation parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     The `SpectrogramParams` used to compute this spectrogram
    #[getter]
    fn params(&self) -> PySpectrogramParams {
        self.params.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Spectrogram(shape=({}, {}))",
            self.n_bins(),
            self.n_frames()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    const fn __len__(&self) -> usize {
        self.n_frames()
    }

    /// Get the transpose of the spectrogram data.
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     Transposed 2D `NumPy` array with shape (`n_frames`, `n_bins`)
    #[getter]
    #[allow(non_snake_case)]
    fn T<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.py_data.bind(py);
        // Use numpy's .T property for transpose
        arr.getattr("T").map(|t| t.clone().into_any())
    }

    #[pyo3(signature = (dtype), text_signature = "($self, dtype)")]
    fn astype<'py>(
        &'py self,
        py: Python<'py>,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.py_data.bind(py);
        arr.call_method1("astype", (dtype,))
            .map(pyo3::Bound::into_any)
    }

    #[pyo3(signature = (dtype=None), text_signature = "($self, dtype=None)")]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.py_data.bind(py);

        if let Some(dt) = dtype {
            // Convert to requested dtype
            let casted = arr.call_method1("astype", (dt,))?;
            Ok(casted)
        } else {
            // Return as-is (f64)
            Ok(arr.clone().into_any())
        }
    }

    fn __getitem__<'py>(
        &'py self,
        py: Python<'py>,
        idx: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let arr = self.py_data.bind(py);
        let sliced: Bound<'py, PyAny> = arr.get_item(idx)?;
        Ok(sliced.unbind())
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

    /// Export the spectrogram data as a DLPack capsule for tensor exchange.
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
    /// >>> spec = sg.compute_mel_power_spectrogram(samples, params, n_mels=128)
    /// >>>
    /// >>> # conversion to PyTorch
    /// >>> tensor = torch.from_dlpack(spec)
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

        // Use the Python-allocated array directly (true !)
        let arr = self.py_data.bind(py).clone();

        create_dlpack_capsule(py, &arr, flags)
    }
}

/// Register the spectrogram class with the Python module.
pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PySpectrogram>()?;
    Ok(())
}
