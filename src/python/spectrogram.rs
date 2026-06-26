//! Python spectrogram result class.

use ndarray::Array2;
use num_complex::Complex;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::{AmpScaleSpec, Spectrogram, SpectrogramParams};

use super::params::PySpectrogramParams;

/// Owned, Python-allocated spectrogram data in either single or double precision.
///
/// The variant is chosen at compute time based on the requested `dtype`. The
/// underlying `numpy` array lives on the Python heap so it can be shared with
/// downstream frameworks with no extra copy (see DLPack support below).
pub(crate) enum PyArrayData {
    F32(Py<PyArray2<f32>>),
    F64(Py<PyArray2<f64>>),
}

impl PyArrayData {
    /// NumPy dtype name for the stored array (`"float32"` / `"float64"`).
    pub(crate) const fn dtype(&self) -> &'static str {
        match self {
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
        }
    }

    /// Bind the stored array and erase its element type, returning a fresh
    /// `Bound` handle valid for `py`.
    pub(crate) fn bind<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Self::F32(a) => a.bind(py).clone().into_any(),
            Self::F64(a) => a.bind(py).clone().into_any(),
        }
    }
}

/// Owned, Python-allocated *complex* result data in either single or double
/// precision (`complex64` / `complex128`).
///
/// Mirrors [`PyArrayData`] for complex-valued results (e.g. the raw STFT). The
/// underlying `numpy` array lives on the Python heap for zero-copy sharing.
pub(crate) enum PyComplexArrayData {
    C64(Py<PyArray2<Complex<f32>>>),
    C128(Py<PyArray2<Complex<f64>>>),
}

impl PyComplexArrayData {
    /// Real-precision dtype name backing the complex array (`"float32"` /
    /// `"float64"`), matching [`PyScalar::DTYPE`].
    pub(crate) const fn dtype(&self) -> &'static str {
        match self {
            Self::C64(_) => "float32",
            Self::C128(_) => "float64",
        }
    }

    /// Bind the stored array and erase its element type.
    pub(crate) fn bind<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Self::C64(a) => a.bind(py).clone().into_any(),
            Self::C128(a) => a.bind(py).clone().into_any(),
        }
    }
}

/// Validate the DLPack protocol arguments shared by every result class and
/// return the capsule flags to use (`IS_COPIED` when `copy == Some(true)`).
fn validate_dlpack_args(
    stream: Option<&Bound<'_, PyAny>>,
    max_version: Option<(u32, u32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<u64> {
    use crate::python::dlpack::DLPACK_FLAG_BITMASK_IS_COPIED;

    if stream.is_some() {
        return Err(pyo3::exceptions::PyBufferError::new_err(
            "stream must be None for CPU tensors",
        ));
    }
    if let Some((major, minor)) = max_version {
        if major < 1 {
            return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                "Unsupported DLPack version: {major}.{minor}"
            )));
        }
    }
    if let Some((dev_type, dev_id)) = dl_device {
        if dev_type != 1 || dev_id != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Only CPU device (1, 0) is supported",
            ));
        }
    }
    let mut flags = 0u64;
    if copy == Some(true) {
        flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
    }
    Ok(flags)
}

/// Build a DLPack capsule for a real-valued (`float32`/`float64`) result array,
/// reporting the native element type. Shared by every real result class.
pub(crate) fn real_dlpack<'py>(
    py: Python<'py>,
    data: &PyArrayData,
    stream: Option<&Bound<'py, PyAny>>,
    max_version: Option<(u32, u32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
    use crate::python::dlpack::{DLDataType, DLDataTypeCode, create_dlpack_capsule};

    let flags = validate_dlpack_args(stream, max_version, dl_device, copy)?;
    match data {
        PyArrayData::F32(a) => create_dlpack_capsule(
            py,
            &a.bind(py).clone(),
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
            flags,
        ),
        PyArrayData::F64(a) => create_dlpack_capsule(
            py,
            &a.bind(py).clone(),
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 64,
                lanes: 1,
            },
            flags,
        ),
    }
}

/// Build a DLPack capsule for a complex-valued (`complex64`/`complex128`)
/// result array. Used by [`super::params::PyStftResult`].
pub(crate) fn complex_dlpack<'py>(
    py: Python<'py>,
    data: &PyComplexArrayData,
    stream: Option<&Bound<'py, PyAny>>,
    max_version: Option<(u32, u32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
    use crate::python::dlpack::{DLDataType, DLDataTypeCode, create_dlpack_capsule};

    let flags = validate_dlpack_args(stream, max_version, dl_device, copy)?;
    match data {
        PyComplexArrayData::C64(a) => create_dlpack_capsule(
            py,
            &a.bind(py).clone(),
            DLDataType {
                code: DLDataTypeCode::kDLComplex as u8,
                bits: 64,
                lanes: 1,
            },
            flags,
        ),
        PyComplexArrayData::C128(a) => create_dlpack_capsule(
            py,
            &a.bind(py).clone(),
            DLDataType {
                code: DLDataTypeCode::kDLComplex as u8,
                bits: 128,
                lanes: 1,
            },
            flags,
        ),
    }
}

/// Sealed helper trait tying a Rust scalar to its `numpy` dtype and array
/// constructor. Implemented for `f32` and `f64` only.
pub(crate) trait PyScalar: crate::Sample + numpy::Element {
    /// NumPy dtype name (`"float32"` / `"float64"`).
    const DTYPE: &'static str;

    /// NumPy complex dtype name pairing with `DTYPE` (`"complex64"` /
    /// `"complex128"`).
    const COMPLEX_DTYPE: &'static str;

    /// Move an owned `Array2<Self>` onto the Python heap, wrapped in the
    /// matching [`PyArrayData`] variant (no copy).
    fn into_array_data(py: Python<'_>, a: Array2<Self>) -> PyArrayData;

    /// Move an owned `Array2<Complex<Self>>` onto the Python heap, wrapped in
    /// the matching [`PyComplexArrayData`] variant (no copy).
    fn into_complex_array_data(py: Python<'_>, a: Array2<Complex<Self>>) -> PyComplexArrayData;
}

impl PyScalar for f32 {
    const DTYPE: &'static str = "float32";
    const COMPLEX_DTYPE: &'static str = "complex64";

    fn into_array_data(py: Python<'_>, a: Array2<Self>) -> PyArrayData {
        PyArrayData::F32(PyArray2::from_owned_array(py, a).unbind())
    }

    fn into_complex_array_data(py: Python<'_>, a: Array2<Complex<Self>>) -> PyComplexArrayData {
        PyComplexArrayData::C64(PyArray2::from_owned_array(py, a).unbind())
    }
}

impl PyScalar for f64 {
    const DTYPE: &'static str = "float64";
    const COMPLEX_DTYPE: &'static str = "complex128";

    fn into_array_data(py: Python<'_>, a: Array2<Self>) -> PyArrayData {
        PyArrayData::F64(PyArray2::from_owned_array(py, a).unbind())
    }

    fn into_complex_array_data(py: Python<'_>, a: Array2<Complex<Self>>) -> PyComplexArrayData {
        PyComplexArrayData::C128(PyArray2::from_owned_array(py, a).unbind())
    }
}

/// Spectrogram computation result.
///
/// Contains the spectrogram data as a `NumPy` array along with frequency and time axes and the parameters used to create it.
///
/// The data is stored natively in the precision requested at compute time
/// (`float32` or `float64`); inspect [`PySpectrogram::dtype`] to find out which.
#[pyclass(name = "Spectrogram", skip_from_py_object)]
pub struct PySpectrogram {
    py_data: PyArrayData,
    // Extracted metadata (no longer storing full Spectrogram to avoid duplication)
    frequencies: Vec<f64>,
    times: Vec<f64>,
    params: SpectrogramParams,
    db_range: Option<(f64, f64)>,
}

impl PySpectrogram {
    /// Create a PySpectrogram from computed Rust spectrogram.
    /// Extracts metadata and transfers data ownership to Python (!).
    pub(crate) fn from_spectrogram<FreqScale, AmpScale, T>(
        py: Python<'_>,
        spec: Spectrogram<FreqScale, AmpScale, T>,
    ) -> Self
    where
        FreqScale: Copy + Clone + 'static,
        AmpScale: AmpScaleSpec + 'static,
        T: PyScalar,
    {
        // Extract metadata before consuming spectrogram
        let frequencies = spec.frequencies().to_vec();
        let times = spec.times().to_vec();
        let params = spec.params().clone();
        let db_range = spec.db_range();

        // Transfer ownership of Array2 to Python (NO COPY!)
        let array = spec.into_data();
        let py_data = T::into_array_data(py, array);

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
    /// numpy.typing.NDArray
    ///     2D `NumPy` array with shape (`n_bins`, `n_frames`). The element type
    ///     is ``float32`` or ``float64`` depending on the ``dtype`` requested
    ///     when the spectrogram was computed (see :attr:`dtype`).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        // Return the Python-allocated array (no copy!)
        self.py_data.bind(py)
    }

    /// Get the NumPy dtype name of the stored data.
    ///
    /// Returns
    /// -------
    /// str
    ///     ``"float32"`` or ``"float64"``.
    #[getter]
    const fn dtype(&self) -> &'static str {
        self.py_data.dtype()
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
            "Spectrogram(shape=({}, {}), dtype={})",
            self.n_bins(),
            self.n_frames(),
            self.dtype(),
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
    /// numpy.typing.NDArray
    ///     Transposed 2D `NumPy` array with shape (`n_frames`, `n_bins`)
    #[getter]
    #[allow(non_snake_case)]
    fn T<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Use numpy's .T property for transpose
        self.py_data.bind(py).getattr("T")
    }

    #[pyo3(signature = (dtype), text_signature = "($self, dtype)")]
    fn astype<'py>(
        &self,
        py: Python<'py>,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.py_data.bind(py).call_method1("astype", (dtype,))
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
            arr.call_method1("astype", (dt,))
        } else {
            // Return as-is (native dtype)
            Ok(arr)
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, idx: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
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
    /// The exported tensor reports the native element type of the spectrogram
    /// (``float32`` or ``float64``), matching :attr:`dtype`.
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
        real_dlpack(py, &self.py_data, stream, max_version, dl_device, copy)
    }
}

/// Register the spectrogram class with the Python module.
pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PySpectrogram>()?;
    Ok(())
}
