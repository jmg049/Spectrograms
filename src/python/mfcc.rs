//! Python MFCC result class.

use pyo3::prelude::*;

use crate::{Mfcc, MfccParams};

use super::params::PyMfccParams;
use super::spectrogram::{PyArrayData, PyScalar, real_dlpack};

/// MFCC (Mel-Frequency Cepstral Coefficients) result.
///
/// Carries the coefficient matrix as a native-precision NumPy array
/// (`float32` or `float64`, see :attr:`dtype`) plus the parameters used to
/// compute it. The data lives on the Python heap so it can be shared zero-copy
/// with array libraries via ``__array__`` / ``__dlpack__``.
#[pyclass(name = "Mfcc", skip_from_py_object)]
pub struct PyMfcc {
    data: PyArrayData,
    params: MfccParams,
    n_bins: usize,
    n_frames: usize,
}

impl PyMfcc {
    /// Build a `PyMfcc` from a computed Rust [`Mfcc`], moving its data onto the
    /// Python heap (no copy).
    pub(crate) fn from_mfcc<T: PyScalar>(py: Python<'_>, mfcc: Mfcc<T>) -> Self {
        let n_bins = mfcc.data.nrows();
        let n_frames = mfcc.data.ncols();
        let params = *mfcc.params();
        let data = T::into_array_data(py, mfcc.data);
        Self {
            data,
            params,
            n_bins,
            n_frames,
        }
    }
}

#[pymethods]
impl PyMfcc {
    /// MFCC coefficient matrix as a NumPy array with shape (`n_bins`, `n_frames`).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.data.bind(py)
    }

    /// NumPy dtype name of the stored data (`"float32"` / `"float64"`).
    #[getter]
    const fn dtype(&self) -> &'static str {
        self.data.dtype()
    }

    /// Number of time frames.
    #[getter]
    const fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of cepstral coefficients (rows).
    #[getter]
    const fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Shape of the coefficient matrix as (`n_bins`, `n_frames`).
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        (self.n_bins, self.n_frames)
    }

    /// The `MfccParams` used to compute these coefficients.
    #[getter]
    fn params(&self) -> PyMfccParams {
        PyMfccParams::from(self.params)
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

    /// Export the MFCC data as a DLPack capsule for tensor exchange.
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
            "Mfcc(shape=({}, {}), dtype={})",
            self.n_bins,
            self.n_frames,
            self.dtype()
        )
    }
}
