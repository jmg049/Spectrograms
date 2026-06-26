//! Shared `dtype`-selection machinery for the dtype-aware Python compute functions.
//!
//! Every standalone compute function whose underlying Rust implementation is
//! generic over the scalar `T` exposes a `dtype="float32"|"float64"` parameter.
//! The helpers here parse that string, force input NumPy arrays to the matching
//! precision, and move owned results back onto the Python heap as numpy arrays
//! of the matching element type.

// These helpers are deliberately `pub(crate)` so the sibling Python binding
// modules can share them; the `redundant_pub_crate` nursery lint would prefer
// `pub` inside this private module, but crate visibility is the intent.
#![allow(clippy::redundant_pub_crate)]

use ndarray::{Array1, Array2};
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use super::spectrogram::PyScalar;

/// Output precision requested by the Python caller.
#[derive(Clone, Copy)]
pub(crate) enum Dtype {
    F32,
    F64,
}

/// Parse a user-supplied dtype string into [`Dtype`].
///
/// Accepts the common spellings (`"float64"`, `"float32"`, `"f64"`, `"f32"`,
/// and the `numpy` aliases `"double"` / `"single"`). `None` defaults to
/// double precision. Anything else raises `ValueError`.
pub(crate) fn parse_dtype(dtype: Option<&str>) -> PyResult<Dtype> {
    match dtype.unwrap_or("float64") {
        "float64" | "f64" | "double" => Ok(Dtype::F64),
        "float32" | "f32" | "single" => Ok(Dtype::F32),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype {other:?}; expected 'float32' or 'float64'"
        ))),
    }
}

/// Move an owned 2D array onto the Python heap as a numpy array (type-erased).
pub(crate) fn array2_to_py<T: Element>(py: Python<'_>, a: Array2<T>) -> Py<PyAny> {
    PyArray2::from_owned_array(py, a).into_any().unbind()
}

/// Move an owned 1D array onto the Python heap as a numpy array (type-erased).
pub(crate) fn array1_to_py<T: Element>(py: Python<'_>, a: Array1<T>) -> Py<PyAny> {
    PyArray1::from_owned_array(py, a).into_any().unbind()
}

/// Move an owned `Vec` onto the Python heap as a 1D numpy array (type-erased).
pub(crate) fn vec1_to_py<T: Element>(py: Python<'_>, v: Vec<T>) -> Py<PyAny> {
    PyArray1::from_vec(py, v).into_any().unbind()
}

/// Force `obj` to a contiguous 1D numpy array of element `T` and run `f` on the
/// non-empty slice view.
pub(crate) fn with_real_1d<T, R, F>(py: Python<'_>, obj: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    T: PyScalar,
    F: FnOnce(&NonEmptySlice<T>) -> PyResult<R>,
{
    let np = py.import("numpy")?;
    let array_any = np.call_method1("ascontiguousarray", (obj, T::DTYPE))?;
    let array = array_any.cast::<PyArray1<T>>()?;
    let readonly = array.readonly();
    let slice = readonly.as_slice()?;
    let ne = NonEmptySlice::new(slice)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("input array must not be empty"))?;
    f(ne)
}

/// Force `obj` to a contiguous 1D complex numpy array of element `Complex<T>`
/// and run `f` on the non-empty slice view.
pub(crate) fn with_complex_1d<T, R, F>(py: Python<'_>, obj: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    T: PyScalar,
    Complex<T>: Element,
    F: FnOnce(&NonEmptySlice<Complex<T>>) -> PyResult<R>,
{
    let np = py.import("numpy")?;
    let array_any = np.call_method1("ascontiguousarray", (obj, T::COMPLEX_DTYPE))?;
    let array = array_any.cast::<PyArray1<Complex<T>>>()?;
    let readonly = array.readonly();
    let slice = readonly.as_slice()?;
    let ne = NonEmptySlice::new(slice)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("input array must not be empty"))?;
    f(ne)
}

/// Force `obj` to a contiguous 1D numpy array of element `T`, returning an owned
/// `Vec<T>`.
pub(crate) fn real_1d_vec<T: PyScalar>(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<T>> {
    let np = py.import("numpy")?;
    let array_any = np.call_method1("ascontiguousarray", (obj, T::DTYPE))?;
    let array = array_any.cast::<PyArray1<T>>()?;
    let readonly = array.readonly();
    Ok(readonly.as_slice()?.to_vec())
}

/// Force `obj` (numpy array or `__array__`-able) to a contiguous, standard-layout
/// 2D numpy array of element `T`, returning an owned `Array2<T>`.
pub(crate) fn real_2d_owned<T: PyScalar>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Array2<T>> {
    let np = py.import("numpy")?;
    let array_any = np.call_method1("ascontiguousarray", (obj, T::DTYPE))?;
    let array = array_any.cast::<PyArray2<T>>()?;
    let readonly = array.readonly();
    Ok(readonly.as_array().to_owned())
}

/// Force `obj` to a contiguous 2D complex numpy array of element `Complex<T>`,
/// returning an owned `Array2<Complex<T>>`.
pub(crate) fn complex_2d_owned<T>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Array2<Complex<T>>>
where
    T: PyScalar,
    Complex<T>: Element,
{
    let np = py.import("numpy")?;
    let array_any = np.call_method1("ascontiguousarray", (obj, T::COMPLEX_DTYPE))?;
    let array = array_any.cast::<PyArray2<Complex<T>>>()?;
    let readonly = array.readonly();
    Ok(readonly.as_array().to_owned())
}
