//! Python bindings for 2D FFT operations.
//!
//! This module provides Python wrappers for 2D FFT functions that work with
//! 2D `NumPy` arrays (images) and array-like objects (e.g., Spectrogram).

use pyo3::prelude::*;

use super::dtype::{
    Dtype, array2_to_py, complex_2d_owned, parse_dtype, real_1d_vec, real_2d_owned, vec1_to_py,
};
use super::spectrogram::PyScalar;
use crate::fft2d as rust_fft2d;
use crate::fft2d::Fft2dPlanner as RustFft2dPlanner;
use crate::image_ops;

/// Compute 2D FFT of a real-valued 2D array.
///
/// Accepts numpy arrays, Spectrogram objects, or any object implementing __array__().
///
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input 2D array (e.g., image) with shape (nrows, ncols)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.complex64]
///     Complex 2D array with shape (nrows, ncols/2 + 1) due to Hermitian symmetry
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> image = np.random.randn(128, 128)
/// >>> spectrum = sg.fft2d(image)
/// >>> spectrum.shape
/// (128, 65)
#[pyfunction]
#[inline]
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(data: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn fft2d(py: Python, data: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    fn run<T>(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>
    where
        T: PyScalar,
        num_complex::Complex<T>: numpy::Element,
    {
        let owned = real_2d_owned::<T>(py, data)?;
        let result = py.detach(|| rust_fft2d::fft2d(&owned.view()))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, data),
        Dtype::F64 => run::<f64>(py, data),
    }
}

/// Compute inverse 2D FFT from frequency domain back to spatial domain.
///
/// Parameters
/// ----------
/// spectrum : numpy.typing.NDArray[numpy.complex64]
///     Complex frequency array with shape (nrows, ncols/2 + 1)
/// `output_ncols` : int
///     Number of columns in the output (must match original image width)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Real 2D array with shape (nrows, `output_ncols`)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> image = np.random.randn(128, 128)
/// >>> spectrum = sg.fft2d(image)
/// >>> reconstructed = sg.ifft2d(spectrum, 128)
/// >>> np.allclose(image, reconstructed)
/// True
#[pyfunction]
#[inline]
#[pyo3(signature = (spectrum: "numpy.typing.NDArray[numpy.complex64]", output_ncols: "int", dtype: "str" = None), text_signature = "(spectrum: numpy.typing.NDArray[numpy.complex64], output_ncols: int, dtype: str = \"float64\")")]
pub fn ifft2d(
    py: Python,
    spectrum: &Bound<'_, PyAny>,
    output_ncols: usize,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T>(py: Python<'_>, spectrum: &Bound<'_, PyAny>, output_ncols: usize) -> PyResult<Py<PyAny>>
    where
        T: PyScalar,
        num_complex::Complex<T>: numpy::Element,
    {
        let owned = complex_2d_owned::<T>(py, spectrum)?;
        let result = py.detach(|| rust_fft2d::ifft2d(&owned, output_ncols))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, spectrum, output_ncols),
        Dtype::F64 => run::<f64>(py, spectrum, output_ncols),
    }
}

/// Compute 2D power spectrum (squared magnitude).
///
/// Accepts numpy arrays, Spectrogram objects, or any object implementing __array__().
///
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input 2D array with shape (nrows, ncols)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Power spectrum with shape (nrows, ncols/2 + 1)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> image = np.ones((64, 64))
/// >>> power = sg.power_spectrum_2d(image)
/// >>> power[0, 0]  # DC component should have all energy
/// 16777216.0
#[pyfunction]
#[inline]
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(data: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn power_spectrum_2d(
    py: Python,
    data: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let owned = real_2d_owned::<T>(py, data)?;
        let result = py.detach(|| rust_fft2d::power_spectrum_2d(&owned.view()))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, data),
        Dtype::F64 => run::<f64>(py, data),
    }
}

/// Compute 2D magnitude spectrum.
///
/// Accepts numpy arrays, Spectrogram objects, or any object implementing __array__().
///
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input 2D array with shape (nrows, ncols)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Magnitude spectrum with shape (nrows, ncols/2 + 1)
#[pyfunction]
#[inline]
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(data: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn magnitude_spectrum_2d(
    py: Python,
    data: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let owned = real_2d_owned::<T>(py, data)?;
        let result = py.detach(|| rust_fft2d::magnitude_spectrum_2d(&owned.view()))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, data),
        Dtype::F64 => run::<f64>(py, data),
    }
}

/// Shift zero-frequency component to center.
///
/// Accepts numpy arrays, Spectrogram objects, or any object implementing __array__().
///
/// Parameters
/// ----------
/// arr : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input 2D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array with DC component at center
#[pyfunction]
#[inline]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(arr: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn fftshift(py: Python, arr: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let owned = real_2d_owned::<T>(py, arr)?;
        Ok(array2_to_py(py, rust_fft2d::fftshift(owned)))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, arr),
        Dtype::F64 => run::<f64>(py, arr),
    }
}

/// Inverse of fftshift - shift center back to corners.
///
/// Accepts numpy arrays, Spectrogram objects, or any object implementing __array__().
///
/// Parameters
/// ----------
/// arr : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input 2D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array with DC component at corners
#[pyfunction]
#[inline]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(arr: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn ifftshift(py: Python, arr: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let owned = real_2d_owned::<T>(py, arr)?;
        Ok(array2_to_py(py, rust_fft2d::ifftshift(owned)))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, arr),
        Dtype::F64 => run::<f64>(py, arr),
    }
}

/// Shift zero-frequency component to center for 1D arrays.
///
/// Parameters
/// ----------
/// arr : list[float] or numpy.typing.NDArray[numpy.float64]
///     Input 1D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array with DC component at center
#[pyfunction]
#[inline]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(arr: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn fftshift_1d(py: Python, arr: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let v = real_1d_vec::<T>(py, arr)?;
        Ok(vec1_to_py(py, rust_fft2d::fftshift_1d(v)))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, arr),
        Dtype::F64 => run::<f64>(py, arr),
    }
}

/// Inverse of fftshift for 1D arrays.
///
/// Parameters
/// ----------
/// arr : list[float] or numpy.typing.NDArray[numpy.float64]
///     Input 1D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array
#[pyfunction]
#[inline]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(arr: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn ifftshift_1d(py: Python, arr: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let v = real_1d_vec::<T>(py, arr)?;
        Ok(vec1_to_py(py, rust_fft2d::ifftshift_1d(v)))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, arr),
        Dtype::F64 => run::<f64>(py, arr),
    }
}

/// Compute FFT sample frequencies.
///
/// Returns the sample frequencies (in cycles per unit of the sample spacing) for FFT output.
///
/// Parameters
/// ----------
/// n : int
///     Window length (number of samples)
/// d : float, optional
///     Sample spacing (inverse of sampling rate). Default is 1.0.
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Array of length n containing the frequency bin centers in cycles per unit
///     (float64 or float32 depending on `dtype`)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> # For temporal modulation at 16kHz sample rate with 100 frames
/// >>> hop_size = 128
/// >>> sample_rate = 16000.0
/// >>> frame_period = hop_size / sample_rate
/// >>> freqs_hz = sg.fftfreq(100, frame_period)
/// >>> # Returns frequencies in Hz
#[pyfunction]
#[inline]
#[pyo3(signature = (n, d = 1.0, dtype = None), text_signature = "(n: int, d: float = 1.0, dtype: str = \"float64\")")]
pub fn fftfreq(py: Python, n: usize, d: f64, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match parse_dtype(dtype)? {
        Dtype::F32 => Ok(vec1_to_py(py, rust_fft2d::fftfreq::<f32>(n, d))),
        Dtype::F64 => Ok(vec1_to_py(py, rust_fft2d::fftfreq::<f64>(n, d))),
    }
}

/// Compute FFT sample frequencies for real FFT.
///
/// Returns only the positive frequencies for a real-to-complex FFT.
///
/// Parameters
/// ----------
/// n : int
///     Window length (number of samples in original real signal)
/// d : float, optional
///     Sample spacing (inverse of sampling rate). Default is 1.0.
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Array of length n/2 + 1 containing the positive frequency bin centers
///     (float64 or float32 depending on `dtype`)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> # For 8 samples
/// >>> freqs = sg.rfftfreq(8, 1.0)
/// >>> # Returns: [0.0, 0.125, 0.25, 0.375, 0.5]
#[pyfunction]
#[inline]
#[pyo3(signature = (n, d = 1.0, dtype = None), text_signature = "(n: int, d: float = 1.0, dtype: str = \"float64\")")]
pub fn rfftfreq(py: Python, n: usize, d: f64, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match parse_dtype(dtype)? {
        Dtype::F32 => Ok(vec1_to_py(py, rust_fft2d::rfftfreq::<f32>(n, d))),
        Dtype::F64 => Ok(vec1_to_py(py, rust_fft2d::rfftfreq::<f64>(n, d))),
    }
}

/// Create 2D Gaussian kernel for blurring.
///
/// Parameters
/// ----------
/// size : int
///     Kernel size (must be odd, e.g., 3, 5, 7, 9)
/// sigma : float
///     Standard deviation of the Gaussian
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Normalized Gaussian kernel with shape (size, size), float64 or float32
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> kernel = sg.gaussian_kernel_2d(5, 1.0)
/// >>> kernel.shape
/// (5, 5)
/// >>> kernel.sum()  # Should be ~1.0
/// 1.0
#[pyfunction]
#[inline]
#[pyo3(signature = (size: "int", sigma: "float", dtype: "str" = None), text_signature = "(size: int, sigma: float, dtype: str = \"float64\")")]
pub fn gaussian_kernel_2d(
    py: Python,
    size: usize,
    sigma: f64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let size = std::num::NonZeroUsize::new(size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("size must be a non-zero odd integer")
    })?;
    match parse_dtype(dtype)? {
        Dtype::F32 => {
            let result = py.detach(|| image_ops::gaussian_kernel_2d::<f32>(size, sigma))?;
            Ok(array2_to_py(py, result))
        }
        Dtype::F64 => {
            let result = py.detach(|| image_ops::gaussian_kernel_2d::<f64>(size, sigma))?;
            Ok(array2_to_py(py, result))
        }
    }
}

/// Convolve 2D image with kernel using FFT.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64]
///     Input image with shape (nrows, ncols)
/// kernel : numpy.typing.NDArray[numpy.float64]
///     Convolution kernel (must be smaller than image)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Convolved image (same size as input)
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> image = np.random.randn(256, 256)
/// >>> kernel = sg.gaussian_kernel_2d(9, 2.0)
/// >>> blurred = sg.convolve_fft(image, kernel)
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", kernel: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], kernel: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn convolve_fft(
    py: Python,
    image: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        kernel: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let kernel = real_2d_owned::<T>(py, kernel)?;
        let result = py.detach(|| image_ops::convolve_fft(&image.view(), &kernel.view()))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image, kernel),
        Dtype::F64 => run::<f64>(py, image, kernel),
    }
}

/// Apply low-pass filter to suppress high frequencies.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input image
/// `cutoff_fraction` : float
///     Cutoff radius as fraction (0.0 to 1.0)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Filtered image
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float, dtype: str = \"float64\")")]
pub fn lowpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    cutoff_fraction: f64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        cutoff_fraction: f64,
    ) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let result = py.detach(|| image_ops::lowpass_filter(&image.view(), cutoff_fraction))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image, cutoff_fraction),
        Dtype::F64 => run::<f64>(py, image, cutoff_fraction),
    }
}

/// Apply high-pass filter to suppress low frequencies.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64]
///     Input image
/// `cutoff_fraction` : float
///     Cutoff radius as fraction (0.0 to 1.0)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Filtered image with edges emphasized
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float, dtype: str = \"float64\")")]
pub fn highpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    cutoff_fraction: f64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        cutoff_fraction: f64,
    ) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let result = py.detach(|| image_ops::highpass_filter(&image.view(), cutoff_fraction))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image, cutoff_fraction),
        Dtype::F64 => run::<f64>(py, image, cutoff_fraction),
    }
}

/// Apply band-pass filter to keep frequencies in a range.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input image
/// `low_cutoff` : float
///     Lower cutoff as fraction (0.0 to 1.0)
/// `high_cutoff` : float
///     Upper cutoff as fraction (0.0 to 1.0)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Filtered image
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", low_cutoff: "float", high_cutoff: "float", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], low_cutoff: float, high_cutoff: float, dtype: str = \"float64\")")]
pub fn bandpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    low_cutoff: f64,
    high_cutoff: f64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        low_cutoff: f64,
        high_cutoff: f64,
    ) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let result =
            py.detach(|| image_ops::bandpass_filter(&image.view(), low_cutoff, high_cutoff))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image, low_cutoff, high_cutoff),
        Dtype::F64 => run::<f64>(py, image, low_cutoff, high_cutoff),
    }
}

/// Detect edges using high-pass filtering.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64] or Spectrogram
///     Input image
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Edge-detected image
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], dtype: str = \"float64\")")]
pub fn detect_edges_fft(
    py: Python,
    image: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let result = py.detach(|| image_ops::detect_edges_fft(&image.view()))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image),
        Dtype::F64 => run::<f64>(py, image),
    }
}

/// Sharpen image by enhancing high frequencies.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64]
///     Input image
/// amount : float
///     Sharpening strength (typical range: 0.5 to 2.0)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Sharpened image
#[pyfunction]
#[inline]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", amount: "float", dtype: "str" = None), text_signature = "(image: numpy.typing.NDArray[numpy.float64], amount: float, dtype: str = \"float64\")")]
pub fn sharpen_fft(
    py: Python,
    image: &Bound<'_, PyAny>,
    amount: f64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    fn run<T: PyScalar>(py: Python<'_>, image: &Bound<'_, PyAny>, amount: f64) -> PyResult<Py<PyAny>> {
        let image = real_2d_owned::<T>(py, image)?;
        let result = py.detach(|| image_ops::sharpen_fft(&image.view(), amount))?;
        Ok(array2_to_py(py, result))
    }
    match parse_dtype(dtype)? {
        Dtype::F32 => run::<f32>(py, image, amount),
        Dtype::F64 => run::<f64>(py, image, amount),
    }
}

/// 2D FFT planner for efficient batch processing.
///
/// Caches FFT plans internally to avoid repeated setup overhead when
/// processing multiple arrays with the same dimensions.
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> import numpy as np
/// >>> planner = sg.Fft2dPlanner()
/// >>> for _ in range(10):
/// ...     image = np.random.randn(128, 128)
/// ...     spectrum = planner.fft2d(image)
/// Precision-tagged storage for the generic Rust [`RustFft2dPlanner`].
///
/// The Rust planner caches FFT plans per dimension, so its precision is fixed
/// at construction time; we therefore hold one monomorphic planner per dtype.
enum PlannerInner {
    F32(RustFft2dPlanner<f32>),
    F64(RustFft2dPlanner<f64>),
}

#[pyclass(name = "Fft2dPlanner", skip_from_py_object)]
pub struct PyFft2dPlanner {
    inner: PlannerInner,
    dtype: Dtype,
}

#[pymethods]
impl PyFft2dPlanner {
    /// Create a new 2D FFT planner.
    ///
    /// Parameters
    /// ----------
    /// dtype : str
    ///     Working precision: "float64" (default) or "float32". Fixed for the
    ///     lifetime of the planner.
    #[new]
    #[pyo3(signature = (dtype = None), text_signature = "(dtype: str = \"float64\")")]
    fn new(dtype: Option<&str>) -> PyResult<Self> {
        let d = parse_dtype(dtype)?;
        let inner = match d {
            Dtype::F32 => PlannerInner::F32(RustFft2dPlanner::<f32>::new()),
            Dtype::F64 => PlannerInner::F64(RustFft2dPlanner::<f64>::new()),
        };
        Ok(Self { inner, dtype: d })
    }

    /// The working precision of this planner ("float32" or "float64").
    #[getter]
    const fn dtype(&self) -> &'static str {
        match self.dtype {
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
        }
    }

    /// Compute 2D FFT using cached plans.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.typing.NDArray[numpy.float64]
    ///     Input 2D array with shape (nrows, ncols)
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.complex128]
    ///     Complex 2D array with shape (nrows, ncols/2 + 1), complex128 or complex64
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn fft2d(&mut self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match &mut self.inner {
            PlannerInner::F32(p) => {
                let owned = real_2d_owned::<f32>(py, data)?;
                let result = py.detach(|| p.fft2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
            PlannerInner::F64(p) => {
                let owned = real_2d_owned::<f64>(py, data)?;
                let result = py.detach(|| p.fft2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
        }
    }

    /// Compute inverse 2D FFT using cached plans.
    ///
    /// Parameters
    /// ----------
    /// spectrum : numpy.typing.NDArray[numpy.complex128]
    ///     Complex frequency array
    /// `output_ncols` : int
    ///     Number of columns in output
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     Real 2D array, float64 or float32
    #[pyo3(signature = (spectrum: "numpy.typing.NDArray[numpy.complex128]", output_ncols: "int"), text_signature = "(spectrum: numpy.typing.NDArray[numpy.complex128], output_ncols: int)")]
    fn ifft2d(
        &mut self,
        py: Python,
        spectrum: &Bound<'_, PyAny>,
        output_ncols: usize,
    ) -> PyResult<Py<PyAny>> {
        match &mut self.inner {
            PlannerInner::F32(p) => {
                let owned = complex_2d_owned::<f32>(py, spectrum)?;
                let result = py.detach(|| p.ifft2d(&owned.view(), output_ncols))?;
                Ok(array2_to_py(py, result))
            }
            PlannerInner::F64(p) => {
                let owned = complex_2d_owned::<f64>(py, spectrum)?;
                let result = py.detach(|| p.ifft2d(&owned.view(), output_ncols))?;
                Ok(array2_to_py(py, result))
            }
        }
    }

    /// Compute 2D power spectrum using cached plans.
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn power_spectrum_2d(&mut self, py: Python, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match &mut self.inner {
            PlannerInner::F32(p) => {
                let owned = real_2d_owned::<f32>(py, data)?;
                let result = py.detach(|| p.power_spectrum_2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
            PlannerInner::F64(p) => {
                let owned = real_2d_owned::<f64>(py, data)?;
                let result = py.detach(|| p.power_spectrum_2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
        }
    }

    /// Compute 2D magnitude spectrum using cached plans.
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn magnitude_spectrum_2d(
        &mut self,
        py: Python,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &mut self.inner {
            PlannerInner::F32(p) => {
                let owned = real_2d_owned::<f32>(py, data)?;
                let result = py.detach(|| p.magnitude_spectrum_2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
            PlannerInner::F64(p) => {
                let owned = real_2d_owned::<f64>(py, data)?;
                let result = py.detach(|| p.magnitude_spectrum_2d(&owned.view()))?;
                Ok(array2_to_py(py, result))
            }
        }
    }
}

/// Register 2D FFT functions and classes with the Python module.
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register functions
    m.add_function(wrap_pyfunction!(fft2d, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2d, m)?)?;
    m.add_function(wrap_pyfunction!(power_spectrum_2d, m)?)?;
    m.add_function(wrap_pyfunction!(magnitude_spectrum_2d, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift_1d, m)?)?;
    m.add_function(wrap_pyfunction!(fftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq, m)?)?;

    // Register image processing functions
    m.add_function(wrap_pyfunction!(gaussian_kernel_2d, m)?)?;
    m.add_function(wrap_pyfunction!(convolve_fft, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(detect_edges_fft, m)?)?;
    m.add_function(wrap_pyfunction!(sharpen_fft, m)?)?;

    // Register planner class
    m.add_class::<PyFft2dPlanner>()?;

    Ok(())
}
