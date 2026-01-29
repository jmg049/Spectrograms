//! Python bindings for 2D FFT operations.
//!
//! This module provides Python wrappers for 2D FFT functions that work with
//! 2D `NumPy` arrays (images) and array-like objects (e.g., Spectrogram).

use numpy::{
    Complex64, PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

use crate::fft2d as rust_fft2d;
use crate::fft2d::Fft2dPlanner as RustFft2dPlanner;
use crate::image_ops;

/// Helper function to convert array-like objects to PyReadonlyArray2
/// by calling their __array__() method, mimicking numpy's behavior
fn extract_array<'py>(
    _py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    // Try direct extraction first (for actual numpy arrays)
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr);
    }

    // If that fails, try calling __array__() method (for Spectrogram and other array-like objects)
    if obj.hasattr("__array__")? {
        let array_result = obj.call_method0("__array__")?;
        return array_result
            .extract::<PyReadonlyArray2<f64>>()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("Failed to extract array: {e}"))
            });
    }

    // Neither worked - return an error
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Object must be a numpy array or implement __array__()",
    ))
}

/// Helper function to convert array-like objects to PyReadonlyArray1 (1D)
/// by calling their __array__() method, mimicking numpy's behavior
fn extract_array_1d<'py>(
    _py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    // Try direct extraction first (for actual numpy arrays)
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr);
    }

    // If that fails, try calling __array__() method
    if obj.hasattr("__array__")? {
        let array_result = obj.call_method0("__array__")?;
        return array_result
            .extract::<PyReadonlyArray1<f64>>()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("Failed to extract 1D array: {e}"))
            });
    }

    // Neither worked - return an error
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Object must be a 1D numpy array or implement __array__()",
    ))
}

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
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn fft2d(py: Python, data: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<Complex64>>> {
    let data_arr = extract_array(py, data)?;
    let data_view = data_arr.as_array();

    let result = py.detach(|| rust_fft2d::fft2d(&data_view))?;

    // Convert Complex<f64> to Complex64 for Python
    let result_complex64 = result.mapv(|c| Complex64::new(c.re, c.im));

    Ok(result_complex64.to_pyarray(py).unbind())
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
#[pyo3(signature = (spectrum: "numpy.typing.NDArray[numpy.complex64]", output_ncols: "int"), text_signature = "(spectrum: numpy.typing.NDArray[numpy.complex64], output_ncols: int)")]
pub fn ifft2d(
    py: Python,
    spectrum: PyReadonlyArray2<Complex64>,
    output_ncols: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let spectrum_arr = spectrum.as_array();

    // Convert Complex64 to Complex<f64>
    let spectrum_f64 = spectrum_arr.mapv(|c| num_complex::Complex::new(c.re as f64, c.im as f64));

    let result = py.detach(|| rust_fft2d::ifft2d(&spectrum_f64, output_ncols))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn power_spectrum_2d(py: Python, data: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    let data_arr = extract_array(py, data)?;
    let data_view = data_arr.as_array();

    let result = py.detach(|| rust_fft2d::power_spectrum_2d(&data_view))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn magnitude_spectrum_2d(py: Python, data: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    let data_arr = extract_array(py, data)?;
    let data_view = data_arr.as_array();
    let result = py.detach(|| rust_fft2d::magnitude_spectrum_2d(&data_view))?;
    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn fftshift(py: Python, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    let arr_data = extract_array(py, arr)?;
    let arr_owned = arr_data.as_array().to_owned();
    let result = rust_fft2d::fftshift(arr_owned);
    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn ifftshift(py: Python, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    let arr_data = extract_array(py, arr)?;
    let arr_owned = arr_data.as_array().to_owned();

    let result = rust_fft2d::ifftshift(arr_owned);

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn fftshift_1d(py: Python, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
    let arr_data = extract_array_1d(py, arr)?;
    let arr_vec = arr_data.as_slice()?.to_vec();
    let result = rust_fft2d::fftshift_1d(arr_vec);
    Ok(PyArray1::from_vec(py, result).unbind())
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
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn ifftshift_1d(py: Python, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
    let arr_data = extract_array_1d(py, arr)?;
    let arr_vec = arr_data.as_slice()?.to_vec();
    let result = rust_fft2d::ifftshift_1d(arr_vec);
    Ok(PyArray1::from_vec(py, result).unbind())
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
#[pyo3(signature = (n, d = 1.0), text_signature = "(n: int, d: float = 1.0)")]
pub fn fftfreq(py: Python, n: usize, d: f64) -> Py<PyArray<f64, numpy::Ix1>> {
    let freqs = rust_fft2d::fftfreq(n, d);
    numpy::PyArray1::from_vec(py, freqs).unbind()
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
///
/// Examples
/// --------
/// >>> import spectrograms as sg
/// >>> # For 8 samples
/// >>> freqs = sg.rfftfreq(8, 1.0)
/// >>> # Returns: [0.0, 0.125, 0.25, 0.375, 0.5]
#[pyfunction]
#[inline]
#[pyo3(signature = (n, d = 1.0), text_signature = "(n: int, d: float = 1.0)")]
pub fn rfftfreq(py: Python, n: usize, d: f64) -> Py<numpy::PyArray<f64, numpy::Ix1>> {
    let freqs = rust_fft2d::rfftfreq(n, d);
    numpy::PyArray1::from_vec(py, freqs).unbind()
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
///     Normalized Gaussian kernel with shape (size, size)
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
#[pyo3(signature = (size: "int", sigma: "float"), text_signature = "(size: int, sigma: float)")]
pub fn gaussian_kernel_2d(py: Python, size: usize, sigma: f64) -> PyResult<Py<PyArray2<f64>>> {
    let size = std::num::NonZeroUsize::new(size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("size must be a non-zero odd integer")
    })?;
    let result = py.detach(|| image_ops::gaussian_kernel_2d(size, sigma))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", kernel: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], kernel: numpy.typing.NDArray[numpy.float64])")]
pub fn convolve_fft(
    py: Python,
    image: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let kernel_arr = extract_array(py, kernel)?;
    let image_view = image_arr.as_array();
    let kernel_view = kernel_arr.as_array();

    let result = py.detach(|| image_ops::convolve_fft(&image_view, &kernel_view))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float)")]
pub fn lowpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    cutoff_fraction: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let image_view = image_arr.as_array();

    let result = py.detach(|| image_ops::lowpass_filter(&image_view, cutoff_fraction))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float)")]
pub fn highpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    cutoff_fraction: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let image_view = image_arr.as_array();

    let result = py.detach(|| image_ops::highpass_filter(&image_view, cutoff_fraction))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", low_cutoff: "float", high_cutoff: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], low_cutoff: float, high_cutoff: float)")]
pub fn bandpass_filter(
    py: Python,
    image: &Bound<'_, PyAny>,
    low_cutoff: f64,
    high_cutoff: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let image_view = image_arr.as_array();

    let result = py.detach(|| image_ops::bandpass_filter(&image_view, low_cutoff, high_cutoff))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(image: numpy.typing.NDArray[numpy.float64])")]
pub fn detect_edges_fft(py: Python, image: &Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let image_view = image_arr.as_array();

    let result = py.detach(|| image_ops::detect_edges_fft(&image_view))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", amount: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], amount: float)")]
pub fn sharpen_fft(
    py: Python,
    image: &Bound<'_, PyAny>,
    amount: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = extract_array(py, image)?;
    let image_view = image_arr.as_array();

    let result = py.detach(|| image_ops::sharpen_fft(&image_view, amount))?;

    Ok(result.to_pyarray(py).unbind())
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
#[pyclass(name = "Fft2dPlanner", skip_from_py_object)]
pub struct PyFft2dPlanner {
    inner: RustFft2dPlanner,
}

#[pymethods]
impl PyFft2dPlanner {
    /// Create a new 2D FFT planner.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustFft2dPlanner::new(),
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
    /// numpy.typing.NDArray[numpy.complex64]
    ///     Complex 2D array with shape (nrows, ncols/2 + 1)
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn fft2d(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let data_arr = data.as_array();

        let result = py.detach(|| self.inner.fft2d(&data_arr))?;

        // Convert Complex<f64> to Complex64
        let result_complex64 = result.mapv(|c| Complex64::new(c.re, c.im));

        Ok(result_complex64.to_pyarray(py).unbind())
    }

    /// Compute inverse 2D FFT using cached plans.
    ///
    /// Parameters
    /// ----------
    /// spectrum : numpy.typing.NDArray[numpy.complex64]
    ///     Complex frequency array
    /// `output_ncols` : int
    ///     Number of columns in output
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     Real 2D array
    #[pyo3(signature = (spectrum: "numpy.typing.NDArray[numpy.complex64]", output_ncols: "int"), text_signature = "(spectrum: numpy.typing.NDArray[numpy.complex64], output_ncols: int)")]
    fn ifft2d(
        &mut self,
        py: Python,
        spectrum: PyReadonlyArray2<Complex64>,
        output_ncols: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let spectrum_arr = spectrum.as_array();
        let spectrum_f64 =
            spectrum_arr.mapv(|c| num_complex::Complex::new(c.re as f64, c.im as f64));

        let result = py.detach(|| self.inner.ifft2d(&spectrum_f64.view(), output_ncols))?;

        Ok(result.to_pyarray(py).unbind())
    }

    /// Compute 2D power spectrum using cached plans.
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn power_spectrum_2d(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let data_arr = data.as_array();

        let result = py.detach(|| self.inner.power_spectrum_2d(&data_arr))?;

        Ok(result.to_pyarray(py).unbind())
    }

    /// Compute 2D magnitude spectrum using cached plans.
    #[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
    fn magnitude_spectrum_2d(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let data_arr = data.as_array();

        let result = py.detach(|| self.inner.magnitude_spectrum_2d(&data_arr.view()))?;

        Ok(result.to_pyarray(py).unbind())
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
