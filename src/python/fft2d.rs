//! Python bindings for 2D FFT operations.
//!
//! This module provides Python wrappers for 2D FFT functions that work with
//! 2D `NumPy` arrays (images).

use numpy::{Complex64, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::fft2d as rust_fft2d;
use crate::fft2d::Fft2dPlanner as RustFft2dPlanner;
use crate::image_ops;

/// Compute 2D FFT of a real-valued 2D array.
///
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64]
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
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn fft2d(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<Complex64>>> {
    let data_arr = data.as_array();

    let result = py.detach(|| rust_fft2d::fft2d(&data_arr))?;

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
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64]
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
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn power_spectrum_2d(py: Python, data: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let data_arr = data.as_array();

    let result = py.detach(|| rust_fft2d::power_spectrum_2d(&data_arr))?;

    Ok(result.to_pyarray(py).unbind())
}

/// Compute 2D magnitude spectrum.
///
/// Parameters
/// ----------
/// data : numpy.typing.NDArray[numpy.float64]
///     Input 2D array with shape (nrows, ncols)
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Magnitude spectrum with shape (nrows, ncols/2 + 1)
#[pyfunction]
#[pyo3(signature = (data: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(data: numpy.typing.NDArray[numpy.float64])")]
pub fn magnitude_spectrum_2d(
    py: Python,
    data: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let data_arr = data.as_array();
    let result = py.detach(|| rust_fft2d::magnitude_spectrum_2d(&data_arr))?;
    Ok(result.to_pyarray(py).unbind())
}

/// Shift zero-frequency component to center.
///
/// Parameters
/// ----------
/// arr : numpy.typing.NDArray[numpy.float64]
///     Input 2D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array with DC component at center
#[pyfunction]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn fftshift(py: Python, arr: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
    let arr_owned = arr.as_array().to_owned();
    let result = rust_fft2d::fftshift(arr_owned);
    result.to_pyarray(py).unbind()
}

/// Inverse of fftshift - shift center back to corners.
///
/// Parameters
/// ----------
/// arr : numpy.typing.NDArray[numpy.float64]
///     Input 2D array
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Shifted array with DC component at corners
#[pyfunction]
#[pyo3(signature = (arr: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(arr: numpy.typing.NDArray[numpy.float64])")]
pub fn ifftshift(py: Python, arr: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
    let arr_owned = arr.as_array().to_owned();

    let result = rust_fft2d::ifftshift(arr_owned);

    result.to_pyarray(py).unbind()
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", kernel: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], kernel: numpy.typing.NDArray[numpy.float64])")]
pub fn convolve_fft(
    py: Python,
    image: PyReadonlyArray2<f64>,
    kernel: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();
    let kernel_arr = kernel.as_array();

    let result = py.detach(|| image_ops::convolve_fft(&image_arr, &kernel_arr))?;

    Ok(result.to_pyarray(py).unbind())
}

/// Apply low-pass filter to suppress high frequencies.
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
///     Filtered image
#[pyfunction]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float)")]
pub fn lowpass_filter(
    py: Python,
    image: PyReadonlyArray2<f64>,
    cutoff_fraction: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();

    let result = py.detach(|| image_ops::lowpass_filter(&image_arr, cutoff_fraction))?;

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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", cutoff_fraction: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], cutoff_fraction: float)")]
pub fn highpass_filter(
    py: Python,
    image: PyReadonlyArray2<f64>,
    cutoff_fraction: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();

    let result = py.detach(|| image_ops::highpass_filter(&image_arr, cutoff_fraction))?;

    Ok(result.to_pyarray(py).unbind())
}

/// Apply band-pass filter to keep frequencies in a range.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64]
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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", low_cutoff: "float", high_cutoff: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], low_cutoff: float, high_cutoff: float)")]
pub fn bandpass_filter(
    py: Python,
    image: PyReadonlyArray2<f64>,
    low_cutoff: f64,
    high_cutoff: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();

    let result = py.detach(|| image_ops::bandpass_filter(&image_arr, low_cutoff, high_cutoff))?;

    Ok(result.to_pyarray(py).unbind())
}

/// Detect edges using high-pass filtering.
///
/// Parameters
/// ----------
/// image : numpy.typing.NDArray[numpy.float64]
///     Input image
///
/// Returns
/// -------
/// numpy.typing.NDArray[numpy.float64]
///     Edge-detected image
#[pyfunction]
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]"), text_signature = "(image: numpy.typing.NDArray[numpy.float64])")]
pub fn detect_edges_fft(py: Python, image: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();

    let result = py.detach(|| image_ops::detect_edges_fft(&image_arr))?;

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
#[pyo3(signature = (image: "numpy.typing.NDArray[numpy.float64]", amount: "float"), text_signature = "(image: numpy.typing.NDArray[numpy.float64], amount: float)")]
pub fn sharpen_fft(
    py: Python,
    image: PyReadonlyArray2<f64>,
    amount: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let image_arr = image.as_array();

    let result = py.detach(|| image_ops::sharpen_fft(&image_arr, amount))?;

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
#[pyclass(name = "Fft2dPlanner")]
pub struct PyFft2dPlanner {
    inner: RustFft2dPlanner,
}

#[pymethods]
impl PyFft2dPlanner {
    /// Create a new 2D FFT planner.
    #[new]
    pub fn new() -> Self {
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
    pub fn fft2d(
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
    pub fn ifft2d(
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
    pub fn power_spectrum_2d(
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
    pub fn magnitude_spectrum_2d(
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
