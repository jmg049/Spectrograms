//! 2D FFT operations for image and spatial data processing.
//!
//! This module provides 2D Fast Fourier Transform operations optimized for
//! image processing, spatial filtering, and 2D signal analysis.
//!
//! # Overview
//!
//! The 2D FFT transforms a 2D real-valued array (like an image) into the
//! frequency domain, enabling efficient convolution, filtering, and analysis.
//!
//! # Examples
//!
//! ```rust
//! use spectrograms::fft2d::fft2d;
//! use ndarray::Array2;
//!
//! // Create a 128x128 image
//! let image = Array2::<f64>::zeros((128, 128));
//!
//! // Compute 2D FFT
//! let spectrum = fft2d(&image.view()).unwrap();
//!
//! // Output shape is 128 x 65 due to Hermitian symmetry
//! assert_eq!(spectrum.shape(), &[128, 65]);
//! ```
//!
//! # Output Shape
//!
//! Due to Hermitian symmetry in real-to-complex FFTs, the output shape is
//! `(nrows, ncols/2 + 1)` for an input of shape `(nrows, ncols)`.

use ndarray::{Array2, ArrayView2};
use num_complex::Complex;

use crate::fft_backend::{C2rPlan2d, C2rPlanner2d, R2cPlan2d, R2cPlanner2d, r2c_output_size_2d};
use crate::{SpectrogramError, SpectrogramResult};

#[cfg(feature = "fftw")]
use crate::fft_backend::fftw_backend::FftwPlanner;

#[cfg(feature = "realfft")]
use crate::fft_backend::realfft_backend::RealFftPlanner;

/// Compute 2D FFT of a real-valued array (e.g., image).
///
/// Transforms a 2D real array into the frequency domain using a real-to-complex
/// FFT. The output exploits Hermitian symmetry, so only half the frequencies
/// are stored along the column dimension.
///
/// # Arguments
///
/// * `data` - Input 2D array with shape `(nrows, ncols)`
///
/// # Returns
///
/// Complex 2D array with shape `(nrows, ncols/2 + 1)` containing the frequency
/// domain representation.
///
/// # Errors
///
/// Returns `SpectrogramError` if the input array is invalid or processing fails.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::fft2d;
/// use ndarray::Array2;
///
/// let img = Array2::<f64>::zeros((64, 64));
/// let spectrum = fft2d(&img.view()).unwrap();
/// assert_eq!(spectrum.shape(), &[64, 33]); // 64 x (64/2 + 1)
/// ```
///
/// # Performance
///
/// For batch processing of multiple arrays with the same dimensions, use
/// [`Fft2dPlanner`] to reuse FFT plans and avoid repeated setup overhead.
#[inline]
pub fn fft2d(data: &ArrayView2<f64>) -> SpectrogramResult<Array2<Complex<f64>>> {
    let (nrows, ncols) = (data.nrows(), data.ncols());

    if nrows == 0 || ncols == 0 {
        return Err(SpectrogramError::invalid_input(
            "array dimensions must be > 0",
        ));
    }

    // Ensure array is contiguous and row-major
    // Tricky to convert it here due to ownership and cloning - better to require caller to provide standard layout
    if !data.is_standard_layout() {
        return Err(SpectrogramError::invalid_input(
            "array must be contiguous and row-major (standard layout)",
        ));
    }

    let out_shape = r2c_output_size_2d(nrows, ncols);

    #[cfg(feature = "fftw")]
    let mut planner = FftwPlanner::new();

    #[cfg(feature = "realfft")]
    let mut planner = RealFftPlanner::new();

    let mut plan = planner.plan_r2c_2d(nrows, ncols)?;

    let input_slice = data
        .as_slice()
        .ok_or_else(|| SpectrogramError::invalid_input("array must be contiguous"))?;

    let mut output = vec![Complex::new(0.0, 0.0); out_shape.0 * out_shape.1];

    plan.process(input_slice, &mut output)?;

    Array2::from_shape_vec(out_shape, output)
        .map_err(|e| SpectrogramError::invalid_input(format!("failed to reshape output: {e}")))
}

/// Compute inverse 2D FFT from frequency domain back to spatial domain.
///
/// Transforms a complex 2D frequency array back into a real spatial array.
/// This is the inverse operation of [`fft2d`].
///
/// # Arguments
///
/// * `spectrum` - Complex frequency array with shape `(nrows, ncols/2 + 1)`
/// * `output_ncols` - Number of columns in the output (must be even or odd as per original)
///
/// # Returns
///
/// Real 2D array with shape `(nrows, output_ncols)`.
///
/// # Errors
///
/// Returns `SpectrogramError` if the input array is invalid or processing fails.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::{fft2d, ifft2d};
/// use spectrograms::nzu;
/// use ndarray::Array2;
///
/// let original = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
///     (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
/// });
///
/// let spectrum = fft2d(&original.view()).unwrap();
/// let reconstructed = ifft2d(&spectrum, 64).unwrap();
///
/// // Check roundtrip accuracy
/// for ((i, j), &val) in original.indexed_iter() {
///     assert!((reconstructed[[i, j]] - val).abs() < 1e-10);
/// }
/// ```
#[inline]
pub fn ifft2d(
    spectrum: &Array2<Complex<f64>>,
    output_ncols: usize,
) -> SpectrogramResult<Array2<f64>> {
    let nrows = spectrum.nrows();

    if nrows == 0 || output_ncols == 0 {
        return Err(SpectrogramError::invalid_input("dimensions must be > 0"));
    }

    // Verify input shape matches expected frequency domain shape
    let expected_ncols = output_ncols / 2 + 1;
    if spectrum.ncols() != expected_ncols {
        return Err(SpectrogramError::dimension_mismatch(
            expected_ncols,
            spectrum.ncols(),
        ));
    }

    if !spectrum.is_standard_layout() {
        return Err(SpectrogramError::invalid_input(
            "array must be contiguous and row-major (standard layout)",
        ));
    }

    #[cfg(feature = "fftw")]
    let mut planner = FftwPlanner::new();

    #[cfg(feature = "realfft")]
    let mut planner = RealFftPlanner::new();

    let mut plan = planner.plan_c2r_2d(nrows, output_ncols)?;

    let input_slice = spectrum
        .as_slice()
        .ok_or_else(|| SpectrogramError::invalid_input("array must be contiguous"))?;

    let mut output = vec![0.0; nrows * output_ncols];

    plan.process(input_slice, &mut output)?;

    Array2::from_shape_vec((nrows, output_ncols), output)
        .map_err(|e| SpectrogramError::invalid_input(format!("failed to reshape output: {e}")))
}

/// Compute 2D power spectrum (squared magnitude).
///
/// # Arguments
///
/// * `data` - Input 2D array with shape `(nrows, ncols)`
///
/// # Returns
/// Returns `|FFT(x)|²` for each frequency component, useful for analyzing
/// energy distribution across frequencies.
///
/// # Errors
///
/// Returns `SpectrogramError` if the input array is invalid or processing fails.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::power_spectrum_2d;
/// use ndarray::Array2;
///
/// let img = Array2::<f64>::ones((32, 32));
/// let power = power_spectrum_2d(&img.view()).unwrap();
///
/// // DC component (0,0) should have all the energy
/// assert!(power[[0, 0]] > 1000.0);
/// ```
#[inline]
pub fn power_spectrum_2d(data: &ArrayView2<f64>) -> SpectrogramResult<Array2<f64>> {
    let spectrum = fft2d(data)?;
    let power = spectrum.mapv(|c| c.norm_sqr());
    Ok(power)
}

/// Compute 2D magnitude spectrum.
///
/// # Arguments
///
/// * `data` - Input 2D array with shape `(nrows, ncols)`
///
/// # Returns
///
/// Real 2D array with shape `(nrows, ncols/2 + 1)` containing the magnitude spectrum.
///
/// # Errors
///
/// Returns `SpectrogramError` if the input array is invalid or processing fails.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::magnitude_spectrum_2d;
/// use ndarray::Array2;
///
/// let img = Array2::<f64>::zeros((32, 32));
/// let magnitude = magnitude_spectrum_2d(&img.view()).unwrap();
/// ```
#[inline]
pub fn magnitude_spectrum_2d(data: &ArrayView2<f64>) -> SpectrogramResult<Array2<f64>> {
    let spectrum = fft2d(data)?;
    let magnitude = spectrum.mapv(num_complex::Complex::norm);
    Ok(magnitude)
}

/// Shift zero-frequency component to center (like `numpy.fft.fftshift`).
///
/// This rearranges the FFT output so that the DC (zero-frequency) component
/// is at the center of the array, which is useful for visualization.
///
/// # Arguments
///
/// * `arr` - Input 2D array to shift
///
/// # Returns
///
/// 2D array with quadrants swapped to center the zero-frequency component.
///
/// # Note
///
/// This function works on the **full** FFT output. Since [`fft2d`] returns
/// a reduced spectrum due to Hermitian symmetry, you may need to expand it
/// first for a full centered view, or use this on magnitude/power spectra.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::{power_spectrum_2d, fftshift};
/// use ndarray::{Array2, ArrayView2};
///
/// let img = Array2::<f64>::ones((32, 32));
/// let power = power_spectrum_2d(&img.view()).unwrap();
/// let shifted = fftshift(power);
/// ```
#[inline]
#[must_use]
pub fn fftshift<T: Clone>(arr: Array2<T>) -> Array2<T> {
    let (nrows, ncols) = arr.dim();
    let row_half = nrows / 2;
    let col_half = ncols / 2;

    let mut result = arr.clone();

    // Swap quadrants:
    // Original layout:  [0,0] [0,1]
    //                   [1,0] [1,1]
    // After shift:      [1,1] [1,0]
    //                   [0,1] [0,0]

    // Quadrant 0,0 -> shifted position (row_half, col_half)
    for i in 0..row_half {
        for j in 0..col_half {
            result[[i + row_half, j + col_half]] = arr[[i, j]].clone();
        }
    }

    // Quadrant 0,1 -> shifted position (row_half, 0)
    for i in 0..row_half {
        for j in col_half..ncols {
            result[[i + row_half, j - col_half]] = arr[[i, j]].clone();
        }
    }

    // Quadrant 1,0 -> shifted position (0, col_half)
    for i in row_half..nrows {
        for j in 0..col_half {
            result[[i - row_half, j + col_half]] = arr[[i, j]].clone();
        }
    }

    // Quadrant 1,1 -> shifted position (0, 0)
    for i in row_half..nrows {
        for j in col_half..ncols {
            result[[i - row_half, j - col_half]] = arr[[i, j]].clone();
        }
    }

    result
}

/// Inverse of [`fftshift`] - shift center back to corners.
///
/// Undoes the frequency shift performed by [`fftshift`].
///
/// # Arguments
///
/// * `arr` - Input 2D array to inverse shift
///
/// # Returns
///
/// 2D array with quadrants swapped back to original positions.
#[inline]
#[must_use]
pub fn ifftshift<T: Clone>(arr: Array2<T>) -> Array2<T> {
    let (nrows, ncols) = arr.dim();
    let row_half = nrows.div_ceil(2); // Ceiling division for odd sizes
    let col_half = ncols.div_ceil(2);

    let mut result = arr.clone();

    // Inverse swap of quadrants
    for i in 0..row_half {
        for j in 0..col_half {
            result[[i, j]] = arr[[i + nrows - row_half, j + ncols - col_half]].clone();
        }
    }

    for i in 0..row_half {
        for j in col_half..ncols {
            result[[i, j]] = arr[[i + nrows - row_half, j - col_half]].clone();
        }
    }

    for i in row_half..nrows {
        for j in 0..col_half {
            result[[i, j]] = arr[[i - row_half, j + ncols - col_half]].clone();
        }
    }

    for i in row_half..nrows {
        for j in col_half..ncols {
            result[[i, j]] = arr[[i - row_half, j - col_half]].clone();
        }
    }

    result
}

/// Shift zero-frequency component to center for 1D array.
///
/// # Arguments
///
/// * `arr` - Input 1D array to shift
///
/// # Returns
///
/// 1D array with zero-frequency component at center.
#[inline]
#[must_use]
pub fn fftshift_1d<T: Clone>(arr: Vec<T>) -> Vec<T> {
    let n = arr.len();
    let half = n / 2;

    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&arr[half..]);
    result.extend_from_slice(&arr[..half]);
    result
}

/// Inverse of fftshift for 1D arrays.
///
/// # Arguments
///
/// * `arr` - Input 1D array to inverse shift
///
/// # Returns
///
/// 1D array with quadrants swapped back to original positions.
#[inline]
#[must_use]
pub fn ifftshift_1d<T: Clone>(arr: Vec<T>) -> Vec<T> {
    let n = arr.len();
    let half = n.div_ceil(2);

    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&arr[n - half..]);
    result.extend_from_slice(&arr[..n - half]);
    result
}

/// Compute FFT sample frequencies (like `numpy.fft.fftfreq`).
///
/// Returns the sample frequencies (in cycles per unit of the sample spacing) for FFT output.
/// For an FFT of length `n` with sample spacing `d`:
/// - Frequencies go from 0 to (n-1)//(2*n*d) for positive frequencies
/// - Then wrap to negative frequencies
///
/// # Arguments
///
/// * `n` - Window length (number of samples)
/// * `d` - Sample spacing (inverse of sampling rate). Default is 1.0.
///
/// # Returns
///
/// Vector of length `n` containing the frequency bin centers in cycles per unit.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::fftfreq;
///
/// // For 8 samples with spacing 1.0
/// let freqs = fftfreq(8, 1.0);
/// // Returns: [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
///
/// // For temporal modulation at 16kHz sample rate with 100 frames
/// let freqs_hz = fftfreq(100, 1.0 / 16000.0);
/// // Returns frequencies in Hz
/// ```
#[inline]
#[must_use]
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let mut freqs = Vec::with_capacity(n);
    let n_f64 = n as f64;
    let n_half = n.div_ceil(2);

    // Positive frequencies: 0, 1, 2, ..., (n-1)/2
    for i in 0..n_half {
        freqs.push(i as f64 / (n_f64 * d));
    }

    // Negative frequencies: -n/2, ..., -2, -1
    for i in n_half..n {
        freqs.push((i as f64 - n_f64) / (n_f64 * d));
    }

    freqs
}

/// Compute FFT sample frequencies for real FFT (like `numpy.fft.rfftfreq`).
///
/// Returns only the positive frequencies for a real-to-complex FFT.
/// For an FFT of length `n` with sample spacing `d`, returns `n/2 + 1` frequencies.
///
/// # Arguments
///
/// * `n` - Window length (number of samples in original real signal)
/// * `d` - Sample spacing (inverse of sampling rate). Default is 1.0.
///
/// # Returns
///
/// Vector of length `n/2 + 1` containing the positive frequency bin centers.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::rfftfreq;
///
/// // For 8 samples
/// let freqs = rfftfreq(8, 1.0);
/// // Returns: [0.0, 0.125, 0.25, 0.375, 0.5]
/// ```
#[inline]
#[must_use]
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let n_out = n / 2 + 1;
    let mut freqs = Vec::with_capacity(n_out);
    let n_f64 = n as f64;

    for i in 0..n_out {
        freqs.push(i as f64 / (n_f64 * d));
    }

    freqs
}

/// 2D FFT planner for efficient batch processing.
///
/// Caches FFT plans internally, avoiding repeated setup overhead when
/// processing multiple arrays with the same dimensions.
///
/// # Examples
///
/// ```rust
/// use spectrograms::fft2d::Fft2dPlanner;
/// use ndarray::Array2;
///
/// let mut planner = Fft2dPlanner::new();
///
/// // Process multiple images with same size
/// for _ in 0..10 {
///     let img = Array2::<f64>::zeros((128, 128));
///     let spectrum = planner.fft2d(&img.view()).unwrap();
/// }
/// ```
pub struct Fft2dPlanner {
    #[cfg(feature = "fftw")]
    inner: FftwPlanner,

    #[cfg(feature = "realfft")]
    inner: RealFftPlanner,
}

impl Fft2dPlanner {
    /// Create a new 2D FFT planner.
    ///
    /// # Returns
    ///
    /// A new `Fft2dPlanner` instance.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "fftw")]
            inner: FftwPlanner::new(),

            #[cfg(feature = "realfft")]
            inner: RealFftPlanner::new(),
        }
    }

    /// Compute 2D FFT using cached plans.
    ///
    /// Reuses plans for arrays with dimensions seen before, avoiding setup overhead.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D array with shape `(nrows, ncols)`
    ///
    /// # Returns
    ///
    /// Complex 2D array with shape `(nrows, ncols/2 + 1)` containing the frequency domain representation.
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError` if the input array is invalid or processing fails.
    #[inline]
    pub fn fft2d(&mut self, data: &ArrayView2<f64>) -> SpectrogramResult<Array2<Complex<f64>>> {
        let (nrows, ncols) = (data.nrows(), data.ncols());

        if nrows == 0 || ncols == 0 {
            return Err(SpectrogramError::invalid_input(
                "array dimensions must be > 0",
            ));
        }

        if !data.is_standard_layout() {
            return Err(SpectrogramError::invalid_input(
                "array must be contiguous and row-major (standard layout)",
            ));
        }

        let out_shape = r2c_output_size_2d(nrows, ncols);
        let mut plan = self.inner.plan_r2c_2d(nrows, ncols)?;

        let input_slice = data
            .as_slice()
            .ok_or_else(|| SpectrogramError::invalid_input("array must be contiguous"))?;

        let mut output = vec![Complex::new(0.0, 0.0); out_shape.0 * out_shape.1];
        plan.process(input_slice, &mut output)?;

        Array2::from_shape_vec(out_shape, output)
            .map_err(|e| SpectrogramError::invalid_input(format!("failed to reshape output: {e}")))
    }

    /// Compute inverse 2D FFT using cached plans.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex frequency array with shape `(nrows, ncols/2 + 1)`
    /// * `output_ncols` - Number of columns in the output (must match original)
    ///
    /// # Returns
    ///
    /// Real 2D array with shape `(nrows, output_ncols)`.
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError` if the input array is invalid or processing fails.    
    #[inline]
    pub fn ifft2d(
        &mut self,
        spectrum: &ArrayView2<Complex<f64>>,
        output_ncols: usize,
    ) -> SpectrogramResult<Array2<f64>> {
        let nrows = spectrum.nrows();

        if nrows == 0 || output_ncols == 0 {
            return Err(SpectrogramError::invalid_input("dimensions must be > 0"));
        }

        let expected_ncols = output_ncols / 2 + 1;
        if spectrum.ncols() != expected_ncols {
            return Err(SpectrogramError::dimension_mismatch(
                expected_ncols,
                spectrum.ncols(),
            ));
        }

        if !spectrum.is_standard_layout() {
            return Err(SpectrogramError::invalid_input(
                "array must be contiguous and row-major (standard layout)",
            ));
        }

        let mut plan = self.inner.plan_c2r_2d(nrows, output_ncols)?;

        let input_slice = spectrum
            .as_slice()
            .ok_or_else(|| SpectrogramError::invalid_input("array must be contiguous"))?;

        let mut output = vec![0.0; nrows * output_ncols];
        plan.process(input_slice, &mut output)?;

        Array2::from_shape_vec((nrows, output_ncols), output)
            .map_err(|e| SpectrogramError::invalid_input(format!("failed to reshape output: {e}")))
    }

    /// Compute 2D power spectrum using cached plans.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D array with shape `(nrows, ncols)`
    ///
    /// # Returns
    ///
    /// Real 2D array with shape `(nrows, ncols/2 + 1)` containing the power spectrum.
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError` if the input array is invalid or processing fails.
    #[inline]
    pub fn power_spectrum_2d(&mut self, data: &ArrayView2<f64>) -> SpectrogramResult<Array2<f64>> {
        let spectrum = self.fft2d(data)?;
        let power = spectrum.mapv(|c| c.norm_sqr());
        Ok(power)
    }

    /// Compute 2D magnitude spectrum using cached plans.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D array with shape `(nrows, ncols)`
    ///
    /// # Returns
    ///
    /// Real 2D array with shape `(nrows, ncols/2 + 1)` containing the magnitude spectrum.
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError` if the input array is invalid or processing fails.
    #[inline]
    pub fn magnitude_spectrum_2d(
        &mut self,
        data: &ArrayView2<f64>,
    ) -> SpectrogramResult<Array2<f64>> {
        let spectrum = self.fft2d(data)?;
        let magnitude = spectrum.mapv(num_complex::Complex::norm);
        Ok(magnitude)
    }
}

impl Default for Fft2dPlanner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft2d_zeros() {
        let data = Array2::<f64>::zeros((32, 32));
        let result = fft2d(&data.view());
        assert!(result.is_ok());
        let spectrum = result.unwrap();
        assert_eq!(spectrum.shape(), &[32, 17]); // 32 x (32/2 + 1)

        // All values should be ~zero
        for val in spectrum.iter() {
            assert!(val.norm() < 1e-10);
        }
    }

    #[test]
    fn test_fft2d_ones() {
        let data = Array2::<f64>::ones((32, 32));
        let spectrum = fft2d(&data.view()).unwrap();

        // DC component should have all energy
        assert!(spectrum[[0, 0]].norm() > 1000.0);

        // Other components should be ~zero
        for i in 1..32 {
            assert!(spectrum[[i, 0]].norm() < 1e-10);
        }
    }

    #[test]
    fn test_roundtrip() {
        let original = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        let spectrum = fft2d(&original.view()).unwrap();
        let reconstructed = ifft2d(&spectrum, 64).unwrap();

        for ((i, j), &val) in original.indexed_iter() {
            assert!((reconstructed[[i, j]] - val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_planner_reuse() {
        let mut planner = Fft2dPlanner::new();

        for _ in 0..5 {
            let data = Array2::<f64>::zeros((32, 32));
            let result = planner.fft2d(&data.view());
            assert!(result.is_ok());
        }
    }
}
