//! Image processing operations using 2D FFTs.
//!
//! This module provides common image processing tasks that leverage 2D FFTs
//! for efficient computation:
//! - Convolution and correlation
//! - Spatial filtering (low-pass, high-pass, band-pass)
//! - Edge detection in frequency domain
//! - Image sharpening and blurring
//!
//! # Performance
//!
//! FFT-based convolution is faster than spatial convolution for large kernels
//! (typically > 7x7), due to the convolution theorem: multiplication in the
//! frequency domain is equivalent to convolution in the spatial domain.
//!
//! # Examples
//!
//! ```rust
//! use spectrograms::image_ops::{gaussian_kernel_2d, convolve_fft};
//! use spectrograms::nzu;
//! use ndarray::Array2;
//!
//! // Create a 256x256 image
//! let image = Array2::<f64>::zeros((256, 256));
//!
//! // Apply Gaussian blur
//! let kernel = gaussian_kernel_2d(nzu!(9), 2.0).unwrap();
//! let blurred = convolve_fft(&image.view(), &kernel.view()).unwrap();
//! ```

use ndarray::{Array2, ArrayView2};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

use crate::fft2d::{fft2d, ifft2d};
use crate::{SpectrogramError, SpectrogramResult};

/// Convolve 2D image with kernel using FFT (faster for large kernels).
///
/// Performs 2D convolution via the convolution theorem: multiplication in the
/// frequency domain equals convolution in the spatial domain. This is typically
/// faster than spatial convolution for kernels larger than ~7x7.
///
/// # Arguments
///
/// * `image` - Input image (nrows x ncols)
/// * `kernel` - Convolution kernel (must be smaller than or equal to image size)
///
/// # Returns
///
/// Convolved image (same size as input)
///
/// # Errors
///
/// Returns an error if:
/// - Kernel dimensions exceed image dimensions
/// - Kernel dimensions are zero
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::{gaussian_kernel_2d, convolve_fft};
/// use spectrograms::nzu;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_shape_fn((128, 128), |(i, j)| {
///     ((i as f64 - 64.0).powi(2) + (j as f64 - 64.0).powi(2)).sqrt()
/// });
///
/// // Gaussian blur with 5x5 kernel, sigma=1.0
/// let kernel = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();
/// let blurred = convolve_fft(&image.view(), &kernel.view()).unwrap();
/// ```
///
/// # Performance
///
/// For repeated convolutions with the same kernel size, consider using a
/// planner from the [`fft2d`](mod@crate::fft2d) module to cache FFT plans.
#[inline]
pub fn convolve_fft(
    image: &ArrayView2<f64>,
    kernel: &ArrayView2<f64>,
) -> SpectrogramResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    // Validate kernel size
    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SpectrogramError::invalid_input(
            "kernel dimensions must not exceed image dimensions",
        ));
    }

    if ker_rows == 0 || ker_cols == 0 {
        return Err(SpectrogramError::invalid_input(
            "kernel dimensions must be > 0",
        ));
    }

    // Pad kernel to image size with proper centering for FFT
    // The kernel center should be at (0, 0) in the padded array for correct convolution
    let padded_kernel = pad_kernel_for_fft(kernel, (img_rows, img_cols));

    // FFT both image and kernel
    let img_freq = fft2d(image)?;
    let kernel_freq = fft2d(&padded_kernel.view())?;

    // Multiply in frequency domain (element-wise)
    let result_freq = &img_freq * &kernel_freq;

    // Inverse FFT back to spatial domain
    let result = ifft2d(&result_freq, img_cols)?;

    Ok(result)
}

/// Pad kernel for FFT convolution with proper phase shifting.
///
/// For FFT convolution to work correctly, the kernel's center should be
/// at position (0, 0) after padding. This means we need to place the kernel's
/// center at (0, 0) and wrap the rest of the kernel around using periodic
/// boundary conditions.
fn pad_kernel_for_fft(kernel: &ArrayView2<f64>, target_shape: (usize, usize)) -> Array2<f64> {
    let (target_rows, target_cols) = target_shape;
    let (ker_rows, ker_cols) = kernel.dim();

    let mut result = Array2::<f64>::zeros(target_shape);

    // Kernel center position
    let ker_center_row = ker_rows / 2;
    let ker_center_col = ker_cols / 2;

    // Place kernel values in padded array with center at (0, 0)
    for i in 0..ker_rows {
        for j in 0..ker_cols {
            // Calculate offset from kernel center
            let row_offset = i as isize - ker_center_row as isize;
            let col_offset = j as isize - ker_center_col as isize;

            // Wrap around to place center at (0, 0)
            let target_row = row_offset.rem_euclid(target_rows as isize) as usize;
            let target_col = col_offset.rem_euclid(target_cols as isize) as usize;

            result[[target_row, target_col]] = kernel[[i, j]];
        }
    }

    result
}

/// Create 2D Gaussian kernel for blurring.
///
/// Generates a normalized Gaussian kernel suitable for image smoothing.
/// The kernel is normalized so its elements sum to 1.0.
///
/// # Arguments
///
/// * `size` - Kernel size (must be odd and > 0, e.g., 3, 5, 7, 9)
/// * `sigma` - Standard deviation of the Gaussian (must be > 0, controls blur amount)
///
/// # Returns
///
/// A `size x size` Gaussian kernel, normalized to sum to 1.0.
///
/// # Errors
///
/// Returns an error if:
/// - `size` is zero or even
/// - `sigma` is zero or negative
///
/// # Examples
///
/// ```rust
/// use spectrograms::{nzu, image_ops::gaussian_kernel_2d};
///
/// // 5x5 Gaussian with sigma=1.0
/// let kernel = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();
/// assert_eq!(kernel.shape(), &[5, 5]);
///
/// // Kernel should be normalized
/// let sum: f64 = kernel.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
#[inline]
pub fn gaussian_kernel_2d(size: NonZeroUsize, sigma: f64) -> SpectrogramResult<Array2<f64>> {
    if size.get().is_multiple_of(2) {
        return Err(SpectrogramError::invalid_input(
            "kernel size must be odd and > 0",
        ));
    }
    if sigma <= 0.0 {
        return Err(SpectrogramError::invalid_input("sigma must be > 0"));
    }
    let size = size.get();
    let center = (size / 2) as f64;
    let variance = sigma * sigma;
    let coeff = 1.0 / (2.0 * PI * variance);

    let mut kernel = Array2::<f64>::zeros((size, size));

    for i in 0..size {
        for j in 0..size {
            let x = i as f64 - center;
            let y = j as f64 - center;
            let exponent = -(x * x + y * y) / (2.0 * variance);
            kernel[[i, j]] = coeff * exponent.exp();
        }
    }

    // Normalize to sum to 1.0
    let sum: f64 = kernel.iter().sum();
    kernel.mapv_inplace(|v| v / sum);

    Ok(kernel)
}

/// Create a circular low-pass filter mask in frequency domain.
///
/// The mask is designed for non-shifted FFT output where DC is at (0, 0).
/// Frequencies are measured by their distance from DC, wrapping around
/// the array boundaries.
///
/// # Arguments
///
/// * `shape` - Shape of the frequency domain array `(nrows, ncols)`
/// * `cutoff_fraction` - Cutoff radius as fraction of image size (0.0 to 1.0)
///
/// # Returns
///
/// Binary mask (0.0 or 1.0) where 1.0 indicates frequencies to keep.
fn create_lowpass_mask(shape: (usize, usize), cutoff_fraction: f64) -> Array2<f64> {
    let (nrows, ncols) = shape;
    let mut mask = Array2::<f64>::zeros(shape);

    // DC is at (0, 0), so we measure distance from corners with wrapping
    let max_freq_row = (nrows / 2) as f64;
    let max_freq_col = (ncols / 2) as f64;
    let max_radius = (max_freq_row.min(max_freq_col) * cutoff_fraction).powi(2);

    for i in 0..nrows {
        for j in 0..ncols {
            // Distance from DC at (0, 0) with periodic wrapping
            let freq_row = if i <= nrows / 2 {
                i as f64
            } else {
                (i as f64 - nrows as f64).abs()
            };
            let freq_col = if j <= ncols / 2 {
                j as f64
            } else {
                (j as f64 - ncols as f64).abs()
            };

            let dist_sq = freq_col.mul_add(freq_col, freq_row.powi(2));
            if dist_sq <= max_radius {
                mask[[i, j]] = 1.0;
            }
        }
    }

    mask
}

/// Apply low-pass filter (suppress high frequencies).
///
/// Removes high-frequency components, effectively smoothing the image.
/// This is useful for noise reduction and blurring.
///
/// # Arguments
///
/// * `image` - Input image
/// * `cutoff_fraction` - Cutoff radius as fraction of image size (0.0 to 1.0)
///   - 0.1 = keep only 10% of lowest frequencies (strong blur)
///   - 0.5 = keep 50% of frequencies (moderate blur)
///   - 1.0 = keep all frequencies (no effect)
///
///
/// # Returns
///
/// Smoothed image after low-pass filtering.
///
/// # Errors
///
/// Returns an error if `cutoff_fraction` is not between 0.0 and 1.0.
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::lowpass_filter;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_elem((128, 128), 1.0);
/// let smoothed = lowpass_filter(&image.view(), 0.3).unwrap();
/// ```
#[inline]
pub fn lowpass_filter(
    image: &ArrayView2<f64>,
    cutoff_fraction: f64,
) -> SpectrogramResult<Array2<f64>> {
    if !(0.0..=1.0).contains(&cutoff_fraction) {
        return Err(SpectrogramError::invalid_input(
            "cutoff_fraction must be between 0.0 and 1.0",
        ));
    }

    let spectrum = fft2d(image)?;
    let mask = create_lowpass_mask(spectrum.dim(), cutoff_fraction);

    // Apply mask (element-wise multiplication)
    let filtered = &spectrum * &mask.mapv(|v| num_complex::Complex::new(v, 0.0));

    ifft2d(&filtered, image.ncols())
}

/// Apply high-pass filter (suppress low frequencies).
///
/// Removes low-frequency components, emphasizing edges and fine details.
/// This is useful for edge detection and sharpening.
///
/// # Arguments
///
/// * `image` - Input image
/// * `cutoff_fraction` - Cutoff radius as fraction of image size (0.0 to 1.0)
///   - 0.1 = remove lowest 10% of frequencies (strong edge emphasis)
///   - 0.5 = remove 50% of frequencies (moderate edge emphasis)
///
/// # Returns
///
/// Image after high-pass filtering.
///
/// # Errors
///
/// Returns an error if `cutoff_fraction` is not between 0.0 and 1.0.
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::highpass_filter;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_elem((128, 128), 1.0);
/// let edges = highpass_filter(&image.view(), 0.1).unwrap();
/// ```
#[inline]
pub fn highpass_filter(
    image: &ArrayView2<f64>,
    cutoff_fraction: f64,
) -> SpectrogramResult<Array2<f64>> {
    if !(0.0..=1.0).contains(&cutoff_fraction) {
        return Err(SpectrogramError::invalid_input(
            "cutoff_fraction must be between 0.0 and 1.0",
        ));
    }

    let spectrum = fft2d(image)?;
    let lowpass_mask = create_lowpass_mask(spectrum.dim(), cutoff_fraction);

    // High-pass = 1 - low-pass
    let highpass_mask = lowpass_mask.mapv(|v| 1.0 - v);

    // Apply mask
    let filtered = &spectrum * &highpass_mask.mapv(|v| num_complex::Complex::new(v, 0.0));

    ifft2d(&filtered, image.ncols())
}

/// Apply band-pass filter (keep frequencies in a specific range).
///
/// Keeps only frequencies within a specified band, removing both very low
/// and very high frequencies.
///
/// # Arguments
///
/// * `image` - Input image
/// * `low_cutoff` - Lower cutoff as fraction (0.0 to 1.0)
/// * `high_cutoff` - Upper cutoff as fraction (0.0 to 1.0), must be > `low_cutoff`
///
/// # Returns
///
/// Image after band-pass filtering.
///
/// # Errors
///
/// Returns an error if:
/// - `low_cutoff` or `high_cutoff` are not between 0.0 and 1.0
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::bandpass_filter;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_elem((128, 128), 1.0);
/// // Keep only middle frequencies (0.2 to 0.6)
/// let filtered = bandpass_filter(&image.view(), 0.2, 0.6).unwrap();
/// ```
#[inline]
pub fn bandpass_filter(
    image: &ArrayView2<f64>,
    low_cutoff: f64,
    high_cutoff: f64,
) -> SpectrogramResult<Array2<f64>> {
    if !(0.0..=1.0).contains(&low_cutoff) || !(0.0..=1.0).contains(&high_cutoff) {
        return Err(SpectrogramError::invalid_input(
            "cutoff fractions must be between 0.0 and 1.0",
        ));
    }

    if low_cutoff >= high_cutoff {
        return Err(SpectrogramError::invalid_input(
            "high_cutoff must be greater than low_cutoff",
        ));
    }

    let spectrum = fft2d(image)?;

    let low_mask = create_lowpass_mask(spectrum.dim(), low_cutoff);
    let high_mask = create_lowpass_mask(spectrum.dim(), high_cutoff);

    // Band-pass = high_mask - low_mask (frequencies between the two)
    let bandpass_mask = &high_mask - &low_mask;

    // Apply mask
    let filtered = &spectrum * &bandpass_mask.mapv(|v| num_complex::Complex::new(v, 0.0));

    ifft2d(&filtered, image.ncols())
}

/// Detect edges in an image using high-pass filtering.
///
/// Applies a high-pass filter to emphasize edges and rapid intensity changes.
/// This is a simple edge detection method that works in the frequency domain.
///
/// # Arguments
///
/// * `image` - Input image
///
/// # Returns
///
/// Image highlighting edges after high-pass filtering.
///
/// # Errors
///
/// Returns an error if high-pass filtering fails.
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::detect_edges_fft;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_elem((128, 128), 1.0);
/// let edges = detect_edges_fft(&image.view()).unwrap();
/// ```
///
/// # Note
///
/// For more sophisticated edge detection, consider using spatial domain
/// operators like Sobel or Canny. This FFT-based method is useful for
/// quick frequency-domain analysis.
#[inline]
pub fn detect_edges_fft(image: &ArrayView2<f64>) -> SpectrogramResult<Array2<f64>> {
    // High-pass filter with low cutoff removes DC and low frequencies
    highpass_filter(image, 0.1)
}

/// Sharpen an image by enhancing high frequencies.
///
/// Adds weighted high-frequency components back to the original image,
/// enhancing edges and fine details.
///
/// # Arguments
///
/// * `image` - Input image
/// * `amount` - Sharpening strength (typical range: 0.5 to 2.0)
///   - 0.5 = subtle sharpening
///   - 1.0 = moderate sharpening
///   - 2.0 = strong sharpening
///
/// # Returns
///
/// Sharpened image = original + amount x `high_frequencies`
///
/// # Errors
///
/// Returns an error if `amount` is negative.
///
/// # Examples
///
/// ```rust
/// use spectrograms::image_ops::sharpen_fft;
/// use ndarray::Array2;
///
/// let image = Array2::<f64>::from_elem((128, 128), 1.0);
/// let sharpened = sharpen_fft(&image.view(), 1.0).unwrap();
/// ```
#[inline]
pub fn sharpen_fft(image: &ArrayView2<f64>, amount: f64) -> SpectrogramResult<Array2<f64>> {
    if amount < 0.0 {
        return Err(SpectrogramError::invalid_input("amount must be >= 0"));
    }

    // Extract high-frequency components
    let high_freq = highpass_filter(image, 0.2)?;

    // Add weighted high frequencies to original
    Ok(image + &(high_freq * amount))
}

#[cfg(test)]
mod tests {
    use crate::nzu;

    use super::*;

    #[test]
    fn test_gaussian_kernel_normalized() {
        let kernel = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "kernel should sum to 1.0");
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let kernel = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();
        let center = 2;

        // Check symmetry around center
        for i in 0..5 {
            for j in 0..5 {
                let di = (i as isize - center as isize).unsigned_abs();
                let dj = (j as isize - center as isize).unsigned_abs();
                let mirrored_i = if i <= center {
                    center + di
                } else {
                    center - di
                };
                let mirrored_j = if j <= center {
                    center + dj
                } else {
                    center - dj
                };

                if mirrored_i < 5 && mirrored_j < 5 {
                    assert!(
                        (kernel[[i, j]] - kernel[[mirrored_i, mirrored_j]]).abs() < 1e-10,
                        "kernel should be symmetric"
                    );
                }
            }
        }
    }

    #[test]
    fn test_convolve_fft_identity() {
        let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| i as f64 + j as f64);

        // Identity kernel (1 at center, 0 elsewhere)
        let mut kernel = Array2::<f64>::zeros((3, 3));
        kernel[[1, 1]] = 1.0;

        let result = convolve_fft(&image.view(), &kernel.view()).unwrap();

        // Convolution with identity should return approximately original
        for i in 1..63 {
            for j in 1..63 {
                assert!((result[[i, j]] - image[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_lowpass_removes_high_freq() {
        // Create image with high-frequency pattern
        let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
            ((i as f64 * 0.5).sin() + (j as f64 * 0.5).cos()) * 50.0
        });

        let filtered = lowpass_filter(&image.view(), 0.2).unwrap();

        // High frequencies should be reduced (smaller variance)
        let original_var: f64 = image.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);
        let filtered_var: f64 = filtered.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);

        assert!(
            filtered_var < original_var,
            "low-pass should reduce variance"
        );
    }

    #[test]
    fn test_highpass_emphasizes_edges() {
        // Constant image (no edges)
        let constant = Array2::<f64>::from_elem((64, 64), 100.0);
        let edges = highpass_filter(&constant.view(), 0.1).unwrap();

        // High-pass of constant should be ~zero
        let max_val = edges.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_val < 1.0, "high-pass of constant should be ~zero");
    }

    #[test]
    fn test_bandpass_bounds() {
        let image = Array2::<f64>::from_elem((64, 64), 1.0);

        // Valid bandpass
        assert!(bandpass_filter(&image.view(), 0.2, 0.8).is_ok());

        // Invalid: low >= high
        assert!(bandpass_filter(&image.view(), 0.8, 0.2).is_err());
        assert!(bandpass_filter(&image.view(), 0.5, 0.5).is_err());
        // Invalid: out of bounds
        assert!(bandpass_filter(&image.view(), -0.1, 0.5).is_err());
        assert!(bandpass_filter(&image.view(), 0.5, 1.5).is_err());
    }
}
