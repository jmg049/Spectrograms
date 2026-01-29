#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(unused_unsafe)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)] // False positives with NonZeroUsize conversions
#![allow(clippy::cast_possible_wrap)] // False positives with NonZeroUsize conversions
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::tuple_array_conversions)] // False positives with ndarray indexing
#![allow(clippy::identity_op)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::needless_pass_by_value)] // False positives with PyO3
#![allow(clippy::trivially_copy_pass_by_ref)] // False positives with PyO3 (likes of __repr__ and any pymethod requires &self)
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::multiple_unsafe_ops_per_block)]
#![allow(clippy::doc_markdown)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exhaustive_structs)]
#![warn(clippy::missing_inline_in_public_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::iter_cloned_collect)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

//! # Spectrograms - FFT-Based Computations
//!
//! High-performance FFT-based computations for audio and image processing.
//!
//! # Overview
//!
//! This library provides:
//! - **1D FFTs**: For time-series and audio signals
//! - **2D FFTs**: For images and spatial data
//! - **Spectrograms**: Time-frequency representations (STFT, Mel, ERB, CQT)
//! - **Image operations**: Convolution, filtering, edge detection
//! - **Two backends**: `RealFFT` (pure Rust) or FFTW (fastest)
//! - **Plan-based API**: Reusable plans for batch processing
//!
//! # Domain Organization
//!
//! The library is organized by application domain:
//!
//! - [`audio`] - Audio processing (spectrograms, MFCC, chroma, pitch analysis)
//! - [`image`] - Image processing (convolution, filtering, frequency analysis)
//! - [`mod@fft`] - Core FFT operations (1D and 2D transforms)
//!
//! All functionality is also exported at the crate root for convenience.
//!
//! # Audio Processing
//!
//! Compute various types of spectrograms:
//! - Linear-frequency spectrograms
//! - Mel-frequency spectrograms
//! - ERB spectrograms
//! - Logarithmic-frequency spectrograms
//! - CQT (Constant-Q Transform)
//!
//! With multiple amplitude scales:
//! - Power (`|X|²`)
//! - Magnitude (`|X|`)
//! - Decibels (`10·log₁₀(power)`)
//!
//! # Image Processing
//!
//! Frequency-domain operations for images:
//! - 2D FFT and inverse FFT
//! - Convolution via FFT (faster for large kernels)
//! - Spatial filtering (low-pass, high-pass, band-pass)
//! - Edge detection
//! - Sharpening and blurring
//!
//! # Features
//!
//! - **Two FFT backends**: `RealFFT` (default, pure Rust) or FFTW (fastest performance)
//! - **Plan-based computation**: Reuse FFT plans for efficient batch processing
//! - **Comprehensive window functions**: Hanning, Hamming, Blackman, Kaiser, Gaussian, etc.
//! - **Type-safe API**: Compile-time guarantees for spectrogram types
//!
//! # Quick Start
//!
//! ## Audio: Compute a Mel Spectrogram
//!
//! ```
//! use spectrograms::*;
//! use std::f64::consts::PI;
//! use non_empty_slice::NonEmptyVec;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Generate a sine wave at 440 Hz
//! let sample_rate = 16000.0;
//! let samples_vec: Vec<f64> = (0..16000)
//!     .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
//!     .collect();
//! let samples = NonEmptyVec::new(samples_vec).unwrap();
//!
//! // Set up parameters
//! let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, sample_rate)?;
//! let mel = MelParams::new(nzu!(80), 0.0, 8000.0)?;
//!
//! // Compute Mel spectrogram
//! let spec = MelPowerSpectrogram::compute(samples.as_ref(), &params, &mel, None)?;
//! println!("Computed {} bins x {} frames", spec.n_bins(), spec.n_frames());
//! # Ok(())
//! # }
//! ```
//!
//! ## Image: Apply Gaussian Blur via FFT
//!
//! ```
//! use spectrograms::image_ops::*;
//! use spectrograms::nzu;
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a 256x256 image
//! let image = Array2::<f64>::from_shape_fn((256, 256), |(i, j)| {
//!     ((i as f64 - 128.0).powi(2) + (j as f64 - 128.0).powi(2)).sqrt()
//! });
//!
//! // Apply Gaussian blur
//! let kernel = gaussian_kernel_2d(nzu!(9), 2.0)?;
//! let blurred = convolve_fft(&image.view(), &kernel.view())?;
//! # Ok(())
//! # }
//! ```
//!
//! ## General: 2D FFT
//!
//! ```
//! use spectrograms::fft2d::*;
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Array2::<f64>::zeros((128, 128));
//! let spectrum = fft2d(&data.view())?;
//! let power = power_spectrum_2d(&data.view())?;
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flags
//!
//! The library requires exactly one FFT backend:
//!
//! - `realfft` (default): Pure-Rust FFT implementation, no system dependencies
//! - `fftw`: Uses FFTW C library for fastest performance (requires system install)
//!
//! # Examples
//!
//! ## Mel Spectrogram
//!
//! ```
//! use spectrograms::*;
//! use non_empty_slice::non_empty_vec;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let samples = non_empty_vec![0.0; nzu!(16000)];
//!
//! let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, 16000.0)?;
//! let mel = MelParams::new(nzu!(80), 0.0, 8000.0)?;
//! let db = LogParams::new(-80.0)?;
//!
//! let spec = MelDbSpectrogram::compute(samples.as_ref(), &params, &mel, Some(&db))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Efficient Batch Processing
//!
//! ```
//! use spectrograms::*;
//! use non_empty_slice::non_empty_vec;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let signals = vec![non_empty_vec![0.0; nzu!(16000)], non_empty_vec![0.0; nzu!(16000)]];
//!
//! let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, 16000.0)?;
//!
//! // Create plan once, reuse for all signals
//! let planner = SpectrogramPlanner::new();
//! let mut plan = planner.linear_plan::<Power>(&params, None)?;
//!
//! for signal in &signals {
//!     let spec = plan.compute(&signal)?;
//!     // Process spec...
//! }
//! # Ok(())
//! # }
//! ```

mod chroma;
mod cqt;
mod erb;
mod error;
pub mod fft2d;
mod fft_backend;
pub mod image_ops;
mod mfcc;
mod spectrogram;
mod window;

#[cfg(feature = "python")]
mod python;

// ============================================================================
// Domain-Specific Module Organization
// ============================================================================

/// Audio processing utilities (spectrograms, MFCC, chroma, etc.)
///
/// This module contains all audio-related functionality:
/// - Spectrogram computation (Linear, Mel, ERB, CQT)
/// - MFCC (Mel-Frequency Cepstral Coefficients)
/// - Chromagram (pitch class profiles)
/// - Window functions
///
/// # Examples
///
/// ```
/// use spectrograms::{nzu, audio::*};
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let samples = non_empty_vec![0.0; nzu!(16000)];
/// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
/// let params = SpectrogramParams::new(stft, 16000.0)?;
/// let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;
/// # Ok(())
/// # }
/// ```
pub mod audio {
    pub use crate::chroma::*;
    pub use crate::cqt::*;
    pub use crate::erb::*;
    pub use crate::mfcc::*;
    pub use crate::spectrogram::*;
    pub use crate::window::*;
}

/// Image processing utilities (convolution, filtering, etc.)
///
/// This module contains image processing operations using 2D FFTs:
/// - Convolution and correlation
/// - Spatial filtering (low-pass, high-pass, band-pass)
/// - Edge detection
/// - Sharpening and blurring
///
/// # Examples
///
/// ```
/// use spectrograms::image::*;
/// use spectrograms::nzu;
/// use ndarray::Array2;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = Array2::<f64>::zeros((128, 128));
/// let kernel = gaussian_kernel_2d(nzu!(5), 1.0)?;
/// let blurred = convolve_fft(&image.view(), &kernel.view())?;
/// # Ok(())
/// # }
/// ```
pub mod image {
    pub use crate::image_ops::*;
}

/// Core FFT operations (1D and 2D)
///
/// This module provides direct access to FFT functions:
/// - 1D FFT: `fft()`, `rfft()`, `irfft()`
/// - 2D FFT: `fft2d()`, `ifft2d()`
/// - STFT: `stft()`, `istft()`
/// - Power/magnitude spectra
///
/// # Examples
///
/// ```
/// use spectrograms::{nzu, fft::*};
/// use ndarray::Array2;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // 1D FFT
/// let signal = non_empty_vec![0.0; nzu!(1024)];
/// let spectrum = rfft(&signal, nzu!(1024))?;
///
/// // 2D FFT
/// let image = Array2::<f64>::zeros((128, 128));
/// let spectrum_2d = fft2d(&image.view())?;
/// # Ok(())
/// # }
/// ```
pub mod fft {
    pub use crate::fft2d::*;
    pub use crate::spectrogram::{
        fft, irfft, istft, magnitude_spectrum, power_spectrum, rfft, stft,
    };
}

// Re-export everything at top level for backward compatibility
pub use chroma::{
    ChromaNorm, ChromaParams, Chromagram, N_CHROMA, chromagram, chromagram_from_spectrogram,
};
pub use cqt::{CqtParams, CqtResult, cqt};
pub use erb::{ErbParams, GammatoneParams};
pub use error::{SpectrogramError, SpectrogramResult};
pub use fft_backend::{C2rPlan, C2rPlanner, R2cPlan, R2cPlanner, r2c_output_size};
pub use fft2d::*;
pub use image_ops::*;
pub use mfcc::{Mfcc, MfccParams, mfcc, mfcc_from_log_mel};
pub use spectrogram::*;
pub use window::{
    WindowType, blackman_window, gaussian_window, hamming_window, hanning_window, kaiser_window,
    rectangular_window,
};
#[macro_export]
macro_rules! nzu {
    ($rate:expr) => {{
        const RATE: usize = $rate;
        const { assert!(RATE > 0, "non zero usize must be greater than 0") };
        // SAFETY: We just asserted RATE > 0 at compile time
        unsafe { ::core::num::NonZeroUsize::new_unchecked(RATE) }
    }};
}

#[cfg(all(feature = "fftw", feature = "realfft"))]
compile_error!(
    "Features 'fftw' and 'realfft' are mutually exclusive. Please enable only one of them."
);

#[cfg(not(any(feature = "fftw", feature = "realfft")))]
compile_error!("At least one FFT backend feature must be enabled: 'fftw' or 'realfft'.");

#[cfg(feature = "realfft")]
pub use fft_backend::realfft_backend::*;

#[cfg(feature = "fftw")]
pub use fft_backend::fftw_backend::*;

/// Python module definition for `PyO3`.
///
/// This module is only available when the `python` feature is enabled.
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _spectrograms(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(py, m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

pub(crate) fn min_max_single_pass<A: AsRef<[f64]>>(data: A) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &val in data.as_ref() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }
    (min_val, max_val)
}
