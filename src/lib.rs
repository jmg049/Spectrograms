#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exhaustive_structs)]
#![warn(clippy::missing_inline_in_public_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::iter_cloned_collect)]
#![allow(clippy::needless_pass_by_value)] // False positives with PyO3
#![warn(clippy::indexing_slicing)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::multiple_unsafe_ops_per_block)]

//! # Spectrograms
//! A focused Rust library for computing spectrograms with a simple, unified API.
//!
//! # Overview
//!
//! This library provides efficient computation of various types of spectrograms:
//! - Linear-frequency spectrograms
//! - Mel-frequency spectrograms
//! - ERB spectrograms (planned)
//! - Logarithmic-frequency spectrograms (planned)
//!
//! With support for multiple amplitude scales:
//! - Power (`|X|²`)
//! - Magnitude (`|X|`)
//! - Decibels (`10·log₁₀(power)`)
//!
//! # Features
//!
//! - **Two FFT backends**: FFTW (default, fastest) or pure-Rust `RealFFT`
//! - **Plan-based computation**: Reuse FFT plans for efficient batch processing
//! - **Comprehensive window functions**: Hanning, Hamming, Blackman, Kaiser, Gaussian, etc.
//! - **Type-safe API**: Compile-time guarantees for spectrogram types
//! - **Zero-copy design**: Efficient memory usage with minimal allocations
//!
//! # Quick Start
//!
//! ```
//! use spectrograms::*;
//! use std::f64::consts::PI;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Generate a sine wave
//! let sample_rate = 16000.0;
//! let samples: Vec<f64> = (0..16000)
//!     .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
//!     .collect();
//!
//! // Set up parameters
//! let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, sample_rate)?;
//!
//! // Compute power spectrogram
//! let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;
//!
//! println!("Computed {} bins x {} frames", spec.n_bins(), spec.n_frames());
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flags
//!
//! The library requires exactly one FFT backend:
//!
//! - `fftw` (default): Uses FFTW for fastest performance
//! - `realfft`: Pure-Rust FFT implementation
//!
//! # Examples
//!
//! ## Mel Spectrogram
//!
//! ```
//! use spectrograms::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let samples = vec![0.0; 16000];
//!
//! let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, 16000.0)?;
//! let mel = MelParams::new(80, 0.0, 8000.0)?;
//! let db = LogParams::new(-80.0)?;
//!
//! let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Efficient Batch Processing
//!
//! ```
//! use spectrograms::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let signals = vec![vec![0.0; 16000], vec![0.0; 16000]];
//!
//! let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
//! let params = SpectrogramParams::new(stft, 16000.0)?;
//!
//! // Create plan once, reuse for all signals
//! let planner = SpectrogramPlanner::new();
//! let mut plan = planner.linear_plan::<Power>(&params, None)?;
//!
//! for signal in signals {
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
mod fft_backend;
mod mfcc;
mod spectrogram;
mod window;

// #[cfg(feature = "python")]
mod python;

pub use chroma::{
    ChromaNorm, ChromaParams, Chromagram, N_CHROMA, chromagram, chromagram_from_spectrogram,
};
pub use cqt::{CqtParams, CqtResult, cqt};
pub use erb::{ErbParams, GammatoneParams};
pub use error::{SpectrogramError, SpectrogramResult};
pub use fft_backend::{C2rPlan, C2rPlanner, *};
pub use mfcc::{Mfcc, MfccParams, mfcc, mfcc_from_log_mel};
pub use spectrogram::*;
pub use window::WindowType;

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
