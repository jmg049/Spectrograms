//! Python bindings for the spectrograms library.
//!
//! This module provides PyO3-based Python bindings that expose the full
//! functionality of the spectrograms library to Python users.

use pyo3::prelude::*;

mod error;
mod functions;
mod params;
mod planner;
mod spectrogram;

pub use error::*;

/// Register the Python module with `PyO3`.
///
/// This function is called from the `_spectrograms` module definition in lib.rs.
pub fn register_module(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception types
    m.add("SpectrogramError", py.get_type::<PySpectrogramError>())?;
    m.add("InvalidInputError", py.get_type::<PyInvalidInputError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type::<PyDimensionMismatchError>(),
    )?;
    m.add("FFTBackendError", py.get_type::<PyFFTBackendError>())?;
    m.add("InternalError", py.get_type::<PyInternalError>())?;

    // Register parameter classes
    params::register(py, m)?;

    // Register spectrogram result class
    spectrogram::register(py, m)?;

    // Register planner and plan classes
    planner::register(py, m)?;

    // Register convenience functions
    functions::register(py, m)?;

    Ok(())
}
