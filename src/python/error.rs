//! Python exception types for the spectrograms library.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::SpectrogramError;

// Base exception type
create_exception!(
    spectrograms,
    PySpectrogramError,
    PyException,
    "Base exception for all spectrogram errors."
);

// Specific exception types
create_exception!(
    spectrograms,
    PyInvalidInputError,
    PySpectrogramError,
    "Exception raised when invalid input is provided."
);

create_exception!(
    spectrograms,
    PyDimensionMismatchError,
    PySpectrogramError,
    "Exception raised when array dimensions don't match expected values."
);

create_exception!(
    spectrograms,
    PyFFTBackendError,
    PySpectrogramError,
    "Exception raised when an error occurs in the FFT backend."
);

create_exception!(
    spectrograms,
    PyInternalError,
    PySpectrogramError,
    "Exception raised when an internal error occurs."
);

/// Convert a Rust `SpectrogramError` to a Python exception.
///
/// This implementation allows Rust errors to be automatically converted to
/// Python exceptions when crossing the language boundary.
impl From<SpectrogramError> for PyErr {
    #[inline]
    fn from(err: SpectrogramError) -> Self {
        match err {
            SpectrogramError::InvalidInput(msg) => PyInvalidInputError::new_err(msg),
            SpectrogramError::DimensionMismatch { expected, got } => {
                PyDimensionMismatchError::new_err(format!(
                    "Dimension mismatch: expected {expected}, got {got}"
                ))
            }
            SpectrogramError::FftBackendError { backend, msg } => {
                PyFFTBackendError::new_err(format!("{backend} FFT backend error: {msg}"))
            }
            SpectrogramError::InternalError(msg) => PyInternalError::new_err(msg),
        }
    }
}
