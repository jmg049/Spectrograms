/// Error types for the spectrogram library.
use thiserror::Error;

pub type SpectrogramResult<T> = Result<T, SpectrogramError>;

/// Represents errors that can occur in the spectrogram library.
///
/// Variants:
/// - `InvalidInput`: Indicates that the provided input is invalid.
/// - `DimensionMismatch`: Indicates a mismatch in expected and actual dimensions.
/// - `FftBackendError`: Represents errors originating from the FFT backend.
///
#[derive(Debug, Error)]
pub enum SpectrogramError {
    /// The input provided to the spectrogram function is invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// The dimensions provided do not match the expected dimensions.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// An error originating from the FFT backend.
    #[error("{backend} -- FFT backend error: {msg}")]
    FftBackendError { backend: &'static str, msg: String },
    /// An internal error occurred within the spectrogram library.
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl SpectrogramError {
    /// Create an FFT backend error based on the given backend and message.
    ///
    /// # Arguments
    /// * `backend` - The name of the FFT backend (e.g., "fftw", "realfft").
    /// * `msg` - The error message from the backend.
    ///
    /// # Returns
    /// A `SpectrogramError::FftBackendError` variant containing the backend name and message.
    #[inline]
    pub fn fft_backend<S: Into<String>>(backend: &'static str, msg: S) -> Self {
        Self::FftBackendError {
            backend,
            msg: msg.into(),
        }
    }

    /// Create an invalid input error.
    ///
    /// # Arguments
    /// * `msg` - The error message describing the invalid input.
    ///
    /// # Type Parameters
    /// * `S` - A type that can be converted into a `String`.
    ///
    /// # Returns
    /// A `SpectrogramError::InvalidInput` variant containing the message.
    #[inline]
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a dimension mismatch error.
    ///
    /// # Arguments
    /// * `expected` - The expected dimension size.
    /// * `got` - The actual dimension size received.
    ///
    /// # Returns
    /// A `SpectrogramError::DimensionMismatch` variant containing the expected and actual sizes.
    #[inline]
    #[must_use] 
    pub const fn dimension_mismatch(expected: usize, got: usize) -> Self {
        Self::DimensionMismatch { expected, got }
    }

    /// Create an internal error.
    ///
    /// # Arguments
    /// * `msg` - The error message describing the internal error.
    ///
    /// # Type Parameters
    /// * `S` - A type that can be converted into a `String`.
    ///
    /// # Returns
    /// A `SpectrogramError::InternalError` variant containing the message.
    #[inline]
    pub fn internal_error<S: Into<String>>(msg: S) -> Self {
        Self::InternalError(msg.into())
    }
}
