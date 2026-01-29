use core::fmt::Display;
use core::str::FromStr;
use std::{
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use non_empty_slice::NonEmptyVec;

use crate::{SpectrogramError, make_window};

/// Window functions for spectral analysis and filtering.
///
/// Different window types provide different trade-offs between frequency resolution
/// and spectral leakage in FFT-based analysis.
#[derive(Default, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive] // Allow adding new window types in future versions
pub enum WindowType {
    /// Rectangular window (no windowing) - best frequency resolution but high leakage.
    Rectangular,
    /// Hanning window - good general-purpose window with moderate leakage.
    #[default]
    Hanning,
    /// Hamming window - similar to Hanning but slightly different coefficients.
    Hamming,
    /// Blackman window - low leakage but wider main lobe.
    Blackman,
    /// Kaiser window - parameterizable trade-off between resolution and leakage.
    Kaiser {
        /// Beta parameter controlling the trade-off between main lobe width and side lobe level
        beta: f64,
    },
    /// Gaussian window - smooth roll-off with parameterizable width.
    Gaussian {
        /// Standard deviation parameter controlling the window width
        std: f64,
    },
    /// Custom pre-computed window coefficients.
    ///
    /// The length must match the FFT size used in `make_window`.
    /// Use `WindowType::custom()` to create a custom window from a vector of coefficients.
    Custom {
        /// Pre-computed window coefficients
        #[cfg_attr(feature = "serde", serde(with = "arc_vec_serde"))]
        coefficients: Arc<Vec<f64>>,
        /// Size of the window (must match n_fft)
        size: NonZeroUsize,
    },
}

#[cfg(feature = "serde")]
mod arc_vec_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(arc: &Arc<Vec<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        arc.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<Vec<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<f64>::deserialize(deserializer).map(Arc::new)
    }
}

impl WindowType {
    /// Create a custom window from pre-computed coefficients.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Pre-computed window coefficients. The length must match
    ///   the FFT size that will be used with this window.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Coefficients are empty
    /// - Any coefficient is non-finite (NaN or infinity)
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::WindowType;
    ///
    /// // Create a simple triangular window
    /// let coeffs = vec![0.0, 0.5, 1.0, 0.5, 0.0];
    /// let window = WindowType::custom(coeffs).unwrap();
    /// ```
    #[inline]
    pub fn custom(coefficients: Vec<f64>) -> Result<Self, SpectrogramError> {
        Self::custom_with_normalization(coefficients, None)
    }

    /// Create a custom window from pre-computed coefficients with optional normalization.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Pre-computed window coefficients. The length must match
    ///   the FFT size that will be used with this window.
    /// * `normalize` - Optional normalization mode:
    ///   - `None`: No normalization (use coefficients as-is)
    ///   - `Some("sum")`: Normalize so sum equals 1.0
    ///   - `Some("peak")`: Normalize so maximum value equals 1.0
    ///   - `Some("energy")`: Normalize so sum of squares equals 1.0
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Coefficients are empty
    /// - Any coefficient is non-finite (NaN or infinity)
    /// - Unknown normalization mode is specified
    /// - Normalization would divide by zero (e.g., all-zero window)
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::WindowType;
    ///
    /// // Create a window normalized to unit sum
    /// let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    /// let window = WindowType::custom_with_normalization(coeffs, Some("sum")).unwrap();
    ///
    /// // Create a window normalized to unit peak
    /// let coeffs = vec![0.0, 0.5, 1.0, 0.5, 0.0];
    /// let window = WindowType::custom_with_normalization(coeffs, Some("peak")).unwrap();
    /// ```
    #[inline]
    pub fn custom_with_normalization(
        mut coefficients: Vec<f64>,
        normalize: Option<&str>,
    ) -> Result<Self, SpectrogramError> {
        let size = NonZeroUsize::new(coefficients.len()).ok_or_else(|| {
            SpectrogramError::invalid_input("Custom window coefficients cannot be empty")
        })?;

        // Validate all coefficients are finite
        for (i, &coef) in coefficients.iter().enumerate() {
            if !coef.is_finite() {
                return Err(SpectrogramError::invalid_input(format!(
                    "Window coefficient at index {i} is not finite: {coef}"
                )));
            }
        }

        // Apply normalization if requested
        if let Some(norm_mode) = normalize {
            match norm_mode {
                "sum" => {
                    let sum: f64 = coefficients.iter().sum();
                    if sum == 0.0 {
                        return Err(SpectrogramError::invalid_input(
                            "Cannot normalize window by sum: sum is zero",
                        ));
                    }
                    for coef in &mut coefficients {
                        *coef /= sum;
                    }
                }
                "peak" | "max" => {
                    let max = coefficients
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max == 0.0 {
                        return Err(SpectrogramError::invalid_input(
                            "Cannot normalize window by peak: maximum is zero",
                        ));
                    }
                    for coef in &mut coefficients {
                        *coef /= max;
                    }
                }
                "energy" | "rms" => {
                    let energy: f64 = coefficients.iter().map(|x| x * x).sum();
                    if energy == 0.0 {
                        return Err(SpectrogramError::invalid_input(
                            "Cannot normalize window by energy: energy is zero",
                        ));
                    }
                    let norm = energy.sqrt();
                    for coef in &mut coefficients {
                        *coef /= norm;
                    }
                }
                _ => {
                    return Err(SpectrogramError::invalid_input(format!(
                        "Unknown normalization mode '{norm_mode}'. Valid modes: 'sum', 'peak', 'energy'"
                    )));
                }
            }
        }

        Ok(Self::Custom {
            coefficients: Arc::new(coefficients),
            size,
        })
    }

    #[must_use]
    #[inline]
    pub const fn is_parameterized(&self) -> bool {
        matches!(self, Self::Kaiser { .. } | Self::Gaussian { .. })
    }

    #[must_use]
    #[inline]
    pub const fn parameter_value(&self) -> Option<f64> {
        match self {
            Self::Kaiser { beta } => Some(*beta),
            Self::Gaussian { std } => Some(*std),
            _ => None,
        }
    }
}

#[must_use]
#[inline]
pub fn hanning_window(n: NonZeroUsize) -> NonEmptyVec<f64> {
    make_window(WindowType::Hanning, n)
}

#[must_use]
#[inline]
pub fn hamming_window(n: NonZeroUsize) -> NonEmptyVec<f64> {
    make_window(WindowType::Hamming, n)
}

#[must_use]
#[inline]
pub fn blackman_window(n: NonZeroUsize) -> NonEmptyVec<f64> {
    make_window(WindowType::Blackman, n)
}

#[must_use]
#[inline]
pub fn rectangular_window(n: NonZeroUsize) -> NonEmptyVec<f64> {
    make_window(WindowType::Rectangular, n)
}

#[must_use]
#[inline]
pub fn kaiser_window(n: NonZeroUsize, beta: f64) -> NonEmptyVec<f64> {
    make_window(WindowType::Kaiser { beta }, n)
}

#[must_use]
#[inline]
pub fn gaussian_window(n: NonZeroUsize, std: f64) -> NonEmptyVec<f64> {
    make_window(WindowType::Gaussian { std }, n)
}

impl Display for WindowType {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Rectangular => write!(f, "Rectangular"),
            Self::Hanning => write!(f, "Hanning"),
            Self::Hamming => write!(f, "Hamming"),
            Self::Blackman => write!(f, "Blackman"),
            Self::Kaiser { beta } => write!(f, "Kaiser(beta={beta})"),
            Self::Gaussian { std } => write!(f, "Gaussian(std={std})"),
            Self::Custom { size, .. } => write!(f, "Custom(n={size})"),
        }
    }
}

// Cache the compiled regex for window type parsing
static WINDOW_REGEX: OnceLock<regex::Regex> = OnceLock::new();

impl FromStr for WindowType {
    type Err = SpectrogramError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(SpectrogramError::invalid_input(
                "Input must not be empty. Must be one of ['rectangular', 'hanning', 'hamming', 'blackman', 'gaussian', 'kaiser']",
            ));
        }

        let reg = WINDOW_REGEX.get_or_init(|| {
            let pattern = r"^(?:(?P<name>rect|rectangle|hann|hanning|hamm|hamming|blackman)|(?P<param_name>kaiser|gaussian)=(?P<param>\d+(\.\d+)?))$";
            regex::RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()
                .expect("hardcoded window regex should compile")
        });

        let normalised = s.trim();
        match reg.captures(normalised) {
            Some(caps) => {
                if let Some(name) = caps.name("name") {
                    match name.as_str().to_ascii_lowercase().as_str() {
                        "rect" | "rectangle" => Ok(Self::Rectangular),
                        "hann" | "hanning" => Ok(Self::Hanning),
                        "hamm" | "hamming" => Ok(Self::Hamming),
                        "blackman" => Ok(Self::Blackman),
                        _ => Err(SpectrogramError::invalid_input(format!(
                            "Unrecognized window name: {}",
                            name.as_str()
                        ))),
                    }
                } else if let (Some(param_name), Some(param)) =
                    (caps.name("param_name"), caps.name("param"))
                {
                    let value: f64 = param.as_str().parse().map_err(|_| {
                        SpectrogramError::invalid_input(format!(
                            "Invalid numeric parameter '{}'",
                            param.as_str()
                        ))
                    })?;

                    match param_name.as_str().to_ascii_lowercase().as_str() {
                        "kaiser" => Ok(Self::Kaiser { beta: value }),
                        "gaussian" => Ok(Self::Gaussian { std: value }),
                        _ => Err(SpectrogramError::invalid_input(format!(
                            "Unrecognized parameterized window: {}",
                            param_name.as_str()
                        ))),
                    }
                } else {
                    Err(SpectrogramError::invalid_input(
                        "Invalid window specification: regex matched but no valid capture group found",
                    ))
                }
            }
            None => Err(SpectrogramError::invalid_input(format!(
                "Invalid window specification '{s}'"
            ))),
        }
    }
}
