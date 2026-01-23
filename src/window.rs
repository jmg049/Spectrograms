use core::fmt::Display;
use core::str::FromStr;

use crate::SpectrogramError;

/// Window functions for spectral analysis and filtering.
///
/// Different window types provide different trade-offs between frequency resolution
/// and spectral leakage in FFT-based analysis.
#[derive(Default, Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
}

impl Display for WindowType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Rectangular => write!(f, "Rectangular"),
            Self::Hanning => write!(f, "Hanning"),
            Self::Hamming => write!(f, "Hamming"),
            Self::Blackman => write!(f, "Blackman"),
            Self::Kaiser { beta } => write!(f, "Kaiser(beta={beta})"),
            Self::Gaussian { std } => write!(f, "Gaussian(std={std})"),
        }
    }
}

impl FromStr for WindowType {
    type Err = SpectrogramError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(SpectrogramError::invalid_input(
                "Input must not be empty. Must be one of ['rectangular', 'hanning', 'hamming', 'blackman', 'gaussian', 'kaiser']",
            ));
        }
        let pattern = r"^(?:(?P<name>rect|rectangle|hann|hanning|hamm|hamming|blackman)|(?P<param_name>kaiser|gaussian)=(?P<param>\d+(\.\d+)?))$";
        let reg = regex::RegexBuilder::new(pattern)
            .case_insensitive(true)
            .build()
            .map_err(|reg_err| {
                SpectrogramError::invalid_input(format!(
                    "Failed to build regex for parsing. Reason {reg_err}"
                ))
            })?;
        let normalised = s.trim();
        match reg.captures(normalised) {
            Some(caps) => {
                if let Some(name) = caps.name("name") {
                    match name.as_str().to_ascii_lowercase().as_str() {
                        "rect" | "rectangle" => Ok(Self::Rectangular),
                        "hann" | "hanning" => Ok(Self::Hanning),
                        "hamm" | "hamming" => Ok(Self::Hamming),
                        "blackman" => Ok(Self::Blackman),
                        _ => unreachable!("regex guarantees exhaustiveness"),
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
                        _ => unreachable!("regex guarantees exhaustiveness"),
                    }
                } else {
                    unreachable!("regex guarantees one capture branch")
                }
            }
            None => Err(SpectrogramError::invalid_input(format!(
                "Invalid window specification '{s}'"
            ))),
        }
    }
}
