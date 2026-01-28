use std::{num::NonZeroUsize, ops::Deref};

/// Mel-Frequency Cepstral Coefficients (MFCC) computation.
///
/// MFCCs are widely used features in speech and audio processing, representing
/// the short-term power spectrum of a sound on a mel scale.
use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::{SpectrogramError, SpectrogramResult, StftParams, nzu};

/// MFCC computation parameters.
///
/// # Fields
///
/// - `n_mfcc`: Number of MFCC coefficients to compute
/// - `include_c0`: Whether to include the 0th coefficient (energy)
/// - `lifter`: Lifter parameter for cepstral liftering (0 = no liftering)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfccParams {
    /// Number of MFCC coefficients to compute
    n_mfcc: NonZeroUsize,
    /// Whether to include the 0th coefficient (energy)
    include_c0: bool,
    /// Lifter parameter for cepstral liftering (0 = no liftering)
    lifter: usize,
}

impl Default for MfccParams {
    #[inline]
    fn default() -> Self {
        Self {
            n_mfcc: nzu!(13),
            include_c0: true,
            lifter: 22,
        }
    }
}

impl MfccParams {
    /// Create new MFCC parameters.
    ///
    /// # Arguments
    ///
    /// * `n_mfcc` - Number of MFCC coefficients to return
    ///
    /// # Returns
    ///
    /// An `MfccParams` instance with default settings.
    #[inline]
    #[must_use]
    pub const fn new(n_mfcc: NonZeroUsize) -> Self {
        Self {
            n_mfcc,
            include_c0: true,
            lifter: 22,
        }
    }

    /// Standard MFCC parameters for speech recognition (13 coefficients).
    ///
    /// # Returns
    ///
    /// An `MfccParams` instance with 13 coefficients, C0 included, and lifter of 22.
    #[inline]
    #[must_use]
    pub const fn speech_standard() -> Self {
        Self::new(nzu!(13))
    }

    /// Set whether to include the 0th coefficient.
    ///
    /// The 0th coefficient represents the overall energy and is sometimes excluded.
    ///
    /// # Arguments
    ///
    /// * `include_c0` - `true` to include C0, `false` to exclude it
    ///
    /// # Returns
    ///
    /// The updated `MfccParams` with the specified C0 inclusion.
    #[inline]
    #[must_use]
    pub const fn with_c0(mut self, include_c0: bool) -> Self {
        self.include_c0 = include_c0;
        self
    }

    /// Set the lifter parameter for cepstral liftering.
    ///
    /// Liftering applies a sinusoidal weighting to emphasize mid-range coefficients.
    /// Common values: 22 (default), 0 (no liftering).
    ///
    /// # Arguments
    ///
    /// * `lifter` - Lifter value (0 = no liftering)
    ///
    /// # Returns
    ///
    /// The lifter value (0 = no liftering).
    #[inline]
    #[must_use]
    pub const fn with_lifter(mut self, lifter: usize) -> Self {
        self.lifter = lifter;
        self
    }

    /// Get the number of MFCC coefficients.
    ///
    /// # Returns
    ///
    /// The number of MFCC coefficients to compute.
    #[inline]
    #[must_use]
    pub const fn n_mfcc(&self) -> NonZeroUsize {
        self.n_mfcc
    }

    /// Get whether C0 is included.
    ///
    /// # Returns
    ///
    /// `true` if C0 is included, `false` otherwise.
    #[inline]
    #[must_use]
    pub const fn include_c0(&self) -> bool {
        self.include_c0
    }

    /// Get the lifter parameter.
    ///
    /// # Returns
    ///
    /// The lifter value (0 = no liftering).
    #[inline]
    #[must_use]
    pub const fn lifter(&self) -> usize {
        self.lifter
    }
}

/// MFCC features representation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mfcc {
    /// MFCC coefficient matrix with shape (`n_mfcc`, `n_frames`)
    pub data: Array2<f64>,
    /// Parameters used to compute these MFCCs
    params: MfccParams,
}

impl AsRef<Array2<f64>> for Mfcc {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for Mfcc {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Mfcc {
    /// Get the number of MFCC coefficients.
    ///
    /// # Returns
    ///
    /// The number of coefficients (rows in the data matrix).
    #[inline]
    #[must_use]
    pub fn n_coefficients(&self) -> NonZeroUsize {
        // safety: data has at least one row since n_mfcc is NonZeroUsize
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of frames.
    ///
    /// # Returns
    ///
    /// The number of frames (columns in the data matrix).
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        // safety: data has at least one column since there is at least one frame
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the parameters used to compute these MFCCs.
    ///
    /// # Returns
    ///
    /// A reference to the `MfccParams`.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &MfccParams {
        &self.params
    }
}

/// Compute MFCCs from a log mel spectrogram.
///
/// This applies a Discrete Cosine Transform (DCT) to the log mel spectrogram
/// to decorrelate the features and compress the representation.
///
/// # Arguments
///
/// * `log_mel_spec` - 2D array with shape (`n_mels`, `n_frames`) in log scale
/// * `params` - MFCC computation parameters
///
/// # Returns
///
/// An `Mfcc` structure with shape (`n_mfcc`, `n_frames`).
///
/// # Errors
///
/// Returns an error if `n_mfcc` is greater than `n_mels`.
#[inline]
pub fn mfcc_from_log_mel(
    log_mel_spec: &Array2<f64>,
    params: &MfccParams,
) -> SpectrogramResult<Mfcc> {
    let n_mels = log_mel_spec.nrows();
    let n_frames = log_mel_spec.ncols();

    if params.n_mfcc.get() > n_mels {
        return Err(SpectrogramError::invalid_input("n_mfcc must be <= n_mels"));
    }

    // Apply DCT-II to each frame
    let mut mfcc_data = Array2::<f64>::zeros((params.n_mfcc.get(), n_frames));

    // Preallocate buffer to avoid per-frame allocation
    let mut mel_frame = vec![0.0; n_mels];

    for frame_idx in 0..n_frames {
        // Extract mel spectrum for this frame into reusable buffer
        for i in 0..n_mels {
            mel_frame[i] = log_mel_spec[[i, frame_idx]];
        }

        // Apply DCT-II
        let dct_coeffs = dct_ii(&mel_frame);

        // Store first n_mfcc coefficients
        for (coeff_idx, &val) in dct_coeffs.iter().enumerate().take(params.n_mfcc.get()) {
            mfcc_data[[coeff_idx, frame_idx]] = val;
        }
    }

    // Apply liftering if requested
    if params.lifter > 0 {
        apply_liftering(&mut mfcc_data, params.lifter);
    }

    // Optionally remove C0
    let final_data = if !params.include_c0 && params.n_mfcc > nzu!(1) {
        // Remove first row (C0)
        mfcc_data.slice(ndarray::s![1.., ..]).to_owned()
    } else {
        mfcc_data
    };

    Ok(Mfcc {
        data: final_data,
        params: *params,
    })
}

/// Compute Discrete Cosine Transform (DCT-II).
///
/// DCT-II is used to decorrelate mel-filterbank energies.
fn dct_ii(input: &[f64]) -> NonEmptyVec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];

    for (k, sample) in output.iter_mut().enumerate().take(n) {
        *sample = input.iter().enumerate().fold(0.0, |acc, (i, &val)| {
            val.mul_add(
                (std::f64::consts::PI * k as f64 * (i as f64 + 0.5) / n as f64).cos(),
                acc,
            )
        });
    }
    // safety: output is non-empty since n > 0
    unsafe { NonEmptyVec::new_unchecked(output) }
}

/// Apply cepstral liftering to MFCC coefficients.
///
/// Liftering applies a sinusoidal weighting to emphasize mid-range coefficients.
fn apply_liftering(mfcc: &mut Array2<f64>, lifter: usize) {
    let n_mfcc = mfcc.nrows();
    let n_frames = mfcc.ncols();

    // Compute lifter weights
    let mut weights = vec![0.0; n_mfcc];
    for (i, w) in weights.iter_mut().enumerate().take(n_mfcc) {
        *w = (lifter as f64 / 2.0)
            .mul_add((std::f64::consts::PI * i as f64 / lifter as f64).sin(), 1.0);
    }

    // Apply weights to each frame
    for frame_idx in 0..n_frames {
        for (coeff_idx, &weight) in weights.iter().enumerate().take(n_mfcc) {
            mfcc[[coeff_idx, frame_idx]] *= weight;
        }
    }
}

/// Compute MFCCs directly from audio samples.
///
/// This is a convenience function that computes a mel spectrogram, converts to
/// log scale, and extracts MFCC features in a single call.
///
/// # Arguments
///
/// * `samples` - Audio samples (any type that can be converted to a slice)
/// * `stft_params` - STFT parameters
/// * `sample_rate` - Sample rate in Hz
/// * `n_mels` - Number of mel bands
/// * `mfcc_params` - MFCC computation parameters
///
/// # Returns
///
/// An `Mfcc` structure containing the coefficients.
///
/// # Errors
///
/// Returns an error if any step of the computation fails.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let samples = non_empty_vec![0.0; nzu!(16000)];
/// let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true)?;
/// let mfcc_params = MfccParams::speech_standard();
///
/// let mfccs = mfcc(&samples, &stft, 16000.0, nzu!(40), &mfcc_params)?;
///
/// assert_eq!(mfccs.n_coefficients(), nzu!(13));
/// println!("MFCCs: {} coefficients x {} frames",
///          mfccs.n_coefficients(), mfccs.n_frames());
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn mfcc(
    samples: &NonEmptySlice<f64>,
    stft_params: &StftParams,
    sample_rate: f64,
    n_mels: NonZeroUsize,
    mfcc_params: &MfccParams,
) -> SpectrogramResult<Mfcc> {
    use crate::{LogParams, MelDbSpectrogram, MelParams, SpectrogramParams};

    // Create parameters
    let params = SpectrogramParams::new(*stft_params, sample_rate)?;
    let mel = MelParams::new(n_mels, 0.0, sample_rate / 2.0)?;
    let db = LogParams::new(-80.0)?;

    // Compute log mel spectrogram
    let log_mel_spec = MelDbSpectrogram::compute(samples, &params, &mel, Some(&db))?;

    // Extract MFCCs from log mel spectrogram
    mfcc_from_log_mel(log_mel_spec.data(), mfcc_params)
}
