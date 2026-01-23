use std::ops::Deref;

/// Chroma feature extraction for music information retrieval.
///
/// Chroma features represent the energy distribution across the 12 pitch classes
/// of the Western musical scale, providing a robust representation of harmonic content.
use ndarray::Array2;

use crate::{SpectrogramError, SpectrogramResult, StftParams};

/// Number of pitch classes in Western music.
pub const N_CHROMA: usize = 12;

/// Chroma feature parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChromaParams {
    /// Reference tuning frequency in Hz (typically 440.0 for A4)
    tuning: f64,
    /// Number of octaves to consider
    n_octaves: usize,
    /// Minimum frequency in Hz
    f_min: f64,
    /// Maximum frequency in Hz
    f_max: f64,
    /// Normalization strategy
    norm: ChromaNorm,
}

/// Normalization strategy for chroma features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChromaNorm {
    /// No normalization
    None,
    /// L1 normalization (sum to 1)
    L1,
    /// L2 normalization (Euclidean norm = 1)
    #[default]
    L2,
    /// Max normalization (max value = 1)
    Max,
}

impl Default for ChromaParams {
    fn default() -> Self {
        Self {
            tuning: 440.0,
            n_octaves: 7,
            f_min: 32.7,   // C1
            f_max: 4186.0, // C8
            norm: ChromaNorm::L2,
        }
    }
}

impl ChromaParams {
    /// Create new chroma parameters.
    ///
    /// # Arguments
    ///
    /// * `tuning` - Reference tuning frequency in Hz (e.g., 440.0 for A4)
    /// * `f_min` - Minimum frequency in Hz
    /// * `f_max` - Maximum frequency in Hz
    /// * `norm` - Normalization strategy
    pub fn new(tuning: f64, f_min: f64, f_max: f64, norm: ChromaNorm) -> SpectrogramResult<Self> {
        if !(tuning > 0.0 && tuning.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "tuning must be finite and > 0",
            ));
        }
        if !(f_min > 0.0 && f_min.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and > 0",
            ));
        }
        if !(f_max > f_min) {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        // Calculate number of octaves
        let n_octaves = ((f_max / f_min).log2().ceil() as usize).max(1);

        Ok(Self {
            tuning,
            n_octaves,
            f_min,
            f_max,
            norm,
        })
    }

    /// Create standard chroma parameters for music analysis.
    ///
    /// Uses A4=440Hz tuning, covering C1 to C8.
    pub fn music_standard() -> SpectrogramResult<Self> {
        Self::new(440.0, 32.7, 4186.0, ChromaNorm::L2)
    }

    /// Set the normalization strategy.
    #[must_use]
    pub const fn with_norm(mut self, norm: ChromaNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Get the tuning frequency.
    #[must_use]
    pub const fn tuning(&self) -> f64 {
        self.tuning
    }

    /// Get the minimum frequency.
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

    /// Get the maximum frequency.
    #[must_use]
    pub const fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Get the number of octaves.
    #[must_use]
    pub const fn n_octaves(&self) -> usize {
        self.n_octaves
    }
}

/// Chromagram representation with 12 pitch classes.
#[derive(Debug, Clone)]
pub struct Chromagram {
    /// Chroma feature matrix with shape (12, `n_frames`)
    pub data: Array2<f64>,
    /// Parameters used to compute this chromagram
    params: ChromaParams,
}

impl Chromagram {
    /// Get the number of frames.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.data.ncols()
    }

    /// Get the number of chroma bins (always 12).
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.data.nrows()
    }

    /// Get the parameters used to compute this chromagram.
    #[must_use]
    pub const fn params(&self) -> &ChromaParams {
        &self.params
    }

    /// Get the chroma labels.
    #[must_use]
    pub const fn labels() -> [&'static str; 12] {
        [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ]
    }
}

impl AsRef<Array2<f64>> for Chromagram {
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for Chromagram {
    type Target = Array2<f64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Build a chroma filterbank matrix.
///
/// Maps FFT bins to the 12 pitch classes using a weighted projection.
pub fn build_chroma_filterbank(
    sample_rate: f64,
    n_fft: usize,
    params: &ChromaParams,
) -> SpectrogramResult<Array2<f64>> {
    use std::f64::consts::LN_2;

    if sample_rate <= 0.0 || !sample_rate.is_finite() {
        return Err(SpectrogramError::invalid_input(
            "sample_rate must be finite and > 0",
        ));
    }
    if n_fft == 0 {
        return Err(SpectrogramError::invalid_input("n_fft must be > 0"));
    }

    let n_bins = n_fft / 2 + 1;
    let freq_resolution = sample_rate / n_fft as f64;

    // Frequency for each FFT bin
    let fft_freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * freq_resolution).collect();

    // Build filterbank matrix: (12, n_bins)
    let mut filterbank = Array2::<f64>::zeros((N_CHROMA, n_bins));

    // A4 = 440 Hz is typically the 9th semitone (A) at octave 4
    // MIDI note 69 = A4
    // Reference: C0 = 16.35 Hz = MIDI 12
    let a4_midi = 69.0;
    let a4_freq = params.tuning;

    for (bin_idx, &freq) in fft_freqs.iter().enumerate() {
        if freq < params.f_min || freq > params.f_max || freq <= 0.0 {
            continue;
        }

        // Convert frequency to MIDI note number (continuous)
        let midi_note = a4_midi + 12.0 * (freq / a4_freq).ln() / LN_2;

        // Map to pitch class (0-11, where 0=C)
        let pitch_class = midi_note.rem_euclid(12.0);

        // Use a Gaussian-like weighting centered on the pitch class
        // This spreads energy to neighboring pitch classes
        for chroma_idx in 0..N_CHROMA {
            let chroma_center = chroma_idx as f64;

            // Circular distance on pitch class circle
            let dist = (pitch_class - chroma_center).abs();
            let circular_dist = dist.min(12.0 - dist);

            // Gaussian window with sigma = 1 semitone
            let sigma = 1.0;
            let weight = (-0.5 * (circular_dist / sigma).powi(2)).exp();

            filterbank[[chroma_idx, bin_idx]] = weight;
        }
    }

    // Normalize each chroma bin (row) to unit sum
    for chroma_idx in 0..N_CHROMA {
        let row_sum: f64 = (0..n_bins).map(|i| filterbank[[chroma_idx, i]]).sum();
        if row_sum > 0.0 {
            for bin_idx in 0..n_bins {
                filterbank[[chroma_idx, bin_idx]] /= row_sum;
            }
        }
    }

    Ok(filterbank)
}

/// Compute chromagram from a magnitude or power spectrogram.
///
/// # Arguments
///
/// * `spectrogram` - 2D array with shape (`frequency_bins`, `time_frames`)
/// * `sample_rate` - Sample rate in Hz
/// * `n_fft` - FFT size
/// * `params` - Chroma feature parameters
///
/// # Returns
///
/// A `Chromagram` with shape (12, `n_frames`).
pub fn chromagram_from_spectrogram(
    spectrogram: &Array2<f64>,
    sample_rate: f64,
    n_fft: usize,
    params: &ChromaParams,
) -> SpectrogramResult<Chromagram> {
    let n_bins = spectrogram.nrows();
    let n_frames = spectrogram.ncols();

    let expected_bins = n_fft / 2 + 1;
    if n_bins != expected_bins {
        return Err(SpectrogramError::dimension_mismatch(expected_bins, n_bins));
    }

    // Build chroma filterbank
    let filterbank = build_chroma_filterbank(sample_rate, n_fft, params)?;

    // Apply filterbank: chroma = filterbank @ spectrogram
    let mut chroma_data = Array2::<f64>::zeros((N_CHROMA, n_frames));

    for frame_idx in 0..n_frames {
        for chroma_idx in 0..N_CHROMA {
            let mut sum = 0.0;
            for bin_idx in 0..n_bins {
                sum += filterbank[[chroma_idx, bin_idx]] * spectrogram[[bin_idx, frame_idx]];
            }
            chroma_data[[chroma_idx, frame_idx]] = sum;
        }
    }

    // Apply normalization
    apply_chroma_normalization(&mut chroma_data, params.norm);

    Ok(Chromagram {
        data: chroma_data,
        params: *params,
    })
}

/// Apply normalization to chroma features.
fn apply_chroma_normalization(chroma: &mut Array2<f64>, norm: ChromaNorm) {
    let n_frames = chroma.ncols();

    match norm {
        ChromaNorm::None => {}
        ChromaNorm::L1 => {
            for frame_idx in 0..n_frames {
                let sum: f64 = (0..N_CHROMA).map(|i| chroma[[i, frame_idx]]).sum();
                if sum > 0.0 {
                    for chroma_idx in 0..N_CHROMA {
                        chroma[[chroma_idx, frame_idx]] /= sum;
                    }
                }
            }
        }
        ChromaNorm::L2 => {
            for frame_idx in 0..n_frames {
                let sum_sq: f64 = (0..N_CHROMA).map(|i| chroma[[i, frame_idx]].powi(2)).sum();
                let norm = sum_sq.sqrt();
                if norm > 0.0 {
                    for chroma_idx in 0..N_CHROMA {
                        chroma[[chroma_idx, frame_idx]] /= norm;
                    }
                }
            }
        }
        ChromaNorm::Max => {
            for frame_idx in 0..n_frames {
                let max_val = (0..N_CHROMA)
                    .map(|i| chroma[[i, frame_idx]])
                    .fold(0.0, f64::max);
                if max_val > 0.0 {
                    for chroma_idx in 0..N_CHROMA {
                        chroma[[chroma_idx, frame_idx]] /= max_val;
                    }
                }
            }
        }
    }
}

/// Compute chromagram directly from audio samples.
///
/// This is a convenience function that computes an STFT and then extracts
/// chroma features in a single call.
///
/// # Arguments
///
/// * `samples` - Audio samples (any type that can be converted to a slice)
/// * `stft_params` - STFT parameters (`n_fft`, `hop_size`, window, centering)
/// * `sample_rate` - Sample rate in Hz
/// * `chroma_params` - Chroma feature parameters
///
/// # Returns
///
/// A `Chromagram` with 12 pitch classes.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let samples = vec![0.0; 16000];
/// let stft = StftParams::new(2048, 512, WindowType::Hanning, true)?;
/// let chroma_params = ChromaParams::music_standard()?;
///
/// let chromagram = chromagram(&samples, &stft, 16000.0, &chroma_params)?;
///
/// assert_eq!(chromagram.n_bins(), 12);
/// println!("Chromagram: {} pitch classes x {} frames",
///          chromagram.n_bins(), chromagram.n_frames());
/// # Ok(())
/// # }
/// ```
pub fn chromagram<S: AsRef<[f64]>>(
    samples: S,
    stft_params: &StftParams,
    sample_rate: f64,
    chroma_params: &ChromaParams,
) -> SpectrogramResult<Chromagram> {
    use crate::{SpectrogramParams, SpectrogramPlanner};

    let params = SpectrogramParams::new(*stft_params, sample_rate)?;

    // Compute STFT
    let planner = SpectrogramPlanner::new();
    let stft_result = planner.compute_stft(samples, &params)?;

    // Convert complex STFT to magnitude
    let mut magnitude_spec = Array2::<f64>::zeros(stft_result.data.dim());
    for ((i, j), val) in stft_result.data.indexed_iter() {
        magnitude_spec[[i, j]] = val.norm();
    }

    // Compute chromagram from magnitude spectrogram
    chromagram_from_spectrogram(
        &magnitude_spec,
        sample_rate,
        stft_result.params.n_fft(),
        chroma_params,
    )
}
