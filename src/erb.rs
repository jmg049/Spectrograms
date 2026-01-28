use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec, non_empty_vec};
/// ERB (Equivalent Rectangular Bandwidth) scale implementation.
///
/// The ERB scale is based on psychoacoustic measurements of human hearing
/// and represents critical bandwidths at different frequencies.
use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult, nzu};

/// ERB filterbank parameters
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErbParams {
    /// Number of ERB bands
    n_filters: NonZeroUsize,
    /// Minimum frequency (Hz)
    f_min: f64,
    /// Maximum frequency (Hz)
    f_max: f64,
}
pub type GammatoneParams = ErbParams;

impl ErbParams {
    /// Create new ERB parameters.
    ///
    /// # Arguments
    ///
    /// * `n_filters` - Number of ERB filters
    /// * `f_min` - Minimum frequency in Hz
    /// * `f_max` - Maximum frequency in Hz
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<Self>` - Ok with ErbParams if valid
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if:
    /// * `n_filters` < 2
    /// * `f_min` < 0 or not finite
    /// * `f_max` <= `f_min`
    #[inline]
    pub fn new(n_filters: NonZeroUsize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        if n_filters < nzu!(2) {
            return Err(SpectrogramError::invalid_input(
                "n_filters must be >= 2 (single filter would cause division by zero)",
            ));
        }
        if f_min < 0.0 || f_min.is_infinite() {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and >= 0",
            ));
        }
        if f_max <= f_min {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_filters,
            f_min,
            f_max,
        })
    }

    pub(crate) const unsafe fn new_unchecked(
        n_filters: NonZeroUsize,
        f_min: f64,
        f_max: f64,
    ) -> Self {
        Self {
            n_filters,
            f_min,
            f_max,
        }
    }

    /// Get the number of ERB filters.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Number of filters
    #[inline]
    #[must_use]
    pub const fn n_filters(&self) -> NonZeroUsize {
        self.n_filters
    }

    /// Get the minimum frequency.
    ///
    /// # Returns
    ///
    /// `f64` - Minimum frequency in Hz
    #[inline]
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

    /// Get the maximum frequency.
    ///
    /// # Returns
    ///
    /// `f64` - Maximum frequency in Hz
    #[inline]
    #[must_use]
    pub const fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Standard ERB parameters for speech (40 filters, 0-8000 Hz).
    ///
    /// # Returns
    ///
    /// `Self - ErbParams with standard speech settings
    #[inline]
    #[must_use]
    pub const fn speech_standard() -> Self {
        // safety: parameters are valid
        unsafe { Self::new_unchecked(nzu!(40), 0.0, 8000.0) }
    }

    /// Standard ERB parameters for music (64 filters, 0-Nyquist).
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<Self>` - Ok with ErbParams if valid
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if:
    /// * `sample_rate` <= 0
    #[inline]
    pub fn music_standard(sample_rate: f64) -> SpectrogramResult<Self> {
        Self::new(nzu!(64), 0.0, sample_rate / 2.0)
    }
}

/// Convert frequency to ERB scale (Glasberg & Moore, 1990).
///
/// ERB(f) = 24.7 * (4.37 * f / 1000 + 1)
///
/// # Arguments
///
/// * `hz` - Frequency in Hz
///
/// # Returns
///
/// `f64` - Corresponding ERB value
#[inline]
#[must_use]
pub const fn hz_to_erb(hz: f64) -> f64 {
    24.7 * (4.37 * hz / 1000.0 + 1.0)
}

/// Convert ERB scale to frequency.
///
/// # Arguments
///
/// * `erb` - ERB value
///
/// # Returns
///
/// `f64` - Corresponding frequency in Hz
#[inline]
#[must_use]
pub const fn erb_to_hz(erb: f64) -> f64 {
    (erb / 24.7 - 1.0) * 1000.0 / 4.37
}

/// ERB filterbank for spectrogram computation.
#[derive(Debug, Clone)]
pub struct ErbFilterbank {
    /// Filter center frequencies
    center_freqs: NonEmptyVec<f64>,
    /// Pre-computed gammatone filter responses (power transfer function)
    /// Matrix dimensions: n_filters x n_bins
    /// Each entry is |H(f)|^2 for applying to power spectrum
    response_matrix: NonEmptyVec<NonEmptyVec<f64>>,
}

impl ErbFilterbank {
    /// Generate ERB filterbank with pre-computed frequency responses.
    pub(crate) fn generate(
        params: &ErbParams,
        sample_rate: f64,
        n_fft: NonZeroUsize,
    ) -> SpectrogramResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SpectrogramError::invalid_input("sample_rate must be > 0"));
        }

        // Convert frequency range to ERB scale
        let erb_min = hz_to_erb(params.f_min);
        let erb_max = hz_to_erb(params.f_max);

        // Linearly space in ERB scale
        let erb_step = (erb_max - erb_min) / (params.n_filters.get() - 1) as f64;

        let center_freqs: Vec<f64> = (0..params.n_filters.get())
            .map(|i| (i as f64).mul_add(erb_step, erb_min))
            .map(erb_to_hz)
            .collect::<Vec<f64>>();

        // safety: center_freqs is non-empty since n_filters > 0
        let center_freqs = unsafe { NonEmptyVec::new_unchecked(center_freqs) };
        // Pre-compute gammatone frequency responses
        // We compute |H(f)|^2 for each (filter, freq_bin) pair
        let n_bins = n_fft.get() / 2 + 1; // Number of FFT bins (rfft)
        let freq_resolution = sample_rate / n_fft.get() as f64;

        let mut response_matrix = Vec::with_capacity(params.n_filters.get());

        for &center_freq in &center_freqs {
            // Gammatone bandwidth (with 1.019 factor from literature)
            let erb_bandwidth = 24.7 * (4.37 * center_freq / 1000.0 + 1.0);
            let bandwidth = 1.019 * erb_bandwidth;

            let mut filter_response = Vec::with_capacity(n_bins);

            for bin_idx in 0..n_bins {
                let freq = bin_idx as f64 * freq_resolution;

                // Gammatone frequency response: H(f) = 1 / (1 + j*(f-fc)/b)^4
                // We need |H(f)|^2 for applying to power spectrum
                let denom = Complex::new(1.0, (freq - center_freq) / bandwidth);
                let denom_squared = denom * denom;
                let denom_fourth = denom_squared * denom_squared;

                // |1 / denom_fourth|^2 = 1 / |denom_fourth|^2
                let response_power = 1.0 / denom_fourth.norm_sqr();

                filter_response.push(response_power);
            }
            // safety: filter_response is non-empty since n_bins > 0
            let filter_response = unsafe { NonEmptyVec::new_unchecked(filter_response) };
            response_matrix.push(filter_response);
        }

        // safety: response_matrix is non-empty since n_filters > 0
        let response_matrix = unsafe { NonEmptyVec::new_unchecked(response_matrix) };

        Ok(Self {
            center_freqs,
            response_matrix,
        })
    }

    /// Get center frequencies.
    ///
    /// # Returns
    ///
    /// `&NonEmptySlice<f64>` - Slice of center frequencies in Hz
    #[inline]
    #[must_use]
    pub fn center_frequencies(&self) -> &NonEmptySlice<f64> {
        &self.center_freqs
    }

    /// Get the number of filters.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Number of filters
    #[inline]
    #[must_use]
    pub const fn num_filters(&self) -> NonZeroUsize {
        self.response_matrix.len()
    }

    /// Apply filterbank to power spectrum using pre-computed responses.
    ///
    /// This efficiently computes the output of the gammatone filterbank by
    /// multiplying the pre-computed power transfer functions |H(f)|^2 with
    /// the input power spectrum and summing.
    ///
    /// # Arguments
    ///
    /// * `power_spectrum` - Non-empty slice of power spectrum values (|X(f)|^2)
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<NonEmptyVec<f64>>` - Filterbank output values
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::DimensionMismatch` if the length of
    #[inline]
    pub fn apply_to_power_spectrum(
        &self,
        power_spectrum: &NonEmptySlice<f64>,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        let n_bins = power_spectrum.len();
        let mut output = non_empty_vec![0.0; self.response_matrix.len()];

        for (filter_idx, filter_response) in self.response_matrix.iter().enumerate() {
            if filter_response.len() != n_bins {
                return Err(SpectrogramError::dimension_mismatch(
                    n_bins.get(),
                    filter_response.len().get(),
                ));
            }

            // Compute weighted sum: output[i] = sum_k (|H_i(f_k)|^2 * |X(f_k)|^2)
            let mut sum = 0.0;
            for (bin_idx, &response_power) in filter_response.iter().enumerate() {
                sum += response_power * power_spectrum[bin_idx];
            }

            output[filter_idx] = sum;
        }

        Ok(output)
    }
}
