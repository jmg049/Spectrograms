/// ERB (Equivalent Rectangular Bandwidth) scale implementation.
///
/// The ERB scale is based on psychoacoustic measurements of human hearing
/// and represents critical bandwidths at different frequencies.
use num_complex::Complex;
use std::f64::consts::PI;

use crate::{SpectrogramError, SpectrogramResult};

/// ERB filterbank parameters
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErbParams {
    /// Number of ERB bands
    n_filters: usize,
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
    pub fn new(n_filters: usize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        if n_filters == 0 {
            return Err(SpectrogramError::invalid_input("n_filters must be > 0"));
        }
        if !(f_min >= 0.0 && f_min.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and >= 0",
            ));
        }
        if !(f_max > f_min) {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_filters,
            f_min,
            f_max,
        })
    }

    /// Get the number of ERB filters.
    #[must_use] 
    pub const fn n_filters(&self) -> usize {
        self.n_filters
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

    /// Standard ERB parameters for speech (40 filters, 0-8000 Hz).
    pub fn speech_standard() -> SpectrogramResult<Self> {
        Self::new(40, 0.0, 8000.0)
    }

    /// Standard ERB parameters for music (64 filters, 0-Nyquist).
    pub fn music_standard(sample_rate: f64) -> SpectrogramResult<Self> {
        Self::new(64, 0.0, sample_rate / 2.0)
    }
}

/// Convert frequency to ERB scale (Glasberg & Moore, 1990).
///
/// ERB(f) = 24.7 * (4.37 * f / 1000 + 1)
#[inline]
pub fn hz_to_erb(hz: f64) -> f64 {
    24.7 * (4.37 * hz / 1000.0 + 1.0)
}

/// Convert ERB scale to frequency.
#[inline]
pub fn erb_to_hz(erb: f64) -> f64 {
    (erb / 24.7 - 1.0) * 1000.0 / 4.37
}

/// ERB filterbank for spectrogram computation.
#[derive(Debug, Clone)]
pub struct ErbFilterbank {
    /// Filter center frequencies
    center_freqs: Vec<f64>,
    /// Gammatone filter coefficients
    filters: Vec<GammatoneFilter>,
}

impl ErbFilterbank {
    /// Generate ERB filterbank.
    pub(crate) fn generate(params: &ErbParams, sample_rate: f64) -> SpectrogramResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SpectrogramError::invalid_input("sample_rate must be > 0"));
        }

        // Convert frequency range to ERB scale
        let erb_min = hz_to_erb(params.f_min);
        let erb_max = hz_to_erb(params.f_max);

        // Linearly space in ERB scale
        let erb_step = (erb_max - erb_min) / (params.n_filters - 1) as f64;

        let mut center_freqs = Vec::with_capacity(params.n_filters);
        let mut filters = Vec::with_capacity(params.n_filters);

        for i in 0..params.n_filters {
            let erb = (i as f64).mul_add(erb_step, erb_min);
            let freq = erb_to_hz(erb);

            center_freqs.push(freq);
            filters.push(GammatoneFilter::new(freq, sample_rate));
        }

        Ok(Self {
            center_freqs,
            filters,
        })
    }

    /// Get center frequencies.
    pub fn center_frequencies(&self) -> &[f64] {
        &self.center_freqs
    }

    /// Get the number of filters.
    pub const fn num_filters(&self) -> usize {
        self.filters.len()
    }

    /// Apply filterbank to frequency spectrum.
    ///
    /// Approximates gammatone filtering in the frequency domain.
    pub fn apply_to_spectrum(
        &self,
        fft_bins: &[f64],
        sample_rate: f64,
        n_fft: usize,
    ) -> SpectrogramResult<Vec<f64>> {
        let mut output = vec![0.0; self.filters.len()];

        // Frequency resolution
        let freq_resolution = sample_rate / n_fft as f64;

        for (filter_idx, filter) in self.filters.iter().enumerate() {
            let center_freq = self.center_freqs[filter_idx];

            // Sum power in the ERB bandwidth around center frequency
            let bandwidth = filter.bandwidth();
            let f_low = (center_freq - bandwidth / 2.0).max(0.0);
            let f_high = (center_freq + bandwidth / 2.0).min(sample_rate / 2.0);

            let bin_low = (f_low / freq_resolution).floor() as usize;
            let bin_high = ((f_high / freq_resolution).ceil() as usize).min(fft_bins.len());

            let mut sum = 0.0;
            for bin_idx in bin_low..bin_high {
                sum += fft_bins[bin_idx];
            }

            output[filter_idx] = sum;
        }

        Ok(output)
    }
}

/// Gammatone filter representation.
#[derive(Debug, Clone)]
struct GammatoneFilter {
    center_freq: f64,
    sample_rate: f64,
}

impl GammatoneFilter {
    const fn new(center_freq: f64, sample_rate: f64) -> Self {
        Self {
            center_freq,
            sample_rate,
        }
    }

    /// Get the ERB bandwidth for this filter.
    fn bandwidth(&self) -> f64 {
        // ERB bandwidth formula
        24.7 * (4.37 * self.center_freq / 1000.0 + 1.0)
    }

    /// Generate gammatone impulse response.
    ///
    /// h(t) = t^(n-1) * exp(-2πb*t) * cos(2πf*t + φ)
    #[allow(dead_code)]
    fn impulse_response(&self, length: usize) -> Vec<Complex<f64>> {
        let mut response = Vec::with_capacity(length);
        let bandwidth = self.bandwidth();

        let order = 4.0; // Standard gammatone order
        let phase = 0.0; // Initial phase

        for n in 0..length {
            let t = n as f64 / self.sample_rate;

            // Envelope: t^(order-1) * exp(-2πb*t)
            let envelope = t.powf(order - 1.0) * (-2.0 * PI * bandwidth * t).exp();

            // Carrier: cos(2πf*t + phase)
            let phase_val = (2.0 * PI * self.center_freq).mul_add(t, phase);
            let real = envelope * phase_val.cos();
            let imag = envelope * phase_val.sin();

            response.push(Complex::new(real, imag));
        }

        response
    }
}
