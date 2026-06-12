use std::num::NonZeroUsize;

use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec, non_empty_vec};
/// ERB (Equivalent Rectangular Bandwidth) scale implementation.
///
/// The ERB scale is based on psychoacoustic measurements of human hearing
/// and represents critical bandwidths at different frequencies.
use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult, nzu};

/// Center-frequency spacing strategy for the ERB filterbank.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ErbSpacing {
    /// Linear spacing on the ERB scale (Glasberg & Moore, 1990).
    /// `ERB(f) = 24.7 * (4.37 * f / 1000 + 1)`, bands spaced uniformly in
    /// this domain and ordered low → high. Default.
    #[default]
    Linear,
    /// Geometric spacing using the Apple TR #35 / Patterson-Holdsworth formula
    /// (`earQ=9.26449`, `minBW=24.7`), ordered low → high.
    /// Matches the ViSQOL C++ reference implementation exactly.
    AppleTr35,
}

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
    /// Center-frequency spacing strategy
    spacing: ErbSpacing,
    /// If `Some(floor_db)`, the output matrix is converted to dB in-place with
    /// the given noise floor.  Any band energy below `10^(floor_db/10)` is
    /// clamped to `floor_db`.  `None` leaves the output as linear RMS values.
    db_floor: Option<f64>,
}
pub type GammatoneParams = ErbParams;

impl ErbParams {
    /// Create new ERB parameters with linear spacing (default).
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
            spacing: ErbSpacing::Linear,
            db_floor: None,
        })
    }

    /// Return a copy of these parameters with the given spacing strategy.
    #[inline]
    #[must_use]
    pub const fn with_spacing(self, spacing: ErbSpacing) -> Self {
        Self { spacing, ..self }
    }

    /// Return a copy of these parameters that converts the output to dB.
    ///
    /// Band energies below `10^(floor_db / 10)` are clamped to `floor_db`.
    #[inline]
    #[must_use]
    pub const fn with_db_floor(self, floor_db: f64) -> Self {
        Self {
            db_floor: Some(floor_db),
            ..self
        }
    }

    /// Return the spacing strategy.
    #[inline]
    #[must_use]
    pub const fn spacing(&self) -> ErbSpacing {
        self.spacing
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
            spacing: ErbSpacing::Linear,
            db_floor: None,
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

/// Glasberg & Moore / Apple TR #35 earQ parameter.
const EAR_Q: f64 = 9.26449;
/// Glasberg & Moore / Apple TR #35 minimum bandwidth in Hz.
const MIN_BW: f64 = 24.7;

/// Compute `n` center frequencies using the Apple TR #35 / Patterson-Holdsworth
/// geometric spacing formula (`earQ=9.26449`, `minBW=24.7`).
///
/// Returned order is low → high (consistent with linear ERB spacing).
/// This matches the final band ordering of the ViSQOL C++ reference
/// (`gammatone_spectrogram_builder.cc`).
fn apple_tr35_center_freqs(n: usize, low_freq: f64, high_freq: f64) -> Vec<f64> {
    let shift = EAR_Q * MIN_BW; // ≈ 228.733

    let a = -shift;
    let d = high_freq + shift;
    // e = log((low_freq + shift) / (high_freq + shift)) / n
    let e = ((low_freq + shift).ln() - (high_freq + shift).ln()) / n as f64;

    // C++ iterates i = 0..n-1 with index (i+1), giving high→low order.
    // We reverse immediately to produce low→high.
    let mut cfs: Vec<f64> = (0..n)
        .map(|i| a + f64::exp((i as f64 + 1.0) * e) * d)
        .collect();
    cfs.reverse();
    cfs
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

        let center_freqs_vec: Vec<f64> = match params.spacing {
            ErbSpacing::Linear => {
                // Convert frequency range to ERB scale and space linearly.
                let erb_min = hz_to_erb(params.f_min);
                let erb_max = hz_to_erb(params.f_max);
                let erb_step = (erb_max - erb_min) / (params.n_filters.get() - 1) as f64;
                (0..params.n_filters.get())
                    .map(|i| erb_to_hz((i as f64).mul_add(erb_step, erb_min)))
                    .collect()
            }
            ErbSpacing::AppleTr35 => {
                apple_tr35_center_freqs(params.n_filters.get(), params.f_min, params.f_max)
            }
        };

        // safety: center_freqs is non-empty since n_filters > 0
        let center_freqs = unsafe { NonEmptyVec::new_unchecked(center_freqs_vec) };
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

// ─── Time-domain IIR gammatone spectrogram ───────────────────────────────────
//
// Matches the C++ ViSQOL reference implementation:
//   gammatone_spectrogram_builder.cc  +  signal_filter.cc
//   +  equivalent_rectangular_bandwidth.cc
//
// The frequency-domain ErbFilterbank above is an approximation; this section
// implements the exact 4-th order cascaded IIR gammatone filter bank used by
// the C++ pipeline.  This is required to produce feature vectors that are
// compatible with the SVR model trained on C++ output.

/// Per-channel coefficients for the 4 cascaded 2nd-order IIR sections.
struct IirBandCoeffs {
    /// Numerator triplets [a0, a1, 0.0] — one per section (a2 is always 0).
    a: [[f64; 3]; 4],
    /// Shared denominator [1.0, b1, b2] for all 4 sections.
    b: [f64; 3],
}

/// Gain normalisation factor for a single center frequency.
/// Derived from C++ `EquivalentRectangularBandwidth::MakeFilters`.
fn iir_gain(cf: f64, b_val: f64, t: f64) -> f64 {
    use std::f64::consts::PI;
    let angle = 2.0 * PI * cf * t;
    let (cos1, sin1) = (angle.cos(), angle.sin());

    // exp(4j·π·cf·T)
    let x_exp = Complex::new(f64::cos(2.0 * angle), f64::sin(2.0 * angle));
    let exp_bt_neg = f64::exp(-b_val * t);

    // x01 = -2T·exp(4j·π·cf·T),  x02 = 2T·exp(-BT)·(cos1 + j·sin1)
    let x01: Complex<f64> = x_exp * (-2.0 * t);
    let x02: Complex<f64> = Complex::new(cos1, sin1) * (2.0 * t * exp_bt_neg);

    let s1 = f64::sqrt(3.0 - 2.0 * 2.0_f64.sqrt()); // √2 − 1 ≈ 0.414
    let s2 = f64::sqrt(3.0 + 2.0 * 2.0_f64.sqrt()); // 1 + √2 ≈ 2.414

    let x1 = x01 + x02 * (cos1 - s1 * sin1);
    let x2 = x01 + x02 * (cos1 + s1 * sin1);
    let x3 = x01 + x02 * (cos1 - s2 * sin1);
    let x4 = x01 + x02 * (cos1 + s2 * sin1);

    // x5 = -2·exp(-2BT) − 2·xExp + 2·(1+xExp)·exp(-BT)
    let exp_2bt_neg = exp_bt_neg * exp_bt_neg;
    let x5 = Complex::new(-2.0 * exp_2bt_neg, 0.0) - x_exp * 2.0
        + (Complex::new(1.0, 0.0) + x_exp) * (2.0 * exp_bt_neg);

    ((x1 * x2 * x3 * x4) / x5.powi(4)).norm()
}

/// Build IIR gammatone filter coefficients for every band in `center_freqs`.
fn make_iir_bank(center_freqs: &[f64], sample_rate: f64) -> Vec<IirBandCoeffs> {
    use std::f64::consts::PI;
    let t = 1.0 / sample_rate;

    center_freqs
        .iter()
        .map(|&cf| {
            // ERB bandwidth (Apple TR #35 / Glasberg & Moore — numerically identical)
            let erb = cf / EAR_Q + MIN_BW;
            let b_val = 1.019 * 2.0 * PI * erb;

            let exp_bt = f64::exp(-b_val * t);
            let angle = 2.0 * PI * cf * t;
            let (cos1, sin1) = (angle.cos(), angle.sin());

            let b1 = -2.0 * cos1 * exp_bt; // B1
            let b2 = f64::exp(-2.0 * b_val * t); // B2

            let s1 = f64::sqrt(3.0 - 2.0 * 2.0_f64.sqrt());
            let s2 = f64::sqrt(3.0 + 2.0 * 2.0_f64.sqrt());
            let b_sin = sin1 * t;

            let a11 = -exp_bt * (t * cos1 + b_sin * s2);
            let a12 = -exp_bt * (t * cos1 - b_sin * s2);
            let a13 = -exp_bt * (t * cos1 + b_sin * s1);
            let a14 = -exp_bt * (t * cos1 - b_sin * s1);

            let gain = iir_gain(cf, b_val, t);
            let a0 = t;

            IirBandCoeffs {
                a: [
                    [a0 / gain, a11 / gain, 0.0], // section 1: gain-normalised
                    [a0, a12, 0.0],
                    [a0, a13, 0.0],
                    [a0, a14, 0.0],
                ],
                b: [1.0, b1, b2],
            }
        })
        .collect()
}

/// Direct Form II Transposed 2nd-order IIR section.
///
/// Matches C++ `SignalFilter::Filter` with a 3-element denominator [1, b1, b2]
/// and 3-element numerator [a0, a1, 0].  State is `[z0, z1]`.
#[inline]
fn apply_iir2_section(
    a: &[f64; 3],
    b: &[f64; 3],
    signal: &[f64],
    state: &mut [f64; 2],
) -> Vec<f64> {
    let (a0, a1) = (a[0], a[1]);
    let (b1, b2) = (b[1], b[2]);
    let mut out = vec![0.0_f64; signal.len()];
    let (mut z0, mut z1) = (state[0], state[1]);
    for (&x, y_out) in signal.iter().zip(out.iter_mut()) {
        let y = a0 * x + z0;
        z0 = a1 * x + z1 - b1 * y;
        z1 = -b2 * y; // a2 == 0
        *y_out = y;
    }
    state[0] = z0;
    state[1] = z1;
    out
}

/// Apply all 4 cascaded IIR sections to a windowed frame, returning the
/// RMS energy (= `sqrt(mean(x²))`).  State is reset to zero before each
/// call, matching C++ `filter_bank_.ResetFilterConditions()` per frame.
#[inline]
fn iir4_rms(coeffs: &IirBandCoeffs, windowed_frame: &[f64]) -> f64 {
    let mut state = [0.0_f64; 2];
    let s1 = apply_iir2_section(&coeffs.a[0], &coeffs.b, windowed_frame, &mut state);
    state = [0.0; 2];
    let s2 = apply_iir2_section(&coeffs.a[1], &coeffs.b, &s1, &mut state);
    state = [0.0; 2];
    let s3 = apply_iir2_section(&coeffs.a[2], &coeffs.b, &s2, &mut state);
    state = [0.0; 2];
    let s4 = apply_iir2_section(&coeffs.a[3], &coeffs.b, &s3, &mut state);

    let mean_sq = s4.iter().map(|x| x * x).sum::<f64>() / windowed_frame.len() as f64;
    mean_sq.sqrt()
}

/// Hann window: `w[i] = 0.5 − 0.5·cos(2π·i / (N−1))`.
/// Matches the C++ `AnalysisWindow::ApplyHannWindow`.
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 - 0.5 * f64::cos(2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64))
        .collect()
}

/// Build a gammatone magnitude spectrogram using the **time-domain IIR filter
/// bank** from the C++ ViSQOL reference implementation.
///
/// Unlike [`ErbFilterbank`] (which multiplies STFT bins by a pre-computed
/// power response), this function applies actual 4th-order cascaded IIR
/// gammatone filters to each windowed frame, then computes `sqrt(mean(x²))`
/// per band — reproducing the exact feature vectors used to train the ViSQOL
/// SVR model.
///
/// # Arguments
///
/// * `samples` – Mono, f64 audio samples.
/// * `sample_rate` – Sample rate in Hz.
/// * `frame_size` – Frame length in samples (e.g. 3840 for 48 kHz audio mode).
/// * `hop_size` – Hop in samples (e.g. 960 for 25 % hop).
/// * `erb_params` – Band count, min/max frequency, and spacing strategy.
///   Use [`ErbSpacing::AppleTr35`] for C++-compatible center frequencies.
///
/// # Returns
///
/// `(spectrogram, center_freqs)` where `spectrogram` has shape
/// `[n_bands, n_frames]` and `n_frames = 1 + ⌊(n_samples − frame_size) / hop_size⌋`.
///
/// # Errors
///
/// Returns `SpectrogramError::InvalidInput` if `sample_rate ≤ 0` or if the
/// signal is shorter than `frame_size`.
pub fn gammatone_iir_spectrogram(
    samples: &[f64],
    sample_rate: f64,
    frame_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    erb_params: &ErbParams,
) -> SpectrogramResult<(Array2<f64>, Vec<f64>)> {
    if sample_rate <= 0.0 {
        return Err(SpectrogramError::invalid_input("sample_rate must be > 0"));
    }
    let frame_size = frame_size.get();
    let hop_size = hop_size.get();

    if samples.len() < frame_size {
        return Err(SpectrogramError::invalid_input(
            "signal is shorter than frame_size",
        ));
    }

    let center_freqs = match erb_params.spacing() {
        ErbSpacing::AppleTr35 => apple_tr35_center_freqs(
            erb_params.n_filters().get(),
            erb_params.f_min(),
            erb_params.f_max(),
        ),
        ErbSpacing::Linear => {
            let erb_min = hz_to_erb(erb_params.f_min());
            let erb_max = hz_to_erb(erb_params.f_max());
            let step = (erb_max - erb_min) / (erb_params.n_filters().get() - 1) as f64;
            (0..erb_params.n_filters().get())
                .map(|i| erb_to_hz((i as f64).mul_add(step, erb_min)))
                .collect()
        }
    };

    let n_bands = erb_params.n_filters().get();
    let filter_bank = make_iir_bank(&center_freqs, sample_rate);
    let window = hann_window(frame_size);

    let n_frames = 1 + (samples.len() - frame_size) / hop_size;
    let mut out = Array2::<f64>::zeros((n_bands, n_frames));

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let end = start + frame_size;
        let windowed: Vec<f64> = samples[start..end]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        for (band, coeffs) in filter_bank.iter().enumerate() {
            out[(band, frame_idx)] = iir4_rms(coeffs, &windowed);
        }
    }

    if let Some(floor_db) = erb_params.db_floor {
        let eps = 10.0_f64.powf(floor_db / 10.0);
        out.mapv_inplace(|x| 10.0 * x.max(eps).log10());
    }

    Ok((out, center_freqs))
}
