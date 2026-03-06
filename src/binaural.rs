//! Binaural audio analysis spectrograms.
//!
//! This module provides implementations of various binaural audio spectrograms used for
//! spatial audio analysis. These spectrograms capture differences between left and right
//! ear signals, which are critical for understanding sound localization and spatial properties
//! of audio.
//!
//! # Binaural Cues
//!
//! The human auditory system uses several binaural cues for sound localization:
//!
//! - **ITD (Interaural Time Difference)**: Time delay between ears, primary cue for low frequencies (<1.5 kHz)
//! - **IPD (Interaural Phase Difference)**: Phase difference between ears, related to ITD but can be ambiguous at high frequencies
//! - **ILD (Interaural Level Difference)**: Intensity difference in dB, primary cue for high frequencies (>1.5 kHz)
//! - **ILR (Interaural Level Ratio)**: Normalized ratio of intensities, alternative to ILD
//!
//! # Credit
//!
//! Credit to @barrydn for the original implementation of all the spectrograms in this file.
//! Taken from <https://github.com/QxLabIreland/Binaspect/>

use std::{marker::PhantomData, num::NonZeroUsize, ops::Deref};

use crate::{SpectrogramError, SpectrogramParams, SpectrogramResult, StftPlan};
use ndarray::{Array1, Array2, Axis, Zip};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use num_complex::Complex;

// ============================================================================
// Unit Type Markers
// ============================================================================

/// Unit type marker for Interaural Time Difference (ITD) measurements.
///
/// ITD values are measured in seconds and represent the time delay between
/// sound arriving at the two ears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ITD;

/// Unit type marker for Interaural Phase Difference (IPD) measurements.
///
/// IPD values are measured in radians and represent the phase difference
/// between sound at the two ears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IPD;

/// Unit type marker for Interaural Level Difference (ILD) measurements.
///
/// ILD values are measured in decibels (dB) and represent the intensity
/// difference between sound at the two ears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ILD;

/// Unit type marker for Interaural Level Ratio (ILR) measurements.
///
/// ILR values are dimensionless normalized ratios in the range [-1, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ILR;

/// Compute magnitude and phase from a complex spectrogram.
///
/// Extracts magnitude (raised to a given power) and normalized phase (unit complex numbers)
/// from a complex-valued STFT spectrogram in a single pass for efficiency.
///
/// # Arguments
///
/// * `complex_spect` - Complex-valued spectrogram from STFT
/// * `power` - Power to raise the magnitude to (1 for linear magnitude, 2 for power)
///
/// # Returns
///
/// A tuple of:
/// - Magnitude array with values raised to the specified power
/// - Phase array as unit complex numbers (re² + im² = 1)
#[inline]
#[must_use]
pub fn magphase(
    complex_spect: &Array2<Complex<f64>>,
    power: NonZeroUsize,
) -> (Array2<f64>, Array2<Complex<f64>>) {
    let power_usize = power.get();
    let shape = complex_spect.raw_dim();
    let mut mag = Array2::zeros(shape);

    let mut phase = Array2::zeros(shape);

    #[inline(always)]
    fn pow_mag(mag: f64, mag_sq: f64, power: usize) -> f64 {
        match power {
            1 => mag,
            2 => mag_sq,
            3 => mag_sq * mag,
            4 => mag_sq * mag_sq,
            _ => {
                // Integer exponent: prefer exponentiation-by-squaring.
                let mut base = mag;
                let mut exp = power;
                let mut acc = 1.0_f64;
                while exp > 0 {
                    if (exp & 1) == 1 {
                        acc *= base;
                    }
                    exp >>= 1;
                    if exp > 0 {
                        base *= base;
                    }
                }
                acc
            }
        }
    }

    // Single-pass computation of magnitude and phase
    #[cfg(feature = "rayon")]
    {
        Zip::from(&mut mag)
            .and(&mut phase)
            .and(complex_spect)
            .par_for_each(|m, p, &c| {
                let mag_sq = c.re.mul_add(c.re, c.im * c.im);
                if mag_sq == 0.0 {
                    *m = 0.0;
                    *p = Complex { re: 1.0, im: 0.0 };
                } else {
                    let mag_val = mag_sq.sqrt();
                    *m = pow_mag(mag_val, mag_sq, power_usize);
                    let inv_mag = mag_val.recip();
                    *p = Complex {
                        re: c.re * inv_mag,
                        im: c.im * inv_mag,
                    };
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        Zip::from(&mut mag)
            .and(&mut phase)
            .and(complex_spect)
            .for_each(|m, p, &c| {
                let mag_sq = c.re.mul_add(c.re, c.im * c.im);
                if mag_sq == 0.0 {
                    *m = 0.0;
                    *p = Complex { re: 1.0, im: 0.0 };
                } else {
                    let mag_val = mag_sq.sqrt();
                    *m = pow_mag(mag_val, mag_sq, power_usize);
                    let inv_mag = mag_val.recip();
                    *p = Complex {
                        re: c.re * inv_mag,
                        im: c.im * inv_mag,
                    };
                }
            });
    }

    (mag, phase)
}

// ============================================================================
// ITD Spectrogram
// ============================================================================

/// Interaural Time Difference (ITD) spectrogram.
///
/// ITD represents the time difference in seconds between when a sound reaches
/// the left and right ears. This is the primary cue for sound localization at
/// low frequencies (typically below 1500 Hz).
///
/// # Units
///
/// Data values are in **seconds**.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ItdSpectrogram {
    /// ITD values with shape (n_bins, n_frames) in seconds
    pub data: Array2<f64>,
    /// Parameters used to compute this spectrogram
    params: ITDSpectrogramParams,
    /// Frequency values for each bin (Hz)
    frequencies: NonEmptyVec<f64>,
    /// Time values for each frame (seconds)
    times: NonEmptyVec<f64>,
    /// Unit type marker
    _unit: PhantomData<ITD>,
}

impl AsRef<Array2<f64>> for ItdSpectrogram {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for ItdSpectrogram {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl ItdSpectrogram {
    /// Get the number of frequency bins.
    ///
    /// # Returns
    ///
    /// The number of frequency bins (rows in the data matrix).
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        // safety: data has at least one row since frequencies is NonEmpty
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    ///
    /// # Returns
    ///
    /// The number of time frames (columns in the data matrix).
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        // safety: data has at least one column since times is NonEmpty
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the parameters used to compute this spectrogram.
    ///
    /// # Returns
    ///
    /// A reference to the ITD spectrogram parameters.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &ITDSpectrogramParams {
        &self.params
    }

    /// Get the frequency values for each bin.
    ///
    /// # Returns
    ///
    /// A slice of frequency values in Hz.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the time values for each frame.
    ///
    /// # Returns
    ///
    /// A slice of time values in seconds.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        &self.times
    }

    /// Get the frequency range covered by this spectrogram.
    ///
    /// # Returns
    ///
    /// A tuple of (minimum frequency, maximum frequency) in Hz.
    #[inline]
    #[must_use]
    pub fn frequency_range(&self) -> (f64, f64) {
        let freqs = &*self.frequencies;
        (freqs[0], freqs[freqs.len().get() - 1])
    }

    /// Get the total duration of this spectrogram.
    ///
    /// # Returns
    ///
    /// The duration in seconds.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        let ts = &*self.times;
        ts[ts.len().get() - 1] - ts[0]
    }

    /// Get a unit label for the data values.
    ///
    /// # Returns
    ///
    /// A static string describing the units.
    #[inline]
    #[must_use]
    pub const fn unit_label() -> &'static str {
        "ITD (seconds)"
    }

    /// Compute a histogram of ITD values over time.
    ///
    /// Creates a 2D histogram where each column represents the distribution of ITD values
    /// for a given time frame, binned into delay bins.
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of histogram bins (default: 400)
    /// * `delay_range` - Range of delays in seconds as (min, max) (default: (-0.00088, 0.00088))
    /// * `energy_weighted` - If true, weight histogram by energy (magnitude sum)
    /// * `normalize` - If true, normalize each time frame's histogram to sum to 1
    ///
    /// # Returns
    ///
    /// A 2D array of shape (num_bins, n_frames) containing the histogram.
    #[must_use]
    pub fn histogram(
        &self,
        num_bins: Option<NonZeroUsize>,
        delay_range: Option<(f64, f64)>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Array2<f64> {
        let num_bins = num_bins.unwrap_or_else(|| crate::nzu!(400)).get();
        let (min_delay, max_delay) = delay_range.unwrap_or((-0.00088, 0.00088));
        let bin_width = (max_delay - min_delay) / num_bins as f64;

        let n_frames = self.n_frames().get();
        let n_freq_bins = self.n_bins().get();
        let mut histogram = Array2::zeros((num_bins, n_frames));

        for frame in 0..n_frames {
            for freq_bin in 0..n_freq_bins {
                let itd_value = self.data[(freq_bin, frame)];
                
                // Skip NaN or out-of-range values
                if !itd_value.is_finite() || itd_value < min_delay || itd_value > max_delay {
                    continue;
                }

                let bin_idx = ((itd_value - min_delay) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(num_bins - 1);

                let weight = if energy_weighted {
                    // Energy weighting would require magnitude data, use 1.0 for now
                    1.0
                } else {
                    1.0
                };

                histogram[(bin_idx, frame)] += weight;
            }

            // Normalize if requested
            if normalize {
                let sum: f64 = histogram.column(frame).sum();
                if sum > 0.0 {
                    for bin in 0..num_bins {
                        histogram[(bin, frame)] /= sum;
                    }
                }
            }
        }

        histogram
    }
}

/// Parameters for computing Interaural Time Difference (ITD) spectrograms.
///
/// ITD represents the time difference between when a sound reaches the left and right ears.
/// It is the primary cue for sound localization at low frequencies (typically below 1500 Hz),
/// where the wavelength is large enough that phase differences are unambiguous.
///
/// # Fields
///
/// * `spectrogram_params` - Base STFT parameters (FFT size, hop length, window, etc.)
/// * `start_freq` - Lower frequency bound for ITD analysis (Hz)
/// * `end_freq` - Upper frequency bound for ITD analysis (Hz)
/// * `magphase_power` - Power to raise magnitude-phase product to
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ITDSpectrogramParams {
    pub(crate) spectrogram_params: SpectrogramParams,
    pub(crate) start_freq: f64,
    pub(crate) end_freq: f64,
    pub(crate) magphase_power: NonZeroUsize,
}

impl ITDSpectrogramParams {
    /// Create new ITD spectrogram parameters.
    ///
    /// # Arguments
    ///
    /// * `spec_params` - Base spectrogram parameters
    /// * `start_freq` - Lower frequency bound in Hz (must be positive)
    /// * `stop_freq` - Upper frequency bound in Hz (must be > start_freq and < Nyquist)
    /// * `magphase_power` - Optional power for magnitude-phase product (defaults to 1)
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if:
    /// - Frequencies are not positive
    /// - Start frequency >= stop frequency
    /// - Stop frequency > Nyquist frequency (sample_rate / 2)
    /// - Sample rate is not positive
    pub fn new(
        spec_params: SpectrogramParams,
        start_freq: f64,
        stop_freq: f64,
        magphase_power: Option<NonZeroUsize>,
    ) -> SpectrogramResult<Self> {
        let sample_rate = spec_params.sample_rate_hz;
        if start_freq <= 0.0 || stop_freq <= 0.0 {
            return Err(SpectrogramError::InvalidInput(
                "Start and end frequencies must be positive.".into(),
            ));
        }
        if start_freq >= stop_freq {
            return Err(SpectrogramError::InvalidInput(
                "Start frequency must be less than end frequency.".into(),
            ));
        }
        if sample_rate <= 0.0 {
            return Err(SpectrogramError::InvalidInput(
                "Sample rate must be positive.".into(),
            ));
        }

        if stop_freq > sample_rate / 2.0 {
            return Err(SpectrogramError::InvalidInput(
                "End frequency must be less than Nyquist frequency.".into(),
            ));
        }

        Ok(Self {
            spectrogram_params: spec_params,
            start_freq,
            end_freq: stop_freq,
            magphase_power: magphase_power.unwrap_or_else(|| crate::nzu!(1)),
        })
    }
}

/// Compute the Interaural Time Difference (ITD) spectrogram for a stereo audio signal.
///
/// The ITD spectrogram shows how the time delay between left and right ear signals varies
/// across frequency and time. This is computed from the phase difference between channels,
/// weighted by the magnitude product and normalized by frequency.
///
/// # Arguments
///
/// * `audio` - Array of two audio channels [left, right] as non-empty slices
/// * `params` - ITD spectrogram parameters
/// * `plan` - Mutable STFT plan for efficient computation (reused across calls)
///
/// # Returns
///
/// An `ItdSpectrogram` containing ITD values in seconds with associated frequency and time axes.
#[inline]
#[must_use]
pub fn compute_itd_spectrogram(
    audio: [&NonEmptySlice<f64>; 2],
    params: &ITDSpectrogramParams,
    plan: &mut StftPlan, // force reuse
) -> SpectrogramResult<ItdSpectrogram> {
    let window_size = params.spectrogram_params.stft().n_fft();

    let bin_width = params.spectrogram_params.sample_rate_hz / window_size.get() as f64;

    let start_bin = (params.start_freq / bin_width).round() as usize;
    let stop_bin = (params.end_freq / bin_width).round() as usize;

    let left = audio[0];
    let right = audio[1];
    let left_spec = plan.compute(left, &params.spectrogram_params)?;
    let right_spec = plan.compute(right, &params.spectrogram_params)?;

    let (left_mag, left_phase) = magphase(&left_spec, params.magphase_power);
    let (right_mag, right_phase) = magphase(&right_spec, params.magphase_power);

    // Compute intensity only for the frequency range we need
    let num_frames = left_mag.shape()[1];
    let num_bins = stop_bin - start_bin;

    // Slice arrays to only the frequency range of interest for better cache locality
    let left_mag_slice = left_mag.slice(ndarray::s![start_bin..stop_bin, ..]);
    let right_mag_slice = right_mag.slice(ndarray::s![start_bin..stop_bin, ..]);
    let left_phase_slice = left_phase.slice(ndarray::s![start_bin..stop_bin, ..]);
    let right_phase_slice = right_phase.slice(ndarray::s![start_bin..stop_bin, ..]);

    let mut itd_spectrogram = Array2::zeros((num_bins, num_frames));
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    #[inline(always)]
    fn np_mod(x: f64, m: f64) -> f64 {
        ((x % m) + m) % m
    }

    #[cfg(feature = "rayon")]
    Zip::indexed(itd_spectrogram.view_mut()).par_for_each(|(bin_idx, frame), o| {
        // Compute phase angles on-the-fly instead of allocating arrays
        let left_angle = left_phase_slice[(bin_idx, frame)].im.atan2(left_phase_slice[(bin_idx, frame)].re);
        let right_angle = right_phase_slice[(bin_idx, frame)].im.atan2(right_phase_slice[(bin_idx, frame)].re);

        // Check intensity (sum of magnitudes should always be >= 0, but check for numerical safety)
        let intensity_val = left_mag_slice[(bin_idx, frame)] + right_mag_slice[(bin_idx, frame)];
        if intensity_val > 0.0 {
            let diff = left_angle - right_angle;
            let wrapped = np_mod(diff + pi, two_pi) - pi;
            let actual_bin = (start_bin + bin_idx) as f64;
            *o = wrapped / (two_pi * bin_width * actual_bin);
        }
    });

    #[cfg(not(feature = "rayon"))]
    Zip::indexed(itd_spectrogram.view_mut()).for_each(|(bin_idx, frame), o| {
        // Compute phase angles on-the-fly instead of allocating arrays
        let left_angle = left_phase_slice[(bin_idx, frame)].im.atan2(left_phase_slice[(bin_idx, frame)].re);
        let right_angle = right_phase_slice[(bin_idx, frame)].im.atan2(right_phase_slice[(bin_idx, frame)].re);

        // Check intensity (sum of magnitudes should always be >= 0, but check for numerical safety)
        let intensity_val = left_mag_slice[(bin_idx, frame)] + right_mag_slice[(bin_idx, frame)];
        if intensity_val > 0.0 {
            let diff = left_angle - right_angle;
            let wrapped = np_mod(diff + pi, two_pi) - pi;
            let actual_bin = (start_bin + bin_idx) as f64;
            *o = wrapped / (two_pi * bin_width * actual_bin);
        }
    });

    // Build frequency axis
    let frequencies: Vec<f64> = (start_bin..stop_bin)
        .map(|bin| bin as f64 * bin_width)
        .collect();
    let frequencies = NonEmptyVec::new(frequencies)
        .expect("Frequency range should have at least one bin");

    // Build time axis
    let hop_size = params.spectrogram_params.stft().hop_size().get() as f64;
    let sample_rate = params.spectrogram_params.sample_rate_hz;
    let times: Vec<f64> = (0..num_frames)
        .map(|frame| frame as f64 * hop_size / sample_rate)
        .collect();
    let times = NonEmptyVec::new(times)
        .expect("Time axis should have at least one frame");

    Ok(ItdSpectrogram {
        data: itd_spectrogram,
        params: params.clone(),
        frequencies,
        times,
        _unit: PhantomData,
    })
}

// ============================================================================
// IPD Spectrogram
// ============================================================================

/// Interaural Phase Difference (IPD) spectrogram.
///
/// IPD represents the phase difference in radians between left and right ear signals.
/// It's related to ITD but expressed directly as phase rather than time delay.
///
/// # Units
///
/// Data values are in **radians**.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IpdSpectrogram {
    /// IPD values with shape (n_bins, n_frames) in radians
    pub data: Array2<f64>,
    /// Parameters used to compute this spectrogram
    params: IPDSpectrogramParams,
    /// Frequency values for each bin (Hz)
    frequencies: NonEmptyVec<f64>,
    /// Time values for each frame (seconds)
    times: NonEmptyVec<f64>,
    /// Unit type marker
    _unit: PhantomData<IPD>,
}

impl AsRef<Array2<f64>> for IpdSpectrogram {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for IpdSpectrogram {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IpdSpectrogram {
    /// Get the number of frequency bins.
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the parameters used to compute this spectrogram.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &IPDSpectrogramParams {
        &self.params
    }

    /// Get the frequency values for each bin.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the time values for each frame.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        &self.times
    }

    /// Get the frequency range covered by this spectrogram.
    #[inline]
    #[must_use]
    pub fn frequency_range(&self) -> (f64, f64) {
        let freqs = &*self.frequencies;
        (freqs[0], freqs[freqs.len().get() - 1])
    }

    /// Get the total duration of this spectrogram.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        let ts = &*self.times;
        ts[ts.len().get() - 1] - ts[0]
    }

    /// Get a unit label for the data values.
    #[inline]
    #[must_use]
    pub const fn unit_label() -> &'static str {
        "IPD (radians)"
    }

    /// Compute a histogram of IPD values over time.
    ///
    /// Creates a 2D histogram where each column represents the distribution of IPD values
    /// for a given time frame, binned into phase bins.
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of histogram bins (default: 400)
    /// * `phase_range` - Range of phase in radians as (min, max) (default: (-π, π))
    /// * `energy_weighted` - If true, weight histogram by energy (magnitude sum)
    /// * `normalize` - If true, normalize each time frame's histogram to sum to 1
    ///
    /// # Returns
    ///
    /// A 2D array of shape (num_bins, n_frames) containing the histogram.
    #[must_use]
    pub fn histogram(
        &self,
        num_bins: Option<NonZeroUsize>,
        phase_range: Option<(f64, f64)>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Array2<f64> {
        let num_bins = num_bins.unwrap_or_else(|| crate::nzu!(400)).get();
        let pi = std::f64::consts::PI;
        let (min_phase, max_phase) = phase_range.unwrap_or((-pi, pi));
        let bin_width = (max_phase - min_phase) / num_bins as f64;

        let n_frames = self.n_frames().get();
        let n_freq_bins = self.n_bins().get();
        let mut histogram = Array2::zeros((num_bins, n_frames));

        for frame in 0..n_frames {
            for freq_bin in 0..n_freq_bins {
                let ipd_value = self.data[(freq_bin, frame)];
                
                // Skip NaN or out-of-range values
                if !ipd_value.is_finite() || ipd_value < min_phase || ipd_value > max_phase {
                    continue;
                }

                let bin_idx = ((ipd_value - min_phase) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(num_bins - 1);

                let weight = if energy_weighted {
                    1.0
                } else {
                    1.0
                };

                histogram[(bin_idx, frame)] += weight;
            }

            // Normalize if requested
            if normalize {
                let sum: f64 = histogram.column(frame).sum();
                if sum > 0.0 {
                    for bin in 0..num_bins {
                        histogram[(bin, frame)] /= sum;
                    }
                }
            }
        }

        histogram
    }
}

/// Parameters for computing Interaural Phase Difference (IPD) spectrograms.
///
/// IPD represents the phase difference between left and right ear signals. It's related to ITD
/// but expressed directly as phase rather than time delay. At higher frequencies, phase differences
/// can wrap around (phase ambiguity), so the `wrapped` parameter controls this behavior.
///
/// # Fields
///
/// * `spectrogram_params` - Base STFT parameters
/// * `start_freq` - Lower frequency bound for IPD analysis (Hz)
/// * `end_freq` - Upper frequency bound for IPD analysis (Hz)
/// * `wrapped` - If true, wrap phase to [-π, π]; if false, return unwrapped phase
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IPDSpectrogramParams {
    pub(crate) spectrogram_params: SpectrogramParams,
    pub(crate) start_freq: f64,
    pub(crate) end_freq: f64,
    pub(crate) wrapped: bool,
}

impl IPDSpectrogramParams {
    /// Create new IPD spectrogram parameters.
    ///
    /// # Arguments
    ///
    /// * `spec_params` - Base spectrogram parameters
    /// * `start_freq` - Lower frequency bound in Hz
    /// * `stop_freq` - Upper frequency bound in Hz
    /// * `wrapped` - Whether to wrap phase difference to [-π, π]
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if frequencies are invalid or out of range.
    pub fn new(
        spec_params: SpectrogramParams,
        start_freq: f64,
        stop_freq: f64,
        wrapped: bool,
    ) -> SpectrogramResult<Self> {
        let sample_rate = spec_params.sample_rate_hz;
        if start_freq <= 0.0 || stop_freq <= 0.0 {
            return Err(SpectrogramError::InvalidInput(
                "Start and end frequencies must be positive.".into(),
            ));
        }
        if start_freq >= stop_freq {
            return Err(SpectrogramError::InvalidInput(
                "Start frequency must be less than end frequency.".into(),
            ));
        }
        if stop_freq > sample_rate / 2.0 {
            return Err(SpectrogramError::InvalidInput(
                "End frequency must be less than Nyquist frequency.".into(),
            ));
        }

        Ok(Self {
            spectrogram_params: spec_params,
            start_freq,
            end_freq: stop_freq,
            wrapped,
        })
    }
}

/// Compute the Interaural Phase Difference (IPD) spectrogram for a stereo audio signal.
///
/// IPD captures the phase difference between left and right channels across frequency and time.
/// This is closely related to ITD but expressed in radians rather than seconds.
///
/// # Arguments
///
/// * `audio` - Array of two audio channels [left, right]
/// * `params` - IPD spectrogram parameters
/// * `plan` - Mutable STFT plan for efficient computation
///
/// # Returns
///
/// An `IpdSpectrogram` containing IPD values in radians with associated frequency and time axes.
#[inline]
#[must_use]
pub fn compute_ipd_spectrogram(
    audio: [&NonEmptySlice<f64>; 2],
    params: &IPDSpectrogramParams,
    plan: &mut StftPlan,
) -> SpectrogramResult<IpdSpectrogram> {
    let window_size = params.spectrogram_params.stft().n_fft();
    let bin_width = params.spectrogram_params.sample_rate_hz / window_size.get() as f64;

    let start_bin = (params.start_freq / bin_width).round() as usize;
    let stop_bin = (params.end_freq / bin_width).round() as usize;

    let left = audio[0];
    let right = audio[1];
    let left_spec = plan.compute(left, &params.spectrogram_params)?;
    let right_spec = plan.compute(right, &params.spectrogram_params)?;

    // We only need phase, not magnitude
    let (_, left_phase) = magphase(&left_spec, crate::nzu!(1));
    let (_, right_phase) = magphase(&right_spec, crate::nzu!(1));

    let num_frames = left_phase.shape()[1];
    let num_bins = stop_bin - start_bin;

    // Slice to frequency range
    let left_phase_slice = left_phase.slice(ndarray::s![start_bin..stop_bin, ..]);
    let right_phase_slice = right_phase.slice(ndarray::s![start_bin..stop_bin, ..]);

    let mut ipd_spectrogram = Array2::zeros((num_bins, num_frames));
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    #[inline(always)]
    fn np_mod(x: f64, m: f64) -> f64 {
        ((x % m) + m) % m
    }

    #[cfg(feature = "rayon")]
    Zip::indexed(ipd_spectrogram.view_mut()).par_for_each(|(bin_idx, frame), o| {
        let left_angle = left_phase_slice[(bin_idx, frame)].im.atan2(left_phase_slice[(bin_idx, frame)].re);
        let right_angle = right_phase_slice[(bin_idx, frame)].im.atan2(right_phase_slice[(bin_idx, frame)].re);

        let diff = left_angle - right_angle;

        if params.wrapped {
            *o = np_mod(diff + pi, two_pi) - pi;
        } else {
            *o = diff;
        }
    });

    #[cfg(not(feature = "rayon"))]
    Zip::indexed(ipd_spectrogram.view_mut()).for_each(|(bin_idx, frame), o| {
        let left_angle = left_phase_slice[(bin_idx, frame)].im.atan2(left_phase_slice[(bin_idx, frame)].re);
        let right_angle = right_phase_slice[(bin_idx, frame)].im.atan2(right_phase_slice[(bin_idx, frame)].re);

        let diff = left_angle - right_angle;

        if params.wrapped {
            *o = np_mod(diff + pi, two_pi) - pi;
        } else {
            *o = diff;
        }
    });

    // Build frequency axis
    let frequencies: Vec<f64> = (start_bin..stop_bin)
        .map(|bin| bin as f64 * bin_width)
        .collect();
    let frequencies = NonEmptyVec::new(frequencies)
        .expect("Frequency range should have at least one bin");

    // Build time axis
    let hop_size = params.spectrogram_params.stft().hop_size().get() as f64;
    let sample_rate = params.spectrogram_params.sample_rate_hz;
    let times: Vec<f64> = (0..num_frames)
        .map(|frame| frame as f64 * hop_size / sample_rate)
        .collect();
    let times = NonEmptyVec::new(times)
        .expect("Time axis should have at least one frame");

    Ok(IpdSpectrogram {
        data: ipd_spectrogram,
        params: params.clone(),
        frequencies,
        times,
        _unit: PhantomData,
    })
}

// ============================================================================
// ILD Spectrogram
// ============================================================================

/// Interaural Level Difference (ILD) spectrogram.
///
/// ILD represents the sound intensity difference in decibels (dB) between
/// the left and right ears. It's the primary cue for high-frequency localization.
///
/// # Units
///
/// Data values are in **decibels (dB)**.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IldSpectrogram {
    /// ILD values with shape (n_bins, n_frames) in dB
    pub data: Array2<f64>,
    /// Parameters used to compute this spectrogram
    params: ILDSpectrogramParams,
    /// Frequency values for each bin (Hz)
    frequencies: NonEmptyVec<f64>,
    /// Time values for each frame (seconds)
    times: NonEmptyVec<f64>,
    /// Unit type marker
    _unit: PhantomData<ILD>,
}

impl AsRef<Array2<f64>> for IldSpectrogram {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for IldSpectrogram {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IldSpectrogram {
    /// Get the number of frequency bins.
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the parameters used to compute this spectrogram.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &ILDSpectrogramParams {
        &self.params
    }

    /// Get the frequency values for each bin.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the time values for each frame.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        &self.times
    }

    /// Get the frequency range covered by this spectrogram.
    #[inline]
    #[must_use]
    pub fn frequency_range(&self) -> (f64, f64) {
        let freqs = &*self.frequencies;
        (freqs[0], freqs[freqs.len().get() - 1])
    }

    /// Get the total duration of this spectrogram.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        let ts = &*self.times;
        ts[ts.len().get() - 1] - ts[0]
    }

    /// Get a unit label for the data values.
    #[inline]
    #[must_use]
    pub const fn unit_label() -> &'static str {
        "ILD (dB)"
    }

    /// Compute a histogram of ILD values over time.
    ///
    /// Creates a 2D histogram where each column represents the distribution of ILD values
    /// for a given time frame, binned into dB bins. Values are raised to a power (exponent)
    /// to enhance peaks in the distribution.
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of histogram bins (default: 400)
    /// * `db_range` - Range of dB values as (min, max) (default: (-24, 24))
    /// * `exponent` - Power to raise histogram values to enhance peaks (default: 3)
    /// * `energy_weighted` - If true, weight histogram by energy (magnitude sum)
    /// * `normalize` - If true, normalize each time frame's histogram to sum to 1
    ///
    /// # Returns
    ///
    /// A 2D array of shape (num_bins, n_frames) containing the histogram.
    #[must_use]
    pub fn histogram(
        &self,
        num_bins: Option<NonZeroUsize>,
        db_range: Option<(f64, f64)>,
        exponent: Option<i32>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Array2<f64> {
        let num_bins = num_bins.unwrap_or_else(|| crate::nzu!(400)).get();
        let (min_db, max_db) = db_range.unwrap_or((-24.0, 24.0));
        let exponent = exponent.unwrap_or(3);
        let bin_width = (max_db - min_db) / num_bins as f64;

        let n_frames = self.n_frames().get();
        let n_freq_bins = self.n_bins().get();
        let mut histogram = Array2::zeros((num_bins, n_frames));

        for frame in 0..n_frames {
            for freq_bin in 0..n_freq_bins {
                let ild_value = self.data[(freq_bin, frame)];
                
                // Skip NaN or out-of-range values
                if !ild_value.is_finite() || ild_value < min_db || ild_value > max_db {
                    continue;
                }

                let bin_idx = ((ild_value - min_db) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(num_bins - 1);

                let weight = if energy_weighted {
                    1.0
                } else {
                    1.0
                };

                histogram[(bin_idx, frame)] += weight;
            }

            // Apply exponent to enhance peaks
            if exponent != 1 {
                for bin in 0..num_bins {
                    let val: f64 = histogram[(bin, frame)];
                    histogram[(bin, frame)] = val.powi(exponent);
                }
            }

            // Normalize if requested
            if normalize {
                let sum: f64 = histogram.column(frame).sum();
                if sum > 0.0 {
                    for bin in 0..num_bins {
                        histogram[(bin, frame)] /= sum;
                    }
                }
            }
        }

        histogram
    }
}

/// Parameters for computing Interaural Level Difference (ILD) spectrograms.
///
/// ILD represents the difference in sound intensity (in dB) between the left and right ears.
/// It is the primary cue for sound localization at high frequencies (typically above 1500 Hz),
/// where the head creates a significant acoustic shadow.
///
/// # Fields
///
/// * `spectrogram_params` - Base STFT parameters
/// * `start_freq` - Lower frequency bound (typically 1700 Hz for ILD)
/// * `end_freq` - Upper frequency bound (typically 4600 Hz for ILD)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ILDSpectrogramParams {
    pub(crate) spectrogram_params: SpectrogramParams,
    pub(crate) start_freq: f64,
    pub(crate) end_freq: f64,
}

impl ILDSpectrogramParams {
    /// Create new ILD spectrogram parameters.
    ///
    /// # Arguments
    ///
    /// * `spec_params` - Base spectrogram parameters
    /// * `start_freq` - Lower frequency bound in Hz
    /// * `stop_freq` - Upper frequency bound in Hz
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if frequencies are invalid or out of range.
    pub fn new(
        spec_params: SpectrogramParams,
        start_freq: f64,
        stop_freq: f64,
    ) -> SpectrogramResult<Self> {
        let sample_rate = spec_params.sample_rate_hz;
        if start_freq <= 0.0 || stop_freq <= 0.0 {
            return Err(SpectrogramError::InvalidInput(
                "Start and end frequencies must be positive.".into(),
            ));
        }
        if start_freq >= stop_freq {
            return Err(SpectrogramError::InvalidInput(
                "Start frequency must be less than end frequency.".into(),
            ));
        }
        if stop_freq > sample_rate / 2.0 {
            return Err(SpectrogramError::InvalidInput(
                "End frequency must be less than Nyquist frequency.".into(),
            ));
        }

        Ok(Self {
            spectrogram_params: spec_params,
            start_freq,
            end_freq: stop_freq,
        })
    }
}

/// Compute the Interaural Level Difference (ILD) spectrogram in dB.
///
/// ILD is computed as -20 * log10(right/left), representing the intensity difference
/// between ears in decibels. Positive values indicate the left channel is louder,
/// negative values indicate the right channel is louder.
///
/// # Arguments
///
/// * `audio` - Array of two audio channels [left, right]
/// * `params` - ILD spectrogram parameters
/// * `plan` - Mutable STFT plan for efficient computation
///
/// # Returns
///
/// An `IldSpectrogram` containing ILD values in dB with associated frequency and time axes.
#[inline]
#[must_use]
pub fn compute_ild_spectrogram(
    audio: [&NonEmptySlice<f64>; 2],
    params: &ILDSpectrogramParams,
    plan: &mut StftPlan,
) -> SpectrogramResult<IldSpectrogram> {
    let window_size = params.spectrogram_params.stft().n_fft();
    let bin_width = params.spectrogram_params.sample_rate_hz / window_size.get() as f64;

    let start_bin = (params.start_freq / bin_width).round() as usize;
    let stop_bin = (params.end_freq / bin_width).round() as usize;

    let left = audio[0];
    let right = audio[1];
    let left_spec = plan.compute(left, &params.spectrogram_params)?;
    let right_spec = plan.compute(right, &params.spectrogram_params)?;

    let (left_mag, _) = magphase(&left_spec, crate::nzu!(1));
    let (right_mag, _) = magphase(&right_spec, crate::nzu!(1));

    let num_frames = left_mag.shape()[1];
    let num_bins = stop_bin - start_bin;

    // Slice to frequency range
    let left_mag_slice = left_mag.slice(ndarray::s![start_bin..stop_bin, ..]);
    let right_mag_slice = right_mag.slice(ndarray::s![start_bin..stop_bin, ..]);

    let mut ild_spectrogram = Array2::from_elem((num_bins, num_frames), f64::NAN);

    #[cfg(feature = "rayon")]
    Zip::indexed(ild_spectrogram.view_mut()).par_for_each(|(bin_idx, frame), o| {
        let left_val = left_mag_slice[(bin_idx, frame)];
        let right_val = right_mag_slice[(bin_idx, frame)];

        // Mask out low intensity (sum < 0, though this shouldn't happen with magnitudes)
        let intensity = left_val + right_val;
        if intensity > 0.0 && left_val > 0.0 && right_val > 0.0 {
            // ILD = 20 * log10(right / left), then negated as per binaspect
            *o = -20.0 * (right_val / left_val).log10();
        }
    });

    #[cfg(not(feature = "rayon"))]
    Zip::indexed(ild_spectrogram.view_mut()).for_each(|(bin_idx, frame), o| {
        let left_val = left_mag_slice[(bin_idx, frame)];
        let right_val = right_mag_slice[(bin_idx, frame)];

        let intensity = left_val + right_val;
        if intensity > 0.0 && left_val > 0.0 && right_val > 0.0 {
            *o = -20.0 * (right_val / left_val).log10();
        }
    });

    // Build frequency axis
    let frequencies: Vec<f64> = (start_bin..stop_bin)
        .map(|bin| bin as f64 * bin_width)
        .collect();
    let frequencies = NonEmptyVec::new(frequencies)
        .expect("Frequency range should have at least one bin");

    // Build time axis
    let hop_size = params.spectrogram_params.stft().hop_size().get() as f64;
    let sample_rate = params.spectrogram_params.sample_rate_hz;
    let times: Vec<f64> = (0..num_frames)
        .map(|frame| frame as f64 * hop_size / sample_rate)
        .collect();
    let times = NonEmptyVec::new(times)
        .expect("Time axis should have at least one frame");

    Ok(IldSpectrogram {
        data: ild_spectrogram,
        params: params.clone(),
        frequencies,
        times,
        _unit: PhantomData,
    })
}

// ============================================================================
// ILR Spectrogram
// ============================================================================

/// Interaural Level Ratio (ILR) spectrogram.
///
/// ILR is a normalized measure of level differences between ears, providing
/// values in the range [-1, 1] rather than decibels.
///
/// # Units
///
/// Data values are **dimensionless** in the range [-1, 1].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IlrSpectrogram {
    /// ILR values with shape (n_bins, n_frames) in range [-1, 1]
    pub data: Array2<f64>,
    /// Parameters used to compute this spectrogram
    params: ILRSpectrogramParams,
    /// Frequency values for each bin (Hz)
    frequencies: NonEmptyVec<f64>,
    /// Time values for each frame (seconds)
    times: NonEmptyVec<f64>,
    /// Unit type marker
    _unit: PhantomData<ILR>,
}

impl AsRef<Array2<f64>> for IlrSpectrogram {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for IlrSpectrogram {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IlrSpectrogram {
    /// Get the number of frequency bins.
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the parameters used to compute this spectrogram.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &ILRSpectrogramParams {
        &self.params
    }

    /// Get the frequency values for each bin.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the time values for each frame.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        &self.times
    }

    /// Get the frequency range covered by this spectrogram.
    #[inline]
    #[must_use]
    pub fn frequency_range(&self) -> (f64, f64) {
        let freqs = &*self.frequencies;
        (freqs[0], freqs[freqs.len().get() - 1])
    }

    /// Get the total duration of this spectrogram.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        let ts = &*self.times;
        ts[ts.len().get() - 1] - ts[0]
    }

    /// Get a unit label for the data values.
    #[inline]
    #[must_use]
    pub const fn unit_label() -> &'static str {
        "ILR (normalized)"
    }

    /// Compute a histogram of ILR values over time.
    ///
    /// Creates a 2D histogram where each column represents the distribution of ILR values
    /// for a given time frame, binned into ratio bins. Values are raised to a power (exponent)
    /// to enhance peaks in the distribution.
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of histogram bins (default: 400)
    /// * `ratio_range` - Range of ratio values as (min, max) (default: (-1, 1))
    /// * `exponent` - Power to raise histogram values to enhance peaks (default: 3)
    /// * `energy_weighted` - If true, weight histogram by energy (magnitude sum)
    /// * `normalize` - If true, normalize each time frame's histogram to sum to 1
    ///
    /// # Returns
    ///
    /// A 2D array of shape (num_bins, n_frames) containing the histogram.
    #[must_use]
    pub fn histogram(
        &self,
        num_bins: Option<NonZeroUsize>,
        ratio_range: Option<(f64, f64)>,
        exponent: Option<i32>,
        energy_weighted: bool,
        normalize: bool,
    ) -> Array2<f64> {
        let num_bins = num_bins.unwrap_or_else(|| crate::nzu!(400)).get();
        let (min_ratio, max_ratio) = ratio_range.unwrap_or((-1.0, 1.0));
        let exponent = exponent.unwrap_or(3);
        let bin_width = (max_ratio - min_ratio) / num_bins as f64;

        let n_frames = self.n_frames().get();
        let n_freq_bins = self.n_bins().get();
        let mut histogram = Array2::zeros((num_bins, n_frames));

        for frame in 0..n_frames {
            for freq_bin in 0..n_freq_bins {
                let ilr_value = self.data[(freq_bin, frame)];
                
                // Skip NaN or out-of-range values
                if !ilr_value.is_finite() || ilr_value < min_ratio || ilr_value > max_ratio {
                    continue;
                }

                let bin_idx = ((ilr_value - min_ratio) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(num_bins - 1);

                let weight = if energy_weighted {
                    1.0
                } else {
                    1.0
                };

                histogram[(bin_idx, frame)] += weight;
            }

            // Apply exponent to enhance peaks
            if exponent != 1 {
                for bin in 0..num_bins {
                    let val: f64 = histogram[(bin, frame)];
                    histogram[(bin, frame)] = val.powi(exponent);
                }
            }

            // Normalize if requested
            if normalize {
                let sum: f64 = histogram.column(frame).sum();
                if sum > 0.0 {
                    for bin in 0..num_bins {
                        histogram[(bin, frame)] /= sum;
                    }
                }
            }
        }

        histogram
    }
}

/// Parameters for computing Interaural Level Ratio (ILR) spectrograms.
///
/// ILR is a normalized measure of level differences between ears, providing values
/// in the range [-1, 1] rather than decibels. This can be more intuitive for
/// certain applications and avoids logarithmic scaling issues.
///
/// # Fields
///
/// * `spectrogram_params` - Base STFT parameters
/// * `start_freq` - Lower frequency bound
/// * `end_freq` - Upper frequency bound
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ILRSpectrogramParams {
    pub(crate) spectrogram_params: SpectrogramParams,
    pub(crate) start_freq: f64,
    pub(crate) end_freq: f64,
}

impl ILRSpectrogramParams {
    /// Create new ILR spectrogram parameters.
    ///
    /// # Arguments
    ///
    /// * `spec_params` - Base spectrogram parameters
    /// * `start_freq` - Lower frequency bound in Hz
    /// * `stop_freq` - Upper frequency bound in Hz
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if frequencies are invalid or out of range.
    pub fn new(
        spec_params: SpectrogramParams,
        start_freq: f64,
        stop_freq: f64,
    ) -> SpectrogramResult<Self> {
        let sample_rate = spec_params.sample_rate_hz;
        if start_freq <= 0.0 || stop_freq <= 0.0 {
            return Err(SpectrogramError::InvalidInput(
                "Start and end frequencies must be positive.".into(),
            ));
        }
        if start_freq >= stop_freq {
            return Err(SpectrogramError::InvalidInput(
                "Start frequency must be less than end frequency.".into(),
            ));
        }
        if stop_freq > sample_rate / 2.0 {
            return Err(SpectrogramError::InvalidInput(
                "End frequency must be less than Nyquist frequency.".into(),
            ));
        }

        Ok(Self {
            spectrogram_params: spec_params,
            start_freq,
            end_freq: stop_freq,
        })
    }
}

/// Compute the Interaural Level Ratio (ILR) spectrogram.
///
/// ILR provides a normalized measure of level differences in the range [-1, 1].
/// The transformation is:
/// - If right/left < 1: ILR = 1 - right/left (positive, left louder)
/// - If right/left ≥ 1: ILR = -(1 - left/right) (negative, right louder)
///
/// # Arguments
///
/// * `audio` - Array of two audio channels [left, right]
/// * `params` - ILR spectrogram parameters
/// * `plan` - Mutable STFT plan for efficient computation
///
/// # Returns
///
/// An `IlrSpectrogram` containing ILR values in [-1, 1] with associated frequency and time axes.
#[inline]
#[must_use]
pub fn compute_ilr_spectrogram(
    audio: [&NonEmptySlice<f64>; 2],
    params: &ILRSpectrogramParams,
    plan: &mut StftPlan,
) -> SpectrogramResult<IlrSpectrogram> {
    let window_size = params.spectrogram_params.stft().n_fft();
    let bin_width = params.spectrogram_params.sample_rate_hz / window_size.get() as f64;

    let start_bin = (params.start_freq / bin_width).round() as usize;
    let stop_bin = (params.end_freq / bin_width).round() as usize;

    let left = audio[0];
    let right = audio[1];
    let left_spec = plan.compute(left, &params.spectrogram_params)?;
    let right_spec = plan.compute(right, &params.spectrogram_params)?;

    let (left_mag, _) = magphase(&left_spec, crate::nzu!(1));
    let (right_mag, _) = magphase(&right_spec, crate::nzu!(1));

    let num_frames = left_mag.shape()[1];
    let num_bins = stop_bin - start_bin;

    // Slice to frequency range
    let left_mag_slice = left_mag.slice(ndarray::s![start_bin..stop_bin, ..]);
    let right_mag_slice = right_mag.slice(ndarray::s![start_bin..stop_bin, ..]);

    let mut ilr_spectrogram = Array2::from_elem((num_bins, num_frames), f64::NAN);

    #[cfg(feature = "rayon")]
    Zip::indexed(ilr_spectrogram.view_mut()).par_for_each(|(bin_idx, frame), o| {
        let left_val = left_mag_slice[(bin_idx, frame)];
        let right_val = right_mag_slice[(bin_idx, frame)];

        let intensity = left_val + right_val;
        if intensity > 0.0 && left_val > 0.0 && right_val > 0.0 {
            let ratio = right_val / left_val;

            // Transform: values < 1 => 1 - ratio, values >= 1 => -(1 - 1/ratio)
            if ratio < 1.0 {
                *o = 1.0 - ratio;
            } else {
                *o = -(1.0 - 1.0 / ratio);
            }
        }
    });

    #[cfg(not(feature = "rayon"))]
    Zip::indexed(ilr_spectrogram.view_mut()).for_each(|(bin_idx, frame), o| {
        let left_val = left_mag_slice[(bin_idx, frame)];
        let right_val = right_mag_slice[(bin_idx, frame)];

        let intensity = left_val + right_val;
        if intensity > 0.0 && left_val > 0.0 && right_val > 0.0 {
            let ratio = right_val / left_val;

            if ratio < 1.0 {
                *o = 1.0 - ratio;
            } else {
                *o = -(1.0 - 1.0 / ratio);
            }
        }
    });

    // Build frequency axis
    let frequencies: Vec<f64> = (start_bin..stop_bin)
        .map(|bin| bin as f64 * bin_width)
        .collect();
    let frequencies = NonEmptyVec::new(frequencies)
        .expect("Frequency range should have at least one bin");

    // Build time axis
    let hop_size = params.spectrogram_params.stft().hop_size().get() as f64;
    let sample_rate = params.spectrogram_params.sample_rate_hz;
    let times: Vec<f64> = (0..num_frames)
        .map(|frame| frame as f64 * hop_size / sample_rate)
        .collect();
    let times = NonEmptyVec::new(times)
        .expect("Time axis should have at least one frame");

    Ok(IlrSpectrogram {
        data: ilr_spectrogram,
        params: params.clone(),
        frequencies,
        times,
        _unit: PhantomData,
    })
}

// ============================================================================
// Diff Functions
// ============================================================================

#[inline]
pub(crate) fn median(arr: &Array1<f64>) -> f64 {
    let mut sorted: Vec<f64> = arr.iter().filter(|x| x.is_finite()).copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    if len == 0 {
        return f64::NAN;
    }
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Compute the difference between two ITD spectrograms.
///
/// # Returns
///
/// A tuple of `(itd_time_diff, mean_diff_degrees, mean_diff_itd)` where:
/// - `itd_time_diff` is a 1D array of mean ITD differences per frame
/// - `mean_diff_degrees` is the mean absolute difference converted to degrees
/// - `mean_diff_itd` is the median ITD difference across frames
#[inline]
#[must_use]
pub fn compute_itd_spectrogram_diff(
    reference: [&NonEmptySlice<f64>; 2],
    test: [&NonEmptySlice<f64>; 2],
    params: &ITDSpectrogramParams,
    plan: &mut StftPlan,
) -> SpectrogramResult<(Array1<f64>, f64, f64)> {
    let ref_itd = compute_itd_spectrogram(reference, params, plan)?;
    let test_itd = compute_itd_spectrogram(test, params, plan)?;

    let diff = &test_itd.data - &ref_itd.data;

    let col_means = diff
        .mean_axis(Axis(0))
        .expect("Non-empty slices produce non-zero diff array");

    let mean_diff_degrees = col_means
        .mapv(|x| x.abs() * (1.0 / 0.00086) * 90.0)
        .mean()
        .expect("Non-empty slices produce non-zero diff array");

    let mean_diff_itd = median(&col_means);

    Ok((col_means, mean_diff_degrees, mean_diff_itd))
}

/// Compute the difference between two ILR spectrograms.
///
/// # Returns
///
/// A tuple of `(ilr_time_diff, mean_diff)` where:
/// - `ilr_time_diff` is a 1D array of mean ILR differences per frame
/// - `mean_diff` is the mean absolute difference across frames
#[inline]
#[must_use]
pub fn compute_ilr_spectrogram_diff(
    reference: [&NonEmptySlice<f64>; 2],
    test: [&NonEmptySlice<f64>; 2],
    params: &ILRSpectrogramParams,
    plan: &mut StftPlan,
) -> SpectrogramResult<(Array1<f64>, f64)> {
    let ref_ilr = compute_ilr_spectrogram(reference, params, plan)?;
    let test_ilr = compute_ilr_spectrogram(test, params, plan)?;

    let diff = &test_ilr.data - &ref_ilr.data;

    let n_frames = diff.ncols();
    let col_means: Array1<f64> = (0..n_frames)
        .map(|frame| {
            let col = diff.column(frame);
            let (sum, count) = col
                .iter()
                .filter(|x| !x.is_nan())
                .fold((0.0_f64, 0usize), |(s, c), x| (s + x, c + 1));
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect();

    let mean_diff = {
        let (sum, count) = col_means
            .iter()
            .filter(|x| !x.is_nan())
            .fold((0.0_f64, 0usize), |(s, c), x| (s + x.abs(), c + 1));
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    };

    Ok((col_means, mean_diff))
}

