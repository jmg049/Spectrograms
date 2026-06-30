//! Pluggable per-frame spectrogram sources.
//!
//! [`SpectrogramSource`] abstracts "a thing that turns a mono signal into a
//! `[n_bands × n_frames]` feature matrix". It deliberately makes **no**
//! representation privileged: the crate implements it for every
//! [`SpectrogramPlan`](crate::SpectrogramPlan) (linear, log-Hz, mel, ERB, CQT)
//! and for the time-domain IIR gammatone bank ([`GammatoneSource`]), and any
//! downstream consumer can stay generic over the representation while callers
//! choose — or supply their own implementation for a representation the crate
//! does not provide.

use std::num::NonZeroUsize;

use ndarray::Array2;
use non_empty_slice::NonEmptySlice;

use crate::chroma::{ChromaParams, N_CHROMA, chromagram};
use crate::cqt::{CqtParams, cqt};
use crate::erb::{ErbParams, gammatone_center_frequencies, gammatone_iir_spectrogram};
use crate::error::{SpectrogramError, SpectrogramResult};
use crate::mfcc::{MfccParams, mfcc};
use crate::sample::Sample;
use crate::spectrogram::{AmpScaleSpec, SpectrogramPlan, StftParams};

/// A source of frame-wise spectrogram features.
///
/// Implementors map a mono signal of scalar type `T` to a `[n_bands ×
/// n_frames]` matrix (bands in rows, frames in columns). Frame `k` covers
/// samples starting at `k * hop` where `hop = hop_seconds() * sample_rate()`.
///
/// The crate provides several families of implementors:
/// - every [`SpectrogramPlan`] (mel / ERB / linear / log-Hz / CQT via plan),
/// - [`GammatoneSource`] for the time-domain IIR gammatone bank,
/// - [`CqtSource`] for the standalone CQT kernel path,
/// - [`ChromaSource`] for chroma (pitch-class) features,
/// - [`MfccSource`] for mel-frequency cepstral coefficients.
///
/// Implement it yourself to plug in any other representation.
pub trait SpectrogramSource<T: Sample> {
    /// Compute the `[n_bands × n_frames]` feature matrix for `samples`.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` is empty or too short for the configured
    /// framing, or if the underlying transform fails.
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>>;

    /// Number of frequency bands (rows of the output matrix).
    fn n_bands(&self) -> usize;

    /// Centre frequency (Hz) of each band, low→high. Length equals [`Self::n_bands`].
    fn center_frequencies(&self) -> Vec<f64>;

    /// Sample rate (Hz) the source expects its input to be at.
    fn sample_rate(&self) -> f64;

    /// Seconds between successive frames (`hop_size / sample_rate`).
    fn hop_seconds(&self) -> f64;
}

/// Every compiled [`SpectrogramPlan`] is a [`SpectrogramSource`].
///
/// The amplitude scale is whatever the plan was built with (`Power`,
/// `Magnitude`, `Decibels`); consumers that need a particular scale should
/// build the plan accordingly.
impl<F, A, T> SpectrogramSource<T> for SpectrogramPlan<F, A, T>
where
    F: Copy + Clone + 'static,
    A: AmpScaleSpec + 'static,
    T: Sample,
{
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>> {
        let samples = NonEmptySlice::new(samples)
            .ok_or_else(|| SpectrogramError::invalid_input("samples must be non-empty"))?;
        Ok(self.compute(samples)?.into_data())
    }

    fn n_bands(&self) -> usize {
        self.freq_axis().frequencies().len().get()
    }

    fn center_frequencies(&self) -> Vec<f64> {
        self.freq_axis().frequencies().as_slice().to_vec()
    }

    fn sample_rate(&self) -> f64 {
        self.params().sample_rate_hz()
    }

    fn hop_seconds(&self) -> f64 {
        self.params().frame_period_seconds()
    }
}

// ─── GammatoneSource ─────────────────────────────────────────────────────────

/// [`SpectrogramSource`] backed by the time-domain IIR gammatone filter bank
/// ([`gammatone_iir_spectrogram`]).
///
/// Set a dB floor on the [`ErbParams`] (e.g. `ErbParams::with_db_floor(-45.0)`)
/// for decibel-scaled output.
#[derive(Clone, Debug)]
pub struct GammatoneSource {
    sample_rate: f64,
    frame_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    params: ErbParams,
}

impl GammatoneSource {
    /// Create a gammatone source.
    ///
    /// * `sample_rate` – rate (Hz) the input signal is expected at.
    /// * `frame_size` – analysis frame length in samples.
    /// * `hop_size` – hop between frames in samples.
    /// * `params` – band count, frequency range, spacing, and optional dB floor.
    #[must_use]
    pub const fn new(
        sample_rate: f64,
        frame_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        params: ErbParams,
    ) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            params,
        }
    }

    /// The gammatone parameters this source uses.
    #[must_use]
    pub const fn params(&self) -> &ErbParams {
        &self.params
    }
}

impl<T: Sample> SpectrogramSource<T> for GammatoneSource {
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>> {
        let (matrix, _center_freqs) = gammatone_iir_spectrogram(
            samples,
            self.sample_rate,
            self.frame_size,
            self.hop_size,
            &self.params,
        )?;
        Ok(matrix)
    }

    fn n_bands(&self) -> usize {
        self.params.n_filters().get()
    }

    fn center_frequencies(&self) -> Vec<f64> {
        gammatone_center_frequencies(&self.params)
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn hop_seconds(&self) -> f64 {
        self.hop_size.get() as f64 / self.sample_rate
    }
}

// ─── CqtSource ───────────────────────────────────────────────────────────────

/// [`SpectrogramSource`] backed by the standalone CQT transform ([`cqt`]).
///
/// Returns the magnitude of the complex CQT coefficients. Bins whose centre
/// frequency would exceed the Nyquist limit are dropped automatically.
#[derive(Clone, Debug)]
pub struct CqtSource {
    sample_rate: f64,
    hop_size: NonZeroUsize,
    params: CqtParams,
}

impl CqtSource {
    /// Create a CQT source.
    ///
    /// * `sample_rate` – rate (Hz) the input signal is expected at.
    /// * `hop_size` – hop between frames in samples.
    /// * `params` – CQT frequency and resolution parameters.
    #[must_use]
    pub fn new(sample_rate: f64, hop_size: NonZeroUsize, params: CqtParams) -> Self {
        Self { sample_rate, hop_size, params }
    }

    /// The CQT parameters this source uses.
    #[must_use]
    pub const fn params(&self) -> &CqtParams {
        &self.params
    }
}

impl<T: Sample> SpectrogramSource<T> for CqtSource {
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>> {
        let samples = NonEmptySlice::new(samples)
            .ok_or_else(|| SpectrogramError::invalid_input("samples must be non-empty"))?;
        Ok(cqt(samples, self.sample_rate, &self.params, self.hop_size)?.to_magnitude())
    }

    fn n_bands(&self) -> usize {
        let nyquist = self.sample_rate / 2.0;
        (0..self.params.num_bins().get())
            .filter(|&i| self.params.bin_frequency(i) < nyquist)
            .count()
    }

    fn center_frequencies(&self) -> Vec<f64> {
        let nyquist = self.sample_rate / 2.0;
        (0..self.params.num_bins().get())
            .map(|i| self.params.bin_frequency(i))
            .filter(|&f| f < nyquist)
            .collect()
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn hop_seconds(&self) -> f64 {
        self.hop_size.get() as f64 / self.sample_rate
    }
}

// ─── ChromaSource ─────────────────────────────────────────────────────────────

/// [`SpectrogramSource`] backed by the chroma filterbank ([`chromagram`]).
///
/// Always produces 12 bands (one per pitch class, C through B).
/// [`SpectrogramSource::center_frequencies`] returns the 12 pitch-class
/// frequencies of the lowest octave (`f_min * 2^(i/12)` for i = 0..12).
#[derive(Clone, Debug)]
pub struct ChromaSource {
    sample_rate: f64,
    stft_params: StftParams,
    params: ChromaParams,
}

impl ChromaSource {
    /// Create a chroma source.
    ///
    /// * `sample_rate` – rate (Hz) the input signal is expected at.
    /// * `stft_params` – STFT configuration used by the chroma filterbank.
    /// * `params` – chroma parameters (tuning, frequency range, normalization).
    #[must_use]
    pub fn new(sample_rate: f64, stft_params: StftParams, params: ChromaParams) -> Self {
        Self { sample_rate, stft_params, params }
    }

    /// The chroma parameters this source uses.
    #[must_use]
    pub const fn params(&self) -> &ChromaParams {
        &self.params
    }
}

impl<T: Sample> SpectrogramSource<T> for ChromaSource {
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>> {
        let samples = NonEmptySlice::new(samples)
            .ok_or_else(|| SpectrogramError::invalid_input("samples must be non-empty"))?;
        Ok(chromagram(samples, &self.stft_params, self.sample_rate, &self.params)?.data)
    }

    fn n_bands(&self) -> usize {
        N_CHROMA
    }

    fn center_frequencies(&self) -> Vec<f64> {
        let f_min = self.params.f_min();
        (0..N_CHROMA)
            .map(|i| f_min * (i as f64 / N_CHROMA as f64).exp2())
            .collect()
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn hop_seconds(&self) -> f64 {
        self.stft_params.hop_size().get() as f64 / self.sample_rate
    }
}

// ─── MfccSource ──────────────────────────────────────────────────────────────

/// [`SpectrogramSource`] backed by the MFCC pipeline ([`mfcc`]).
///
/// Produces `n_mfcc` bands. [`SpectrogramSource::center_frequencies`] returns
/// coefficient indices (0.0, 1.0, 2.0, …) since MFCC coefficients do not map
/// to specific centre frequencies.
#[derive(Clone, Debug)]
pub struct MfccSource {
    sample_rate: f64,
    stft_params: StftParams,
    n_mels: NonZeroUsize,
    params: MfccParams,
}

impl MfccSource {
    /// Create an MFCC source.
    ///
    /// * `sample_rate` – rate (Hz) the input signal is expected at.
    /// * `stft_params` – STFT configuration used by the underlying mel spectrogram.
    /// * `n_mels` – number of mel filter bands.
    /// * `params` – MFCC parameters (coefficient count, C0 inclusion, liftering).
    #[must_use]
    pub fn new(
        sample_rate: f64,
        stft_params: StftParams,
        n_mels: NonZeroUsize,
        params: MfccParams,
    ) -> Self {
        Self { sample_rate, stft_params, n_mels, params }
    }

    /// The MFCC parameters this source uses.
    #[must_use]
    pub const fn params(&self) -> &MfccParams {
        &self.params
    }
}

impl<T: Sample> SpectrogramSource<T> for MfccSource {
    fn compute_matrix(&mut self, samples: &[T]) -> SpectrogramResult<Array2<T>> {
        let samples = NonEmptySlice::new(samples)
            .ok_or_else(|| SpectrogramError::invalid_input("samples must be non-empty"))?;
        Ok(mfcc(samples, &self.stft_params, self.sample_rate, self.n_mels, &self.params)?.data)
    }

    fn n_bands(&self) -> usize {
        self.params.n_mfcc().get()
    }

    fn center_frequencies(&self) -> Vec<f64> {
        (0..self.params.n_mfcc().get()).map(|i| i as f64).collect()
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn hop_seconds(&self) -> f64 {
        self.stft_params.hop_size().get() as f64 / self.sample_rate
    }
}
