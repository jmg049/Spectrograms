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

use crate::erb::{ErbParams, gammatone_center_frequencies, gammatone_iir_spectrogram};
use crate::error::{SpectrogramError, SpectrogramResult};
use crate::sample::Sample;
use crate::spectrogram::{AmpScaleSpec, SpectrogramPlan};

/// A source of frame-wise spectrogram features.
///
/// Implementors map a mono signal of scalar type `T` to a `[n_bands ×
/// n_frames]` matrix (bands in rows, frames in columns). Frame `k` covers
/// samples starting at `k * hop` where `hop = hop_seconds() * sample_rate()`.
///
/// The crate provides two families of implementors:
/// - every [`SpectrogramPlan`] (so mel / ERB / linear / log-Hz / CQT all work),
/// - [`GammatoneSource`] for the time-domain IIR gammatone bank.
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
