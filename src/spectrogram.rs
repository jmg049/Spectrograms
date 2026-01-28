use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

use ndarray::{Array1, Array2};
use non_empty_slice::{NonEmptySlice, NonEmptyVec, non_empty_vec};
use num_complex::Complex;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::cqt::CqtKernel;
use crate::erb::ErbFilterbank;
use crate::{
    CqtParams, ErbParams, R2cPlan, SpectrogramError, SpectrogramResult, WindowType,
    min_max_single_pass, nzu,
};
const EPS: f64 = 1e-12;

//
// ========================
// Sparse Matrix for efficient filterbank multiplication
// ========================
//

/// Row-wise sparse matrix optimized for matrix-vector multiplication.
///
/// This structure stores sparse data as a vector of vectors, where each row maintains
/// its own list of non-zero values and corresponding column indices. This is more
/// flexible than traditional CSR format and allows efficient row-by-row construction.
///
/// This structure is designed for matrices with very few non-zero values per row,
/// such as mel filterbanks (triangular filters) and logarithmic frequency mappings
/// (linear interpolation between 1-2 adjacent bins).
///
/// For typical spectrograms:
/// - `LogHz` interpolation: Only 1-2 non-zeros per row (~99% sparse)
/// - Mel filterbank: ~10-50 non-zeros per row depending on FFT size (~90-98% sparse)
///
/// By storing only non-zero values, we avoid wasting CPU cycles multiplying by zero,
/// which can provide 10-100x speedup compared to dense matrix multiplication.
#[derive(Debug, Clone)]
struct SparseMatrix {
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
    /// Non-zero values for each row (row-major order)
    values: Vec<Vec<f64>>,
    /// Column indices for each non-zero value
    indices: Vec<Vec<usize>>,
}

impl SparseMatrix {
    /// Create a new sparse matrix with the given dimensions.
    fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            values: vec![Vec::new(); nrows],
            indices: vec![Vec::new(); nrows],
        }
    }

    /// Set a value in the matrix. Only stores if value is non-zero.
    ///
    /// # Panics (debug mode only)
    /// Panics in debug builds if row or col are out of bounds.
    fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(
            row < self.nrows && col < self.ncols,
            "SparseMatrix index out of bounds: ({}, {}) for {}x{} matrix",
            row,
            col,
            self.nrows,
            self.ncols
        );
        if row >= self.nrows || col >= self.ncols {
            return;
        }

        // Only store non-zero values (with small epsilon for numerical stability)
        if value.abs() > 1e-10 {
            self.values[row].push(value);
            self.indices[row].push(col);
        }
    }

    /// Get the number of rows.
    const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Get the number of columns.
    const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Perform sparse matrix-vector multiplication: out = self * input
    /// This is much faster than dense multiplication when the matrix is sparse.
    #[inline]
    fn multiply_vec(&self, input: &[f64], out: &mut [f64]) {
        debug_assert_eq!(input.len(), self.ncols);
        debug_assert_eq!(out.len(), self.nrows);

        for (row_idx, (row_values, row_indices)) in
            self.values.iter().zip(&self.indices).enumerate()
        {
            let mut acc = 0.0;
            for (&value, &col_idx) in row_values.iter().zip(row_indices) {
                acc += value * input[col_idx];
            }
            out[row_idx] = acc;
        }
    }
}

// Linear frequency
pub type LinearPowerSpectrogram = Spectrogram<LinearHz, Power>;
pub type LinearMagnitudeSpectrogram = Spectrogram<LinearHz, Magnitude>;
pub type LinearDbSpectrogram = Spectrogram<LinearHz, Decibels>;
pub type LinearSpectrogram<AmpScale> = Spectrogram<LinearHz, AmpScale>;

// Log-frequency (e.g. CQT-style)
pub type LogHzPowerSpectrogram = Spectrogram<LogHz, Power>;
pub type LogHzMagnitudeSpectrogram = Spectrogram<LogHz, Magnitude>;
pub type LogHzDbSpectrogram = Spectrogram<LogHz, Decibels>;
pub type LogHzSpectrogram<AmpScale> = Spectrogram<LogHz, AmpScale>;

// ERB / gammatone
pub type ErbPowerSpectrogram = Spectrogram<Erb, Power>;
pub type ErbMagnitudeSpectrogram = Spectrogram<Erb, Magnitude>;
pub type ErbDbSpectrogram = Spectrogram<Erb, Decibels>;
pub type GammatonePowerSpectrogram = ErbPowerSpectrogram;
pub type GammatoneMagnitudeSpectrogram = ErbMagnitudeSpectrogram;
pub type GammatoneDbSpectrogram = ErbDbSpectrogram;
pub type ErbSpectrogram<AmpScale> = Spectrogram<Erb, AmpScale>;
pub type GammatoneSpectrogram<AmpScale> = ErbSpectrogram<AmpScale>;

// Mel
pub type MelMagnitudeSpectrogram = Spectrogram<Mel, Magnitude>;
pub type MelPowerSpectrogram = Spectrogram<Mel, Power>;
pub type MelDbSpectrogram = Spectrogram<Mel, Decibels>;
pub type LogMelSpectrogram = MelDbSpectrogram;
pub type MelSpectrogram<AmpScale> = Spectrogram<Mel, AmpScale>;

// CQT
pub type CqtPowerSpectrogram = Spectrogram<Cqt, Power>;
pub type CqtMagnitudeSpectrogram = Spectrogram<Cqt, Magnitude>;
pub type CqtDbSpectrogram = Spectrogram<Cqt, Decibels>;
pub type CqtSpectrogram<AmpScale> = Spectrogram<Cqt, AmpScale>;

use crate::fft_backend::r2c_output_size;

/// A spectrogram plan is the compiled, reusable execution object.
///
/// It owns:
/// - FFT plan (reusable)
/// - window samples
/// - mapping (identity / mel filterbank / etc.)
/// - amplitude scaling config
/// - workspace buffers to avoid allocations in hot loops
///
/// It computes one specific spectrogram type: `Spectrogram<FreqScale, AmpScale>`.
///
/// # Type Parameters
///
/// - `FreqScale`: Frequency scale type (e.g. `LinearHz`, `LogHz`, `Mel`, etc.)
/// - `AmpScale`: Amplitude scaling type (e.g. `Power`, `Magnitude`, `Decibels`, etc.)
pub struct SpectrogramPlan<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    params: SpectrogramParams,

    stft: StftPlan,
    mapping: FrequencyMapping<FreqScale>,
    scaling: AmplitudeScaling<AmpScale>,

    freq_axis: FrequencyAxis<FreqScale>,
    workspace: Workspace,

    _amp: PhantomData<AmpScale>,
}

impl<FreqScale, AmpScale> SpectrogramPlan<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    /// Get the spectrogram parameters used to create this plan.
    ///
    /// # Returns
    ///
    /// A reference to the `SpectrogramParams` used in this plan.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &SpectrogramParams {
        &self.params
    }

    /// Get the frequency axis for this spectrogram plan.
    ///
    /// # Returns
    ///
    /// A reference to the `FrequencyAxis<FreqScale>` used in this plan.
    #[inline]
    #[must_use]
    pub const fn freq_axis(&self) -> &FrequencyAxis<FreqScale> {
        &self.freq_axis
    }

    /// Compute a spectrogram for a mono signal.
    ///
    /// This function performs:
    /// - framing + windowing
    /// - FFT per frame
    /// - magnitude/power
    /// - frequency mapping (identity/mel/etc.)
    /// - amplitude scaling (linear or dB)
    ///
    /// It allocates the output `Array2` once, but does not allocate per-frame.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples
    ///
    /// # Returns
    ///
    /// A `Spectrogram<FreqScale, AmpScale>` containing the computed spectrogram.
    ///
    /// # Errors
    ///
    /// Returns an error if STFT computation or mapping fails.
    #[inline]
    pub fn compute(
        &mut self,
        samples: &NonEmptySlice<f64>,
    ) -> SpectrogramResult<Spectrogram<FreqScale, AmpScale>> {
        let n_frames = self.stft.frame_count(samples.len())?;
        let n_bins = self.mapping.output_bins();

        // Create output matrix: (n_bins, n_frames)
        let mut data = Array2::<f64>::zeros((n_bins.get(), n_frames.get()));

        // Ensure workspace is correctly sized
        self.workspace
            .ensure_sizes(self.stft.n_fft, self.stft.out_len, n_bins);

        // Main loop: fill each frame (column)
        // TODO: Parallelize with rayon once thread-safety issues are resolved
        for frame_idx in 0..n_frames.get() {
            self.stft
                .compute_frame_spectrum(samples, frame_idx, &mut self.workspace)?;

            // mapping: spectrum(out_len) -> mapped(n_bins)
            // For CQT, this uses workspace.frame; for others, workspace.spectrum
            // For ERB, we need the complex FFT output (fft_out)
            // We need to borrow workspace fields separately to avoid borrow conflicts
            let Workspace {
                spectrum,
                mapped,
                frame,
                ..
            } = &mut self.workspace;

            self.mapping.apply(spectrum, frame, mapped)?;

            // amplitude scaling in-place on mapped vector
            self.scaling.apply_in_place(mapped)?;

            // write column into output
            for (row, &val) in mapped.iter().enumerate() {
                data[[row, frame_idx]] = val;
            }
        }

        let times = build_time_axis_seconds(&self.params, n_frames);
        let axes = Axes::new(self.freq_axis.clone(), times);

        Ok(Spectrogram::new(data, axes, self.params))
    }

    /// Compute a single frame of the spectrogram.
    ///
    /// This is useful for streaming/online processing where you want to
    /// process audio frame-by-frame without computing the entire spectrogram.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (must contain at least enough samples for the requested frame)
    /// * `frame_idx` - Frame index to compute
    ///
    /// # Returns
    ///
    /// A vector of frequency bin values for the requested frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame index is out of bounds or if STFT computation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let mut plan = planner.linear_plan::<Power>(&params, None)?;
    ///
    /// // Compute just the first frame
    /// let frame = plan.compute_frame(&samples, 0)?;
    /// assert_eq!(frame.len(), nzu!(257)); // n_fft/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_frame(
        &mut self,
        samples: &NonEmptySlice<f64>,
        frame_idx: usize,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        let n_bins = self.mapping.output_bins();

        // Ensure workspace is correctly sized
        self.workspace
            .ensure_sizes(self.stft.n_fft, self.stft.out_len, n_bins);

        // Compute frame spectrum
        self.stft
            .compute_frame_spectrum(samples, frame_idx, &mut self.workspace)?;

        // Apply mapping (using split borrows to avoid borrow conflicts)
        let Workspace {
            spectrum,
            mapped,
            frame,
            ..
        } = &mut self.workspace;

        self.mapping.apply(spectrum, frame, mapped)?;

        // Apply amplitude scaling
        self.scaling.apply_in_place(mapped)?;

        Ok(mapped.clone())
    }

    /// Compute spectrogram into a pre-allocated buffer.
    ///
    /// This avoids allocating the output matrix, which is useful when
    /// you want to reuse buffers or have strict memory requirements.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples
    /// * `output` - Pre-allocated output matrix (must be correct size: `n_bins` x `n_frames`)
    ///
    /// # Returns
    ///
    /// An empty result on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the output buffer dimensions don't match the expected size.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use ndarray::Array2;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let mut plan = planner.linear_plan::<Power>(&params, None)?;
    ///
    /// // Pre-allocate output buffer
    /// let mut output = Array2::<f64>::zeros((257, 63));
    /// plan.compute_into(&samples, &mut output)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_into(
        &mut self,
        samples: &NonEmptySlice<f64>,
        output: &mut Array2<f64>,
    ) -> SpectrogramResult<()> {
        let n_frames = self.stft.frame_count(samples.len())?;
        let n_bins = self.mapping.output_bins();

        // Validate output dimensions
        if output.nrows() != n_bins.get() {
            return Err(SpectrogramError::dimension_mismatch(
                n_bins.get(),
                output.nrows(),
            ));
        }
        if output.ncols() != n_frames.get() {
            return Err(SpectrogramError::dimension_mismatch(
                n_frames.get(),
                output.ncols(),
            ));
        }

        // Ensure workspace is correctly sized
        self.workspace
            .ensure_sizes(self.stft.n_fft, self.stft.out_len, n_bins);

        // Main loop: fill each frame (column)
        for frame_idx in 0..n_frames.get() {
            self.stft
                .compute_frame_spectrum(samples, frame_idx, &mut self.workspace)?;

            // mapping: spectrum(out_len) -> mapped(n_bins)
            // For CQT, this uses workspace.frame; for others, workspace.spectrum
            // For ERB, we need the complex FFT output (fft_out)
            // We need to borrow workspace fields separately to avoid borrow conflicts
            let Workspace {
                spectrum,
                mapped,
                frame,
                ..
            } = &mut self.workspace;

            self.mapping.apply(spectrum, frame, mapped)?;

            // amplitude scaling in-place on mapped vector
            self.scaling.apply_in_place(mapped)?;

            // write column into output
            for (row, &val) in mapped.iter().enumerate() {
                output[[row, frame_idx]] = val;
            }
        }

        Ok(())
    }

    /// Get the expected output dimensions for a given signal length.
    ///
    /// # Arguments
    ///
    /// * `signal_length` - Length of the input signal in samples.
    ///
    /// # Returns
    ///
    /// A tuple `(n_bins, n_frames)` representing the output spectrogram shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal length is too short to produce any frames.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let plan = planner.linear_plan::<Power>(&params, None)?;
    ///
    /// let (n_bins, n_frames) = plan.output_shape(nzu!(16000))?;
    /// assert_eq!(n_bins, nzu!(257));
    /// assert_eq!(n_frames, nzu!(63));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn output_shape(
        &self,
        signal_length: NonZeroUsize,
    ) -> SpectrogramResult<(NonZeroUsize, NonZeroUsize)> {
        let n_frames = self.stft.frame_count(signal_length)?;
        let n_bins = self.mapping.output_bins();
        Ok((n_bins, n_frames))
    }
}

/// STFT (Short-Time Fourier Transform) result containing complex frequency bins.
///
/// This is the raw STFT output before any frequency mapping or amplitude scaling.
///
/// # Fields
///
/// - `data`: Complex STFT matrix with shape (`frequency_bins`, `time_frames`)
/// - `frequencies`: Frequency axis in Hz
/// - `sample_rate`: Sample rate in Hz
/// - `params`: STFT computation parameters
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StftResult {
    /// Complex STFT matrix with shape (`frequency_bins`, `time_frames`)
    pub data: Array2<Complex<f64>>,
    /// Frequency axis in Hz
    pub frequencies: NonEmptyVec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    pub params: StftParams,
}

impl StftResult {
    /// Get the number of frequency bins.
    ///
    /// # Returns
    ///
    /// Number of frequency bins in the STFT result.
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        // safety: nrows() > 0 for NonEmptyVec
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    ///
    /// # Returns
    ///
    /// Number of time frames in the STFT result.
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        // safety: ncols() > 0 for NonEmptyVec
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the frequency resolution in Hz
    ///
    /// # Returns
    ///
    /// Frequency bin width in Hz.
    #[inline]
    #[must_use]
    pub fn frequency_resolution(&self) -> f64 {
        self.sample_rate / self.params.n_fft().get() as f64
    }

    /// Get the time resolution in seconds.
    ///
    /// # Returns
    ///
    /// Time between successive frames in seconds.
    #[inline]
    #[must_use]
    pub fn time_resolution(&self) -> f64 {
        self.params.hop_size().get() as f64 / self.sample_rate
    }

    /// Normalizes self.data to remove the complex aspect of it.
    ///
    /// # Returns
    ///
    /// An Array2<f64> containing the norms of each complex number in self.data.
    #[inline]
    pub fn norm(&self) -> Array2<f64> {
        self.as_ref().mapv(Complex::norm)
    }
}

impl AsRef<Array2<Complex<f64>>> for StftResult {
    #[inline]
    fn as_ref(&self) -> &Array2<Complex<f64>> {
        &self.data
    }
}

impl AsMut<Array2<Complex<f64>>> for StftResult {
    #[inline]
    fn as_mut(&mut self) -> &mut Array2<Complex<f64>> {
        &mut self.data
    }
}

impl Deref for StftResult {
    type Target = Array2<Complex<f64>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for StftResult {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// A planner is an object that can build spectrogram plans.
///
/// In your design, this is where:
/// - FFT plans are created
/// - mapping matrices are compiled
/// - axes are computed
///
/// This allows you to keep plan building separate from the output types.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct SpectrogramPlanner;

impl SpectrogramPlanner {
    /// Create a new spectrogram planner.
    ///
    /// # Returns
    ///
    /// A new `SpectrogramPlanner` instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Compute the Short-Time Fourier Transform (STFT) of a signal.
    ///
    /// This returns the raw complex STFT matrix before any frequency mapping
    /// or amplitude scaling. Useful for applications that need the full complex
    /// spectrum or custom processing.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - STFT computation parameters
    ///
    /// # Returns
    ///
    /// An `StftResult` containing the complex STFT matrix and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if STFT computation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let stft_result = planner.compute_stft(&samples, &params)?;
    ///
    /// println!("STFT: {} bins x {} frames", stft_result.n_bins(), stft_result.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Note
    ///
    /// This method creates a new FFT plan each time. For processing multiple
    /// signals, create a reusable plan with `StftPlan::new()` instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// // One-shot (convenient)
    /// let planner = SpectrogramPlanner::new();
    /// let stft_result = planner.compute_stft(&non_empty_vec![0.0; nzu!(16000)], &params)?;
    ///
    /// // Reusable plan (efficient for batch)
    /// let mut plan = StftPlan::new(&params)?;
    /// for signal in &[non_empty_vec![0.0; nzu!(16000)], non_empty_vec![1.0; nzu!(16000)]] {
    ///     let stft = plan.compute(&signal, &params)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_stft(
        &self,
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
    ) -> SpectrogramResult<StftResult> {
        let mut plan = StftPlan::new(params)?;
        plan.compute(samples, params)
    }

    /// Compute the power spectrum of a single audio frame.
    ///
    /// This is useful for real-time processing or analyzing individual frames.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio frame (length ≤ n_fft, will be zero-padded if shorter)
    /// * `n_fft` - FFT size
    /// * `window` - Window type to apply
    ///
    /// # Returns
    ///
    /// A vector of power values (|X|²) with length `n_fft/2` + 1.
    ///
    /// # Automatic Zero-Padding
    ///
    /// If the input signal is shorter than `n_fft`, it will be automatically
    /// zero-padded to the required length.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` error if the input length exceeds `n_fft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let frame = non_empty_vec![0.0; nzu!(512)];
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let power = planner.compute_power_spectrum(frame.as_ref(), nzu!(512), WindowType::Hanning)?;
    ///
    /// assert_eq!(power.len(), nzu!(257)); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_power_spectrum(
        &self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
        window: WindowType,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        if samples.len() > n_fft {
            return Err(SpectrogramError::invalid_input(format!(
                "Input length ({}) exceeds FFT size ({})",
                samples.len(),
                n_fft
            )));
        }

        let window_samples = make_window(window, n_fft);
        let out_len = r2c_output_size(n_fft.get());

        // Create FFT plan
        #[cfg(feature = "realfft")]
        let mut fft = {
            let mut planner = crate::RealFftPlanner::new();
            let plan = planner.get_or_create(n_fft.get());
            crate::RealFftPlan::new(n_fft.get(), plan)
        };

        #[cfg(feature = "fftw")]
        let mut fft = {
            use std::sync::Arc;
            let plan = crate::FftwPlanner::build_plan(n_fft.get())?;
            crate::FftwPlan::new(Arc::new(plan))
        };

        // Apply window and compute FFT
        let mut windowed = vec![0.0; n_fft.get()];
        for i in 0..samples.len().get() {
            windowed[i] = samples[i] * window_samples[i];
        }
        // The rest is already zero-padded
        let mut fft_out = vec![Complex::new(0.0, 0.0); out_len];
        fft.process(&windowed, &mut fft_out)?;

        // Convert to power
        let power: Vec<f64> = fft_out.iter().map(num_complex::Complex::norm_sqr).collect();
        // safety: power is non-empty since n_fft > 0
        Ok(unsafe { NonEmptyVec::new_unchecked(power) })
    }

    /// Compute the magnitude spectrum of a single audio frame.
    ///
    /// This is useful for real-time processing or analyzing individual frames.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio frame (length ≤ n_fft, will be zero-padded if shorter)
    /// * `n_fft` - FFT size
    /// * `window` - Window type to apply
    ///
    /// # Returns
    ///
    /// A vector of magnitude values (|X|) with length `n_fft/2` + 1.
    ///
    /// # Automatic Zero-Padding
    ///
    /// If the input signal is shorter than `n_fft`, it will be automatically
    /// zero-padded to the required length.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` error if the input length exceeds `n_fft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let frame = non_empty_vec![0.0; nzu!(512)];
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let magnitude = planner.compute_magnitude_spectrum(frame.as_ref(), nzu!(512), WindowType::Hanning)?;
    ///
    /// assert_eq!(magnitude.len(), nzu!(257)); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_magnitude_spectrum(
        &self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
        window: WindowType,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        let power = self.compute_power_spectrum(samples, n_fft, window)?;
        let power = power.iter().map(|&p| p.sqrt()).collect::<Vec<f64>>();
        // safety: power is non-empty since power_spectrum returned successfully
        Ok(unsafe { NonEmptyVec::new_unchecked(power) })
    }

    /// Build a linear-frequency spectrogram plan.
    ///
    /// # Type Parameters
    ///
    /// `AmpScale` determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters
    /// * `db` - Logarithmic scaling parameters (only used if `AmpScale
    /// is `Decibels`)
    ///
    /// # Returns
    ///
    /// A `SpectrogramPlan` configured for linear-frequency spectrogram computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be created due to invalid parameters.
    #[inline]
    pub fn linear_plan<AmpScale>(
        &self,
        params: &SpectrogramParams,
        db: Option<&LogParams>, // only used when AmpScale = Decibels
    ) -> SpectrogramResult<SpectrogramPlan<LinearHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        let stft = StftPlan::new(params)?;
        let mapping = FrequencyMapping::<LinearHz>::new(params)?;
        let scaling = AmplitudeScaling::<AmpScale>::new(db);
        let freq_axis = build_frequency_axis::<LinearHz>(params, &mapping);

        let workspace = Workspace::new(stft.n_fft, stft.out_len, mapping.output_bins());

        Ok(SpectrogramPlan {
            params: *params,
            stft,
            mapping,
            scaling,
            freq_axis,
            workspace,
            _amp: PhantomData,
        })
    }

    /// Build a mel-frequency spectrogram plan.
    ///
    /// This compiles a mel filterbank matrix and caches it inside the plan.
    ///
    /// # Type Parameters
    ///
    /// `AmpScale`: determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters
    /// * `mel` - Mel-specific parameters
    /// * `db` - Logarithmic scaling parameters (only used if `AmpScale` is `Decibels`)
    ///
    /// # Returns
    ///
    /// A `SpectrogramPlan` configured for mel spectrogram computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be created due to invalid parameters.
    #[inline]
    pub fn mel_plan<AmpScale>(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: Option<&LogParams>, // only used when AmpScale = Decibels
    ) -> SpectrogramResult<SpectrogramPlan<Mel, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        // cross-validation: mel range must be compatible with sample rate
        let nyquist = params.nyquist_hz();
        if mel.f_max() > nyquist {
            return Err(SpectrogramError::invalid_input(
                "mel f_max must be <= Nyquist",
            ));
        }

        let stft = StftPlan::new(params)?;
        let mapping = FrequencyMapping::<Mel>::new_mel(params, mel)?;
        let scaling = AmplitudeScaling::<AmpScale>::new(db);
        let freq_axis = build_frequency_axis::<Mel>(params, &mapping);

        let workspace = Workspace::new(stft.n_fft, stft.out_len, mapping.output_bins());

        Ok(SpectrogramPlan {
            params: *params,
            stft,
            mapping,
            scaling,
            freq_axis,
            workspace,
            _amp: PhantomData,
        })
    }

    /// Build an ERB-scale spectrogram plan.
    ///
    /// This creates a spectrogram with ERB-spaced frequency bands using gammatone
    /// filterbank approximation in the frequency domain.
    ///
    /// # Type Parameters
    ///
    /// `AmpScale`: determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters
    /// * `erb` - ERB-specific parameters
    /// * `db` - Logarithmic scaling parameters (only used if `AmpScale` is `Decibels`)
    ///
    /// # Returns
    ///
    /// A `SpectrogramPlan` configured for ERB spectrogram computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be created due to invalid parameters.
    #[inline]
    pub fn erb_plan<AmpScale>(
        &self,
        params: &SpectrogramParams,
        erb: &ErbParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<SpectrogramPlan<Erb, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        // cross-validation: erb range must be compatible with sample rate
        let nyquist = params.nyquist_hz();
        if erb.f_max() > nyquist {
            return Err(SpectrogramError::invalid_input(format!(
                "f_max={} exceeds Nyquist={}",
                erb.f_max(),
                nyquist
            )));
        }

        let stft = StftPlan::new(params)?;
        let mapping = FrequencyMapping::<Erb>::new_erb(params, erb)?;
        let scaling = AmplitudeScaling::<AmpScale>::new(db);
        let freq_axis = build_frequency_axis::<Erb>(params, &mapping);

        let workspace = Workspace::new(stft.n_fft, stft.out_len, mapping.output_bins());

        Ok(SpectrogramPlan {
            params: *params,
            stft,
            mapping,
            scaling,
            freq_axis,
            workspace,
            _amp: PhantomData,
        })
    }

    /// Build a log-frequency plan.
    ///
    /// This creates a spectrogram with logarithmically-spaced frequency bins.
    ///
    /// # Type Parameters
    ///
    /// `AmpScale`: determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters
    /// * `loghz` - LogHz-specific parameters
    /// * `db` - Logarithmic scaling parameters (only used if `AmpScale` is `Decibels`)
    ///
    /// # Returns
    ///
    /// A `SpectrogramPlan` configured for log-frequency spectrogram computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be created due to invalid parameters.
    #[inline]
    pub fn log_hz_plan<AmpScale>(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<SpectrogramPlan<LogHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        // cross-validation: loghz range must be compatible with sample rate
        let nyquist = params.nyquist_hz();
        if loghz.f_max() > nyquist {
            return Err(SpectrogramError::invalid_input(format!(
                "f_max={} exceeds Nyquist={}",
                loghz.f_max(),
                nyquist
            )));
        }

        let stft = StftPlan::new(params)?;
        let mapping = FrequencyMapping::<LogHz>::new_loghz(params, loghz)?;
        let scaling = AmplitudeScaling::<AmpScale>::new(db);
        let freq_axis = build_frequency_axis::<LogHz>(params, &mapping);

        let workspace = Workspace::new(stft.n_fft, stft.out_len, mapping.output_bins());

        Ok(SpectrogramPlan {
            params: *params,
            stft,
            mapping,
            scaling,
            freq_axis,
            workspace,
            _amp: PhantomData,
        })
    }

    /// Build a cqt spectrogram plan.
    ///
    /// # Type Parameters
    ///
    /// `AmpScale`: determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters
    /// * `cqt` - CQT-specific parameters
    /// * `db` - Logarithmic scaling parameters (only used if `AmpScale` is `Decibels`)
    ///
    /// # Returns
    ///
    /// A `SpectrogramPlan` configured for CQT spectrogram computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan cannot be created due to invalid parameters.
    #[inline]
    pub fn cqt_plan<AmpScale>(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: Option<&LogParams>, // only used when AmpScale = Decibels
    ) -> SpectrogramResult<SpectrogramPlan<Cqt, AmpScale>>
    where
        AmpScale: AmpScaleSpec + 'static,
    {
        let stft = StftPlan::new(params)?;
        let mapping = FrequencyMapping::<Cqt>::new(params, cqt)?;
        let scaling = AmplitudeScaling::<AmpScale>::new(db);
        let freq_axis = build_frequency_axis::<Cqt>(params, &mapping);

        let workspace = Workspace::new(stft.n_fft, stft.out_len, mapping.output_bins());

        Ok(SpectrogramPlan {
            params: *params,
            stft,
            mapping,
            scaling,
            freq_axis,
            workspace,
            _amp: PhantomData,
        })
    }
}

/// STFT plan containing reusable FFT plan and buffers.
///
/// This struct is responsible for performing the Short-Time Fourier Transform (STFT)
/// on audio signals based on the provided parameters.
///
/// It encapsulates the FFT plan, windowing function, and internal buffers to efficiently
/// compute the STFT for multiple frames of audio data.
///
/// # Fields
///
/// - `n_fft`: Size of the FFT.
/// - `hop_size`: Hop size between consecutive frames.
/// - `window`: Windowing function samples.
/// - `centre`: Whether to centre the frames with padding.
/// - `out_len`: Length of the FFT output.
/// - `fft`: Boxed FFT plan for real-to-complex transformation.
/// - `fft_out`: Internal buffer for FFT output.
/// - `frame`: Internal buffer for windowed audio frames.
pub struct StftPlan {
    n_fft: NonZeroUsize,
    hop_size: NonZeroUsize,
    window: NonEmptyVec<f64>,
    centre: bool,

    out_len: NonZeroUsize,

    // FFT plan (reused for all frames)
    fft: Box<dyn R2cPlan>,

    // internal scratch
    fft_out: NonEmptyVec<Complex<f64>>,
    frame: NonEmptyVec<f64>,
}

impl StftPlan {
    /// Create a new STFT plan from parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Spectrogram parameters containing STFT config
    ///
    /// # Returns
    ///
    /// A new `StftPlan` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the FFT plan cannot be created.
    #[inline]
    pub fn new(params: &SpectrogramParams) -> SpectrogramResult<Self> {
        let stft = params.stft();
        let n_fft = stft.n_fft();
        let hop_size = stft.hop_size();
        let centre = stft.centre();

        let window = make_window(stft.window(), n_fft);

        let out_len = r2c_output_size(n_fft.get());
        let out_len = NonZeroUsize::new(out_len)
            .ok_or_else(|| SpectrogramError::invalid_input("FFT output length must be non-zero"))?;

        #[cfg(feature = "realfft")]
        let fft = {
            let mut planner = crate::RealFftPlanner::new();
            let plan = planner.get_or_create(n_fft.get());
            let plan = crate::RealFftPlan::new(n_fft.get(), plan);
            Box::new(plan)
        };

        #[cfg(feature = "fftw")]
        let fft = {
            use std::sync::Arc;
            let plan = crate::FftwPlanner::build_plan(n_fft.get())?;
            Box::new(crate::FftwPlan::new(Arc::new(plan)))
        };

        Ok(Self {
            n_fft,
            hop_size,
            window,
            centre,
            out_len,
            fft,
            fft_out: non_empty_vec![Complex::new(0.0, 0.0); out_len],
            frame: non_empty_vec![0.0; n_fft],
        })
    }

    fn frame_count(&self, n_samples: NonZeroUsize) -> SpectrogramResult<NonZeroUsize> {
        // Framing policy:
        // - centre = true: implicit padding of n_fft/2 on both sides
        // - centre = false: no padding
        //
        // Define the number of frames such that each frame has a valid centre sample position.
        let pad = if self.centre { self.n_fft.get() / 2 } else { 0 };
        let padded_len = n_samples.get() + 2 * pad;

        if padded_len < self.n_fft.get() {
            // still produce 1 frame (all padding / partial)
            return Ok(nzu!(1));
        }

        let remaining = padded_len - self.n_fft.get();
        let n_frames = remaining / self.hop_size().get() + 1;
        let n_frames = NonZeroUsize::new(n_frames).ok_or_else(|| {
            SpectrogramError::invalid_input("computed number of frames must be non-zero")
        })?;
        Ok(n_frames)
    }

    /// Compute one frame FFT using internal buffers only.
    fn compute_frame_fft_simple(
        &mut self,
        samples: &NonEmptySlice<f64>,
        frame_idx: usize,
    ) -> SpectrogramResult<()> {
        let out = self.frame.as_mut_slice();
        debug_assert_eq!(out.len(), self.n_fft.get());

        let pad = if self.centre { self.n_fft.get() / 2 } else { 0 };
        let start = frame_idx
            .checked_mul(self.hop_size.get())
            .ok_or_else(|| SpectrogramError::invalid_input("frame index overflow"))?;

        // Fill windowed frame
        for (i, sample) in out.iter_mut().enumerate().take(self.n_fft.get()) {
            let v_idx = start + i;
            let s_idx = v_idx as isize - pad as isize;

            let sample_val = if s_idx < 0 || (s_idx as usize) >= samples.len().get() {
                0.0
            } else {
                samples[s_idx as usize]
            };
            *sample = sample_val * self.window[i];
        }

        // Compute FFT
        let fft_out = self.fft_out.as_mut_slice();
        self.fft.process(out, fft_out)?;

        Ok(())
    }

    /// Compute one frame spectrum into workspace:
    /// - fills windowed frame
    /// - runs FFT
    /// - converts to magnitude/power based on `AmpScale` later
    fn compute_frame_spectrum(
        &mut self,
        samples: &NonEmptySlice<f64>,
        frame_idx: usize,
        workspace: &mut Workspace,
    ) -> SpectrogramResult<()> {
        let out = workspace.frame.as_mut_slice();

        // self.fill_frame(samples, frame_idx, frame)?;
        debug_assert_eq!(out.len(), self.n_fft.get());

        let pad = if self.centre { self.n_fft.get() / 2 } else { 0 };
        let start = frame_idx
            .checked_mul(self.hop_size().get())
            .ok_or_else(|| SpectrogramError::invalid_input("frame index overflow"))?;

        // The "virtual" signal is samples with pad zeros on both sides.
        // Virtual index 0..padded_len
        // Map virtual index to original samples by subtracting pad.
        for (i, sample) in out.iter_mut().enumerate().take(self.n_fft.get()) {
            let v_idx = start + i;
            let s_idx = v_idx as isize - pad as isize;

            let sample_val = if s_idx < 0 || (s_idx as usize) >= samples.len().get() {
                0.0
            } else {
                samples[s_idx as usize]
            };

            *sample = sample_val * self.window[i];
        }
        let fft_out = workspace.fft_out.as_mut_slice();
        // FFT
        self.fft.process(out, fft_out)?;

        // Convert complex spectrum to linear magnitude OR power here? No:
        // Keep "spectrum" as power by default? That would entangle semantics.
        //
        // Instead, we store magnitude^2 (power) as the canonical intermediate,
        // and let AmpScale decide later whether output is magnitude or power.
        //
        // This is consistent and avoids recomputing norms multiple times.
        for (i, c) in workspace.fft_out.iter().enumerate() {
            workspace.spectrum[i] = c.norm_sqr();
        }

        Ok(())
    }

    /// Compute the full STFT for a signal, returning an `StftResult`.
    ///
    /// This is a convenience method that handles frame iteration and
    /// builds the complete STFT matrix.
    ///
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples
    /// * `params` - STFT computation parameters
    ///
    /// # Returns
    ///
    /// An `StftResult` containing the complex STFT matrix and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let mut plan = StftPlan::new(&params)?;
    ///
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft_result = plan.compute(&samples, &params)?;
    ///
    /// println!("STFT: {} bins x {} frames", stft_result.n_bins(), stft_result.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute(
        &mut self,
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
    ) -> SpectrogramResult<StftResult> {
        let n_frames = self.frame_count(samples.len())?;
        let n_bins = self.out_len;

        // Allocate output matrix (frequency_bins x time_frames)
        let mut data = Array2::<Complex<f64>>::zeros((n_bins.get(), n_frames.get()));

        // Compute each frame
        for frame_idx in 0..n_frames.get() {
            self.compute_frame_fft_simple(samples, frame_idx)?;

            // Copy from internal buffer to output
            for (bin_idx, &value) in self.fft_out.iter().enumerate() {
                data[[bin_idx, frame_idx]] = value;
            }
        }

        // Build frequency axis
        let frequencies: Vec<f64> = (0..n_bins.get())
            .map(|k| k as f64 * params.sample_rate_hz() / params.stft().n_fft().get() as f64)
            .collect();
        // SAFETY: n_bins > 0
        let frequencies = unsafe { NonEmptyVec::new_unchecked(frequencies) };

        Ok(StftResult {
            data,
            frequencies,
            sample_rate: params.sample_rate_hz(),
            params: *params.stft(),
        })
    }

    /// Compute a single frame of STFT, returning the complex spectrum.
    ///
    /// This is useful for streaming/online processing where you want to
    /// process audio frame-by-frame.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples
    /// * `frame_idx` - Index of the frame to compute
    ///
    /// # Returns
    ///
    /// A `NonEmptyVec` containing the complex spectrum for the specified frame.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let mut plan = StftPlan::new(&params)?;
    ///
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let (_, n_frames) = plan.output_shape(samples.len())?;
    ///
    /// for frame_idx in 0..n_frames.get() {
    ///     let spectrum = plan.compute_frame_simple(&samples, frame_idx)?;
    ///     // Process spectrum...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_frame_simple(
        &mut self,
        samples: &NonEmptySlice<f64>,
        frame_idx: usize,
    ) -> SpectrogramResult<NonEmptyVec<Complex<f64>>> {
        self.compute_frame_fft_simple(samples, frame_idx)?;
        Ok(self.fft_out.clone())
    }

    /// Compute STFT into a pre-allocated buffer.
    ///
    /// This avoids allocating the output matrix, useful for reusing buffers.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples
    /// * `output` - Pre-allocated output buffer (shape: `n_bins` x `n_frames`)
    ///  
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if dimensions mismatch.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` error if the output buffer has incorrect shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use spectrograms::*;
    /// use ndarray::Array2;
    /// use num_complex::Complex;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let mut plan = StftPlan::new(&params)?;
    ///
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let (n_bins, n_frames) = plan.output_shape(samples.len())?;
    /// let mut output = Array2::<Complex<f64>>::zeros((n_bins.get(), n_frames.get()));
    ///
    /// plan.compute_into(&samples, &mut output)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute_into(
        &mut self,
        samples: &NonEmptySlice<f64>,
        output: &mut Array2<Complex<f64>>,
    ) -> SpectrogramResult<()> {
        let n_frames = self.frame_count(samples.len())?;
        let n_bins = self.out_len;

        // Validate output dimensions
        if output.nrows() != n_bins.get() {
            return Err(SpectrogramError::dimension_mismatch(
                n_bins.get(),
                output.nrows(),
            ));
        }
        if output.ncols() != n_frames.get() {
            return Err(SpectrogramError::dimension_mismatch(
                n_frames.get(),
                output.ncols(),
            ));
        }

        // Compute into pre-allocated buffer
        for frame_idx in 0..n_frames.get() {
            self.compute_frame_fft_simple(samples, frame_idx)?;

            for (bin_idx, &value) in self.fft_out.iter().enumerate() {
                output[[bin_idx, frame_idx]] = value;
            }
        }

        Ok(())
    }

    /// Get the expected output dimensions for a given signal length.
    ///
    /// # Arguments
    ///
    /// * `signal_length` - Length of the input signal in samples
    ///
    /// # Returns
    ///
    /// A tuple `(n_frequency_bins, n_time_frames)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the computed number of frames is invalid.
    #[inline]
    pub fn output_shape(
        &self,
        signal_length: NonZeroUsize,
    ) -> SpectrogramResult<(NonZeroUsize, NonZeroUsize)> {
        let n_frames = self.frame_count(signal_length)?;
        Ok((self.out_len, n_frames))
    }

    /// Get the number of frequency bins in the output.
    ///
    /// # Returns
    ///
    /// The number of frequency bins.
    #[inline]
    #[must_use]
    pub const fn n_bins(&self) -> NonZeroUsize {
        self.out_len
    }

    /// Get the FFT size.
    ///
    /// # Returns
    ///
    /// The FFT size.
    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> NonZeroUsize {
        self.n_fft
    }

    /// Get the hop size.
    ///
    /// # Returns
    ///
    /// The hop size.
    #[inline]
    #[must_use]
    pub const fn hop_size(&self) -> NonZeroUsize {
        self.hop_size
    }
}

#[derive(Debug, Clone)]
enum MappingKind {
    Identity {
        out_len: NonZeroUsize,
    },
    Mel {
        matrix: SparseMatrix,
    }, // shape: (n_mels, out_len)
    LogHz {
        matrix: SparseMatrix,
        frequencies: NonEmptyVec<f64>,
    }, // shape: (n_bins, out_len)
    Erb {
        filterbank: ErbFilterbank,
    },
    Cqt {
        kernel: CqtKernel,
    },
}

/// Typed mapping wrapper.
#[derive(Debug, Clone)]
struct FrequencyMapping<FreqScale> {
    kind: MappingKind,
    _marker: PhantomData<FreqScale>,
}

impl FrequencyMapping<LinearHz> {
    fn new(params: &SpectrogramParams) -> SpectrogramResult<Self> {
        let out_len = r2c_output_size(params.stft().n_fft().get());
        let out_len = NonZeroUsize::new(out_len)
            .ok_or_else(|| SpectrogramError::invalid_input("FFT output length must be non-zero"))?;
        Ok(Self {
            kind: MappingKind::Identity { out_len },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<Mel> {
    fn new_mel(params: &SpectrogramParams, mel: &MelParams) -> SpectrogramResult<Self> {
        let n_fft = params.stft().n_fft();
        let out_len = r2c_output_size(n_fft.get());
        let out_len = NonZeroUsize::new(out_len)
            .ok_or_else(|| SpectrogramError::invalid_input("FFT output length must be non-zero"))?;

        // Validate: mel bins must be <= something sensible
        if mel.n_mels() > nzu!(10_000) {
            return Err(SpectrogramError::invalid_input(
                "n_mels is unreasonably large",
            ));
        }

        let matrix = build_mel_filterbank_matrix(
            params.sample_rate_hz(),
            n_fft,
            mel.n_mels(),
            mel.f_min(),
            mel.f_max(),
            mel.norm(),
        )?;

        // matrix must be (n_mels, out_len)
        if matrix.nrows() != mel.n_mels().get() || matrix.ncols() != out_len.get() {
            return Err(SpectrogramError::invalid_input(
                "mel filterbank matrix shape mismatch",
            ));
        }

        Ok(Self {
            kind: MappingKind::Mel { matrix },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<LogHz> {
    fn new_loghz(params: &SpectrogramParams, loghz: &LogHzParams) -> SpectrogramResult<Self> {
        let n_fft = params.stft().n_fft();
        let out_len = r2c_output_size(n_fft.get());
        let out_len = NonZeroUsize::new(out_len)
            .ok_or_else(|| SpectrogramError::invalid_input("FFT output length must be non-zero"))?;
        // Validate: n_bins must be <= something sensible
        if loghz.n_bins() > nzu!(10_000) {
            return Err(SpectrogramError::invalid_input(
                "n_bins is unreasonably large",
            ));
        }

        let (matrix, frequencies) = build_loghz_matrix(
            params.sample_rate_hz(),
            n_fft,
            loghz.n_bins(),
            loghz.f_min(),
            loghz.f_max(),
        )?;

        // matrix must be (n_bins, out_len)
        if matrix.nrows() != loghz.n_bins().get() || matrix.ncols() != out_len.get() {
            return Err(SpectrogramError::invalid_input(
                "loghz matrix shape mismatch",
            ));
        }

        Ok(Self {
            kind: MappingKind::LogHz {
                matrix,
                frequencies,
            },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<Erb> {
    fn new_erb(params: &SpectrogramParams, erb: &crate::erb::ErbParams) -> SpectrogramResult<Self> {
        let n_fft = params.stft().n_fft();
        let sample_rate = params.sample_rate_hz();

        // Validate: n_filters must be <= something sensible
        if erb.n_filters() > nzu!(10_000) {
            return Err(SpectrogramError::invalid_input(
                "n_filters is unreasonably large",
            ));
        }

        // Generate ERB filterbank with pre-computed frequency responses
        let filterbank = crate::erb::ErbFilterbank::generate(erb, sample_rate, n_fft)?;

        Ok(Self {
            kind: MappingKind::Erb { filterbank },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<Cqt> {
    fn new(params: &SpectrogramParams, cqt: &CqtParams) -> SpectrogramResult<Self> {
        let sample_rate = params.sample_rate_hz();
        let n_fft = params.stft().n_fft();

        // Validate that frequency range is reasonable
        let f_max = cqt.bin_frequency(cqt.num_bins().get().saturating_sub(1));
        if f_max >= sample_rate / 2.0 {
            return Err(SpectrogramError::invalid_input(
                "CQT maximum frequency must be below Nyquist frequency",
            ));
        }

        // Generate CQT kernel using n_fft as the signal length for kernel generation
        let kernel = CqtKernel::generate(cqt, sample_rate, n_fft);

        Ok(Self {
            kind: MappingKind::Cqt { kernel },
            _marker: PhantomData,
        })
    }
}

impl<FreqScale> FrequencyMapping<FreqScale> {
    const fn output_bins(&self) -> NonZeroUsize {
        // safety: all variants ensure output bins > 0 OR rely on a matrix that is guaranteed to have rows > 0
        match &self.kind {
            MappingKind::Identity { out_len } => *out_len,
            // safety: matrix.nrows() > 0
            MappingKind::LogHz { matrix, .. } | MappingKind::Mel { matrix } => unsafe {
                NonZeroUsize::new_unchecked(matrix.nrows())
            },
            MappingKind::Erb { filterbank, .. } => filterbank.num_filters(),
            MappingKind::Cqt { kernel, .. } => kernel.num_bins(),
        }
    }

    fn apply(
        &self,
        spectrum: &NonEmptySlice<f64>,
        frame: &NonEmptySlice<f64>,
        out: &mut NonEmptySlice<f64>,
    ) -> SpectrogramResult<()> {
        match &self.kind {
            MappingKind::Identity { out_len } => {
                if spectrum.len() != *out_len {
                    return Err(SpectrogramError::dimension_mismatch(
                        (*out_len).get(),
                        spectrum.len().get(),
                    ));
                }
                if out.len() != *out_len {
                    return Err(SpectrogramError::dimension_mismatch(
                        (*out_len).get(),
                        out.len().get(),
                    ));
                }
                out.copy_from_slice(spectrum);
                Ok(())
            }
            MappingKind::LogHz { matrix, .. } | MappingKind::Mel { matrix } => {
                let out_bins = matrix.nrows();
                let in_bins = matrix.ncols();

                if spectrum.len().get() != in_bins {
                    return Err(SpectrogramError::dimension_mismatch(
                        in_bins,
                        spectrum.len().get(),
                    ));
                }
                if out.len().get() != out_bins {
                    return Err(SpectrogramError::dimension_mismatch(
                        out_bins,
                        out.len().get(),
                    ));
                }

                // Sparse matrix-vector multiplication: out = matrix * spectrum
                matrix.multiply_vec(spectrum, out);
                Ok(())
            }
            MappingKind::Erb { filterbank } => {
                // Apply ERB filterbank using pre-computed frequency responses
                // The filterbank already has |H(f)|^2 pre-computed, so we just
                // apply it to the power spectrum
                let erb_out = filterbank.apply_to_power_spectrum(spectrum)?;

                if out.len().get() != erb_out.len().get() {
                    return Err(SpectrogramError::dimension_mismatch(
                        erb_out.len().get(),
                        out.len().get(),
                    ));
                }

                out.copy_from_slice(&erb_out);
                Ok(())
            }
            MappingKind::Cqt { kernel } => {
                // CQT works on time-domain windowed frame, not FFT spectrum
                // Apply CQT kernel to get complex coefficients
                let cqt_complex = kernel.apply(frame)?;

                if out.len().get() != cqt_complex.len().get() {
                    return Err(SpectrogramError::dimension_mismatch(
                        cqt_complex.len().get(),
                        out.len().get(),
                    ));
                }

                // Convert complex coefficients to power (|z|^2)
                // This matches the convention where intermediate values are in power domain
                for (i, c) in cqt_complex.iter().enumerate() {
                    out[i] = c.norm_sqr();
                }

                Ok(())
            }
        }
    }

    fn frequencies_hz(&self, params: &SpectrogramParams) -> NonEmptyVec<f64> {
        match &self.kind {
            MappingKind::Identity { out_len } => {
                // Standard R2C bins: k * sr / n_fft
                let n_fft = params.stft().n_fft().get() as f64;
                let sr = params.sample_rate_hz();
                let df = sr / n_fft;

                let mut f = Vec::with_capacity((*out_len).get());
                for k in 0..(*out_len).get() {
                    f.push(k as f64 * df);
                }
                // safety: out_len > 0
                unsafe { NonEmptyVec::new_unchecked(f) }
            }
            MappingKind::Mel { matrix } => {
                // For mel, the axis is defined by the mel band centre frequencies.
                // We compute and store them consistently with how we built the filterbank.
                let n_mels = matrix.nrows();
                // safety: n_mels > 0
                let n_mels = unsafe { NonZeroUsize::new_unchecked(n_mels) };
                mel_band_centres_hz(n_mels, params.sample_rate_hz(), params.nyquist_hz())
            }
            MappingKind::LogHz { frequencies, .. } => {
                // Frequencies are stored when the mapping is created
                frequencies.clone()
            }
            MappingKind::Erb { filterbank, .. } => {
                // ERB center frequencies
                filterbank.center_frequencies().to_non_empty_vec()
            }
            MappingKind::Cqt { kernel, .. } => {
                // CQT center frequencies from the kernel
                kernel.frequencies().to_non_empty_vec()
            }
        }
    }
}

//
// ========================
// Amplitude scaling
// ========================
//

/// Marker trait so we can specialise behaviour by `AmpScale`.
pub trait AmpScaleSpec: Sized + Send + Sync {
    /// Apply conversion from power-domain value to the desired amplitude scale.
    ///
    /// # Arguments
    ///
    /// - `power`: input power-domain value.
    ///
    /// # Returns
    ///
    /// Converted amplitude value.
    fn apply_from_power(power: f64) -> f64;

    /// Apply dB conversion in-place on a power-domain vector.
    ///
    /// This is a no-op for Power and Magnitude scales.
    ///
    /// # Arguments
    ///
    /// - `x`: power-domain values to convert to dB in-place.
    /// - `floor_db`: dB floor value to apply.
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<()>`: Ok on success, error on invalid input.
    ///
    /// # Errors
    ///
    /// - If `floor_db` is not finite.
    fn apply_db_in_place(x: &mut [f64], floor_db: f64) -> SpectrogramResult<()>;
}

impl AmpScaleSpec for Power {
    #[inline]
    fn apply_from_power(power: f64) -> f64 {
        power
    }

    #[inline]
    fn apply_db_in_place(_x: &mut [f64], _floor_db: f64) -> SpectrogramResult<()> {
        Ok(())
    }
}

impl AmpScaleSpec for Magnitude {
    #[inline]
    fn apply_from_power(power: f64) -> f64 {
        power.sqrt()
    }

    #[inline]
    fn apply_db_in_place(_x: &mut [f64], _floor_db: f64) -> SpectrogramResult<()> {
        Ok(())
    }
}

impl AmpScaleSpec for Decibels {
    #[inline]
    fn apply_from_power(power: f64) -> f64 {
        // dB conversion is applied in batch, not here.
        power
    }

    #[inline]
    fn apply_db_in_place(x: &mut [f64], floor_db: f64) -> SpectrogramResult<()> {
        // Convert power -> dB: 10*log10(max(power, eps))
        // where eps is derived from floor_db to ensure consistency
        if !floor_db.is_finite() {
            return Err(SpectrogramError::invalid_input("floor_db must be finite"));
        }

        // Convert floor_db to linear scale to get epsilon
        // e.g., floor_db = -80 dB -> eps = 10^(-80/10) = 1e-8
        let eps = 10.0_f64.powf(floor_db / 10.0);

        for v in x.iter_mut() {
            // Clamp power to epsilon before log to avoid log(0) and ensure floor
            *v = 10.0 * v.max(eps).log10();
        }
        Ok(())
    }
}

/// Amplitude scaling configuration.
///
/// This handles conversion from power-domain intermediate to the desired amplitude scale (Power, Magnitude, Decibels).
#[derive(Debug, Clone)]
struct AmplitudeScaling<AmpScale> {
    db_floor: Option<f64>,
    _marker: PhantomData<AmpScale>,
}

impl<AmpScale> AmplitudeScaling<AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    fn new(db: Option<&LogParams>) -> Self {
        let db_floor = db.map(LogParams::floor_db);
        Self {
            db_floor,
            _marker: PhantomData,
        }
    }

    /// Apply amplitude scaling in-place on a mapped spectrum vector.
    ///
    /// The input vector is assumed to be in the *power* domain (|X|^2),
    /// because the STFT stage produces power as the canonical intermediate.
    ///
    /// - Power: leaves values unchanged.
    /// - Magnitude: sqrt(power).
    /// - Decibels: converts power -> dB and floors at `db_floor`.
    pub fn apply_in_place(&self, x: &mut [f64]) -> SpectrogramResult<()> {
        // Convert from canonical power-domain intermediate into the requested linear domain.
        for v in x.iter_mut() {
            *v = AmpScale::apply_from_power(*v);
        }

        // Apply dB conversion if configured (no-op for Power/Magnitude via trait impls).
        if let Some(floor_db) = self.db_floor {
            AmpScale::apply_db_in_place(x, floor_db)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Workspace {
    spectrum: NonEmptyVec<f64>,         // out_len (power spectrum)
    mapped: NonEmptyVec<f64>,           // n_bins (after mapping)
    frame: NonEmptyVec<f64>,            // n_fft (windowed frame for FFT)
    fft_out: NonEmptyVec<Complex<f64>>, // out_len (FFT output)
}

impl Workspace {
    fn new(n_fft: NonZeroUsize, out_len: NonZeroUsize, n_bins: NonZeroUsize) -> Self {
        Self {
            spectrum: non_empty_vec![0.0; out_len],
            mapped: non_empty_vec![0.0; n_bins],
            frame: non_empty_vec![0.0; n_fft],
            fft_out: non_empty_vec![Complex::new(0.0, 0.0); out_len],
        }
    }

    fn ensure_sizes(&mut self, n_fft: NonZeroUsize, out_len: NonZeroUsize, n_bins: NonZeroUsize) {
        if self.spectrum.len() != out_len {
            self.spectrum.resize(out_len, 0.0);
        }
        if self.mapped.len() != n_bins {
            self.mapped.resize(n_bins, 0.0);
        }
        if self.frame.len() != n_fft {
            self.frame.resize(n_fft, 0.0);
        }
        if self.fft_out.len() != out_len {
            self.fft_out.resize(out_len, Complex::new(0.0, 0.0));
        }
    }
}

fn build_frequency_axis<FreqScale>(
    params: &SpectrogramParams,
    mapping: &FrequencyMapping<FreqScale>,
) -> FrequencyAxis<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    let frequencies = mapping.frequencies_hz(params);
    FrequencyAxis::new(frequencies)
}

fn build_time_axis_seconds(params: &SpectrogramParams, n_frames: NonZeroUsize) -> NonEmptyVec<f64> {
    let dt = params.frame_period_seconds();
    let mut times = Vec::with_capacity(n_frames.get());

    for i in 0..n_frames.get() {
        times.push(i as f64 * dt);
    }

    // safety: times is guaranteed non-empty since n_frames > 0

    unsafe { NonEmptyVec::new_unchecked(times) }
}

/// Generate window function samples.
///
/// Supports various window types including Rectangular, Hanning, Hamming, Blackman, Kaiser, and Gaussian.
///
/// # Arguments
///
/// * `window` - The type of window function to generate.
/// * `n_fft` - The size of the FFT, which determines the length of the window.
///
/// # Returns
///
/// A `NonEmptyVec<f64>` containing the window function samples.
#[inline]
#[must_use]
pub fn make_window(window: WindowType, n_fft: NonZeroUsize) -> NonEmptyVec<f64> {
    let n_fft = n_fft.get();
    let mut w = vec![0.0; n_fft];

    match window {
        WindowType::Rectangular => {
            w.fill(1.0);
        }
        WindowType::Hanning => {
            // Hann: 0.5 - 0.5*cos(2πn/(N-1))
            let n1 = (n_fft - 1) as f64;
            for (n, v) in w.iter_mut().enumerate() {
                *v = 0.5f64.mul_add(-(2.0 * std::f64::consts::PI * (n as f64) / n1).cos(), 0.5);
            }
        }
        WindowType::Hamming => {
            // Hamming: 0.54 - 0.46*cos(2πn/(N-1))
            let n1 = (n_fft - 1) as f64;
            for (n, v) in w.iter_mut().enumerate() {
                *v = 0.46f64.mul_add(-(2.0 * std::f64::consts::PI * (n as f64) / n1).cos(), 0.54);
            }
        }
        WindowType::Blackman => {
            // Blackman: 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
            let n1 = (n_fft - 1) as f64;
            for (n, v) in w.iter_mut().enumerate() {
                let a = 2.0 * std::f64::consts::PI * (n as f64) / n1;
                *v = 0.08f64.mul_add((2.0 * a).cos(), 0.5f64.mul_add(-a.cos(), 0.42));
            }
        }
        WindowType::Kaiser { beta } => {
            (0..n_fft).for_each(|i| {
                let n = i as f64;
                let n_max: f64 = (n_fft - 1) as f64;
                let alpha: f64 = (n - n_max / 2.0) / (n_max / 2.0);
                let bessel_arg = beta * alpha.mul_add(-alpha, 1.0).sqrt();
                // Simplified approximation of modified Bessel function
                let x = 1.0
                    + bessel_arg / 2.0
                        // Normalize by I0(beta) approximation
                        / (1.0 + beta / 2.0);
                w[i] = x;
            });
        }
        WindowType::Gaussian { std } => (0..n_fft).for_each(|i| {
            let n = i as f64;
            let center: f64 = (n_fft - 1) as f64 / 2.0;
            let exponent: f64 = -0.5 * ((n - center) / std).powi(2);
            w[i] = exponent.exp();
        }),
    }

    // safety: window is guaranteed non-empty since n_fft > 0
    unsafe { NonEmptyVec::new_unchecked(w) }
}

/// Convert Hz to mel scale using Slaney formula (librosa default, htk=False).
///
/// Uses a hybrid scale:
/// - Linear below 1000 Hz: mel = hz / (200/3)
/// - Logarithmic above 1000 Hz: mel = 15 + log(hz/1000) / log_step
///
/// This matches librosa's default behavior.
fn hz_to_mel(hz: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0; // ~66.667
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP; // = 15.0
    const LOGSTEP: f64 = 0.06853891945200942; // ln(6.4) / 27

    if hz >= MIN_LOG_HZ {
        // Logarithmic region
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    } else {
        // Linear region
        (hz - F_MIN) / F_SP
    }
}

/// Convert mel to Hz using Slaney formula (librosa default, htk=False).
///
/// Inverse of hz_to_mel.
fn mel_to_hz(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0; // ~66.667
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP; // = 15.0
    const LOGSTEP: f64 = 0.06853891945200942; // ln(6.4) / 27

    if mel >= MIN_LOG_MEL {
        // Logarithmic region
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        // Linear region
        F_MIN + F_SP * mel
    }
}

fn build_mel_filterbank_matrix(
    sample_rate_hz: f64,
    n_fft: NonZeroUsize,
    n_mels: NonZeroUsize,
    f_min: f64,
    f_max: f64,
    norm: MelNorm,
) -> SpectrogramResult<SparseMatrix> {
    if sample_rate_hz <= 0.0 || !sample_rate_hz.is_finite() {
        return Err(SpectrogramError::invalid_input(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    if f_min < 0.0 || f_min.is_infinite() {
        return Err(SpectrogramError::invalid_input("f_min must be >= 0"));
    }
    if f_max <= f_min {
        return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
    }
    if f_max > sample_rate_hz * 0.5 {
        return Err(SpectrogramError::invalid_input("f_max must be <= Nyquist"));
    }
    let n_mels = n_mels.get();
    let n_fft = n_fft.get();
    let out_len = r2c_output_size(n_fft);

    // FFT bin frequencies
    let df = sample_rate_hz / n_fft as f64;

    // Mel points: n_mels + 2 (for triangular edges)
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let n_points = n_mels + 2;
    let step = (mel_max - mel_min) / (n_points - 1) as f64;

    let mut mel_points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        mel_points.push((i as f64).mul_add(step, mel_min));
    }

    let mut hz_points = Vec::with_capacity(n_points);
    for m in &mel_points {
        hz_points.push(mel_to_hz(*m));
    }

    // Convert Hz points into FFT bin indices
    let mut bin_points = Vec::with_capacity(n_points);
    for hz in hz_points {
        let b = (hz / df).floor() as isize;
        let b = b.clamp(0, (out_len - 1) as isize);
        bin_points.push(b as usize);
    }

    // Build filterbank as sparse matrix
    let mut fb = SparseMatrix::new(n_mels, out_len);

    for m in 0..n_mels {
        let left = bin_points[m];
        let centre = bin_points[m + 1];
        let right = bin_points[m + 2];

        if left == centre || centre == right {
            // Degenerate triangle, skip (should be rare if params are sensible)
            continue;
        }

        // Rising slope: left -> centre
        for k in left..centre {
            let v = (k - left) as f64 / (centre - left) as f64;
            fb.set(m, k, v);
        }

        // Falling slope: centre -> right
        for k in centre..right {
            let v = (right - k) as f64 / (right - centre) as f64;
            fb.set(m, k, v);
        }
    }

    // Apply normalization
    match norm {
        MelNorm::None => {
            // No normalization needed
        }
        MelNorm::Slaney => {
            // Slaney-style area normalization: 2 / (hz_max - hz_min) for each triangle
            // NOTE: Uses Hz bandwidth, not mel bandwidth (to match librosa's implementation)
            for m in 0..n_mels {
                let mel_left = mel_points[m];
                let mel_right = mel_points[m + 2];
                let hz_left = mel_to_hz(mel_left);
                let hz_right = mel_to_hz(mel_right);
                let enorm = 2.0 / (hz_right - hz_left);

                // Normalize all values in this row
                for val in &mut fb.values[m] {
                    *val *= enorm;
                }
            }
        }
        MelNorm::L1 => {
            // L1 normalization: sum of weights = 1.0
            for m in 0..n_mels {
                let sum: f64 = fb.values[m].iter().sum();
                if sum > 0.0 {
                    let normalizer = 1.0 / sum;
                    for val in &mut fb.values[m] {
                        *val *= normalizer;
                    }
                }
            }
        }
        MelNorm::L2 => {
            // L2 normalization: L2 norm = 1.0
            for m in 0..n_mels {
                let norm_val: f64 = fb.values[m].iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm_val > 0.0 {
                    let normalizer = 1.0 / norm_val;
                    for val in &mut fb.values[m] {
                        *val *= normalizer;
                    }
                }
            }
        }
    }

    Ok(fb)
}

/// Build a logarithmic frequency interpolation matrix.
///
/// Maps linearly-spaced FFT bins to logarithmically-spaced frequency bins
/// using linear interpolation.
fn build_loghz_matrix(
    sample_rate_hz: f64,
    n_fft: NonZeroUsize,
    n_bins: NonZeroUsize,
    f_min: f64,
    f_max: f64,
) -> SpectrogramResult<(SparseMatrix, NonEmptyVec<f64>)> {
    if sample_rate_hz <= 0.0 || !sample_rate_hz.is_finite() {
        return Err(SpectrogramError::invalid_input(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    if f_min <= 0.0 || f_min.is_infinite() {
        return Err(SpectrogramError::invalid_input(
            "f_min must be finite and > 0",
        ));
    }
    if f_max <= f_min {
        return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
    }
    if f_max > sample_rate_hz * 0.5 {
        return Err(SpectrogramError::invalid_input("f_max must be <= Nyquist"));
    }

    let n_bins = n_bins.get();
    let n_fft = n_fft.get();

    let out_len = r2c_output_size(n_fft);
    let df = sample_rate_hz / n_fft as f64;

    // Generate logarithmically-spaced frequencies
    let log_f_min = f_min.ln();
    let log_f_max = f_max.ln();
    let log_step = (log_f_max - log_f_min) / (n_bins - 1) as f64;

    let mut log_frequencies = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let log_f = (i as f64).mul_add(log_step, log_f_min);
        log_frequencies.push(log_f.exp());
    }
    // safety: n_bins > 0
    let log_frequencies = unsafe { NonEmptyVec::new_unchecked(log_frequencies) };

    // Build interpolation matrix as sparse matrix
    let mut matrix = SparseMatrix::new(n_bins, out_len);

    for (bin_idx, &target_freq) in log_frequencies.iter().enumerate() {
        // Find the two FFT bins that bracket this frequency
        let exact_bin = target_freq / df;
        let lower_bin = exact_bin.floor() as usize;
        let upper_bin = (exact_bin.ceil() as usize).min(out_len - 1);

        if lower_bin >= out_len {
            continue;
        }

        if lower_bin == upper_bin {
            // Exact match
            matrix.set(bin_idx, lower_bin, 1.0);
        } else {
            // Linear interpolation
            let frac = exact_bin - lower_bin as f64;
            matrix.set(bin_idx, lower_bin, 1.0 - frac);
            if upper_bin < out_len {
                matrix.set(bin_idx, upper_bin, frac);
            }
        }
    }

    Ok((matrix, log_frequencies))
}

fn mel_band_centres_hz(
    n_mels: NonZeroUsize,
    sample_rate_hz: f64,
    nyquist_hz: f64,
) -> NonEmptyVec<f64> {
    let f_min = 0.0;
    let f_max = nyquist_hz.min(sample_rate_hz * 0.5);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let n_mels = n_mels.get();
    let step = (mel_max - mel_min) / (n_mels + 1) as f64;

    let mut centres = Vec::with_capacity(n_mels);
    for i in 0..n_mels {
        let mel = (i as f64 + 1.0).mul_add(step, mel_min);
        centres.push(mel_to_hz(mel));
    }
    // safety: centres is guaranteed non-empty since n_mels > 0
    unsafe { NonEmptyVec::new_unchecked(centres) }
}

/// Spectrogram structure holding the computed spectrogram data and metadata.
///
/// # Type Parameters
///
/// * `FreqScale`: The frequency scale type (e.g., `LinearHz`, `Mel`, `LogHz`, etc.).
/// * `AmpScale`: The amplitude scale type (e.g., `Power`, `Magnitude`, `Decibels`).
///
/// # Fields
///
/// * `data`: A 2D array containing the spectrogram data.
/// * `axes`: The axes of the spectrogram (frequency and time).
/// * `params`: The parameters used to compute the spectrogram.
/// * `_amp`: A phantom data marker for the amplitude scale type.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Spectrogram<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    data: Array2<f64>,
    axes: Axes<FreqScale>,
    params: SpectrogramParams,
    #[cfg_attr(feature = "serde", serde(skip))]
    _amp: PhantomData<AmpScale>,
}

impl<FreqScale, AmpScale> Spectrogram<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    /// Get the X-axis label.
    ///
    /// # Returns
    ///
    /// A static string slice representing the X-axis label.
    #[inline]
    #[must_use]
    pub const fn x_axis_label() -> &'static str {
        "Time (s)"
    }

    /// Get the Y-axis label based on the frequency scale type.
    ///
    /// # Returns
    ///
    /// A static string slice representing the Y-axis label.
    #[inline]
    #[must_use]
    pub fn y_axis_label() -> &'static str {
        match std::any::TypeId::of::<FreqScale>() {
            id if id == std::any::TypeId::of::<LinearHz>() => "Frequency (Hz)",
            id if id == std::any::TypeId::of::<Mel>() => "Frequency (Mel)",
            id if id == std::any::TypeId::of::<LogHz>() => "Frequency (Log Hz)",
            id if id == std::any::TypeId::of::<Erb>() => "Frequency (ERB)",
            id if id == std::any::TypeId::of::<Cqt>() => "Frequency (CQT Bins)",
            _ => "Frequency",
        }
    }

    /// Internal constructor. Only callable inside the crate.
    ///
    /// All inputs must already be validated and consistent.
    pub(crate) fn new(data: Array2<f64>, axes: Axes<FreqScale>, params: SpectrogramParams) -> Self {
        debug_assert_eq!(data.nrows(), axes.frequencies().len().get());
        debug_assert_eq!(data.ncols(), axes.times().len().get());

        Self {
            data,
            axes,
            params,
            _amp: PhantomData,
        }
    }

    /// Set spectrogram data matrix
    ///
    /// # Arguments
    ///
    /// * `data` - The new spectrogram data matrix.
    #[inline]
    pub fn set_data(&mut self, data: Array2<f64>) {
        self.data = data;
    }

    /// Spectrogram data matrix
    ///
    /// # Returns
    ///
    /// A reference to the spectrogram data matrix.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &Array2<f64> {
        &self.data
    }

    /// Axes of the spectrogram
    ///
    /// # Returns
    ///
    /// A reference to the axes of the spectrogram.
    #[inline]
    #[must_use]
    pub const fn axes(&self) -> &Axes<FreqScale> {
        &self.axes
    }

    /// Frequency axis in Hz
    ///
    /// # Returns
    ///
    /// A reference to the frequency axis in Hz.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        self.axes.frequencies()
    }

    /// Frequency range in Hz (min, max)
    ///
    /// # Returns
    ///
    /// A tuple containing the minimum and maximum frequencies in Hz.
    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        self.axes.frequency_range()
    }

    /// Time axis in seconds
    ///
    /// # Returns
    ///
    /// A reference to the time axis in seconds.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        self.axes.times()
    }

    /// Spectrogram computation parameters
    ///
    /// # Returns
    ///
    /// A reference to the spectrogram computation parameters.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &SpectrogramParams {
        &self.params
    }

    /// Duration of the spectrogram in seconds
    ///
    /// # Returns
    ///
    /// The duration of the spectrogram in seconds.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.axes.duration()
    }

    /// If this is a dB spectrogram, return the (min, max) dB values. otherwise do the maths to compute dB range.
    ///
    /// # Returns
    ///
    /// The (min, max) dB values of the spectrogram, or `None` if the amplitude scale is unknown.
    #[inline]
    #[must_use]
    pub fn db_range(&self) -> Option<(f64, f64)> {
        let type_self = std::any::TypeId::of::<AmpScale>();

        if type_self == std::any::TypeId::of::<Decibels>() {
            let (min, max) = min_max_single_pass(self.data.as_slice()?);
            Some((min, max))
        } else if type_self == std::any::TypeId::of::<Power>() {
            // Not a dB spectrogram; compute dB range from power values
            let mut min_db = f64::INFINITY;
            let mut max_db = f64::NEG_INFINITY;
            for &v in &self.data {
                let db = 10.0 * (v + EPS).log10();
                if db < min_db {
                    min_db = db;
                }
                if db > max_db {
                    max_db = db;
                }
            }
            Some((min_db, max_db))
        } else if type_self == std::any::TypeId::of::<Magnitude>() {
            // Not a dB spectrogram; compute dB range from magnitude values
            let mut min_db = f64::INFINITY;
            let mut max_db = f64::NEG_INFINITY;

            for &v in &self.data {
                let power = v * v;
                let db = 10.0 * (power + EPS).log10();
                if db < min_db {
                    min_db = db;
                }
                if db > max_db {
                    max_db = db;
                }
            }

            Some((min_db, max_db))
        } else {
            // Unknown AmpScale type; return dummy values
            None
        }
    }

    /// Number of frequency bins
    ///
    /// # Returns
    ///
    /// The number of frequency bins in the spectrogram.
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        // safety: data.nrows() > 0 is guaranteed by construction
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Number of time frames in the spectrogram
    ///
    /// # Returns
    ///
    /// The number of time frames (columns) in the spectrogram.
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        // safety: data.ncols() > 0 is guaranteed by construction
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }
}

impl AsRef<Array2<f64>> for Spectrogram<LinearHz, Power> {
    #[inline]
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for Spectrogram<LinearHz, Power> {
    type Target = Array2<f64>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<AmpScale> Spectrogram<LinearHz, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    /// Compute a linear-frequency spectrogram from audio samples.
    ///
    /// This is a convenience method that creates a planner internally and computes
    /// the spectrogram in one call. For processing multiple signals with the same
    /// parameters, use [`SpectrogramPlanner::linear_plan`] to create a reusable plan.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - Spectrogram computation parameters
    /// * `db` - Optional logarithmic scaling parameters (only used when `AmpScale = Decibels`)
    ///
    /// # Returns
    ///
    /// A linear-frequency spectrogram with the specified amplitude scale.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The samples slice is empty
    /// - Parameters are invalid
    /// - FFT computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn example() -> SpectrogramResult<()> {
    /// // Create a simple test signal
    /// let sample_rate = 16000.0;
    /// let samples_vec: Vec<f64> = (0..16000).map(|i| {
    ///     (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin()
    /// }).collect();
    /// let samples = non_empty_slice::NonEmptyVec::new(samples_vec).unwrap();
    ///
    /// // Set up parameters
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, sample_rate)?;
    ///
    /// // Compute power spectrogram
    /// let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;
    ///
    /// println!("Computed spectrogram: {} bins x {} frames", spec.n_bins(), spec.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute(
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.linear_plan(params, db)?;
        plan.compute(samples)
    }
}

impl<AmpScale> Spectrogram<Mel, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    /// Compute a mel-frequency spectrogram from audio samples.
    ///
    /// This is a convenience method that creates a planner internally and computes
    /// the spectrogram in one call. For processing multiple signals with the same
    /// parameters, use [`SpectrogramPlanner::mel_plan`] to create a reusable plan.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - Spectrogram computation parameters
    /// * `mel` - Mel filterbank parameters
    /// * `db` - Optional logarithmic scaling parameters (only used when `AmpScale = Decibels`)
    ///
    /// # Returns
    ///
    /// A mel-frequency spectrogram with the specified amplitude scale.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The samples slice is empty
    /// - Parameters are invalid
    /// - Mel `f_max` exceeds Nyquist frequency
    /// - FFT computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn example() -> SpectrogramResult<()> {
    /// // Create a simple test signal
    /// let sample_rate = 16000.0;
    /// let samples_vec: Vec<f64> = (0..16000).map(|i| {
    ///     (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin()
    /// }).collect();
    /// let samples = non_empty_slice::NonEmptyVec::new(samples_vec).unwrap();
    ///
    /// // Set up parameters
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, sample_rate)?;
    /// let mel = MelParams::new(nzu!(80), 0.0, 8000.0)?;
    ///
    /// // Compute mel spectrogram in dB scale
    /// let db = LogParams::new(-80.0)?;
    /// let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db))?;
    ///
    /// println!("Computed mel spectrogram: {} mels x {} frames", spec.n_bins(), spec.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute(
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.mel_plan(params, mel, db)?;
        plan.compute(samples)
    }
}

impl<AmpScale> Spectrogram<Erb, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    /// Compute an ERB-frequency spectrogram from audio samples.
    ///
    /// This is a convenience method that creates a planner internally and computes
    /// the spectrogram in one call. For processing multiple signals with the same
    /// parameters, use [`SpectrogramPlanner::erb_plan`] to create a reusable plan.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - Spectrogram computation parameters
    /// * `erb` - ERB frequency scale parameters
    /// * `db` - Optional logarithmic scaling parameters (only used when `AmpScale = Decibels`)
    ///
    /// # Returns
    ///
    /// An ERB-scale spectrogram with the specified amplitude scale.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The samples slice is empty
    /// - Parameters are invalid
    /// - FFT computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let erb = ErbParams::speech_standard();
    ///
    /// let spec = ErbPowerSpectrogram::compute(&samples, &params, &erb, None)?;
    /// assert_eq!(spec.n_bins(), nzu!(40));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute(
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
        erb: &ErbParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.erb_plan(params, erb, db)?;
        plan.compute(samples)
    }
}

impl<AmpScale> Spectrogram<LogHz, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    /// Compute a logarithmic-frequency spectrogram from audio samples.
    ///
    /// This is a convenience method that creates a planner internally and computes
    /// the spectrogram in one call. For processing multiple signals with the same
    /// parameters, use [`SpectrogramPlanner::log_hz_plan`] to create a reusable plan.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - Spectrogram computation parameters
    /// * `loghz` - Logarithmic frequency scale parameters
    /// * `db` - Optional logarithmic scaling parameters (only used when `AmpScale = Decibels`)
    ///
    /// # Returns
    ///
    /// A logarithmic-frequency spectrogram with the specified amplitude scale.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The samples slice is empty
    /// - Parameters are invalid
    /// - FFT computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = non_empty_vec![0.0; nzu!(16000)];
    /// let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let loghz = LogHzParams::new(nzu!(128), 20.0, 8000.0)?;
    ///
    /// let spec = LogHzPowerSpectrogram::compute(&samples, &params, &loghz, None)?;
    /// assert_eq!(spec.n_bins(), nzu!(128));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn compute(
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.log_hz_plan(params, loghz, db)?;
        plan.compute(samples)
    }
}

impl<AmpScale> Spectrogram<Cqt, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    /// Compute a constant-Q transform (CQT) spectrogram from audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (any type that can be converted to a slice)
    /// * `params` - Spectrogram computation parameters
    /// * `cqt` - CQT parameters
    /// * `db` - Optional logarithmic scaling parameters (only used when `AmpScale = Decibels`)
    ///
    /// # Returns
    ///
    /// A CQT spectrogram with the specified amplitude scale.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    #[inline]
    pub fn compute(
        samples: &NonEmptySlice<f64>,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.cqt_plan(params, cqt, db)?;
        plan.compute(samples)
    }
}

// ========================
// Display implementations
// ========================

/// Helper function to get amplitude scale name
fn amp_scale_name<AmpScale>() -> &'static str
where
    AmpScale: AmpScaleSpec + 'static,
{
    match std::any::TypeId::of::<AmpScale>() {
        id if id == std::any::TypeId::of::<Power>() => "Power",
        id if id == std::any::TypeId::of::<Magnitude>() => "Magnitude",
        id if id == std::any::TypeId::of::<Decibels>() => "Decibels",
        _ => "Unknown",
    }
}

/// Helper function to get frequency scale name
fn freq_scale_name<FreqScale>() -> &'static str
where
    FreqScale: Copy + Clone + 'static,
{
    match std::any::TypeId::of::<FreqScale>() {
        id if id == std::any::TypeId::of::<LinearHz>() => "Linear Hz",
        id if id == std::any::TypeId::of::<LogHz>() => "Log Hz",
        id if id == std::any::TypeId::of::<Mel>() => "Mel",
        id if id == std::any::TypeId::of::<Erb>() => "ERB",
        id if id == std::any::TypeId::of::<Cqt>() => "CQT",
        _ => "Unknown",
    }
}

impl<FreqScale, AmpScale> core::fmt::Display for Spectrogram<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (freq_min, freq_max) = self.frequency_range();
        let duration = self.duration();
        let (rows, cols) = self.data.dim();

        // Alternative formatting (#) provides more detailed output with data
        if f.alternate() {
            writeln!(f, "Spectrogram {{")?;
            writeln!(f, "  Frequency Scale: {}", freq_scale_name::<FreqScale>())?;
            writeln!(f, "  Amplitude Scale: {}", amp_scale_name::<AmpScale>())?;
            writeln!(f, "  Shape: {} frequency bins × {} time frames", rows, cols)?;
            writeln!(
                f,
                "  Frequency Range: {:.2} Hz - {:.2} Hz",
                freq_min, freq_max
            )?;
            writeln!(f, "  Duration: {:.3} s", duration)?;
            writeln!(f)?;

            // Display parameters
            writeln!(f, "  Parameters:")?;
            writeln!(f, "    Sample Rate: {} Hz", self.params.sample_rate_hz())?;
            writeln!(f, "    FFT Size: {}", self.params.stft().n_fft())?;
            writeln!(f, "    Hop Size: {}", self.params.stft().hop_size())?;
            writeln!(f, "    Window: {:?}", self.params.stft().window())?;
            writeln!(f, "    Centered: {}", self.params.stft().centre())?;
            writeln!(f)?;

            // Display data statistics
            let data_slice = self.data.as_slice().unwrap_or(&[]);
            if !data_slice.is_empty() {
                let (min_val, max_val) = min_max_single_pass(data_slice);
                let mean = data_slice.iter().sum::<f64>() / data_slice.len() as f64;
                writeln!(f, "  Data Statistics:")?;
                writeln!(f, "    Min: {:.6}", min_val)?;
                writeln!(f, "    Max: {:.6}", max_val)?;
                writeln!(f, "    Mean: {:.6}", mean)?;
                writeln!(f)?;
            }

            // Display actual data (truncated if too large)
            writeln!(f, "  Data Matrix:")?;
            let max_rows_to_display = 8;
            let max_cols_to_display = 8;

            for i in 0..rows.min(max_rows_to_display) {
                write!(f, "    [")?;
                for j in 0..cols.min(max_cols_to_display) {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:9.4}", self.data[[i, j]])?;
                }
                if cols > max_cols_to_display {
                    write!(f, ", ... ({} more)", cols - max_cols_to_display)?;
                }
                writeln!(f, "]")?;
            }

            if rows > max_rows_to_display {
                writeln!(f, "    ... ({} more rows)", rows - max_rows_to_display)?;
            }

            write!(f, "}}")?;
        } else {
            // Default formatting: compact summary
            write!(
                f,
                "Spectrogram<{}, {}>[{}×{}] ({:.2}-{:.2} Hz, {:.3}s)",
                freq_scale_name::<FreqScale>(),
                amp_scale_name::<AmpScale>(),
                rows,
                cols,
                freq_min,
                freq_max,
                duration
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyAxis<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    frequencies: NonEmptyVec<f64>,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<FreqScale>,
}

impl<FreqScale> FrequencyAxis<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    pub(crate) const fn new(frequencies: NonEmptyVec<f64>) -> Self {
        Self {
            frequencies,
            _marker: PhantomData,
        }
    }

    /// Get the frequency values in Hz.
    ///
    /// # Returns
    ///
    /// Returns a non-empty slice of frequencies.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the frequency range (min, max) in Hz.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing the minimum and maximum frequency.
    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        let data = self.frequencies.as_slice();
        let min = data[0];
        let max_idx = data.len().saturating_sub(1); // safe for non-empty
        let max = data[max_idx];
        (min, max)
    }

    /// Get the number of frequency bins.
    ///
    /// # Returns
    ///
    /// Returns the number of frequency bins as a NonZeroUsize.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> NonZeroUsize {
        self.frequencies.len()
    }
}

/// Spectrogram axes container.
///
/// Holds frequency and time axes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Axes<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    freq: FrequencyAxis<FreqScale>,
    times: NonEmptyVec<f64>,
}

impl<FreqScale> Axes<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    pub(crate) const fn new(freq: FrequencyAxis<FreqScale>, times: NonEmptyVec<f64>) -> Self {
        Self { freq, times }
    }

    /// Get the frequency values in Hz.
    ///
    /// # Returns
    ///
    /// Returns a non-empty slice of frequencies.
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        self.freq.frequencies()
    }

    /// Get the time values in seconds.
    ///
    /// # Returns
    ///
    /// Returns a non-empty slice of time values.
    #[inline]
    #[must_use]
    pub fn times(&self) -> &NonEmptySlice<f64> {
        &self.times
    }

    /// Get the frequency range (min, max) in Hz.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing the minimum and maximum frequency.
    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        self.freq.frequency_range()
    }

    /// Get the duration of the spectrogram in seconds.
    ///
    /// # Returns
    ///
    /// Returns the duration in seconds.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        *self.times.last()
    }
}

// Enum types for frequency and amplitude scales

/// Linear frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LinearHz {
    _Phantom,
}

/// Logarithmic frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LogHz {
    _Phantom,
}

/// Mel frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Mel {
    _Phantom,
}

/// ERB/gammatone frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Erb {
    _Phantom,
}
pub type Gammatone = Erb;

/// Constant-Q Transform frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Cqt {
    _Phantom,
}

// Amplitude scales

/// Power amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Power {
    _Phantom,
}

/// Decibel amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Decibels {
    _Phantom,
}

/// Magnitude amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Magnitude {
    _Phantom,
}

/// STFT parameters for spectrogram computation.
///
/// * `n_fft`: Size of the FFT window.
/// * `hop_size`: Number of samples between successive frames.
/// * window: Window function to apply to each frame.
/// * centre: Whether to pad the input signal so that frames are centered.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StftParams {
    n_fft: NonZeroUsize,
    hop_size: NonZeroUsize,
    window: WindowType,
    centre: bool,
}

impl StftParams {
    /// Create new STFT parameters.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - Size of the FFT window
    /// * `hop_size` - Number of samples between successive frames
    /// * `window` - Window function to apply to each frame
    /// * `centre` - Whether to pad the input signal so that frames are centered
    ///
    /// # Errors
    ///
    /// Returns an error if `hop_size` > `n_fft`.
    ///
    /// # Returns
    ///
    /// New `StftParams` instance.
    #[inline]
    pub fn new(
        n_fft: NonZeroUsize,
        hop_size: NonZeroUsize,
        window: WindowType,
        centre: bool,
    ) -> SpectrogramResult<Self> {
        if hop_size.get() > n_fft.get() {
            return Err(SpectrogramError::invalid_input("hop_size must be <= n_fft"));
        }

        Ok(Self {
            n_fft,
            hop_size,
            window,
            centre,
        })
    }

    const unsafe fn new_unchecked(
        n_fft: NonZeroUsize,
        hop_size: NonZeroUsize,
        window: WindowType,
        centre: bool,
    ) -> Self {
        Self {
            n_fft,
            hop_size,
            window,
            centre,
        }
    }

    /// Get the FFT window size.
    ///
    /// # Returns
    ///
    /// The FFT window size.
    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> NonZeroUsize {
        self.n_fft
    }

    /// Get the hop size (samples between successive frames).
    ///
    /// # Returns
    ///
    /// The hop size.
    #[inline]
    #[must_use]
    pub const fn hop_size(&self) -> NonZeroUsize {
        self.hop_size
    }

    /// Get the window function.
    ///
    /// # Returns
    ///
    /// The window function.
    #[inline]
    #[must_use]
    pub const fn window(&self) -> WindowType {
        self.window
    }

    /// Get whether frames are centered (input signal is padded).
    ///
    /// # Returns
    ///
    /// `true` if frames are centered, `false` otherwise.
    #[inline]
    #[must_use]
    pub const fn centre(&self) -> bool {
        self.centre
    }

    /// Create a builder for STFT parameters.
    ///
    /// # Returns
    ///
    /// A `StftParamsBuilder` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::{StftParams, WindowType, nzu};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::builder()
    ///     .n_fft(nzu!(2048))
    ///     .hop_size(nzu!(512))
    ///     .window(WindowType::Hanning)
    ///     .centre(true)
    ///     .build()?;
    ///
    /// assert_eq!(stft.n_fft(), nzu!(2048));
    /// assert_eq!(stft.hop_size(), nzu!(512));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn builder() -> StftParamsBuilder {
        StftParamsBuilder::default()
    }
}

/// Builder for [`StftParams`].
#[derive(Debug, Clone)]
pub struct StftParamsBuilder {
    n_fft: Option<NonZeroUsize>,
    hop_size: Option<NonZeroUsize>,
    window: WindowType,
    centre: bool,
}

impl Default for StftParamsBuilder {
    #[inline]
    fn default() -> Self {
        Self {
            n_fft: None,
            hop_size: None,
            window: WindowType::Hanning,
            centre: true,
        }
    }
}

impl StftParamsBuilder {
    /// Set the FFT window size.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - Size of the FFT window
    ///
    /// # Returns
    ///
    /// The builder with the updated FFT window size.
    #[inline]
    #[must_use]
    pub const fn n_fft(mut self, n_fft: NonZeroUsize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// Set the hop size (samples between successive frames).
    ///
    /// # Arguments
    ///
    /// * `hop_size` - Number of samples between successive frames
    ///
    /// # Returns
    ///
    /// The builder with the updated hop size.
    #[inline]
    #[must_use]
    pub const fn hop_size(mut self, hop_size: NonZeroUsize) -> Self {
        self.hop_size = Some(hop_size);
        self
    }

    /// Set the window function.
    ///
    /// # Arguments
    ///
    /// * `window` - Window function to apply to each frame
    ///
    /// # Returns
    ///
    /// The builder with the updated window function.
    #[inline]
    #[must_use]
    pub const fn window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set whether to center frames (pad input signal).
    #[inline]
    #[must_use]
    pub const fn centre(mut self, centre: bool) -> Self {
        self.centre = centre;
        self
    }

    /// Build the [`StftParams`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `n_fft` or `hop_size` are not set or are zero
    /// - `hop_size` > `n_fft`
    #[inline]
    pub fn build(self) -> SpectrogramResult<StftParams> {
        let n_fft = self
            .n_fft
            .ok_or_else(|| SpectrogramError::invalid_input("n_fft must be set"))?;
        let hop_size = self
            .hop_size
            .ok_or_else(|| SpectrogramError::invalid_input("hop_size must be set"))?;

        StftParams::new(n_fft, hop_size, self.window, self.centre)
    }
}

//
// ========================
// Mel parameters
// ========================
//

/// Mel filterbank normalization strategy.
///
/// Determines how the triangular mel filters are normalized after construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MelNorm {
    /// No normalization (triangular filters with peak = 1.0).
    ///
    /// This is the default and fastest option.
    None,

    /// Slaney-style area normalization (librosa default).
    ///
    /// Each mel filter is divided by its bandwidth in Hz: `2.0 / (f_max - f_min)`.
    /// This ensures constant energy per mel band regardless of bandwidth.
    ///
    /// Use this for compatibility with librosa's default behavior.
    Slaney,

    /// L1 normalization (sum of weights = 1.0).
    ///
    /// Each mel filter's weights are divided by their sum.
    /// Useful when you want each filter to act as a weighted average.
    L1,

    /// L2 normalization (Euclidean norm = 1.0).
    ///
    /// Each mel filter's weights are divided by their L2 norm.
    /// Provides unit-norm filters in the L2 sense.
    L2,
}

impl Default for MelNorm {
    fn default() -> Self {
        Self::None
    }
}

/// Mel filter bank parameters
///
/// * `n_mels`: Number of mel bands
/// * `f_min`: Minimum frequency (Hz)
/// * `f_max`: Maximum frequency (Hz)
/// * `norm`: Filterbank normalization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MelParams {
    n_mels: NonZeroUsize,
    f_min: f64,
    f_max: f64,
    norm: MelNorm,
}

impl MelParams {
    /// Create new mel filter bank parameters.
    ///
    /// # Arguments
    ///
    /// * `n_mels` - Number of mel bands
    /// * `f_min` - Minimum frequency (Hz)
    /// * `f_max` - Maximum frequency (Hz)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `f_min` is not >= 0
    /// - `f_max` is not > `f_min`
    ///
    /// # Returns
    ///
    /// A `MelParams` instance with no normalization (default).
    #[inline]
    pub fn new(n_mels: NonZeroUsize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        Self::with_norm(n_mels, f_min, f_max, MelNorm::None)
    }

    /// Create new mel filter bank parameters with specified normalization.
    ///
    /// # Arguments
    ///
    /// * `n_mels` - Number of mel bands
    /// * `f_min` - Minimum frequency (Hz)
    /// * `f_max` - Maximum frequency (Hz)
    /// * `norm` - Filterbank normalization strategy
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `f_min` is not >= 0
    /// - `f_max` is not > `f_min`
    ///
    /// # Returns
    ///
    /// A `MelParams` instance.
    #[inline]
    pub fn with_norm(
        n_mels: NonZeroUsize,
        f_min: f64,
        f_max: f64,
        norm: MelNorm,
    ) -> SpectrogramResult<Self> {
        if f_min < 0.0 {
            return Err(SpectrogramError::invalid_input("f_min must be >= 0"));
        }

        if f_max <= f_min {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_mels,
            f_min,
            f_max,
            norm,
        })
    }

    const unsafe fn new_unchecked(n_mels: NonZeroUsize, f_min: f64, f_max: f64) -> Self {
        Self {
            n_mels,
            f_min,
            f_max,
            norm: MelNorm::None,
        }
    }

    /// Get the number of mel bands.
    ///
    /// # Returns
    ///
    /// The number of mel bands.
    #[inline]
    #[must_use]
    pub const fn n_mels(&self) -> NonZeroUsize {
        self.n_mels
    }

    /// Get the minimum frequency (Hz).
    ///
    /// # Returns
    ///
    /// The minimum frequency in Hz.
    #[inline]
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

    /// Get the maximum frequency (Hz).
    ///
    /// # Returns
    ///
    /// The maximum frequency in Hz.
    #[inline]
    #[must_use]
    pub const fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Get the filterbank normalization strategy.
    ///
    /// # Returns
    ///
    /// The normalization strategy.
    #[inline]
    #[must_use]
    pub const fn norm(&self) -> MelNorm {
        self.norm
    }

    /// Create standard mel filterbank parameters.
    ///
    /// Uses 128 mel bands from 0 Hz to the Nyquist frequency.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (used to determine `f_max`)
    ///
    /// # Returns
    ///
    /// A `MelParams` instance with standard settings.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is not greater than 0.
    #[inline]
    #[must_use]
    pub const fn standard(sample_rate: f64) -> Self {
        assert!(sample_rate > 0.0);
        // safety: parameters are known to be valid
        unsafe { Self::new_unchecked(nzu!(128), 0.0, sample_rate / 2.0) }
    }

    /// Create mel filterbank parameters optimized for speech.
    ///
    /// Uses 40 mel bands from 0 Hz to 8000 Hz (typical speech bandwidth).
    ///
    /// # Returns
    ///
    /// A `MelParams` instance with speech-optimized settings.
    #[inline]
    #[must_use]
    pub const fn speech_standard() -> Self {
        // safety: parameters are known to be valid
        unsafe { Self::new_unchecked(nzu!(40), 0.0, 8000.0) }
    }
}

//
// ========================
// LogHz parameters
// ========================
//

/// Logarithmic frequency scale parameters
///
/// * `n_bins`: Number of logarithmically-spaced frequency bins
/// * `f_min`: Minimum frequency (Hz)
/// * `f_max`: Maximum frequency (Hz)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogHzParams {
    n_bins: NonZeroUsize,
    f_min: f64,
    f_max: f64,
}

impl LogHzParams {
    /// Create new logarithmic frequency scale parameters.
    ///
    /// # Arguments
    ///
    /// * `n_bins` - Number of logarithmically-spaced frequency bins
    /// * `f_min` - Minimum frequency (Hz)
    /// * `f_max` - Maximum frequency (Hz)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `f_min` is not finite and > 0
    /// - `f_max` is not > `f_min`
    ///
    /// # Returns
    ///
    /// A `LogHzParams` instance.
    #[inline]
    pub fn new(n_bins: NonZeroUsize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        if !(f_min > 0.0 && f_min.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and > 0",
            ));
        }

        if f_max <= f_min {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_bins,
            f_min,
            f_max,
        })
    }

    const unsafe fn new_unchecked(n_bins: NonZeroUsize, f_min: f64, f_max: f64) -> Self {
        Self {
            n_bins,
            f_min,
            f_max,
        }
    }

    /// Get the number of frequency bins.
    ///
    /// # Returns
    ///
    /// The number of frequency bins.
    #[inline]
    #[must_use]
    pub const fn n_bins(&self) -> NonZeroUsize {
        self.n_bins
    }

    /// Get the minimum frequency (Hz).
    ///
    /// # Returns
    ///
    /// The minimum frequency in Hz.
    #[inline]
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

    /// Get the maximum frequency (Hz).
    ///
    /// # Returns
    ///
    /// The maximum frequency in Hz.
    #[inline]
    #[must_use]
    pub const fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Create standard logarithmic frequency parameters.
    ///
    /// Uses 128 log bins from 20 Hz to the Nyquist frequency.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (used to determine `f_max`)
    #[inline]
    #[must_use]
    pub fn standard(sample_rate: f64) -> Self {
        // safety: parameters are known to be valid
        unsafe { Self::new_unchecked(nzu!(128), 20.0, sample_rate / 2.0) }
    }

    /// Create logarithmic frequency parameters optimized for music.
    ///
    /// Uses 84 bins (7 octaves * 12 bins/octave) from 27.5 Hz (A0) to 4186 Hz (C8).
    #[inline]
    #[must_use]
    pub const fn music_standard() -> Self {
        // safety: parameters are known to be valid
        unsafe { Self::new_unchecked(nzu!(84), 27.5, 4186.0) }
    }
}

//
// ========================
// Log scaling parameters
// ========================
//

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogParams {
    floor_db: f64,
}

impl LogParams {
    /// Create new logarithmic scaling parameters.
    ///
    /// # Arguments
    ///
    /// * `floor_db` - Minimum dB value (floor) for logarithmic scaling
    ///
    /// # Errors
    ///
    /// Returns an error if `floor_db` is not finite.
    ///
    /// # Returns
    ///
    /// A `LogParams` instance.
    #[inline]
    pub fn new(floor_db: f64) -> SpectrogramResult<Self> {
        if !floor_db.is_finite() {
            return Err(SpectrogramError::invalid_input("floor_db must be finite"));
        }

        Ok(Self { floor_db })
    }

    /// Get the floor dB value.
    #[inline]
    #[must_use]
    pub const fn floor_db(&self) -> f64 {
        self.floor_db
    }
}

/// Spectrogram computation parameters.
///
/// * `stft`: STFT parameters
/// * `sample_rate_hz`: Sample rate in Hz
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectrogramParams {
    stft: StftParams,
    sample_rate_hz: f64,
}

impl SpectrogramParams {
    /// Create new spectrogram parameters.
    ///
    /// # Arguments
    ///
    /// * `stft` - STFT parameters
    /// * `sample_rate_hz` - Sample rate in Hz
    ///
    /// # Errors
    ///
    /// Returns an error if the sample rate is not positive and finite.
    ///
    /// # Returns
    ///
    /// A `SpectrogramParams` instance.
    #[inline]
    pub fn new(stft: StftParams, sample_rate_hz: f64) -> SpectrogramResult<Self> {
        if !(sample_rate_hz > 0.0 && sample_rate_hz.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "sample_rate_hz must be finite and > 0",
            ));
        }

        Ok(Self {
            stft,
            sample_rate_hz,
        })
    }

    /// Create a builder for spectrogram parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if required parameters are not set or are invalid.
    ///
    /// # Returns
    ///
    /// A builder for [`SpectrogramParams`].
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::{SpectrogramParams, WindowType, nzu};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let params = SpectrogramParams::builder()
    ///     .sample_rate(16000.0)
    ///     .n_fft(nzu!(512))
    ///     .hop_size(nzu!(256))
    ///     .window(WindowType::Hanning)
    ///     .centre(true)
    ///     .build()?;
    ///
    /// assert_eq!(params.sample_rate_hz(), 16000.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn builder() -> SpectrogramParamsBuilder {
        SpectrogramParamsBuilder::default()
    }

    /// Create default parameters for speech processing.
    ///
    /// # Arguments
    ///
    /// * `sample_rate_hz` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A `SpectrogramParams` instance with default settings for music analysis.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample rate is not positive and finite.
    ///
    /// Uses:
    /// - `n_fft`: 512 (32ms at 16kHz)
    /// - `hop_size`: 160 (10ms at 16kHz)
    /// - window: Hanning
    /// - centre: true
    #[inline]
    pub fn speech_default(sample_rate_hz: f64) -> SpectrogramResult<Self> {
        // safety: parameters are known to be valid
        let stft =
            unsafe { StftParams::new_unchecked(nzu!(512), nzu!(160), WindowType::Hanning, true) };

        Self::new(stft, sample_rate_hz)
    }

    /// Create default parameters for music processing.
    ///
    /// # Arguments
    ///
    /// * `sample_rate_hz` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A `SpectrogramParams` instance with default settings for music analysis.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample rate is not positive and finite.
    ///
    /// Uses:
    /// - `n_fft`: 2048 (46ms at 44.1kHz)
    /// - `hop_size`: 512 (11.6ms at 44.1kHz)
    /// - window: Hanning
    /// - centre: true
    #[inline]
    pub fn music_default(sample_rate_hz: f64) -> SpectrogramResult<Self> {
        // safety: parameters are known to be valid
        let stft =
            unsafe { StftParams::new_unchecked(nzu!(2048), nzu!(512), WindowType::Hanning, true) };
        Self::new(stft, sample_rate_hz)
    }

    /// Get the STFT parameters.
    #[inline]
    #[must_use]
    pub const fn stft(&self) -> &StftParams {
        &self.stft
    }

    /// Get the sample rate in Hz.
    #[inline]
    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    /// Get the frame period in seconds.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_period_seconds(&self) -> f64 {
        self.stft.hop_size().get() as f64 / self.sample_rate_hz
    }

    /// Get the Nyquist frequency in Hz.
    #[inline]
    #[must_use]
    pub fn nyquist_hz(&self) -> f64 {
        self.sample_rate_hz * 0.5
    }
}

/// Builder for [`SpectrogramParams`].
#[derive(Debug, Clone)]
pub struct SpectrogramParamsBuilder {
    sample_rate: Option<f64>,
    n_fft: Option<NonZeroUsize>,
    hop_size: Option<NonZeroUsize>,
    window: WindowType,
    centre: bool,
}

impl Default for SpectrogramParamsBuilder {
    #[inline]
    fn default() -> Self {
        Self {
            sample_rate: None,
            n_fft: None,
            hop_size: None,
            window: WindowType::Hanning,
            centre: true,
        }
    }
}

impl SpectrogramParamsBuilder {
    /// Set the sample rate in Hz.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz.
    ///
    /// # Returns
    ///
    /// The updated builder instance.
    #[inline]
    #[must_use]
    pub const fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set the FFT window size.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size.
    ///
    /// # Returns
    ///
    /// The updated builder instance.
    #[inline]
    #[must_use]
    pub const fn n_fft(mut self, n_fft: NonZeroUsize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// Set the hop size (samples between successive frames).
    ///
    /// # Arguments
    ///
    /// * `hop_size` - Hop size in samples.
    ///
    /// # Returns
    ///
    /// The updated builder instance.
    #[inline]
    #[must_use]
    pub const fn hop_size(mut self, hop_size: NonZeroUsize) -> Self {
        self.hop_size = Some(hop_size);
        self
    }

    /// Set the window function.
    ///
    /// # Arguments
    ///
    /// * `window` - Window function to apply to each frame.
    ///
    /// # Returns
    ///
    /// The updated builder instance.
    #[inline]
    #[must_use]
    pub const fn window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set whether to center frames (pad input signal).
    ///
    /// # Arguments
    ///
    /// * `centre` - If true, frames are centered by padding the input signal.
    ///
    /// # Returns
    ///
    /// The updated builder instance.
    #[inline]
    #[must_use]
    pub const fn centre(mut self, centre: bool) -> Self {
        self.centre = centre;
        self
    }

    /// Build the [`SpectrogramParams`].
    ///
    /// # Errors
    ///
    /// Returns an error if required parameters are not set or are invalid.
    ///
    /// # Returns
    ///
    /// A `SpectrogramParams` instance.
    #[inline]
    pub fn build(self) -> SpectrogramResult<SpectrogramParams> {
        let sample_rate = self
            .sample_rate
            .ok_or_else(|| SpectrogramError::invalid_input("sample_rate must be set"))?;
        let n_fft = self
            .n_fft
            .ok_or_else(|| SpectrogramError::invalid_input("n_fft must be set"))?;
        let hop_size = self
            .hop_size
            .ok_or_else(|| SpectrogramError::invalid_input("hop_size must be set"))?;

        let stft = StftParams::new(n_fft, hop_size, self.window, self.centre)?;
        SpectrogramParams::new(stft, sample_rate)
    }
}

//
// ========================
// Standalone FFT Functions
// ========================
//

/// Compute the real-to-complex FFT of a real-valued signal.
///
/// This function performs a forward FFT on real-valued input, returning the
/// complex frequency domain representation. Only the positive frequencies
/// are returned (length = `n_fft/2` + 1) due to conjugate symmetry.
///
/// # Arguments
///
/// * `samples` - Input signal (length ≤ n_fft, will be zero-padded if shorter)
/// * `n_fft` - FFT size
///
/// # Returns
///
/// A vector of complex frequency bins with length `n_fft/2` + 1.
///
/// # Automatic Zero-Padding
///
/// If the input signal is shorter than `n_fft`, it will be automatically
/// zero-padded to the required length. This is standard DSP practice and
/// preserves frequency resolution (bin spacing = sample_rate / n_fft).
///
/// ```
/// use spectrograms::{fft, nzu};
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![1.0, 2.0, 3.0]; // Only 3 samples
/// let spectrum = fft(&signal, nzu!(8))?;   // Automatically padded to 8
/// assert_eq!(spectrum.len(), 5);     // Output: 8/2 + 1 = 5 bins
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `InvalidInput` error if the input length exceeds `n_fft`.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![0.0; nzu!(512)];
/// let spectrum = fft(&signal, nzu!(512))?;
///
/// assert_eq!(spectrum.len(), 257); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn fft(
    samples: &NonEmptySlice<f64>,
    n_fft: NonZeroUsize,
) -> SpectrogramResult<Array1<Complex<f64>>> {
    if samples.len() > n_fft {
        return Err(SpectrogramError::invalid_input(format!(
            "Input length ({}) exceeds FFT size ({})",
            samples.len(),
            n_fft
        )));
    }

    let out_len = r2c_output_size(n_fft.get());

    // Get FFT plan from global cache (or create if first use)
    #[cfg(feature = "realfft")]
    let mut fft = {
        use crate::fft_backend::get_or_create_r2c_plan;
        let plan = get_or_create_r2c_plan(n_fft.get())?;
        // Clone the plan to get our own mutable copy with independent scratch buffer
        // This is cheap - only clones the scratch buffer, not the expensive twiddle factors
        (*plan).clone()
    };

    #[cfg(feature = "fftw")]
    let mut fft = {
        use std::sync::Arc;
        let plan = crate::FftwPlanner::build_plan(n_fft.get())?;
        crate::FftwPlan::new(Arc::new(plan))
    };

    let input = if samples.len() < n_fft {
        let mut padded = vec![0.0; n_fft.get()];
        padded[..samples.len().get()].copy_from_slice(samples);
        // safety: samples.len() < n_fft checked above and n_fft > 0

        unsafe { NonEmptyVec::new_unchecked(padded) }
    } else {
        samples.to_non_empty_vec()
    };

    let mut output = vec![Complex::new(0.0, 0.0); out_len];
    fft.process(&input, &mut output)?;
    let output = Array1::from_vec(output);
    Ok(output)
}

#[inline]
/// Compute the real-valued fft of a signal.
///
/// # Arguments
/// * `samples` - Input signal (length ≤ n_fft, will be zero-padded if shorter)
/// * `n_fft` - FFT size
///
/// # Returns
///
/// An array with length `n_fft/2` + 1.
///
/// # Errors
///
/// Returns `InvalidInput` error if the input length exceeds `n_fft`.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![0.0; nzu!(512)];
/// let rfft_result = rfft(&signal, nzu!(512))?;
/// // equivalent to
/// let fft_result = fft(&signal, nzu!(512))?;
/// let rfft_result = fft_result.mapv(num_complex::Complex::norm);
/// # Ok(())
/// # }
///
pub fn rfft(samples: &NonEmptySlice<f64>, n_fft: NonZeroUsize) -> SpectrogramResult<Array1<f64>> {
    Ok(fft(samples, n_fft)?.mapv(Complex::norm))
}

/// Compute the power spectrum of a signal (|X|²).
///
/// This function applies an optional window function and computes the
/// power spectrum via FFT. The result contains only positive frequencies.
///
/// # Arguments
///
/// * `samples` - Input signal (length ≤ n_fft, will be zero-padded if shorter)
/// * `n_fft` - FFT size
/// * `window` - Optional window function (None for rectangular window)
///
/// # Returns
///
/// A vector of power values with length `n_fft/2` + 1.
///
/// # Automatic Zero-Padding
///
/// If the input signal is shorter than `n_fft`, it will be automatically
/// zero-padded to the required length. This is standard DSP practice and
/// preserves frequency resolution (bin spacing = sample_rate / n_fft).
///
/// ```
/// use spectrograms::{power_spectrum, nzu};
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![1.0, 2.0, 3.0]; // Only 3 samples
/// let power = power_spectrum(&signal, nzu!(8), None)?;
/// assert_eq!(power.len(), nzu!(5));     // Output: 8/2 + 1 = 5 bins
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `InvalidInput` error if the input length exceeds `n_fft`.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![0.0; nzu!(512)];
/// let power = power_spectrum(&signal, nzu!(512), Some(WindowType::Hanning))?;
///
/// assert_eq!(power.len(), nzu!(257)); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn power_spectrum(
    samples: &NonEmptySlice<f64>,
    n_fft: NonZeroUsize,
    window: Option<WindowType>,
) -> SpectrogramResult<NonEmptyVec<f64>> {
    if samples.len() > n_fft {
        return Err(SpectrogramError::invalid_input(format!(
            "Input length ({}) exceeds FFT size ({})",
            samples.len(),
            n_fft
        )));
    }

    let mut windowed = vec![0.0; n_fft.get()];
    windowed[..samples.len().get()].copy_from_slice(samples);

    if let Some(win_type) = window {
        let window_samples = make_window(win_type, n_fft);
        for i in 0..n_fft.get() {
            windowed[i] *= window_samples[i];
        }
    }

    // safety: windowed is non-empty since n_fft > 0
    let windowed = unsafe { NonEmptySlice::new_unchecked(&windowed) };
    let fft_result = fft(windowed, n_fft)?;
    let fft_result = fft_result
        .iter()
        .map(num_complex::Complex::norm_sqr)
        .collect();
    // safety: fft_result is non-empty since fft returned successfully
    Ok(unsafe { NonEmptyVec::new_unchecked(fft_result) })
}

/// Compute the magnitude spectrum of a signal (|X|).
///
/// This function applies an optional window function and computes the
/// magnitude spectrum via FFT. The result contains only positive frequencies.
///
/// # Arguments
///
/// * `samples` - Input signal (length ≤ n_fft, will be zero-padded if shorter)
/// * `n_fft` - FFT size
/// * `window` - Optional window function (None for rectangular window)
///
/// # Automatic Zero-Padding
///
/// If the input signal is shorter than `n_fft`, it will be automatically
/// zero-padded to the required length. This preserves frequency resolution.
///
/// # Errors
///
/// Returns `InvalidInput` error if the input length exceeds `n_fft`.
///
/// # Returns
///
/// A vector of magnitude values with length `n_fft/2` + 1.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![0.0; nzu!(512)];
/// let magnitude = magnitude_spectrum(&signal, nzu!(512), Some(WindowType::Hanning))?;
///
/// assert_eq!(magnitude.len(), nzu!(257)); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn magnitude_spectrum(
    samples: &NonEmptySlice<f64>,
    n_fft: NonZeroUsize,
    window: Option<WindowType>,
) -> SpectrogramResult<NonEmptyVec<f64>> {
    let power = power_spectrum(samples, n_fft, window)?;
    let power = power.iter().map(|&p| p.sqrt()).collect();
    // safety: power is non-empty since power_spectrum returned successfully
    Ok(unsafe { NonEmptyVec::new_unchecked(power) })
}

/// Compute the Short-Time Fourier Transform (STFT) of a signal.
///
/// This function computes the STFT by applying a sliding window and FFT
/// to sequential frames of the input signal.
///
/// # Arguments
///
/// * `samples` - Input signal (any type that can be converted to a slice)
/// * `n_fft` - FFT size
/// * `hop_size` - Number of samples between successive frames
/// * `window` - Window function to apply to each frame
/// * `center` - If true, pad the signal to center frames
///
/// # Returns
///
/// A 2D array with shape (`frequency_bins`, `time_frames`) containing complex STFT values.
///
/// # Errors
///
/// Returns an error if:
/// - `hop_size` > `n_fft`
/// - STFT computation fails
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = non_empty_vec![0.0; nzu!(16000)];
/// let stft_matrix = stft(&signal, nzu!(512), nzu!(256), WindowType::Hanning, true)?;
///
/// println!("STFT: {} bins x {} frames", stft_matrix.nrows(), stft_matrix.ncols());
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn stft(
    samples: &NonEmptySlice<f64>,
    n_fft: NonZeroUsize,
    hop_size: NonZeroUsize,
    window: WindowType,
    center: bool,
) -> SpectrogramResult<Array2<Complex<f64>>> {
    let stft_params = StftParams::new(n_fft, hop_size, window, center)?;
    let params = SpectrogramParams::new(stft_params, 1.0)?; // dummy sample rate

    let planner = SpectrogramPlanner::new();
    let result = planner.compute_stft(samples, &params)?;

    Ok(result.data)
}

/// Compute the inverse real FFT (complex-to-real IFFT).
///
/// This function performs an inverse FFT, converting frequency domain data
/// back to the time domain. Only the positive frequencies need to be provided
/// (length = `n_fft/2` + 1) due to conjugate symmetry.
///
/// # Arguments
///
/// * `spectrum` - Complex frequency bins (length should be `n_fft/2` + 1)
/// * `n_fft` - FFT size (length of the output signal)
///
/// # Returns
///
/// A vector of real-valued time-domain samples with length `n_fft`.
///
/// # Errors
///
/// Returns an error if:
/// - `spectrum` length doesn't match `n_fft/2` + 1
/// - Inverse FFT computation fails
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::{non_empty_vec, NonEmptySlice};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Forward FFT
/// let signal = non_empty_vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
/// let spectrum = fft(&signal, nzu!(8))?;
/// let slice = spectrum.as_slice().unwrap();
/// let spectrum_slice = NonEmptySlice::new(slice).unwrap();
/// // Inverse FFT
/// let reconstructed = irfft(spectrum_slice, nzu!(8))?;
///
/// assert_eq!(reconstructed.len(), nzu!(8));
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn irfft(
    spectrum: &NonEmptySlice<Complex<f64>>,
    n_fft: NonZeroUsize,
) -> SpectrogramResult<NonEmptyVec<f64>> {
    use crate::fft_backend::{C2rPlan, r2c_output_size};

    let n_fft = n_fft.get();
    let expected_len = r2c_output_size(n_fft);
    if spectrum.len().get() != expected_len {
        return Err(SpectrogramError::dimension_mismatch(
            expected_len,
            spectrum.len().get(),
        ));
    }

    // Get inverse FFT plan from global cache (or create if first use)
    #[cfg(feature = "realfft")]
    let mut ifft = {
        use crate::fft_backend::get_or_create_c2r_plan;
        let plan = get_or_create_c2r_plan(n_fft)?;
        // Clone to get our own mutable copy with independent scratch buffer
        (*plan).clone()
    };

    #[cfg(feature = "fftw")]
    let mut ifft = {
        use crate::fft_backend::C2rPlanner;
        let mut planner = crate::FftwPlanner::new();
        planner.plan_c2r(n_fft)?
    };

    let mut output = vec![0.0; n_fft];
    ifft.process(spectrum.as_slice(), &mut output)?;

    // Safety: output is non-empty since n_fft > 0
    Ok(unsafe { NonEmptyVec::new_unchecked(output) })
}

/// Reconstruct a time-domain signal from its STFT using overlap-add.
///
/// This function performs the inverse Short-Time Fourier Transform, converting
/// a 2D complex STFT matrix back to a 1D time-domain signal using overlap-add
/// synthesis with the specified window function.
///
/// # Arguments
///
/// * `stft_matrix` - Complex STFT with shape (`frequency_bins`, `time_frames`)
/// * `n_fft` - FFT size
/// * `hop_size` - Number of samples between successive frames
/// * `window` - Window function to apply (should match forward STFT window)
/// * `center` - If true, assume the forward STFT was centered
///
/// # Returns
///
/// A vector of reconstructed time-domain samples.
///
/// # Errors
///
/// Returns an error if:
/// - `stft_matrix` dimensions are inconsistent with `n_fft`
/// - `hop_size` > `n_fft`
/// - Inverse STFT computation fails
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Generate signal
/// let signal = non_empty_vec![1.0; nzu!(16000)];
///
/// // Forward STFT
/// let stft_matrix = stft(&signal, nzu!(512), nzu!(256), WindowType::Hanning, true)?;
///
/// // Inverse STFT
/// let reconstructed = istft(&stft_matrix, nzu!(512), nzu!(256), WindowType::Hanning, true)?;
///
/// println!("Original: {} samples", signal.len());
/// println!("Reconstructed: {} samples", reconstructed.len());
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn istft(
    stft_matrix: &Array2<Complex<f64>>,
    n_fft: NonZeroUsize,
    hop_size: NonZeroUsize,
    window: WindowType,
    center: bool,
) -> SpectrogramResult<Vec<f64>> {
    use crate::fft_backend::{C2rPlan, C2rPlanner, r2c_output_size};

    let n_bins = stft_matrix.nrows();
    let n_frames = stft_matrix.ncols();

    let expected_bins = r2c_output_size(n_fft.get());
    if n_bins != expected_bins {
        return Err(SpectrogramError::dimension_mismatch(expected_bins, n_bins));
    }
    if hop_size.get() > n_fft.get() {
        return Err(SpectrogramError::invalid_input("hop_size must be <= n_fft"));
    }
    // Create inverse FFT plan
    #[cfg(feature = "realfft")]
    let mut ifft = {
        let mut planner = crate::RealFftPlanner::new();
        planner.plan_c2r(n_fft.get())?
    };

    #[cfg(feature = "fftw")]
    let mut ifft = {
        let mut planner = crate::FftwPlanner::new();
        planner.plan_c2r(n_fft.get())?
    };

    // Generate window
    let window_samples = make_window(window, n_fft);
    let n_fft = n_fft.get();
    let hop_size = hop_size.get();
    // Calculate output length
    let pad = if center { n_fft / 2 } else { 0 };
    let output_len = (n_frames - 1) * hop_size + n_fft;
    let unpadded_len = output_len.saturating_sub(2 * pad);

    // Allocate output buffer and normalization buffer
    let mut output = vec![0.0; output_len];
    let mut norm = vec![0.0; output_len];

    // Overlap-add synthesis
    let mut frame_buffer = vec![Complex::new(0.0, 0.0); n_bins];
    let mut time_frame = vec![0.0; n_fft];

    for frame_idx in 0..n_frames {
        // Extract complex frame from STFT matrix
        for bin_idx in 0..n_bins {
            frame_buffer[bin_idx] = stft_matrix[[bin_idx, frame_idx]];
        }

        // Inverse FFT
        ifft.process(&frame_buffer, &mut time_frame)?;

        // Apply window
        for i in 0..n_fft {
            time_frame[i] *= window_samples[i];
        }

        // Overlap-add into output buffer
        let start = frame_idx * hop_size;
        for i in 0..n_fft {
            let pos = start + i;
            if pos < output_len {
                output[pos] += time_frame[i];
                norm[pos] += window_samples[i] * window_samples[i];
            }
        }
    }

    // Normalize by window energy
    for i in 0..output_len {
        if norm[i] > 1e-10 {
            output[i] /= norm[i];
        }
    }

    // Remove padding if centered
    if center && unpadded_len > 0 {
        let start = pad;
        let end = start + unpadded_len;
        output = output[start..end.min(output_len)].to_vec();
    }

    Ok(output)
}

//
// ========================
// Reusable FFT Plans
// ========================
//

/// A reusable FFT planner for efficient repeated FFT operations.
///
/// This planner caches FFT plans internally, making repeated FFT operations
/// of the same size much more efficient than calling `fft()` repeatedly.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut planner = FftPlanner::new();
///
/// // Process multiple signals of the same size efficiently
/// for _ in 0..100 {
///     let signal = non_empty_vec![0.0; nzu!(512)];
///     let spectrum = planner.fft(&signal, nzu!(512))?;
///     // ... process spectrum ...
/// }
/// # Ok(())
/// # }
/// ```
pub struct FftPlanner {
    #[cfg(feature = "realfft")]
    inner: crate::RealFftPlanner,
    #[cfg(feature = "fftw")]
    inner: crate::FftwPlanner,
}

impl FftPlanner {
    /// Create a new FFT planner with empty cache.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "realfft")]
            inner: crate::RealFftPlanner::new(),
            #[cfg(feature = "fftw")]
            inner: crate::FftwPlanner::new(),
        }
    }

    /// Compute forward FFT, reusing cached plans.
    ///
    /// This is more efficient than calling the standalone `fft()` function
    /// repeatedly for the same FFT size.
    ///
    /// # Automatic Zero-Padding
    ///
    /// If the input signal is shorter than `n_fft`, it will be automatically
    /// zero-padded to the required length.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` error if the input length exceeds `n_fft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = non_empty_vec![1.0; nzu!(512)];
    /// let spectrum = planner.fft(&signal, nzu!(512))?;
    ///
    /// assert_eq!(spectrum.len(), 257); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn fft(
        &mut self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
    ) -> SpectrogramResult<Array1<Complex<f64>>> {
        use crate::fft_backend::{R2cPlan, R2cPlanner, r2c_output_size};

        if samples.len() > n_fft {
            return Err(SpectrogramError::invalid_input(format!(
                "Input length ({}) exceeds FFT size ({})",
                samples.len(),
                n_fft
            )));
        }

        let out_len = r2c_output_size(n_fft.get());
        let mut plan = self.inner.plan_r2c(n_fft.get())?;

        let input = if samples.len() < n_fft {
            let mut padded = vec![0.0; n_fft.get()];
            padded[..samples.len().get()].copy_from_slice(samples);

            // safety: samples.len() < n_fft checked above and n_fft > 0
            unsafe { NonEmptyVec::new_unchecked(padded) }
        } else {
            samples.to_non_empty_vec()
        };

        let mut output = vec![Complex::new(0.0, 0.0); out_len];
        plan.process(&input, &mut output)?;

        let output = Array1::from_vec(output);
        Ok(output)
    }

    /// Compute forward real FFT magnitude
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `n_fft` doesn't match the samples length
    ///
    ///
    #[inline]
    pub fn rfft(
        &mut self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
    ) -> SpectrogramResult<Array1<f64>> {
        let fft_with_complex = fft(samples, n_fft)?;
        Ok(fft_with_complex.mapv(Complex::norm))
    }

    /// Compute inverse FFT, reusing cached plans.
    ///
    /// This is more efficient than calling the standalone `irfft()` function
    /// repeatedly for the same FFT size.
    ///
    /// # Errors
    /// Returns an error if:
    ///
    /// - The calculated expected length of `spectrum` doesn't match its actual length
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::{non_empty_vec, NonEmptySlice};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// // Forward FFT
    /// let signal = non_empty_vec![1.0; nzu!(512)];
    /// let spectrum = planner.fft(&signal, nzu!(512))?;
    ///
    /// // Inverse FFT
    /// let spectrum_slice = NonEmptySlice::new(spectrum.as_slice().unwrap()).unwrap();
    /// let reconstructed = planner.irfft(spectrum_slice, nzu!(512))?;
    ///
    /// assert_eq!(reconstructed.len(), nzu!(512));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn irfft(
        &mut self,
        spectrum: &NonEmptySlice<Complex<f64>>,
        n_fft: NonZeroUsize,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        use crate::fft_backend::{C2rPlan, C2rPlanner, r2c_output_size};

        let expected_len = r2c_output_size(n_fft.get());
        if spectrum.len().get() != expected_len {
            return Err(SpectrogramError::dimension_mismatch(
                expected_len,
                spectrum.len().get(),
            ));
        }

        let mut plan = self.inner.plan_c2r(n_fft.get())?;
        let mut output = vec![0.0; n_fft.get()];
        plan.process(spectrum, &mut output)?;
        // Safety: output is non-empty since n_fft > 0
        let output = unsafe { NonEmptyVec::new_unchecked(output) };
        Ok(output)
    }

    /// Compute power spectrum with optional windowing, reusing cached plans.
    ///
    /// # Automatic Zero-Padding
    ///
    /// If the input signal is shorter than `n_fft`, it will be automatically
    /// zero-padded to the required length.
    ///
    /// # Errors
    /// Returns `InvalidInput` error if the input length exceeds `n_fft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = non_empty_vec![1.0; nzu!(512)];
    /// let power = planner.power_spectrum(&signal, nzu!(512), Some(WindowType::Hanning))?;
    ///
    /// assert_eq!(power.len(), nzu!(257));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn power_spectrum(
        &mut self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
        window: Option<WindowType>,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        if samples.len() > n_fft {
            return Err(SpectrogramError::invalid_input(format!(
                "Input length ({}) exceeds FFT size ({})",
                samples.len(),
                n_fft
            )));
        }

        let mut windowed = vec![0.0; n_fft.get()];
        windowed[..samples.len().get()].copy_from_slice(samples);
        if let Some(win_type) = window {
            let window_samples = make_window(win_type, n_fft);
            for i in 0..n_fft.get() {
                windowed[i] *= window_samples[i];
            }
        }

        // safety: windowed is non-empty since n_fft > 0
        let windowed = unsafe { NonEmptySlice::new_unchecked(&windowed) };
        let fft_result = self.fft(windowed, n_fft)?;
        let f = fft_result
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .collect();
        // safety: fft_result is non-empty since fft returned successfully
        Ok(unsafe { NonEmptyVec::new_unchecked(f) })
    }

    /// Compute magnitude spectrum with optional windowing, reusing cached plans.
    ///
    /// # Automatic Zero-Padding
    ///
    /// If the input signal is shorter than `n_fft`, it will be automatically
    /// zero-padded to the required length.
    ///
    /// # Errors
    /// Returns `InvalidInput` error if the input length exceeds `n_fft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    /// use non_empty_slice::non_empty_vec;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = non_empty_vec![1.0; nzu!(512)];
    /// let magnitude = planner.magnitude_spectrum(&signal, nzu!(512), Some(WindowType::Hanning))?;
    ///
    /// assert_eq!(magnitude.len(), nzu!(257));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn magnitude_spectrum(
        &mut self,
        samples: &NonEmptySlice<f64>,
        n_fft: NonZeroUsize,
        window: Option<WindowType>,
    ) -> SpectrogramResult<NonEmptyVec<f64>> {
        let power = self.power_spectrum(samples, n_fft, window)?;
        let power = power.iter().map(|&p| p.sqrt()).collect::<Vec<f64>>();
        // safety: power is non-empty since power_spectrum returned successfully
        Ok(unsafe { NonEmptyVec::new_unchecked(power) })
    }
}

impl Default for FftPlanner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_basic() {
        // Create a simple 3x5 sparse matrix
        let mut sparse = SparseMatrix::new(3, 5);

        // Row 0: only column 1 has value 2.0
        sparse.set(0, 1, 2.0);

        // Row 1: columns 2 and 3
        sparse.set(1, 2, 0.5);
        sparse.set(1, 3, 1.5);

        // Row 2: columns 0 and 4
        sparse.set(2, 0, 3.0);
        sparse.set(2, 4, 1.0);

        // Test matrix-vector multiplication
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 3];

        sparse.multiply_vec(&input, &mut output);

        // Expected results:
        // Row 0: 2.0 * 2.0 = 4.0
        // Row 1: 0.5 * 3.0 + 1.5 * 4.0 = 1.5 + 6.0 = 7.5
        // Row 2: 3.0 * 1.0 + 1.0 * 5.0 = 3.0 + 5.0 = 8.0
        assert_eq!(output[0], 4.0);
        assert_eq!(output[1], 7.5);
        assert_eq!(output[2], 8.0);
    }

    #[test]
    fn test_sparse_matrix_zeros_ignored() {
        // Verify that zero values are not stored
        let mut sparse = SparseMatrix::new(2, 3);

        sparse.set(0, 0, 1.0);
        sparse.set(0, 1, 0.0); // Should be ignored
        sparse.set(0, 2, 2.0);

        // Only 2 values should be stored in row 0
        assert_eq!(sparse.values[0].len(), 2);
        assert_eq!(sparse.indices[0].len(), 2);

        // The stored indices should be 0 and 2
        assert_eq!(sparse.indices[0], vec![0, 2]);
        assert_eq!(sparse.values[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_loghz_matrix_sparsity() {
        // Verify that LogHz matrices are very sparse (1-2 non-zeros per row)
        let sample_rate = 16000.0;
        let n_fft = nzu!(512);
        let n_bins = nzu!(128);
        let f_min = 20.0;
        let f_max = sample_rate / 2.0;

        let (matrix, _freqs) =
            build_loghz_matrix(sample_rate, n_fft, n_bins, f_min, f_max).unwrap();

        // Each row should have at most 2 non-zero values (linear interpolation)
        for row_idx in 0..matrix.nrows() {
            let nnz = matrix.values[row_idx].len();
            assert!(
                nnz <= 2,
                "Row {} has {} non-zeros, expected at most 2",
                row_idx,
                nnz
            );
            assert!(nnz >= 1, "Row {} has no non-zeros", row_idx);
        }

        // Total non-zeros should be close to n_bins * 2
        let total_nnz: usize = matrix.values.iter().map(|v| v.len()).sum();
        assert!(total_nnz <= n_bins.get() * 2);
        assert!(total_nnz >= n_bins.get()); // At least 1 per row
    }

    #[test]
    fn test_mel_matrix_sparsity() {
        // Verify that Mel matrices are sparse (triangular filters)
        let sample_rate = 16000.0;
        let n_fft = nzu!(512);
        let n_mels = nzu!(40);
        let f_min = 0.0;
        let f_max = sample_rate / 2.0;

        let matrix =
            build_mel_filterbank_matrix(sample_rate, n_fft, n_mels, f_min, f_max, MelNorm::None)
                .unwrap();

        let n_fft_bins = r2c_output_size(n_fft.get());

        // Calculate sparsity
        let total_nnz: usize = matrix.values.iter().map(|v| v.len()).sum();
        let total_elements = n_mels.get() * n_fft_bins;
        let sparsity = 1.0 - (total_nnz as f64 / total_elements as f64);

        // Mel filterbanks should be >80% sparse
        assert!(
            sparsity > 0.8,
            "Mel matrix sparsity is only {:.1}%, expected >80%",
            sparsity * 100.0
        );

        // Each mel filter should have significantly fewer than n_fft_bins non-zeros
        for row_idx in 0..matrix.nrows() {
            let nnz = matrix.values[row_idx].len();
            assert!(
                nnz < n_fft_bins / 2,
                "Mel filter {} is not sparse enough",
                row_idx
            );
        }
    }
}
