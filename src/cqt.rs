use std::{num::NonZeroUsize, ops::Deref};

/// Constant-Q Transform (CQT) implementation.
///
/// The CQT provides logarithmically-spaced frequency bins with constant Q factor,
/// making it ideal for musical analysis where notes are logarithmically spaced.
use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult, WindowType, nzu};

/// CQT parameters
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CqtParams {
    /// Number of bins per octave
    pub bins_per_octave: NonZeroUsize,
    /// Number of octaves to cover
    pub n_octaves: NonZeroUsize,
    /// Minimum frequency (Hz)
    pub f_min: f64,
    /// Q factor (constant quality factor)
    pub q_factor: f64,
    /// Window type for kernel generation
    pub window: WindowType,
    /// Sparsity threshold (0.0 = no sparsity, 0.01 = 1% threshold)
    pub sparsity_threshold: f64,
    /// Whether to normalize kernels
    pub normalize: bool,
}

impl CqtParams {
    /// Create new CQT parameters.
    ///
    /// # Arguments
    ///
    /// * `bins_per_octave` - Number of frequency bins per octave (e.g., 12 for semitones)
    /// * `n_octaves` - Number of octaves to span
    /// * `f_min` - Minimum frequency in Hz
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<Self>` - Ok with CqtParams if valid
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if:
    #[inline]
    pub fn new(
        bins_per_octave: NonZeroUsize,
        n_octaves: NonZeroUsize,
        f_min: f64,
    ) -> SpectrogramResult<Self> {
        if f_min <= 0.0 || f_min.is_infinite() {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and > 0",
            ));
        }

        Ok(Self {
            bins_per_octave,
            n_octaves,
            f_min,
            q_factor: 1.0 / ((1.0 / bins_per_octave.get() as f64).exp2() - 1.0),
            window: WindowType::Hanning,
            sparsity_threshold: 0.01,
            normalize: true,
        })
    }

    #[inline]
    #[must_use]
    /// Create new CQT parameters without validation (unsafe).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `bins_per_octave` and `n_octaves` are non-zero (guaranteed by type)
    /// - `f_min` is > 0 and finite
    pub unsafe fn new_unchecked(
        bins_per_octave: NonZeroUsize,
        n_octaves: NonZeroUsize,
        f_min: f64,
    ) -> Self {
        Self {
            bins_per_octave,
            n_octaves,
            f_min,
            q_factor: 1.0 / ((1.0 / bins_per_octave.get() as f64).exp2() - 1.0),
            window: WindowType::Hanning,
            sparsity_threshold: 0.01,
            normalize: true,
        }
    }

    /// Set the Q factor manually (overrides default based on `bins_per_octave`).
    ///
    /// # Arguments
    ///
    /// * `q_factor` - Desired Q factor (must be > 0)
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<Self>` - Updated CQT parameters
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError::InvalidInput` if `q_factor` is not > 0 or not finite.
    #[inline]
    pub fn with_q_factor(mut self, q_factor: f64) -> SpectrogramResult<Self> {
        if !(q_factor > 0.0 && q_factor.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "q_factor must be finite and > 0",
            ));
        }
        self.q_factor = q_factor;
        Ok(self)
    }

    /// Set the window type for kernel generation.
    ///
    /// # Arguments
    ///
    /// * `window` - Window type to use (e.g., Hanning, Hamming)
    ///
    /// # Returns
    ///
    /// `CqtParams` - Updated CQT parameters
    #[inline]
    #[must_use]
    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set the sparsity threshold for kernel compression.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Sparsity threshold (0.0 = no sparsity, higher values increase sparsity)
    ///
    /// # Returns
    ///
    /// `CqtParams` - Updated CQT parameters
    #[inline]
    #[must_use]
    pub const fn with_sparsity(mut self, threshold: f64) -> Self {
        self.sparsity_threshold = threshold.max(0.0);
        self
    }

    /// Set whether to normalize kernels.
    ///
    /// # Arguments
    ///
    /// * `normalize` - If true, kernels will be normalized to unit energy.
    ///
    /// # Returns
    ///
    /// `CqtParams` - Updated CQT parameters
    #[inline]
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get the total number of frequency bins.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Total number of frequency bins
    #[inline]
    #[must_use]
    pub const fn num_bins(&self) -> NonZeroUsize {
        // safety: bins_per_octave and n_octaves are NonZeroUsize, so their product is also non-zero
        unsafe { NonZeroUsize::new_unchecked(self.bins_per_octave.get() * self.n_octaves.get()) }
    }

    /// Get the center frequency for a given bin index.
    ///
    /// # Returns
    ///
    /// `f64` - Center frequency in Hz
    #[inline]
    #[must_use]
    pub fn bin_frequency(&self, bin_idx: usize) -> f64 {
        self.f_min * (bin_idx as f64 / self.bins_per_octave.get() as f64).exp2()
    }

    /// Get the bandwidth for a given bin index.
    ///
    /// # Returns
    ///
    /// `f64` - Bandwidth in Hz
    #[inline]
    #[must_use]
    pub fn bin_bandwidth(&self, bin_idx: usize) -> f64 {
        self.bin_frequency(bin_idx) / self.q_factor
    }

    /// Get all bin center frequencies.
    ///
    /// # Returns
    ///
    /// `NonEmptyVec<f64>` - Center frequencies in Hz
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> NonEmptyVec<f64> {
        let freqs = (0..self.num_bins().get())
            .map(|i| self.bin_frequency(i))
            .collect();
        // safety: num_bins() is non-zero
        unsafe { NonEmptyVec::new_unchecked(freqs) }
    }

    /// Percussion default config
    ///
    /// # Returns
    ///
    /// `CqtParams` - CQT parameters optimized for percussive sounds
    #[inline]
    #[must_use]
    pub fn percussive() -> Self {
        // safety: The parameters provided are valid and will not cause any panics or invalid states. bins_per_octave and n_octaves are non-zero, and f_min is > 0.
        unsafe { Self::new_unchecked(nzu!(12), nzu!(7), 32.7) }
    }

    /// Onset detection default config
    ///
    /// # Returns
    ///
    /// `CqtParams` - CQT parameters optimized for onset detection
    #[inline]
    #[must_use]
    pub fn onset_detection() -> Self {
        // safety: The parameters provided are valid and will not cause any panics or invalid states. bins_per_octave and n_octaves are non-zero, and f_min is > 0.
        let mut params = unsafe { Self::new_unchecked(nzu!(24), nzu!(6), 55.0) };
        params.q_factor = 0.5; // Lower Q for better time resolution
        params.sparsity_threshold = 0.02;
        params.normalize = true;
        params
    }

    /// Create a CQT configuration optimized for chord detection.
    ///
    /// Uses settings that balance frequency resolution with computational
    /// efficiency for real-time chord detection applications.
    ///
    /// # Returns
    ///
    /// `CqtParams` - CQT parameters optimized for chord detection
    #[inline]
    #[must_use]
    pub fn chord_detection() -> Self {
        // safety: The parameters provided are valid and will not cause any panics or invalid states. bins_per_octave and n_octaves are non-zero, and f_min is > 0.
        let mut params = unsafe { Self::new_unchecked(nzu!(36), nzu!(5), 82.4) };
        params.q_factor = 0.8; // Moderate Q for chord detection
        params.sparsity_threshold = 0.02;
        params.normalize = true;
        params
    }

    /// Create a CQT configuration optimized for harmonic analysis.
    ///
    /// Uses 24 bins per octave for quarter-tone resolution,
    /// providing detailed harmonic analysis capabilities.
    ///
    /// # Returns
    ///
    /// `CqtParams` - CQT parameters optimized for harmonic analysis
    #[inline]
    #[must_use]
    pub fn harmonic() -> Self {
        // safety: The parameters provided are valid and will not cause any panics or invalid states. bins_per_octave and n_octaves are non-zero, and f_min is > 0.
        let mut params = unsafe { Self::new_unchecked(nzu!(24), nzu!(7), 55.0) };
        params.q_factor = 1.0; // Standard Q for harmonic analysis
        params.sparsity_threshold = 0.005;
        params.normalize = true;
        params
    }

    /// Create a CQT configuration optimized for musical analysis.
    ///
    /// Uses 12 bins per octave for chromatic scale analysis,
    /// starting from C1 (32.7 Hz) for full piano range coverage.
    #[inline]
    #[must_use]
    pub fn musical() -> Self {
        // safety: The parameters provided are valid and will not cause any panics or invalid states. bins_per_octave and n_octaves are non-zero, and f_min is > 0.
        let mut params = unsafe { Self::new_unchecked(nzu!(12), nzu!(7), 32.7) };
        params.q_factor = 1.0; // Standard Q for musical analysis
        params.sparsity_threshold = 0.01;
        params.normalize = true;
        params
    }
}

/// CQT kernel data structure for efficient sparse representation.
#[derive(Debug, Clone)]
pub struct CqtKernel {
    /// Complex kernel coefficients for each frequency bin
    kernels: NonEmptyVec<NonEmptyVec<Complex<f64>>>,
    /// Kernel lengths for each frequency bin
    kernel_lengths: Vec<NonZeroUsize>,
    /// FFT size used for convolution (reserved for future use)
    _fft_size: usize,
    /// Center frequencies for each bin
    frequencies: NonEmptyVec<f64>,
}

impl CqtKernel {
    /// Generate CQT kernels for all frequency bins.
    pub(crate) fn generate(
        params: &CqtParams,
        sample_rate: f64,
        signal_length: NonZeroUsize,
    ) -> Self {
        let num_bins = params.num_bins().get();
        let mut kernels = Vec::with_capacity(num_bins);
        let mut frequencies = Vec::with_capacity(num_bins);
        let mut kernel_lengths = Vec::with_capacity(num_bins);

        // Calculate FFT size (next power of 2 for efficiency)
        let fft_size = (signal_length.get() * 2).next_power_of_two();

        for bin_idx in 0..num_bins {
            let center_freq = params.bin_frequency(bin_idx);

            // Check if frequency is within valid range
            if center_freq >= sample_rate / 2.0 {
                break;
            }

            // Calculate kernel length based on bandwidth
            let kernel_length = ((params.q_factor * sample_rate / center_freq).round() as usize)
                .max(1)
                .min(signal_length.get());
            // safety: kernel_length is guaranteed to be > 0
            let kernel_length = unsafe { NonZeroUsize::new_unchecked(kernel_length) };

            // Generate complex exponential kernel
            let mut kernel = Self::generate_kernel_bin(
                center_freq,
                kernel_length,
                sample_rate,
                params.window.clone(),
            );

            // Apply sparsity threshold
            Self::apply_sparsity_threshold(&mut kernel, params.sparsity_threshold);

            // Normalize kernel if requested
            if params.normalize {
                Self::normalize_kernel(&mut kernel);
            }

            kernels.push(kernel);
            frequencies.push(center_freq);
            kernel_lengths.push(kernel_length);
        }

        // safety: kernels is non-empty since num_bins > 0 and at least one frequency is valid
        let kernels = unsafe { NonEmptyVec::new_unchecked(kernels) };
        // safety: frequencies is non-empty since at least one frequency is valid
        let frequencies = unsafe { NonEmptyVec::new_unchecked(frequencies) };

        Self {
            kernels,
            kernel_lengths,
            _fft_size: fft_size,
            frequencies,
        }
    }

    /// Generate a single CQT kernel for a specific frequency bin.
    fn generate_kernel_bin(
        center_freq: f64,
        kernel_length: NonZeroUsize,
        sample_rate: f64,
        window_type: WindowType,
    ) -> NonEmptyVec<Complex<f64>> {
        let mut kernel = Vec::with_capacity(kernel_length.get());

        // Generate window coefficients
        let window = crate::spectrogram::make_window(window_type, kernel_length);

        // Generate complex exponential kernel
        for (n, w) in window.iter().enumerate().take(kernel_length.get()) {
            let t = n as f64 / sample_rate;
            let phase = 2.0 * std::f64::consts::PI * center_freq * t;

            // Complex exponential: e^(i*2*π*f*t)
            let exponential = Complex::new(phase.cos(), phase.sin());

            // Apply window function
            let windowed = exponential * w;

            kernel.push(windowed);
        }
        // safety: kernel is non-empty since kernel_length > 0
        unsafe { NonEmptyVec::new_unchecked(kernel) }
    }

    /// Apply sparsity threshold to reduce kernel size.
    fn apply_sparsity_threshold(kernel: &mut [Complex<f64>], threshold: f64) {
        if threshold <= 0.0 {
            return;
        }

        // Find maximum magnitude in kernel
        let max_magnitude = kernel.iter().map(|c| c.norm()).fold(0.0, f64::max);

        if max_magnitude == 0.0 {
            return;
        }

        let absolute_threshold = max_magnitude * threshold;

        // Apply threshold
        for coefficient in kernel.iter_mut() {
            if coefficient.norm() < absolute_threshold {
                *coefficient = Complex::new(0.0, 0.0);
            }
        }
    }

    /// Normalize a kernel to unit energy.
    fn normalize_kernel(kernel: &mut NonEmptySlice<Complex<f64>>) {
        let energy: f64 = kernel.iter().map(num_complex::Complex::norm_sqr).sum();

        if energy > 0.0 {
            let norm_factor = 1.0 / energy.sqrt();
            for coefficient in kernel.iter_mut() {
                *coefficient *= norm_factor;
            }
        }
    }

    /// Get the frequencies for each bin.
    ///
    /// # Returns
    ///
    /// `&NonEmptySlice<f64>` - Center frequencies in Hz
    #[inline]
    #[must_use]
    pub fn frequencies(&self) -> &NonEmptySlice<f64> {
        &self.frequencies
    }

    /// Get the number of bins.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Number of frequency bins
    #[inline]
    #[must_use]
    pub const fn num_bins(&self) -> NonZeroUsize {
        self.kernels.len()
    }

    /// Apply the CQT kernel to input samples.
    ///
    /// This performs the CQT transform by convolving each kernel with the input signal.
    /// Uses time-domain convolution for efficiency with sparse kernels.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples
    ///
    /// # Returns
    ///
    /// `SpectrogramResult<NonEmptyVec<Complex<f64>>>` - CQT coefficients for each bin
    ///
    /// # Errors
    ///
    /// Returns `SpectrogramError` if input is invalid.
    #[inline]
    pub fn apply(
        &self,
        samples: &NonEmptySlice<f64>,
    ) -> SpectrogramResult<NonEmptyVec<Complex<f64>>> {
        let mut cqt_result = Vec::with_capacity(self.kernels.len().get());

        // For each frequency bin
        for (bin_idx, kernel) in self.kernels.iter().enumerate() {
            let kernel_length = self.kernel_lengths[bin_idx];

            // Time-domain correlation (more efficient for sparse kernels)
            let mut correlation = Complex::new(0.0, 0.0);
            let start_idx = samples.len().get().saturating_sub(kernel_length.get());

            for (k_idx, &k) in kernel.iter().enumerate() {
                let sample_idx = start_idx + k_idx;
                if sample_idx < samples.len().get() {
                    let sample = samples[sample_idx];
                    // Conjugate multiplication for correlation
                    correlation += k.conj() * sample;
                }
            }

            cqt_result.push(correlation);
        }

        // safety: cqt_result is non-empty since self.kernels is non-empty and samples is non-empty
        Ok(unsafe { NonEmptyVec::new_unchecked(cqt_result) })
    }
}

/// CQT result containing complex frequency bins and metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CqtResult {
    /// Complex CQT coefficients with shape (`frequency_bins`, `time_frames`)
    pub data: Array2<Complex<f64>>,
    /// Center frequency for each bin in Hz
    pub frequencies: NonEmptyVec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Hop size in samples
    pub hop_size: NonZeroUsize,
}

impl AsRef<Array2<Complex<f64>>> for CqtResult {
    #[inline]
    fn as_ref(&self) -> &Array2<Complex<f64>> {
        &self.data
    }
}

impl Deref for CqtResult {
    type Target = Array2<Complex<f64>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl CqtResult {
    /// Get the number of frequency bins.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Number of frequency bins
    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> NonZeroUsize {
        // safety: data.nrows() is guaranteed to be > 0 for valid CQT result
        unsafe { NonZeroUsize::new_unchecked(self.data.nrows()) }
    }

    /// Get the number of time frames.
    ///
    /// # Returns
    ///
    /// `NonZeroUsize` - Number of time frames
    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> NonZeroUsize {
        // safety: data.ncols() is guaranteed to be > 0 for valid CQT result
        unsafe { NonZeroUsize::new_unchecked(self.data.ncols()) }
    }

    /// Get the time resolution in seconds.
    ///
    /// # Returns
    ///
    /// `f64` - Time resolution in seconds
    #[inline]
    #[must_use]
    pub fn time_resolution(&self) -> f64 {
        self.hop_size.get() as f64 / self.sample_rate
    }

    /// Convert to magnitude spectrogram.
    ///
    /// # Returns
    ///
    /// `Array2<f64>` - Magnitude spectrogram
    #[inline]
    #[must_use]
    pub fn to_magnitude(&self) -> Array2<f64> {
        let mut magnitude = Array2::<f64>::zeros(self.data.dim());
        for ((i, j), val) in self.data.indexed_iter() {
            magnitude[[i, j]] = val.norm();
        }
        magnitude
    }

    /// Convert to power spectrogram.
    ///
    /// # Returns
    ///
    /// `Array2<f64>` - Power spectrogram (squared magnitude)
    #[inline]
    #[must_use]
    pub fn to_power(&self) -> Array2<f64> {
        let mut power = Array2::<f64>::zeros(self.data.dim());
        for ((i, j), val) in self.data.indexed_iter() {
            power[[i, j]] = val.norm_sqr();
        }
        power
    }
}

/// Compute the Constant-Q Transform (CQT) of a signal.
///
/// The CQT provides logarithmically-spaced frequency bins with constant Q factor,
/// making it ideal for musical analysis. Unlike STFT which has linear frequency
/// spacing, CQT matches the logarithmic nature of musical pitch.
///
/// # Arguments
///
/// * `samples` - Input audio signal (any type that can be converted to a slice)
/// * `sample_rate` - Sample rate in Hz
/// * `params` - CQT parameters (bins per octave, number of octaves, `f_min`, Q factor, etc.)
/// * `hop_size` - Number of samples between successive frames
///
/// # Returns
///
/// A `CqtResult` containing the complex CQT matrix and metadata.
///
/// # Errors
///
/// Returns `SpectrogramError` if input parameters are invalid or computation fails.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
/// use non_empty_slice::non_empty_vec;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let samples = non_empty_vec![0.0; nzu!(16000)];
/// let params = CqtParams::new(nzu!(12), nzu!(7), 32.7)?; // 12 bins/octave, 7 octaves, from C1
///
/// let cqt_result = cqt(&samples, 16000.0, &params, nzu!(512))?;
///
/// println!("CQT: {} bins x {} frames", cqt_result.n_bins(), cqt_result.n_frames());
///
/// // Get magnitude spectrogram
/// let magnitude = cqt_result.to_magnitude();
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn cqt(
    samples: &NonEmptySlice<f64>,
    sample_rate: f64,
    params: &CqtParams,
    hop_size: NonZeroUsize,
) -> SpectrogramResult<CqtResult> {
    // Generate CQT kernels
    // Use a reasonable signal length for kernel generation (we'll apply to frames)
    let kernel_length = samples.len().min(nzu!(16384));
    let kernel = CqtKernel::generate(params, sample_rate, kernel_length);

    let n_bins = kernel.num_bins();
    let frequencies = kernel.frequencies().to_non_empty_vec();

    // Compute number of frames
    let n_frames = if samples.len() < kernel_length {
        1
    } else {
        (samples.len().get() - kernel_length.get()) / hop_size.get() + 1
    };

    // Allocate output matrix
    let mut cqt_data = Array2::<Complex<f64>>::zeros((n_bins.get(), n_frames));

    // Process each frame
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size.get();
        let end = (start + kernel_length.get()).min(samples.len().get());

        if end <= start {
            break;
        }

        let frame = non_empty_slice::non_empty_slice!(&samples[start..end]);

        // Apply CQT kernel to frame
        let cqt_frame = kernel.apply(frame)?;

        // Store in output matrix
        for (bin_idx, &val) in cqt_frame.iter().enumerate() {
            if bin_idx < n_bins.get() {
                cqt_data[[bin_idx, frame_idx]] = val;
            }
        }
    }

    Ok(CqtResult {
        data: cqt_data,
        frequencies,
        sample_rate,
        hop_size,
    })
}
