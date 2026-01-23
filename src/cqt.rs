use std::ops::Deref;

/// Constant-Q Transform (CQT) implementation.
///
/// The CQT provides logarithmically-spaced frequency bins with constant Q factor,
/// making it ideal for musical analysis where notes are logarithmically spaced.
use ndarray::Array2;
use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult, WindowType};

/// CQT parameters
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CqtParams {
    /// Number of bins per octave
    bins_per_octave: usize,
    /// Number of octaves to cover
    n_octaves: usize,
    /// Minimum frequency (Hz)
    f_min: f64,
    /// Q factor (constant quality factor)
    q_factor: f64,
    /// Window type for kernel generation
    window: WindowType,
    /// Sparsity threshold (0.0 = no sparsity, 0.01 = 1% threshold)
    sparsity_threshold: f64,
    /// Whether to normalize kernels
    normalize: bool,
}

impl CqtParams {
    /// Create new CQT parameters.
    ///
    /// # Arguments
    ///
    /// * `bins_per_octave` - Number of frequency bins per octave (e.g., 12 for semitones)
    /// * `n_octaves` - Number of octaves to span
    /// * `f_min` - Minimum frequency in Hz
    pub fn new(bins_per_octave: usize, n_octaves: usize, f_min: f64) -> SpectrogramResult<Self> {
        if bins_per_octave == 0 {
            return Err(SpectrogramError::invalid_input(
                "bins_per_octave must be > 0",
            ));
        }
        if n_octaves == 0 {
            return Err(SpectrogramError::invalid_input("n_octaves must be > 0"));
        }
        if !(f_min > 0.0 && f_min.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and > 0",
            ));
        }

        Ok(Self {
            bins_per_octave,
            n_octaves,
            f_min,
            q_factor: 1.0 / ((1.0 / bins_per_octave as f64).exp2() - 1.0),
            window: WindowType::Hanning,
            sparsity_threshold: 0.01,
            normalize: true,
        })
    }

    /// Set the Q factor manually (overrides default based on `bins_per_octave`).
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
    #[must_use] 
    pub const fn with_window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set the sparsity threshold for kernel compression.
    #[must_use] 
    pub const fn with_sparsity(mut self, threshold: f64) -> Self {
        self.sparsity_threshold = threshold.max(0.0);
        self
    }

    /// Set whether to normalize kernels.
    #[must_use] 
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get the total number of frequency bins.
    #[must_use] 
    pub const fn num_bins(&self) -> usize {
        self.bins_per_octave * self.n_octaves
    }

    /// Get the center frequency for a given bin index.
    #[must_use] 
    pub fn bin_frequency(&self, bin_idx: usize) -> f64 {
        self.f_min * (bin_idx as f64 / self.bins_per_octave as f64).exp2()
    }

    /// Get the bandwidth for a given bin index.
    #[must_use] 
    pub fn bin_bandwidth(&self, bin_idx: usize) -> f64 {
        self.bin_frequency(bin_idx) / self.q_factor
    }

    /// Get all bin center frequencies.
    #[must_use] 
    pub fn frequencies(&self) -> Vec<f64> {
        (0..self.num_bins())
            .map(|i| self.bin_frequency(i))
            .collect()
    }
}

/// CQT kernel data structure for efficient sparse representation.
#[derive(Debug, Clone)]
pub struct CqtKernel {
    /// Complex kernel coefficients for each frequency bin
    kernels: Vec<Vec<Complex<f64>>>,
    /// Kernel lengths for each frequency bin
    kernel_lengths: Vec<usize>,
    /// FFT size used for convolution (reserved for future use)
    _fft_size: usize,
    /// Center frequencies for each bin
    frequencies: Vec<f64>,
}

impl CqtKernel {
    /// Generate CQT kernels for all frequency bins.
    pub(crate) fn generate(
        params: &CqtParams,
        sample_rate: f64,
        signal_length: usize,
    ) -> SpectrogramResult<Self> {
        let num_bins = params.num_bins();
        let mut kernels = Vec::with_capacity(num_bins);
        let mut frequencies = Vec::with_capacity(num_bins);
        let mut kernel_lengths = Vec::with_capacity(num_bins);

        // Calculate FFT size (next power of 2 for efficiency)
        let fft_size = (signal_length * 2).next_power_of_two();

        for bin_idx in 0..num_bins {
            let center_freq = params.bin_frequency(bin_idx);

            // Check if frequency is within valid range
            if center_freq >= sample_rate / 2.0 {
                break;
            }

            // Calculate kernel length based on bandwidth
            let kernel_length = ((params.q_factor * sample_rate / center_freq).round() as usize)
                .max(1)
                .min(signal_length);

            // Generate complex exponential kernel
            let mut kernel =
                Self::generate_kernel_bin(center_freq, kernel_length, sample_rate, params.window)?;

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

        Ok(Self {
            kernels,
            kernel_lengths,
            _fft_size: fft_size,
            frequencies,
        })
    }

    /// Generate a single CQT kernel for a specific frequency bin.
    fn generate_kernel_bin(
        center_freq: f64,
        kernel_length: usize,
        sample_rate: f64,
        window_type: WindowType,
    ) -> SpectrogramResult<Vec<Complex<f64>>> {
        use std::f64::consts::PI;

        let mut kernel = Vec::with_capacity(kernel_length);

        // Generate window coefficients
        let window = crate::spectrogram::make_window(window_type, kernel_length)?;

        // Generate complex exponential kernel
        for (n, w) in window.iter().enumerate().take(kernel_length) {
            let t = n as f64 / sample_rate;
            let phase = 2.0 * PI * center_freq * t;

            // Complex exponential: e^(i*2*π*f*t)
            let exponential = Complex::new(phase.cos(), phase.sin());

            // Apply window function
            let windowed = exponential * w;

            kernel.push(windowed);
        }

        Ok(kernel)
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
    fn normalize_kernel(kernel: &mut [Complex<f64>]) {
        let energy: f64 = kernel.iter().map(num_complex::Complex::norm_sqr).sum();

        if energy > 0.0 {
            let norm_factor = 1.0 / energy.sqrt();
            for coefficient in kernel.iter_mut() {
                *coefficient *= norm_factor;
            }
        }
    }

    /// Get the frequencies for each bin.
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Get the number of bins.
    pub const fn num_bins(&self) -> usize {
        self.kernels.len()
    }

    /// Apply the CQT kernel to input samples.
    ///
    /// This performs the CQT transform by convolving each kernel with the input signal.
    /// Uses time-domain convolution for efficiency with sparse kernels.
    pub fn apply(&self, samples: &[f64]) -> SpectrogramResult<Vec<Complex<f64>>> {
        let mut cqt_result = Vec::with_capacity(self.kernels.len());

        // For each frequency bin
        for (bin_idx, kernel) in self.kernels.iter().enumerate() {
            if kernel.is_empty() {
                cqt_result.push(Complex::new(0.0, 0.0));
                continue;
            }

            let kernel_length = self.kernel_lengths[bin_idx];

            // Time-domain correlation (more efficient for sparse kernels)
            let mut correlation = Complex::new(0.0, 0.0);
            let start_idx = samples.len().saturating_sub(kernel_length);

            for (k_idx, &k) in kernel.iter().enumerate() {
                let sample_idx = start_idx + k_idx;
                if sample_idx < samples.len() {
                    let sample = samples[sample_idx];
                    // Conjugate multiplication for correlation
                    correlation += k.conj() * sample;
                }
            }

            cqt_result.push(correlation);
        }

        Ok(cqt_result)
    }
}

/// CQT result containing complex frequency bins and metadata.
#[derive(Debug, Clone)]
pub struct CqtResult {
    /// Complex CQT coefficients with shape (`frequency_bins`, `time_frames`)
    pub data: Array2<Complex<f64>>,
    /// Center frequency for each bin in Hz
    pub frequencies: Vec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Hop size in samples
    pub hop_size: usize,
}

impl AsRef<Array2<Complex<f64>>> for CqtResult {
    fn as_ref(&self) -> &Array2<Complex<f64>> {
        &self.data
    }
}

impl Deref for CqtResult {
    type Target = Array2<Complex<f64>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl CqtResult {
    /// Get the number of frequency bins.
    #[must_use] 
    pub fn n_bins(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of time frames.
    #[must_use] 
    pub fn n_frames(&self) -> usize {
        self.data.ncols()
    }

    /// Get the time resolution in seconds.
    #[must_use] 
    pub fn time_resolution(&self) -> f64 {
        self.hop_size as f64 / self.sample_rate
    }

    /// Convert to magnitude spectrogram.
    #[must_use] 
    pub fn to_magnitude(&self) -> Array2<f64> {
        let mut magnitude = Array2::<f64>::zeros(self.data.dim());
        for ((i, j), val) in self.data.indexed_iter() {
            magnitude[[i, j]] = val.norm();
        }
        magnitude
    }

    /// Convert to power spectrogram.
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
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let samples = vec![0.0; 16000];
/// let params = CqtParams::new(12, 7, 32.7)?; // 12 bins/octave, 7 octaves, from C1
///
/// let cqt_result = cqt(&samples, 16000.0, &params, 512)?;
///
/// println!("CQT: {} bins x {} frames", cqt_result.n_bins(), cqt_result.n_frames());
///
/// // Get magnitude spectrogram
/// let magnitude = cqt_result.to_magnitude();
/// # Ok(())
/// # }
/// ```
pub fn cqt<S: AsRef<[f64]>>(
    samples: S,
    sample_rate: f64,
    params: &CqtParams,
    hop_size: usize,
) -> SpectrogramResult<CqtResult> {
    let samples = samples.as_ref();
    if samples.is_empty() {
        return Err(SpectrogramError::invalid_input("samples must not be empty"));
    }
    if hop_size == 0 {
        return Err(SpectrogramError::invalid_input("hop_size must be > 0"));
    }

    // Generate CQT kernels
    // Use a reasonable signal length for kernel generation (we'll apply to frames)
    let kernel_length = samples.len().min(16384);
    let kernel = CqtKernel::generate(params, sample_rate, kernel_length)?;

    let n_bins = kernel.num_bins();
    let frequencies = kernel.frequencies().to_vec();

    // Compute number of frames
    let n_frames = if samples.len() < kernel_length {
        1
    } else {
        (samples.len() - kernel_length) / hop_size + 1
    };

    // Allocate output matrix
    let mut cqt_data = Array2::<Complex<f64>>::zeros((n_bins, n_frames));

    // Process each frame
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let end = (start + kernel_length).min(samples.len());

        if end <= start {
            break;
        }

        let frame = &samples[start..end];

        // Apply CQT kernel to frame
        let cqt_frame = kernel.apply(frame)?;

        // Store in output matrix
        for (bin_idx, &val) in cqt_frame.iter().enumerate() {
            if bin_idx < n_bins {
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
