use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

use ndarray::{Array1, Array2};
use num_complex::Complex;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{
    CqtParams, ErbParams, R2cPlan, SpectrogramError, SpectrogramResult, WindowType,
    min_max_single_pass,
};

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
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &SpectrogramParams {
        &self.params
    }

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
    pub fn compute(
        &mut self,
        samples: &[f64],
    ) -> SpectrogramResult<Spectrogram<FreqScale, AmpScale>> {
        if samples.is_empty() {
            return Err(SpectrogramError::invalid_input("samples must not be empty"));
        }

        let n_frames = self.stft.frame_count(samples.len())?;
        let n_bins = self.mapping.output_bins();

        // Create output matrix: (n_bins, n_frames)
        let mut data = Array2::<f64>::zeros((n_bins, n_frames));

        // Ensure workspace is correctly sized
        self.workspace
            .ensure_sizes(self.stft.n_fft, self.stft.out_len, n_bins);

        // Main loop: fill each frame (column)
        // TODO: Parallelize with rayon once thread-safety issues are resolved
        for frame_idx in 0..n_frames {
            self.stft
                .compute_frame_spectrum(samples, frame_idx, &mut self.workspace)?;

            // mapping: spectrum(out_len) -> mapped(n_bins)
            // For CQT, this uses workspace.frame; for others, workspace.spectrum
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

        let times = build_time_axis_seconds(&self.params, n_frames)?;
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
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = vec![0.0; 16000];
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let mut plan = planner.linear_plan::<Power>(&params, None)?;
    ///
    /// // Compute just the first frame
    /// let frame = plan.compute_frame(&samples, 0)?;
    /// assert_eq!(frame.len(), 257); // n_fft/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_frame<S: AsRef<[f64]>>(
        &mut self,
        samples: S,
        frame_idx: usize,
    ) -> SpectrogramResult<Vec<f64>> {
        let samples = samples.as_ref();
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
    /// * `output` - Pre-allocated output matrix (must be correct size: `n_bins` × `n_frames`)
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
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = vec![0.0; 16000];
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
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
    pub fn compute_into<S: AsRef<[f64]>>(
        &mut self,
        samples: S,
        output: &mut Array2<f64>,
    ) -> SpectrogramResult<()> {
        let samples = samples.as_ref();

        if samples.is_empty() {
            return Err(SpectrogramError::invalid_input("samples must not be empty"));
        }

        let n_frames = self.stft.frame_count(samples.len())?;
        let n_bins = self.mapping.output_bins();

        // Validate output dimensions
        if output.nrows() != n_bins {
            return Err(SpectrogramError::dimension_mismatch(n_bins, output.nrows()));
        }
        if output.ncols() != n_frames {
            return Err(SpectrogramError::dimension_mismatch(
                n_frames,
                output.ncols(),
            ));
        }

        // Ensure workspace is correctly sized
        self.workspace
            .ensure_sizes(self.stft.n_fft, self.stft.out_len, n_bins);

        // Main loop: fill each frame (column)
        for frame_idx in 0..n_frames {
            self.stft
                .compute_frame_spectrum(samples, frame_idx, &mut self.workspace)?;

            // mapping: spectrum(out_len) -> mapped(n_bins)
            // For CQT, this uses workspace.frame; for others, workspace.spectrum
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
    /// Returns (`n_bins`, `n_frames`) for the spectrogram that would be computed
    /// from a signal of the given length.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let plan = planner.linear_plan::<Power>(&params, None)?;
    ///
    /// let (n_bins, n_frames) = plan.output_shape(16000)?;
    /// assert_eq!(n_bins, 257);
    /// assert_eq!(n_frames, 63);
    /// # Ok(())
    /// # }
    /// ```
    pub fn output_shape(&self, signal_length: usize) -> SpectrogramResult<(usize, usize)> {
        let n_frames = self.stft.frame_count(signal_length)?;
        let n_bins = self.mapping.output_bins();
        Ok((n_bins, n_frames))
    }
}

/// STFT (Short-Time Fourier Transform) result containing complex frequency bins.
///
/// This is the raw STFT output before any frequency mapping or amplitude scaling.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Complex STFT matrix with shape (`frequency_bins`, `time_frames`)
    pub data: Array2<Complex<f64>>,
    /// Frequency axis in Hz
    pub frequencies: Vec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    pub params: StftParams,
}

impl StftResult {
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

    /// Get the frequency resolution in Hz.
    #[must_use]
    pub fn frequency_resolution(&self) -> f64 {
        self.sample_rate / self.params.n_fft() as f64
    }

    /// Get the time resolution in seconds.
    #[must_use]
    pub fn time_resolution(&self) -> f64 {
        self.params.hop_size() as f64 / self.sample_rate
    }

    /// Normalizes self.data to remove the complex aspect of it.
    pub fn norm(&self) -> Array2<f64> {
        self.as_ref().mapv(Complex::norm)
    }
}

impl AsRef<Array2<Complex<f64>>> for StftResult {
    fn as_ref(&self) -> &Array2<Complex<f64>> {
        &self.data
    }
}

impl AsMut<Array2<Complex<f64>>> for StftResult {
    fn as_mut(&mut self) -> &mut Array2<Complex<f64>> {
        &mut self.data
    }
}

impl Deref for StftResult {
    type Target = Array2<Complex<f64>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for StftResult {
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
pub struct SpectrogramPlanner;

impl SpectrogramPlanner {
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
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = vec![0.0; 16000];
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let stft_result = planner.compute_stft(&samples, &params)?;
    ///
    /// println!("STFT: {} bins x {} frames", stft_result.n_bins(), stft_result.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_stft<S: AsRef<[f64]>>(
        &self,
        samples: S,
        params: &SpectrogramParams,
    ) -> SpectrogramResult<StftResult> {
        let samples = samples.as_ref();
        if samples.is_empty() {
            return Err(SpectrogramError::invalid_input("samples must not be empty"));
        }

        let mut stft_plan = StftPlan::new(params)?;
        let n_frames = stft_plan.frame_count(samples.len())?;
        let n_bins = stft_plan.out_len;

        // Allocate output matrix
        let mut data = Array2::<Complex<f64>>::zeros((n_bins, n_frames));
        let mut workspace = Workspace::new(stft_plan.n_fft, n_bins, n_bins);

        // Compute each frame
        for frame_idx in 0..n_frames {
            // Fill frame and compute FFT
            stft_plan.compute_frame_fft(samples, frame_idx, &mut workspace)?;

            // Copy FFT output to result matrix
            for (bin_idx, &value) in stft_plan.fft_out.iter().enumerate() {
                data[[bin_idx, frame_idx]] = value;
            }
        }

        // Build frequency axis
        let frequencies = (0..n_bins)
            .map(|k| k as f64 * params.sample_rate_hz() / params.stft().n_fft() as f64)
            .collect();

        let stft_params = params.stft();
        Ok(StftResult {
            data,
            frequencies,
            sample_rate: params.sample_rate_hz(),
            params: *stft_params,
        })
    }

    /// Compute the power spectrum of a single audio frame.
    ///
    /// This is useful for real-time processing or analyzing individual frames.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio frame (length should be `n_fft`)
    /// * `n_fft` - FFT size
    /// * `window` - Window type to apply
    ///
    /// # Returns
    ///
    /// A vector of power values (|X|²) with length `n_fft/2` + 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let frame = vec![0.0; 512];
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let power = planner.compute_power_spectrum(&frame, 512, WindowType::Hanning)?;
    ///
    /// assert_eq!(power.len(), 257); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_power_spectrum<S: AsRef<[f64]>>(
        &self,
        samples: S,
        n_fft: usize,
        window: WindowType,
    ) -> SpectrogramResult<Vec<f64>> {
        let samples = samples.as_ref();
        if samples.len() != n_fft {
            return Err(SpectrogramError::dimension_mismatch(n_fft, samples.len()));
        }

        let window_samples = make_window(window, n_fft)?;
        let out_len = r2c_output_size(n_fft);

        // Create FFT plan
        #[cfg(feature = "realfft")]
        let mut fft = {
            let mut planner = crate::RealFftPlanner::new();
            let plan = planner.get_or_create(n_fft);
            crate::RealFftPlan::new(n_fft, plan)
        };

        #[cfg(feature = "fftw")]
        let mut fft = {
            use std::sync::Arc;
            let plan = crate::FftwPlanner::build_plan(n_fft)?;
            crate::FftwPlan::new(Arc::new(plan))
        };

        // Apply window and compute FFT
        let mut windowed = vec![0.0; n_fft];
        for (i, &s) in samples.iter().enumerate() {
            windowed[i] = s * window_samples[i];
        }

        let mut fft_out = vec![Complex::new(0.0, 0.0); out_len];
        fft.process(&windowed, &mut fft_out)?;

        // Convert to power
        let power: Vec<f64> = fft_out.iter().map(num_complex::Complex::norm_sqr).collect();

        Ok(power)
    }

    /// Compute the magnitude spectrum of a single audio frame.
    ///
    /// This is useful for real-time processing or analyzing individual frames.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio frame (length should be `n_fft`)
    /// * `n_fft` - FFT size
    /// * `window` - Window type to apply
    ///
    /// # Returns
    ///
    /// A vector of magnitude values (|X|) with length `n_fft/2` + 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let frame = vec![0.0; 512];
    ///
    /// let planner = SpectrogramPlanner::new();
    /// let magnitude = planner.compute_magnitude_spectrum(&frame, 512, WindowType::Hanning)?;
    ///
    /// assert_eq!(magnitude.len(), 257); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_magnitude_spectrum<S: AsRef<[f64]>>(
        &self,
        samples: S,
        n_fft: usize,
        window: WindowType,
    ) -> SpectrogramResult<Vec<f64>> {
        let power = self.compute_power_spectrum(samples, n_fft, window)?;
        Ok(power.iter().map(|&p| p.sqrt()).collect())
    }

    /// Build a linear-frequency spectrogram plan.
    ///
    /// `AmpScale` determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
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
        let scaling = AmplitudeScaling::<AmpScale>::new(db)?;
        let freq_axis = build_frequency_axis::<LinearHz>(params, &mapping)?;

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
        let scaling = AmplitudeScaling::<AmpScale>::new(db)?;
        let freq_axis = build_frequency_axis::<Mel>(params, &mapping)?;

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
        let scaling = AmplitudeScaling::<AmpScale>::new(db)?;
        let freq_axis = build_frequency_axis::<Erb>(params, &mapping)?;

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
        let scaling = AmplitudeScaling::<AmpScale>::new(db)?;
        let freq_axis = build_frequency_axis::<LogHz>(params, &mapping)?;

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
    /// `AmpScale` determines whether output is:
    /// - Magnitude
    /// - Power
    /// - Decibels
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
        let scaling = AmplitudeScaling::<AmpScale>::new(db)?;
        let freq_axis = build_frequency_axis::<Cqt>(params, &mapping)?;

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

struct StftPlan {
    n_fft: usize,
    hop: usize,
    window: Vec<f64>,
    centre: bool,

    out_len: usize,

    // FFT plan (reused for all frames)
    fft: Box<dyn R2cPlan>,

    // internal scratch
    fft_out: Vec<Complex<f64>>,
    frame: Vec<f64>,
}

impl StftPlan {
    fn new(params: &SpectrogramParams) -> SpectrogramResult<Self> {
        let stft = params.stft();
        let n_fft = stft.n_fft();
        let hop = stft.hop_size();
        let centre = stft.centre();

        let window = make_window(stft.window(), n_fft)?;

        let out_len = r2c_output_size(n_fft);

        #[cfg(feature = "realfft")]
        let fft = {
            let mut planner = crate::RealFftPlanner::new();
            let plan = planner.get_or_create(n_fft);
            let plan = crate::RealFftPlan::new(n_fft, plan);
            Box::new(plan)
        };

        #[cfg(feature = "fftw")]
        let fft = {
            use std::sync::Arc;
            let plan = crate::FftwPlanner::build_plan(n_fft)?;
            Box::new(crate::FftwPlan::new(Arc::new(plan)))
        };

        Ok(Self {
            n_fft,
            hop,
            window,
            centre,
            out_len,
            fft,
            fft_out: vec![Complex::new(0.0, 0.0); out_len],
            frame: vec![0.0; n_fft],
        })
    }

    fn frame_count(&self, n_samples: usize) -> SpectrogramResult<usize> {
        if n_samples == 0 {
            return Err(SpectrogramError::invalid_input("n_samples must be > 0"));
        }

        // Framing policy:
        // - centre = true: implicit padding of n_fft/2 on both sides
        // - centre = false: no padding
        //
        // Define the number of frames such that each frame has a valid centre sample position.
        let pad = if self.centre { self.n_fft / 2 } else { 0 };
        let padded_len = n_samples + 2 * pad;

        if padded_len < self.n_fft {
            // still produce 1 frame (all padding / partial)
            return Ok(1);
        }

        let remaining = padded_len - self.n_fft;
        let n_frames = remaining / self.hop + 1;
        Ok(n_frames)
    }

    /// Compute one frame FFT into internal buffers.
    /// This variant is for computing raw STFT (keeps complex values).
    fn compute_frame_fft(
        &mut self,
        samples: &[f64],
        frame_idx: usize,
        _workspace: &mut Workspace,
    ) -> SpectrogramResult<()> {
        let out = self.frame.as_mut_slice();
        debug_assert_eq!(out.len(), self.n_fft);

        let pad = if self.centre { self.n_fft / 2 } else { 0 };
        let start = frame_idx
            .checked_mul(self.hop)
            .ok_or_else(|| SpectrogramError::invalid_input("frame index overflow"))?;

        // Fill windowed frame
        for i in 0..self.n_fft {
            let v_idx = start + i;
            let s_idx = v_idx as isize - pad as isize;

            let sample = if s_idx < 0 || (s_idx as usize) >= samples.len() {
                0.0
            } else {
                samples[s_idx as usize]
            };

            out[i] = sample * self.window[i];
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
        samples: &[f64],
        frame_idx: usize,
        workspace: &mut Workspace,
    ) -> SpectrogramResult<()> {
        let out = workspace.frame.as_mut_slice();

        // self.fill_frame(samples, frame_idx, frame)?;
        debug_assert_eq!(out.len(), self.n_fft);

        let pad = if self.centre { self.n_fft / 2 } else { 0 };
        let start = frame_idx
            .checked_mul(self.hop)
            .ok_or_else(|| SpectrogramError::invalid_input("frame index overflow"))?;

        // The "virtual" signal is samples with pad zeros on both sides.
        // Virtual index 0..padded_len
        // Map virtual index to original samples by subtracting pad.
        for i in 0..self.n_fft {
            let v_idx = start + i;
            let s_idx = v_idx as isize - pad as isize;

            let sample = if s_idx < 0 || (s_idx as usize) >= samples.len() {
                0.0
            } else {
                samples[s_idx as usize]
            };

            out[i] = sample * self.window[i];
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
}

#[derive(Debug, Clone)]
enum MappingKind {
    Identity {
        out_len: usize,
    },
    Mel {
        matrix: Array2<f64>,
    }, // shape: (n_mels, out_len)
    LogHz {
        matrix: Array2<f64>,
        frequencies: Vec<f64>,
    }, // shape: (n_bins, out_len)
    Erb {
        filterbank: crate::erb::ErbFilterbank,
        sample_rate: f64,
        n_fft: usize,
    },
    Cqt {
        kernel: crate::cqt::CqtKernel,
    },
}

/// Typed mapping wrapper.
#[derive(Debug, Clone)]
struct FrequencyMapping<FreqScale> {
    kind: MappingKind,
    _marker: PhantomData<FreqScale>,
}

impl FrequencyMapping<LinearHz> {
    const fn new(params: &SpectrogramParams) -> SpectrogramResult<Self> {
        let out_len = r2c_output_size(params.stft().n_fft());
        Ok(Self {
            kind: MappingKind::Identity { out_len },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<Mel> {
    fn new_mel(params: &SpectrogramParams, mel: &MelParams) -> SpectrogramResult<Self> {
        let n_fft = params.stft().n_fft();
        let out_len = r2c_output_size(n_fft);

        // Validate: mel bins must be <= something sensible
        if mel.n_mels() > 10_000 {
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
        )?;

        // matrix must be (n_mels, out_len)
        if matrix.nrows() != mel.n_mels() || matrix.ncols() != out_len {
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
        let out_len = r2c_output_size(n_fft);

        // Validate: n_bins must be <= something sensible
        if loghz.n_bins() > 10_000 {
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
        if matrix.nrows() != loghz.n_bins() || matrix.ncols() != out_len {
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
        if erb.n_filters() > 10_000 {
            return Err(SpectrogramError::invalid_input(
                "n_filters is unreasonably large",
            ));
        }

        // Generate ERB filterbank
        let filterbank = crate::erb::ErbFilterbank::generate(erb, sample_rate)?;

        Ok(Self {
            kind: MappingKind::Erb {
                filterbank,
                sample_rate,
                n_fft,
            },
            _marker: PhantomData,
        })
    }
}

impl FrequencyMapping<Cqt> {
    fn new(params: &SpectrogramParams, cqt: &CqtParams) -> SpectrogramResult<Self> {
        let sample_rate = params.sample_rate_hz();
        let n_fft = params.stft().n_fft();

        // Validate that frequency range is reasonable
        let f_max = cqt.bin_frequency(cqt.num_bins().saturating_sub(1));
        if f_max >= sample_rate / 2.0 {
            return Err(SpectrogramError::invalid_input(
                "CQT maximum frequency must be below Nyquist frequency",
            ));
        }

        // Generate CQT kernel using n_fft as the signal length for kernel generation
        let kernel = crate::cqt::CqtKernel::generate(cqt, sample_rate, n_fft)?;

        Ok(Self {
            kind: MappingKind::Cqt { kernel },
            _marker: PhantomData,
        })
    }
}

impl<FreqScale> FrequencyMapping<FreqScale> {
    fn output_bins(&self) -> usize {
        match &self.kind {
            MappingKind::Identity { out_len } => *out_len,
            MappingKind::Mel { matrix } => matrix.nrows(),
            MappingKind::LogHz { matrix, .. } => matrix.nrows(),
            MappingKind::Erb { filterbank, .. } => filterbank.num_filters(),
            MappingKind::Cqt { kernel, .. } => kernel.num_bins(),
        }
    }

    fn apply(&self, spectrum: &[f64], frame: &[f64], out: &mut [f64]) -> SpectrogramResult<()> {
        match &self.kind {
            MappingKind::Identity { out_len } => {
                if spectrum.len() != *out_len {
                    return Err(SpectrogramError::dimension_mismatch(
                        *out_len,
                        spectrum.len(),
                    ));
                }
                if out.len() != *out_len {
                    return Err(SpectrogramError::dimension_mismatch(*out_len, out.len()));
                }
                out.copy_from_slice(spectrum);
                Ok(())
            }
            MappingKind::Mel { matrix } => {
                let out_bins = matrix.nrows();
                let in_bins = matrix.ncols();

                if spectrum.len() != in_bins {
                    return Err(SpectrogramError::dimension_mismatch(
                        in_bins,
                        spectrum.len(),
                    ));
                }
                if out.len() != out_bins {
                    return Err(SpectrogramError::dimension_mismatch(out_bins, out.len()));
                }

                // out = matrix * spectrum
                // matrix: (out_bins, in_bins)
                for r in 0..out_bins {
                    let mut acc = 0.0;
                    for c in 0..in_bins {
                        acc += matrix[[r, c]] * spectrum[c];
                    }
                    out[r] = acc;
                }
                Ok(())
            }
            MappingKind::LogHz { matrix, .. } => {
                let out_bins = matrix.nrows();
                let in_bins = matrix.ncols();

                if spectrum.len() != in_bins {
                    return Err(SpectrogramError::dimension_mismatch(
                        in_bins,
                        spectrum.len(),
                    ));
                }
                if out.len() != out_bins {
                    return Err(SpectrogramError::dimension_mismatch(out_bins, out.len()));
                }

                // out = matrix * spectrum
                // matrix: (out_bins, in_bins)
                for r in 0..out_bins {
                    let mut acc = 0.0;
                    for c in 0..in_bins {
                        acc += matrix[[r, c]] * spectrum[c];
                    }
                    out[r] = acc;
                }
                Ok(())
            }
            MappingKind::Erb {
                filterbank,
                sample_rate,
                n_fft,
            } => {
                // Apply ERB filterbank to power spectrum
                let erb_out = filterbank.apply_to_spectrum(spectrum, *sample_rate, *n_fft)?;

                if out.len() != erb_out.len() {
                    return Err(SpectrogramError::dimension_mismatch(
                        erb_out.len(),
                        out.len(),
                    ));
                }

                out.copy_from_slice(&erb_out);
                Ok(())
            }
            MappingKind::Cqt { kernel } => {
                // CQT works on time-domain windowed frame, not FFT spectrum
                // Apply CQT kernel to get complex coefficients
                let cqt_complex = kernel.apply(frame)?;

                if out.len() != cqt_complex.len() {
                    return Err(SpectrogramError::dimension_mismatch(
                        cqt_complex.len(),
                        out.len(),
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

    fn frequencies_hz(&self, params: &SpectrogramParams) -> SpectrogramResult<Vec<f64>> {
        match &self.kind {
            MappingKind::Identity { out_len } => {
                // Standard R2C bins: k * sr / n_fft
                let n_fft = params.stft().n_fft() as f64;
                let sr = params.sample_rate_hz();
                let df = sr / n_fft;

                let mut f = Vec::with_capacity(*out_len);
                for k in 0..*out_len {
                    f.push(k as f64 * df);
                }
                Ok(f)
            }
            MappingKind::Mel { matrix } => {
                // For mel, the axis is defined by the mel band centre frequencies.
                // We compute and store them consistently with how we built the filterbank.
                let n_mels = matrix.nrows();
                Ok(mel_band_centres_hz(
                    n_mels,
                    params.sample_rate_hz(),
                    params.nyquist_hz(),
                ))
            }
            MappingKind::LogHz { frequencies, .. } => {
                // Frequencies are stored when the mapping is created
                Ok(frequencies.clone())
            }
            MappingKind::Erb { filterbank, .. } => {
                // ERB center frequencies
                Ok(filterbank.center_frequencies().to_vec())
            }
            MappingKind::Cqt { kernel, .. } => {
                // CQT center frequencies from the kernel
                Ok(kernel.frequencies().to_vec())
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
pub trait AmpScaleSpec: Sized {
    fn apply_from_power(power: f64) -> f64;
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
        // Convert power -> dB: 10*log10(power + eps)
        // then floor at floor_db
        if !floor_db.is_finite() {
            return Err(SpectrogramError::invalid_input("floor_db must be finite"));
        }

        const EPS: f64 = 1e-12;
        for v in x.iter_mut() {
            let db = 10.0 * (*v + EPS).log10();
            *v = if db < floor_db { floor_db } else { db };
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AmplitudeScaling<AmpScale> {
    db_floor: Option<f64>,
    _marker: PhantomData<AmpScale>,
}

impl<AmpScale> AmplitudeScaling<AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    fn new(db: Option<&LogParams>) -> SpectrogramResult<Self> {
        let db_floor = db.map(LogParams::floor_db);
        Ok(Self {
            db_floor,
            _marker: PhantomData,
        })
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
    spectrum: Vec<f64>,         // out_len (power spectrum)
    mapped: Vec<f64>,           // n_bins (after mapping)
    frame: Vec<f64>,            // n_fft (windowed frame for FFT)
    fft_out: Vec<Complex<f64>>, // out_len (FFT output)
}

impl Workspace {
    fn new(n_fft: usize, out_len: usize, n_bins: usize) -> Self {
        Self {
            spectrum: vec![0.0; out_len],
            mapped: vec![0.0; n_bins],
            frame: vec![0.0; n_fft],
            fft_out: vec![Complex::new(0.0, 0.0); out_len],
        }
    }

    fn ensure_sizes(&mut self, n_fft: usize, out_len: usize, n_bins: usize) {
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
) -> SpectrogramResult<FrequencyAxis<FreqScale>>
where
    FreqScale: Copy + Clone + 'static,
{
    let frequencies = mapping.frequencies_hz(params)?;
    Ok(FrequencyAxis::new(frequencies))
}

fn build_time_axis_seconds(
    params: &SpectrogramParams,
    n_frames: usize,
) -> SpectrogramResult<Vec<f64>> {
    if n_frames == 0 {
        return Err(SpectrogramError::invalid_input("n_frames must be > 0"));
    }

    let dt = params.frame_period_seconds();
    let mut times = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        times.push(i as f64 * dt);
    }

    Ok(times)
}

pub fn make_window(window: WindowType, n_fft: usize) -> SpectrogramResult<Vec<f64>> {
    if n_fft == 0 {
        return Err(SpectrogramError::invalid_input("n_fft must be > 0"));
    }

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

    Ok(w)
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filterbank_matrix(
    sample_rate_hz: f64,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
) -> SpectrogramResult<Array2<f64>> {
    if sample_rate_hz <= 0.0 || !sample_rate_hz.is_finite() {
        return Err(SpectrogramError::invalid_input(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    if n_fft == 0 {
        return Err(SpectrogramError::invalid_input("n_fft must be > 0"));
    }
    if n_mels == 0 {
        return Err(SpectrogramError::invalid_input("n_mels must be > 0"));
    }
    if !(f_min >= 0.0) {
        return Err(SpectrogramError::invalid_input("f_min must be >= 0"));
    }
    if !(f_max > f_min) {
        return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
    }
    if f_max > sample_rate_hz * 0.5 {
        return Err(SpectrogramError::invalid_input("f_max must be <= Nyquist"));
    }

    let out_len = r2c_output_size(n_fft);

    // FFT bin frequencies
    let df = sample_rate_hz / n_fft as f64;
    let mut fft_freqs = Vec::with_capacity(out_len);
    for k in 0..out_len {
        fft_freqs.push(k as f64 * df);
    }

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
    for m in mel_points {
        hz_points.push(mel_to_hz(m));
    }

    // Convert Hz points into FFT bin indices
    let mut bin_points = Vec::with_capacity(n_points);
    for hz in hz_points {
        let b = (hz / df).floor() as isize;
        let b = b.clamp(0, (out_len - 1) as isize);
        bin_points.push(b as usize);
    }

    // Build filterbank
    let mut fb = Array2::<f64>::zeros((n_mels, out_len));

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
            fb[[m, k]] = v;
        }

        // Falling slope: centre -> right
        for k in centre..right {
            let v = (right - k) as f64 / (right - centre) as f64;
            fb[[m, k]] = v;
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
    n_fft: usize,
    n_bins: usize,
    f_min: f64,
    f_max: f64,
) -> SpectrogramResult<(Array2<f64>, Vec<f64>)> {
    if sample_rate_hz <= 0.0 || !sample_rate_hz.is_finite() {
        return Err(SpectrogramError::invalid_input(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    if n_fft == 0 {
        return Err(SpectrogramError::invalid_input("n_fft must be > 0"));
    }
    if n_bins == 0 {
        return Err(SpectrogramError::invalid_input("n_bins must be > 0"));
    }
    if !(f_min > 0.0 && f_min.is_finite()) {
        return Err(SpectrogramError::invalid_input(
            "f_min must be finite and > 0",
        ));
    }
    if !(f_max > f_min) {
        return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
    }
    if f_max > sample_rate_hz * 0.5 {
        return Err(SpectrogramError::invalid_input("f_max must be <= Nyquist"));
    }

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

    // Build interpolation matrix
    let mut matrix = Array2::<f64>::zeros((n_bins, out_len));

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
            matrix[[bin_idx, lower_bin]] = 1.0;
        } else {
            // Linear interpolation
            let frac = exact_bin - lower_bin as f64;
            matrix[[bin_idx, lower_bin]] = 1.0 - frac;
            if upper_bin < out_len {
                matrix[[bin_idx, upper_bin]] = frac;
            }
        }
    }

    Ok((matrix, log_frequencies))
}

/// A pragmatic, stable mel-centre axis helper.
/// If you want exact centres consistent with your filterbank construction,
/// you can instead compute centres from the mel points used to build the bank.
/// For now, this returns a monotonic axis in Hz with mel spacing.
fn mel_band_centres_hz(n_mels: usize, sample_rate_hz: f64, nyquist_hz: f64) -> Vec<f64> {
    let f_min = 0.0;
    let f_max = nyquist_hz.min(sample_rate_hz * 0.5);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let step = (mel_max - mel_min) / (n_mels + 1) as f64;

    let mut centres = Vec::with_capacity(n_mels);
    for i in 0..n_mels {
        let mel = (i as f64 + 1.0).mul_add(step, mel_min);
        centres.push(mel_to_hz(mel));
    }
    centres
}

#[derive(Debug, Clone)]
pub struct Spectrogram<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    data: Array2<f64>,
    axes: Axes<FreqScale>,
    params: SpectrogramParams,
    _amp: PhantomData<AmpScale>,
}

impl<FreqScale, AmpScale> Spectrogram<FreqScale, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
    FreqScale: Copy + Clone + 'static,
{
    #[must_use]
    pub const fn x_axis_label() -> &'static str {
        "Time (s)"
    }

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
        debug_assert_eq!(data.nrows(), axes.frequencies().len());
        debug_assert_eq!(data.ncols(), axes.times().len());

        Self {
            data,
            axes,
            params,
            _amp: PhantomData,
        }
    }

    #[inline]
    pub fn set_data(&mut self, data: Array2<f64>) {
        self.data = data;
    }

    #[inline]
    #[must_use]
    pub const fn data(&self) -> &Array2<f64> {
        &self.data
    }

    #[inline]
    #[must_use]
    pub const fn axes(&self) -> &Axes<FreqScale> {
        &self.axes
    }

    #[inline]
    #[must_use]
    pub const fn frequencies(&self) -> &[f64] {
        self.axes.frequencies()
    }

    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        self.axes.frequency_range()
    }

    #[inline]
    #[must_use]
    pub const fn times(&self) -> &[f64] {
        self.axes.times()
    }

    #[inline]
    #[must_use]
    pub const fn params(&self) -> &SpectrogramParams {
        &self.params
    }

    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.axes.duration()
    }

    /// If this is a dB spectrogram, return the (min, max) dB values. otherwise do the maths
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

            const EPS: f64 = 1e-12;

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

            const EPS: f64 = 1e-12;

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

    #[inline]
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.data.nrows()
    }

    #[inline]
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.data.ncols()
    }
}

impl AsRef<Array2<f64>> for Spectrogram<LinearHz, Power> {
    fn as_ref(&self) -> &Array2<f64> {
        &self.data
    }
}

impl Deref for Spectrogram<LinearHz, Power> {
    type Target = Array2<f64>;

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
    ///
    /// # fn example() -> SpectrogramResult<()> {
    /// // Create a simple test signal
    /// let sample_rate = 16000.0;
    /// let samples: Vec<f64> = (0..16000).map(|i| {
    ///     (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin()
    /// }).collect();
    ///
    /// // Set up parameters
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, sample_rate)?;
    ///
    /// // Compute power spectrogram
    /// let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;
    ///
    /// println!("Computed spectrogram: {} bins × {} frames", spec.n_bins(), spec.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute<S: AsRef<[f64]>>(
        samples: S,
        params: &SpectrogramParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.linear_plan(params, db)?;
        plan.compute(samples.as_ref())
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
    ///
    /// # fn example() -> SpectrogramResult<()> {
    /// // Create a simple test signal
    /// let sample_rate = 16000.0;
    /// let samples: Vec<f64> = (0..16000).map(|i| {
    ///     (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin()
    /// }).collect();
    ///
    /// // Set up parameters
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, sample_rate)?;
    /// let mel = MelParams::new(80, 0.0, 8000.0)?;
    ///
    /// // Compute mel spectrogram in dB scale
    /// let db = LogParams::new(-80.0)?;
    /// let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db))?;
    ///
    /// println!("Computed mel spectrogram: {} mels × {} frames", spec.n_bins(), spec.n_frames());
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute<S: AsRef<[f64]>>(
        samples: S,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.mel_plan(params, mel, db)?;
        plan.compute(samples.as_ref())
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
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = vec![0.0; 16000];
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let erb = ErbParams::speech_standard()?;
    ///
    /// let spec = ErbPowerSpectrogram::compute(&samples, &params, &erb, None)?;
    /// assert_eq!(spec.n_bins(), 40);
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute<S: AsRef<[f64]>>(
        samples: S,
        params: &SpectrogramParams,
        erb: &ErbParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.erb_plan(params, erb, db)?;
        plan.compute(samples.as_ref())
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
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = vec![0.0; 16000];
    /// let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    /// let params = SpectrogramParams::new(stft, 16000.0)?;
    /// let loghz = LogHzParams::new(128, 20.0, 8000.0)?;
    ///
    /// let spec = LogHzPowerSpectrogram::compute(&samples, &params, &loghz, None)?;
    /// assert_eq!(spec.n_bins(), 128);
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute<S: AsRef<[f64]>>(
        samples: S,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.log_hz_plan(params, loghz, db)?;
        plan.compute(samples.as_ref())
    }
}

impl<AmpScale> Spectrogram<Cqt, AmpScale>
where
    AmpScale: AmpScaleSpec + 'static,
{
    pub fn compute<S: AsRef<[f64]>>(
        samples: S,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: Option<&LogParams>,
    ) -> SpectrogramResult<Self> {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner.cqt_plan(params, cqt, db)?;
        plan.compute(samples.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct FrequencyAxis<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    frequencies: Vec<f64>,
    _marker: PhantomData<FreqScale>,
}

impl<FreqScale> FrequencyAxis<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    pub(crate) fn new(frequencies: Vec<f64>) -> Self {
        debug_assert!(!frequencies.is_empty());
        Self {
            frequencies,
            _marker: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub const fn frequencies(&self) -> &[f64] {
        self.frequencies.as_slice()
    }

    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        let data = self.frequencies.as_slice();
        let min = data[0];
        let max_idx = data.len().saturating_sub(1); // safe for non-empty
        let max = data[max_idx];
        (min, max)
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frequencies.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct Axes<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    freq: FrequencyAxis<FreqScale>,
    times: Vec<f64>,
}

impl<FreqScale> Axes<FreqScale>
where
    FreqScale: Copy + Clone + 'static,
{
    pub(crate) const fn new(freq: FrequencyAxis<FreqScale>, times: Vec<f64>) -> Self {
        debug_assert!(!times.is_empty());
        Self { freq, times }
    }

    #[inline]
    #[must_use]
    pub const fn frequencies(&self) -> &[f64] {
        self.freq.frequencies()
    }

    #[inline]
    #[must_use]
    pub const fn times(&self) -> &[f64] {
        self.times.as_slice()
    }

    #[inline]
    #[must_use]
    pub const fn frequency_range(&self) -> (f64, f64) {
        self.freq.frequency_range()
    }

    #[inline]
    #[must_use]
    pub fn duration(&self) -> f64 {
        let max_idx = self.times.len().saturating_sub(1); // safe for non-empty
        // safety: max_idx is in-bounds as times is non-empty
        unsafe { *self.times.get_unchecked(max_idx) }
    }
}

// Enum types for frequency and amplitude scales

/// Linear frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum LinearHz {
    _Phantom,
}

/// Logarithmic frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum LogHz {
    _Phantom,
}

/// Mel frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum Mel {
    _Phantom,
}

/// ERB/gammatone frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum Erb {
    _Phantom,
}
pub type Gammatone = Erb;

/// Constant-Q Transform frequency scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum Cqt {
    _Phantom,
}

// Amplitude scales

/// Power amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum Power {
    _Phantom,
}

/// Decibel amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub enum Decibels {
    _Phantom,
}

/// Magnitude amplitude scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
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
    pub fn new(
        n_fft: usize,
        hop_size: usize,
        window: WindowType,
        centre: bool,
    ) -> SpectrogramResult<Self> {
        let n_fft =
            NonZeroUsize::new(n_fft).ok_or(SpectrogramError::invalid_input("n_fft must be > 0"))?;

        let hop_size = NonZeroUsize::new(hop_size)
            .ok_or(SpectrogramError::invalid_input("hop_size must be > 0"))?;

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

    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> usize {
        self.n_fft.get()
    }

    #[inline]
    #[must_use]
    pub const fn hop_size(&self) -> usize {
        self.hop_size.get()
    }

    #[inline]
    #[must_use]
    pub const fn window(&self) -> WindowType {
        self.window
    }

    #[inline]
    #[must_use]
    pub const fn centre(&self) -> bool {
        self.centre
    }

    /// Create a builder for STFT parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::{StftParams, WindowType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let stft = StftParams::builder()
    ///     .n_fft(2048)
    ///     .hop_size(512)
    ///     .window(WindowType::Hanning)
    ///     .centre(true)
    ///     .build()?;
    ///
    /// assert_eq!(stft.n_fft(), 2048);
    /// assert_eq!(stft.hop_size(), 512);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn builder() -> StftParamsBuilder {
        StftParamsBuilder::default()
    }
}

/// Builder for [`StftParams`].
#[derive(Debug, Clone)]
pub struct StftParamsBuilder {
    n_fft: Option<usize>,
    hop_size: Option<usize>,
    window: WindowType,
    centre: bool,
}

impl Default for StftParamsBuilder {
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
    #[must_use]
    pub const fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// Set the hop size (samples between successive frames).
    #[must_use]
    pub const fn hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = Some(hop_size);
        self
    }

    /// Set the window function.
    #[must_use]
    pub const fn window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set whether to center frames (pad input signal).
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

/// Mel filter bank parameters
///
/// * `n_mels`: Number of mel bands
/// * `f_min`: Minimum frequency (Hz)
/// * `f_max`: Maximum frequency (Hz)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MelParams {
    n_mels: NonZeroUsize,
    f_min: f64,
    f_max: f64,
}

impl MelParams {
    pub fn new(n_mels: usize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        let n_mels = NonZeroUsize::new(n_mels)
            .ok_or(SpectrogramError::invalid_input("n_mels must be > 0"))?;

        if !(f_min >= 0.0) {
            return Err(SpectrogramError::invalid_input("f_min must be >= 0"));
        }

        if !(f_max > f_min) {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_mels,
            f_min,
            f_max,
        })
    }

    #[inline]
    #[must_use]
    pub const fn n_mels(&self) -> usize {
        self.n_mels.get()
    }

    #[inline]
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

    #[inline]
    #[must_use]
    pub const fn f_max(&self) -> f64 {
        self.f_max
    }

    /// Create standard mel filterbank parameters.
    ///
    /// Uses 128 mel bands from 0 Hz to the Nyquist frequency.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (used to determine `f_max`)
    pub fn standard(sample_rate: f64) -> SpectrogramResult<Self> {
        Self::new(128, 0.0, sample_rate / 2.0)
    }

    /// Create mel filterbank parameters optimized for speech.
    ///
    /// Uses 40 mel bands from 0 Hz to 8000 Hz (typical speech bandwidth).
    pub fn speech_standard() -> SpectrogramResult<Self> {
        Self::new(40, 0.0, 8000.0)
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
    pub fn new(n_bins: usize, f_min: f64, f_max: f64) -> SpectrogramResult<Self> {
        let n_bins = NonZeroUsize::new(n_bins)
            .ok_or(SpectrogramError::invalid_input("n_bins must be > 0"))?;

        if !(f_min > 0.0 && f_min.is_finite()) {
            return Err(SpectrogramError::invalid_input(
                "f_min must be finite and > 0",
            ));
        }

        if !(f_max > f_min) {
            return Err(SpectrogramError::invalid_input("f_max must be > f_min"));
        }

        Ok(Self {
            n_bins,
            f_min,
            f_max,
        })
    }

    #[inline]
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_bins.get()
    }

    #[inline]
    #[must_use]
    pub const fn f_min(&self) -> f64 {
        self.f_min
    }

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
    pub fn standard(sample_rate: f64) -> SpectrogramResult<Self> {
        Self::new(128, 20.0, sample_rate / 2.0)
    }

    /// Create logarithmic frequency parameters optimized for music.
    ///
    /// Uses 84 bins (7 octaves * 12 bins/octave) from 27.5 Hz (A0) to 4186 Hz (C8).
    pub fn music_standard() -> SpectrogramResult<Self> {
        Self::new(84, 27.5, 4186.0)
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
    pub fn new(floor_db: f64) -> SpectrogramResult<Self> {
        if !floor_db.is_finite() {
            return Err(SpectrogramError::invalid_input("floor_db must be finite"));
        }

        Ok(Self { floor_db })
    }

    #[inline]
    #[must_use]
    pub const fn floor_db(&self) -> f64 {
        self.floor_db
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectrogramParams {
    stft: StftParams,
    sample_rate_hz: f64,
}

impl SpectrogramParams {
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
    /// # Examples
    ///
    /// ```
    /// use spectrograms::{SpectrogramParams, WindowType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let params = SpectrogramParams::builder()
    ///     .sample_rate(16000.0)
    ///     .n_fft(512)
    ///     .hop_size(256)
    ///     .window(WindowType::Hanning)
    ///     .centre(true)
    ///     .build()?;
    ///
    /// assert_eq!(params.sample_rate_hz(), 16000.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn builder() -> SpectrogramParamsBuilder {
        SpectrogramParamsBuilder::default()
    }

    /// Create default parameters for speech processing.
    ///
    /// Uses:
    /// - `n_fft`: 512 (32ms at 16kHz)
    /// - `hop_size`: 160 (10ms at 16kHz)
    /// - window: Hanning
    /// - centre: true
    pub fn speech_default(sample_rate_hz: f64) -> SpectrogramResult<Self> {
        let stft = StftParams::new(512, 160, WindowType::Hanning, true)?;
        Self::new(stft, sample_rate_hz)
    }

    /// Create default parameters for music processing.
    ///
    /// Uses:
    /// - `n_fft`: 2048 (46ms at 44.1kHz)
    /// - `hop_size`: 512 (11.6ms at 44.1kHz)
    /// - window: Hanning
    /// - centre: true
    pub fn music_default(sample_rate_hz: f64) -> SpectrogramResult<Self> {
        let stft = StftParams::new(2048, 512, WindowType::Hanning, true)?;
        Self::new(stft, sample_rate_hz)
    }

    #[inline]
    #[must_use]
    pub const fn stft(&self) -> &StftParams {
        &self.stft
    }

    #[inline]
    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    #[inline]
    #[must_use]
    pub fn frame_period_seconds(&self) -> f64 {
        self.stft.hop_size() as f64 / self.sample_rate_hz
    }

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
    n_fft: Option<usize>,
    hop_size: Option<usize>,
    window: WindowType,
    centre: bool,
}

impl Default for SpectrogramParamsBuilder {
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
    #[must_use]
    pub const fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set the FFT window size.
    #[must_use]
    pub const fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// Set the hop size (samples between successive frames).
    #[must_use]
    pub const fn hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = Some(hop_size);
        self
    }

    /// Set the window function.
    #[must_use]
    pub const fn window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set whether to center frames (pad input signal).
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
/// * `samples` - Input signal (any type that can be converted to a slice)
/// * `n_fft` - FFT size (must match samples length)
///
/// # Returns
///
/// A vector of complex frequency bins with length `n_fft/2` + 1.
///
/// # Errors
///
/// Returns an error if:
/// - `n_fft` doesn't match the samples length
/// - FFT computation fails
///
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = vec![0.0; 512];
/// let spectrum = rfft(&signal, 512)?;
///
/// assert_eq!(spectrum.len(), 257); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
pub fn rfft<S: AsRef<[f64]>>(samples: S, n_fft: usize) -> SpectrogramResult<Array1<Complex<f64>>> {
    let samples = samples.as_ref();
    if samples.len() != n_fft {
        return Err(SpectrogramError::dimension_mismatch(n_fft, samples.len()));
    }

    let out_len = r2c_output_size(n_fft);

    // Create FFT plan
    #[cfg(feature = "realfft")]
    let mut fft = {
        let mut planner = crate::RealFftPlanner::new();
        let plan = planner.get_or_create(n_fft);
        crate::RealFftPlan::new(n_fft, plan)
    };

    #[cfg(feature = "fftw")]
    let mut fft = {
        use std::sync::Arc;
        let plan = crate::FftwPlanner::build_plan(n_fft)?;
        crate::FftwPlan::new(Arc::new(plan))
    };

    let input = samples.to_vec();
    let mut output = vec![Complex::new(0.0, 0.0); out_len];
    fft.process(&input, &mut output)?;
    let output = Array1::from_vec(output);
    Ok(output)
}

/// Compute the power spectrum of a signal (|X|²).
///
/// This function applies an optional window function and computes the
/// power spectrum via FFT. The result contains only positive frequencies.
///
/// # Arguments
///
/// * `samples` - Input signal (length should equal `n_fft`)
/// * `n_fft` - FFT size
/// * `window` - Optional window function (None for rectangular window)
///
/// # Returns
///
/// A vector of power values with length `n_fft/2` + 1.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = vec![0.0; 512];
/// let power = power_spectrum(&signal, 512, Some(WindowType::Hanning))?;
///
/// assert_eq!(power.len(), 257); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
pub fn power_spectrum<S: AsRef<[f64]>>(
    samples: S,
    n_fft: usize,
    window: Option<WindowType>,
) -> SpectrogramResult<Vec<f64>> {
    let samples = samples.as_ref();
    if samples.len() != n_fft {
        return Err(SpectrogramError::dimension_mismatch(n_fft, samples.len()));
    }

    let mut windowed = samples.to_vec();

    if let Some(win_type) = window {
        let window_samples = make_window(win_type, n_fft)?;
        for (i, w) in window_samples.iter().enumerate() {
            windowed[i] *= w;
        }
    }

    let fft_result = rfft(&windowed, n_fft)?;
    Ok(fft_result
        .iter()
        .map(num_complex::Complex::norm_sqr)
        .collect())
}

/// Compute the magnitude spectrum of a signal (|X|).
///
/// This function applies an optional window function and computes the
/// magnitude spectrum via FFT. The result contains only positive frequencies.
///
/// # Arguments
///
/// * `samples` - Input signal (length should equal `n_fft`)
/// * `n_fft` - FFT size
/// * `window` - Optional window function (None for rectangular window)
///
/// # Returns
///
/// A vector of magnitude values with length `n_fft/2` + 1.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = vec![0.0; 512];
/// let magnitude = magnitude_spectrum(&signal, 512, Some(WindowType::Hanning))?;
///
/// assert_eq!(magnitude.len(), 257); // 512/2 + 1
/// # Ok(())
/// # }
/// ```
pub fn magnitude_spectrum<S: AsRef<[f64]>>(
    samples: S,
    n_fft: usize,
    window: Option<WindowType>,
) -> SpectrogramResult<Vec<f64>> {
    let power = power_spectrum(samples, n_fft, window)?;
    Ok(power.iter().map(|&p| p.sqrt()).collect())
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
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let signal = vec![0.0; 16000];
/// let stft_matrix = stft(&signal, 512, 256, WindowType::Hanning, true)?;
///
/// println!("STFT: {} bins x {} frames", stft_matrix.nrows(), stft_matrix.ncols());
/// # Ok(())
/// # }
/// ```
pub fn stft<S>(
    samples: S,
    n_fft: usize,
    hop_size: usize,
    window: WindowType,
    center: bool,
) -> SpectrogramResult<Array2<Complex<f64>>>
where
    S: AsRef<[f64]>,
{
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
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Forward FFT
/// let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
/// let spectrum = rfft(&signal, 8)?;
///
/// // Inverse FFT
/// let reconstructed = irfft(spectrum.as_slice().unwrap(), 8)?;
///
/// assert_eq!(reconstructed.len(), 8);
/// # Ok(())
/// # }
/// ```
pub fn irfft<S: AsRef<[Complex<f64>]>>(spectrum: S, n_fft: usize) -> SpectrogramResult<Vec<f64>> {
    use crate::fft_backend::{C2rPlan, C2rPlanner, r2c_output_size};

    let spectrum = spectrum.as_ref();
    let expected_len = r2c_output_size(n_fft);
    if spectrum.len() != expected_len {
        return Err(SpectrogramError::dimension_mismatch(
            expected_len,
            spectrum.len(),
        ));
    }

    // Create inverse FFT plan
    #[cfg(feature = "realfft")]
    let mut ifft = {
        let mut planner = crate::RealFftPlanner::new();
        planner.plan_c2r(n_fft)?
    };

    #[cfg(feature = "fftw")]
    let mut ifft = {
        let mut planner = crate::FftwPlanner::new();
        planner.plan_c2r(n_fft)?
    };

    let mut output = vec![0.0; n_fft];
    ifft.process(spectrum, &mut output)?;

    Ok(output)
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
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Generate signal
/// let signal = vec![1.0; 16000];
///
/// // Forward STFT
/// let stft_matrix = stft(&signal, 512, 256, WindowType::Hanning, true)?;
///
/// // Inverse STFT
/// let reconstructed = istft(&stft_matrix, 512, 256, WindowType::Hanning, true)?;
///
/// println!("Original: {} samples", signal.len());
/// println!("Reconstructed: {} samples", reconstructed.len());
/// # Ok(())
/// # }
/// ```
pub fn istft(
    stft_matrix: &Array2<Complex<f64>>,
    n_fft: usize,
    hop_size: usize,
    window: WindowType,
    center: bool,
) -> SpectrogramResult<Vec<f64>> {
    use crate::fft_backend::{C2rPlan, C2rPlanner, r2c_output_size};

    let n_bins = stft_matrix.nrows();
    let n_frames = stft_matrix.ncols();

    let expected_bins = r2c_output_size(n_fft);
    if n_bins != expected_bins {
        return Err(SpectrogramError::dimension_mismatch(expected_bins, n_bins));
    }

    if hop_size == 0 {
        return Err(SpectrogramError::invalid_input("hop_size must be > 0"));
    }

    // Create inverse FFT plan
    #[cfg(feature = "realfft")]
    let mut ifft = {
        let mut planner = crate::RealFftPlanner::new();
        planner.plan_c2r(n_fft)?
    };

    #[cfg(feature = "fftw")]
    let mut ifft = {
        let mut planner = crate::FftwPlanner::new();
        planner.plan_c2r(n_fft)?
    };

    // Generate window
    let window_samples = make_window(window, n_fft)?;

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
/// of the same size much more efficient than calling `rfft()` repeatedly.
///
/// # Examples
///
/// ```
/// use spectrograms::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut planner = FftPlanner::new();
///
/// // Process multiple signals of the same size efficiently
/// for _ in 0..100 {
///     let signal = vec![0.0; 512];
///     let spectrum = planner.rfft(&signal, 512)?;
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
    /// This is more efficient than calling the standalone `rfft()` function
    /// repeatedly for the same FFT size.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = vec![1.0; 512];
    /// let spectrum = planner.rfft(&signal, 512)?;
    ///
    /// assert_eq!(spectrum.len(), 257); // 512/2 + 1
    /// # Ok(())
    /// # }
    /// ```
    pub fn rfft<S: AsRef<[f64]>>(
        &mut self,
        samples: S,
        n_fft: usize,
    ) -> SpectrogramResult<Vec<Complex<f64>>> {
        use crate::fft_backend::{R2cPlan, R2cPlanner, r2c_output_size};

        let samples = samples.as_ref();
        if samples.len() != n_fft {
            return Err(SpectrogramError::dimension_mismatch(n_fft, samples.len()));
        }

        let out_len = r2c_output_size(n_fft);
        let mut plan = self.inner.plan_r2c(n_fft)?;

        let input = samples.to_vec();
        let mut output = vec![Complex::new(0.0, 0.0); out_len];
        plan.process(&input, &mut output)?;

        Ok(output)
    }

    /// Compute inverse FFT, reusing cached plans.
    ///
    /// This is more efficient than calling the standalone `irfft()` function
    /// repeatedly for the same FFT size.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// // Forward FFT
    /// let signal = vec![1.0; 512];
    /// let spectrum = planner.rfft(&signal, 512)?;
    ///
    /// // Inverse FFT
    /// let reconstructed = planner.irfft(&spectrum, 512)?;
    ///
    /// assert_eq!(reconstructed.len(), 512);
    /// # Ok(())
    /// # }
    /// ```
    pub fn irfft<S: AsRef<[Complex<f64>]>>(
        &mut self,
        spectrum: S,
        n_fft: usize,
    ) -> SpectrogramResult<Vec<f64>> {
        use crate::fft_backend::{C2rPlan, C2rPlanner, r2c_output_size};

        let spectrum = spectrum.as_ref();
        let expected_len = r2c_output_size(n_fft);
        if spectrum.len() != expected_len {
            return Err(SpectrogramError::dimension_mismatch(
                expected_len,
                spectrum.len(),
            ));
        }

        let mut plan = self.inner.plan_c2r(n_fft)?;
        let mut output = vec![0.0; n_fft];
        plan.process(spectrum, &mut output)?;

        Ok(output)
    }

    /// Compute power spectrum with optional windowing, reusing cached plans.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = vec![1.0; 512];
    /// let power = planner.power_spectrum(&signal, 512, Some(WindowType::Hanning))?;
    ///
    /// assert_eq!(power.len(), 257);
    /// # Ok(())
    /// # }
    /// ```
    pub fn power_spectrum<S: AsRef<[f64]>>(
        &mut self,
        samples: S,
        n_fft: usize,
        window: Option<WindowType>,
    ) -> SpectrogramResult<Vec<f64>> {
        let samples = samples.as_ref();
        if samples.len() != n_fft {
            return Err(SpectrogramError::dimension_mismatch(n_fft, samples.len()));
        }

        let mut windowed = samples.to_vec();

        if let Some(win_type) = window {
            let window_samples = make_window(win_type, n_fft)?;
            for (i, w) in window_samples.iter().enumerate() {
                windowed[i] *= w;
            }
        }

        let fft_result = self.rfft(&windowed, n_fft)?;
        Ok(fft_result
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .collect())
    }

    /// Compute magnitude spectrum with optional windowing, reusing cached plans.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectrograms::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut planner = FftPlanner::new();
    ///
    /// let signal = vec![1.0; 512];
    /// let magnitude = planner.magnitude_spectrum(&signal, 512, Some(WindowType::Hanning))?;
    ///
    /// assert_eq!(magnitude.len(), 257);
    /// # Ok(())
    /// # }
    /// ```
    pub fn magnitude_spectrum<S: AsRef<[f64]>>(
        &mut self,
        samples: S,
        n_fft: usize,
        window: Option<WindowType>,
    ) -> SpectrogramResult<Vec<f64>> {
        let power = self.power_spectrum(samples, n_fft, window)?;
        Ok(power.iter().map(|&p| p.sqrt()).collect())
    }
}

impl Default for FftPlanner {
    fn default() -> Self {
        Self::new()
    }
}
