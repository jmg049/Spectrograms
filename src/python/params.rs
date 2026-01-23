//! Python parameter wrapper classes.

use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::{
    ChromaNorm, ChromaParams, CqtParams, ErbParams, LogHzParams, LogParams, MelParams, MfccParams,
    SpectrogramParams, StftParams, WindowType,
};

/// Python wrapper for `WindowType`.
///
/// Represents window functions used for spectral analysis. Different windows provide
/// different trade-offs between frequency resolution and spectral leakage.
#[pyclass(name = "WindowType")]
#[derive(Clone, Copy, Debug)]
pub struct PyWindowType {
    pub(crate) inner: WindowType,
}

#[pymethods]
impl PyWindowType {
    /// Create a rectangular (no) window.
    ///
    /// Best frequency resolution but high spectral leakage.
    #[classmethod]
    const fn rectangular(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: WindowType::Rectangular,
        }
    }

    /// Create a Hanning window.
    ///
    /// Good general-purpose window with moderate leakage.
    #[classmethod]
    const fn hanning(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: WindowType::Hanning,
        }
    }

    /// Create a Hamming window.
    ///
    /// Similar to Hanning but with slightly different coefficients.
    #[classmethod]
    const fn hamming(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: WindowType::Hamming,
        }
    }

    /// Create a Blackman window.
    ///
    /// Low spectral leakage but wider main lobe.
    #[classmethod]
    const fn blackman(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: WindowType::Blackman,
        }
    }

    /// Create a Kaiser window with the given beta parameter.
    ///
    /// # Arguments
    /// * `beta` - Beta parameter controlling the trade-off between main lobe width and side lobe level
    #[classmethod]
    #[pyo3(signature = (beta: "float"), text_signature = "(beta: float)")]
    const fn kaiser(_cls: &Bound<'_, PyType>, beta: f64) -> Self {
        Self {
            inner: WindowType::Kaiser { beta },
        }
    }

    /// Create a Gaussian window with the given standard deviation.
    ///
    /// # Arguments
    /// * `std` - Standard deviation parameter controlling the window width
    #[classmethod]
    #[pyo3(signature = (std: "float"), text_signature = "(std: float)")]
    const fn gaussian(_cls: &Bound<'_, PyType>, std: f64) -> Self {
        Self {
            inner: WindowType::Gaussian { std },
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

impl From<WindowType> for PyWindowType {
    fn from(wt: WindowType) -> Self {
        Self { inner: wt }
    }
}

/// STFT parameters for spectrogram computation.
#[pyclass(name = "StftParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyStftParams {
    pub(crate) inner: StftParams,
}

#[pymethods]
impl PyStftParams {
    /// Create new STFT parameters.
    ///
    /// Parameters
    /// ----------
    /// n_fft : int
    ///     FFT size
    /// hop_size : int
    ///     Hop size between frames
    /// window : WindowType
    ///     Window function
    /// centre : bool, default=True
    ///     Whether to centre frames with padding
    ///
    /// Returns
    /// -------
    /// StftParams
    ///     STFT parameters
    #[new]
    #[pyo3(signature = (
        n_fft: "int",
        hop_size: "int",
        window: "WindowType",
        centre: "bool" = true
    ), text_signature = "(n_fft: int, hop_size: int, window: WindowType, centre: bool = True)")]
    fn new(n_fft: usize, hop_size: usize, window: PyWindowType, centre: bool) -> PyResult<Self> {
        let inner = StftParams::new(n_fft, hop_size, window.inner, centre)?;
        Ok(Self { inner })
    }

    /// FFT size.
    #[getter]
    const fn n_fft(&self) -> usize {
        self.inner.n_fft()
    }

    /// Hop size between frames.
    #[getter]
    const fn hop_size(&self) -> usize {
        self.inner.hop_size()
    }

    /// Window function.
    #[getter]
    const fn window(&self) -> PyWindowType {
        PyWindowType {
            inner: self.inner.window(),
        }
    }

    /// Whether to centre frames with padding.
    #[getter]
    const fn centre(&self) -> bool {
        self.inner.centre()
    }

    fn __repr__(&self) -> String {
        format!(
            "StftParams(n_fft={}, hop_size={}, window={}, centre={})",
            self.n_fft(),
            self.hop_size(),
            self.window().__repr__(),
            self.centre()
        )
    }
}

/// Decibel conversion parameters.

#[pyclass(name = "LogParams")]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PyLogParams {
    pub(crate) inner: LogParams,
}

#[pymethods]
impl PyLogParams {
    /// Parameters
    /// ----------
    /// floor_db : float
    ///     Minimum power in decibels (values below this are clipped)
    #[new]
    #[pyo3(signature = (floor_db: "float"), text_signature = "(floor_db: float)")]
    fn new(floor_db: f64) -> PyResult<Self> {
        let inner = LogParams::new(floor_db)?;
        Ok(Self { inner })
    }

    /// Minimum power in decibels (values below this are clipped).
    #[getter]
    const fn floor_db(&self) -> f64 {
        self.inner.floor_db()
    }

    fn __repr__(&self) -> String {
        format!("LogParams(floor_db={})", self.floor_db())
    }
}

/// Spectrogram computation parameters.
#[pyclass(name = "SpectrogramParams")]
#[derive(Clone, Copy, Debug)]
pub struct PySpectrogramParams {
    pub(crate) inner: SpectrogramParams,
}

#[pymethods]
impl PySpectrogramParams {
    /// Parameters
    /// ----------
    /// stft : StftParams
    ///     STFT parameters
    /// sample_rate : float
    ///     Sample rate in Hz
    #[new]
    #[pyo3(signature = (
        stft: "StftParams",
        sample_rate: "float"
    ), text_signature = "(stft: StftParams, sample_rate: float)")]
    fn new(stft: &PyStftParams, sample_rate: f64) -> PyResult<Self> {
        let inner = SpectrogramParams::new(stft.inner, sample_rate)?;
        Ok(Self { inner })
    }

    /// STFT parameters.
    #[getter]
    const fn stft(&self) -> PyStftParams {
        PyStftParams {
            inner: *self.inner.stft(),
        }
    }

    /// Sample rate in Hz.
    #[getter]
    const fn sample_rate(&self) -> f64 {
        self.inner.sample_rate_hz()
    }

    /// Create default parameters for speech processing.
    ///
    /// Uses `n_fft=512`, `hop_size=160`, Hanning window, centre=true
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Sample rate in Hz
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     SpectrogramParams with standard speech settings
    #[classmethod]
    #[pyo3(signature = (sample_rate: "float"), text_signature = "(sample_rate: float)")]
    fn speech_default(_cls: &Bound<'_, PyType>, sample_rate: f64) -> PyResult<Self> {
        let inner = SpectrogramParams::speech_default(sample_rate)?;
        Ok(Self { inner })
    }

    /// Create default parameters for music processing.
    ///
    /// Uses `n_fft=2048`, `hop_size=512`, Hanning window, centre=true
    ///
    /// Parameters
    /// ----------
    /// sample_rate : float
    ///     Sample rate in Hz
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     SpectrogramParams with standard music settings
    #[classmethod]
    #[pyo3(signature = (sample_rate: "float"), text_signature = "(sample_rate: float)")]
    fn music_default(_cls: &Bound<'_, PyType>, sample_rate: f64) -> PyResult<Self> {
        let inner = SpectrogramParams::music_default(sample_rate)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectrogramParams(sample_rate={}, n_fft={}, hop_size={})",
            self.sample_rate(),
            self.inner.stft().n_fft(),
            self.inner.stft().hop_size()
        )
    }
}

impl From<SpectrogramParams> for PySpectrogramParams {
    fn from(inner: SpectrogramParams) -> Self {
        Self { inner }
    }
}

impl From<PySpectrogramParams> for SpectrogramParams {
    #[inline]
    fn from(py_params: PySpectrogramParams) -> Self {
        py_params.inner
    }
}

/// Mel-scale filterbank parameters.

#[pyclass(name = "MelParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyMelParams {
    pub(crate) inner: MelParams,
}

#[pymethods]
impl PyMelParams {
    /// Mel-scale filterbank parameters.
    ///
    /// Parameters
    /// ----------
    /// n_mels : int
    ///     Number of mel bands
    /// f_min : float
    ///     Minimum frequency in Hz
    /// f_max : float
    ///     Maximum frequency in Hz
    #[new]
    #[pyo3(signature = (
        n_mels: "int",
        f_min: "float",
        f_max: "float"
    ), text_signature = "(n_mels: int, f_min: float, f_max: float)")]
    fn new(n_mels: usize, f_min: f64, f_max: f64) -> PyResult<Self> {
        let inner = MelParams::new(n_mels, f_min, f_max)?;
        Ok(Self { inner })
    }

    /// Number of mel bands.
    #[getter]
    const fn n_mels(&self) -> usize {
        self.inner.n_mels()
    }

    /// Minimum frequency in Hz.
    #[getter]
    const fn f_min(&self) -> f64 {
        self.inner.f_min()
    }

    /// Maximum frequency in Hz.
    #[getter]
    const fn f_max(&self) -> f64 {
        self.inner.f_max()
    }

    fn __repr__(&self) -> String {
        format!(
            "MelParams(n_mels={}, f_min={}, f_max={})",
            self.n_mels(),
            self.f_min(),
            self.f_max()
        )
    }
}

/// ERB-scale (Equivalent Rectangular Bandwidth) filterbank parameters.
#[pyclass(name = "ErbParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyErbParams {
    pub(crate) inner: ErbParams,
}

#[pymethods]
impl PyErbParams {
    /// ERB-scale filterbank parameters.
    ///
    /// Parameters
    /// ----------
    /// n_filters : int
    ///     Number of ERB filters
    /// f_min : float
    ///     Minimum frequency in Hz
    /// f_max : float
    ///     Maximum frequency in Hz
    #[new]
    #[pyo3(signature = (
        n_filters: "int",
        f_min: "float",
        f_max: "float"
    ), text_signature = "(n_filters: int, f_min: float, f_max: float)")]
    fn new(n_filters: usize, f_min: f64, f_max: f64) -> PyResult<Self> {
        let inner = ErbParams::new(n_filters, f_min, f_max)?;
        Ok(Self { inner })
    }

    /// Number of ERB filters.
    #[getter]
    const fn n_filters(&self) -> usize {
        self.inner.n_filters()
    }

    /// Minimum frequency in Hz.
    #[getter]
    const fn f_min(&self) -> f64 {
        self.inner.f_min()
    }

    /// Maximum frequency in Hz.
    #[getter]
    const fn f_max(&self) -> f64 {
        self.inner.f_max()
    }

    fn __repr__(&self) -> String {
        format!(
            "ErbParams(n_filters={}, f_min={}, f_max={})",
            self.n_filters(),
            self.f_min(),
            self.f_max()
        )
    }
}

/// Logarithmic frequency scale parameters.
#[pyclass(name = "LogHzParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyLogHzParams {
    pub(crate) inner: LogHzParams,
}

#[pymethods]
impl PyLogHzParams {
    /// Logarithmic frequency scale parameters.
    ///
    /// Parameters
    /// ----------
    /// n_bins : int
    ///     Number of logarithmically-spaced frequency bins
    /// f_min : float
    ///     Minimum frequency in Hz
    /// f_max : float
    ///     Maximum frequency in Hz
    #[new]
    #[pyo3(signature = (
        n_bins: "int",
        f_min: "float",
        f_max: "float"
    ), text_signature = "(n_bins: int, f_min: float, f_max: float)")]
    fn new(n_bins: usize, f_min: f64, f_max: f64) -> PyResult<Self> {
        let inner = LogHzParams::new(n_bins, f_min, f_max)?;
        Ok(Self { inner })
    }

    /// Number of frequency bins.
    #[getter]
    const fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }

    /// Minimum frequency in Hz.
    #[getter]
    const fn f_min(&self) -> f64 {
        self.inner.f_min()
    }

    /// Maximum frequency in Hz.
    #[getter]
    const fn f_max(&self) -> f64 {
        self.inner.f_max()
    }

    fn __repr__(&self) -> String {
        format!(
            "LogHzParams(n_bins={}, f_min={}, f_max={})",
            self.n_bins(),
            self.f_min(),
            self.f_max()
        )
    }
}

/// Constant-Q Transform parameters.
#[pyclass(name = "CqtParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyCqtParams {
    pub(crate) inner: CqtParams,
}

#[pymethods]
impl PyCqtParams {
    /// Constant-Q Transform parameters.
    ///
    /// Parameters
    /// ----------
    /// bins_per_octave : int
    ///     Number of bins per octave (e.g., 12 for semitones)
    /// n_octaves : int
    ///     Number of octaves to span
    /// f_min : float
    ///     Minimum frequency in Hz
    #[new]
    #[pyo3(signature = (
        bins_per_octave: "int",
        n_octaves: "int",
        f_min: "float"
    ), text_signature = "(bins_per_octave: int, n_octaves: int, f_min: float)")]
    fn new(bins_per_octave: usize, n_octaves: usize, f_min: f64) -> PyResult<Self> {
        let inner = CqtParams::new(bins_per_octave, n_octaves, f_min)?;
        Ok(Self { inner })
    }

    /// Total number of CQT bins.
    #[getter]
    const fn num_bins(&self) -> usize {
        self.inner.num_bins()
    }

    fn __repr__(&self) -> String {
        format!("CqtParams(num_bins={})", self.num_bins())
    }
}

#[pyclass(name = "ChromaNorm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PyChromaNorm {
    pub(crate) inner: ChromaNorm,
}

#[pymethods]
impl PyChromaNorm {
    /// No normalization.
    #[classmethod]
    const fn none(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ChromaNorm::None,
        }
    }

    /// L1 normalization (sum to 1).
    #[classmethod]
    const fn l1(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ChromaNorm::L1,
        }
    }

    /// L2 normalization (Euclidean norm to 1).
    #[classmethod]
    const fn l2(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ChromaNorm::L2,
        }
    }

    /// Max normalization (max value to 1).
    #[classmethod]
    const fn max(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ChromaNorm::Max,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

impl From<ChromaNorm> for PyChromaNorm {
    fn from(inner: ChromaNorm) -> Self {
        Self { inner }
    }
}

impl Into<ChromaNorm> for PyChromaNorm {
    fn into(self) -> ChromaNorm {
        self.inner
    }
}

/// Chromagram (pitch class profile) parameters.
#[pyclass(name = "ChromaParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyChromaParams {
    pub(crate) inner: ChromaParams,
}

#[pymethods]
impl PyChromaParams {
    /// Create new chroma parameters.
    ///
    /// Parameters
    /// ----------
    /// tuning : float, default=440.0
    ///     Reference tuning frequency in Hz (A4)
    /// f_min : float, default=32.7
    ///     Minimum frequency in Hz (C1)
    /// f_max : float, default=4186.0
    ///     Maximum frequency in Hz (C8)
    /// norm : ChromaNorm, optional
    ///     Normalization method: l1, l2, max, or None (default: l2)
    #[new]
    #[pyo3(signature = (
        tuning: "float" = 440.0,
        f_min: "float" = 32.7,
        f_max: "float" = 4186.0,
        norm: "ChromaNorm" = None
    ), text_signature = "(tuning: float = 440.0, f_min: float = 32.7, f_max: float = 4186.0, norm: ChromaNorm = ChromaNorm.None)")]
    fn new(tuning: f64, f_min: f64, f_max: f64, norm: Option<PyChromaNorm>) -> PyResult<Self> {
        let norm = norm.unwrap_or_default();
        let inner = ChromaParams::new(tuning, f_min, f_max, norm.inner)?;
        Ok(Self { inner })
    }

    /// Create standard chroma parameters for music analysis.
    #[classmethod]
    fn music_standard(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let inner = ChromaParams::music_standard()?;
        Ok(Self { inner })
    }

    /// Tuning frequency in Hz (typically 440.0 for A4).
    #[getter]
    const fn tuning(&self) -> f64 {
        self.inner.tuning()
    }

    /// Minimum frequency in Hz.
    #[getter]
    const fn f_min(&self) -> f64 {
        self.inner.f_min()
    }

    /// Maximum frequency in Hz.
    #[getter]
    const fn f_max(&self) -> f64 {
        self.inner.f_max()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChromaParams(tuning={}, f_min={}, f_max={}, norm={:?})",
            self.tuning(),
            self.f_min(),
            self.f_max(),
            self.inner
        )
    }
}

/// MFCC (Mel-Frequency Cepstral Coefficients) parameters.

#[pyclass(name = "MfccParams")]
#[derive(Clone, Copy, Debug)]
pub struct PyMfccParams {
    pub(crate) inner: MfccParams,
}

#[pymethods]
impl PyMfccParams {
    /// Create new MFCC parameters.
    ///
    /// Parameters
    /// ----------
    /// n_mfcc : int, default=13
    ///     Number of MFCC coefficients to compute
    #[new]
    #[pyo3(signature = (n_mfcc: "int" = 13))]
    fn new(n_mfcc: usize) -> PyResult<Self> {
        let inner = MfccParams::new(n_mfcc)?;
        Ok(Self { inner })
    }

    /// Standard MFCC parameters for speech recognition (13 coefficients).
    #[classmethod]
    fn speech_standard(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let inner = MfccParams::speech_standard()?;
        Ok(Self { inner })
    }

    /// Number of MFCC coefficients.
    #[getter]
    const fn n_mfcc(&self) -> usize {
        self.inner.n_mfcc()
    }

    fn __repr__(&self) -> String {
        format!("MfccParams(n_mfcc={})", self.n_mfcc())
    }
}

/// Register all parameter classes with the Python module.
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWindowType>()?;
    m.add_class::<PyStftParams>()?;
    m.add_class::<PyLogParams>()?;
    m.add_class::<PySpectrogramParams>()?;
    m.add_class::<PyMelParams>()?;
    m.add_class::<PyErbParams>()?;
    m.add_class::<PyLogHzParams>()?;
    m.add_class::<PyCqtParams>()?;
    m.add_class::<PyChromaParams>()?;
    m.add_class::<PyMfccParams>()?;
    Ok(())
}
