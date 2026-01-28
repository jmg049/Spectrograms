//! Python spectrogram result class.

use std::{
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
};

use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::{
    AmpScaleSpec, Cqt, Decibels, Gammatone, LinearHz, LogHz, Magnitude, Mel, Power, Spectrogram,
    SpectrogramParams,
};

use super::params::PySpectrogramParams;

macro_rules! dispatch_inner {
    ($self:expr, |$inner:ident| $body:expr) => {
        match $self {
            PySpectrogramInner::LinearPower($inner) => $body,
            PySpectrogramInner::LinearMagnitude($inner) => $body,
            PySpectrogramInner::LinearDb($inner) => $body,

            PySpectrogramInner::LogHzPower($inner) => $body,
            PySpectrogramInner::LogHzMagnitude($inner) => $body,
            PySpectrogramInner::LogHzDb($inner) => $body,

            PySpectrogramInner::GammatonePower($inner) => $body,
            PySpectrogramInner::GammatoneMagnitude($inner) => $body,
            PySpectrogramInner::GammatoneDb($inner) => $body,

            PySpectrogramInner::MelPower($inner) => $body,
            PySpectrogramInner::MelMagnitude($inner) => $body,
            PySpectrogramInner::MelDb($inner) => $body,

            PySpectrogramInner::CqtPower($inner) => $body,
            PySpectrogramInner::CqtMagnitude($inner) => $body,
            PySpectrogramInner::CqtDb($inner) => $body,
        }
    };
}

macro_rules! forward_methods {
    (
        $(
            fn $name:ident (&self $(, $arg:ident : $ty:ty )* ) -> $ret:ty ;
        )*
    ) => {
        $(
            pub fn $name(&self $(, $arg : $ty )* ) -> $ret {
                dispatch_inner!(self, |inner| inner.$name($($arg),*))
            }
        )*
    };
}

#[derive(Debug, Clone)]
enum PySpectrogramInner {
    // Linear
    LinearPower(TypedSpectrogramInner<LinearHz, Power>),
    LinearMagnitude(TypedSpectrogramInner<LinearHz, Magnitude>),
    LinearDb(TypedSpectrogramInner<LinearHz, Decibels>),
    // Log Frequency
    LogHzPower(TypedSpectrogramInner<LogHz, Power>),
    LogHzMagnitude(TypedSpectrogramInner<LogHz, Magnitude>),
    LogHzDb(TypedSpectrogramInner<LogHz, Decibels>),
    // Gammatone
    GammatonePower(TypedSpectrogramInner<Gammatone, Power>),
    GammatoneMagnitude(TypedSpectrogramInner<Gammatone, Magnitude>),
    GammatoneDb(TypedSpectrogramInner<Gammatone, Decibels>),
    // Mel
    MelPower(TypedSpectrogramInner<Mel, Power>),
    MelMagnitude(TypedSpectrogramInner<Mel, Magnitude>),
    MelDb(TypedSpectrogramInner<Mel, Decibels>),
    // Cqt
    CqtPower(TypedSpectrogramInner<Cqt, Power>),
    CqtMagnitude(TypedSpectrogramInner<Cqt, Magnitude>),
    CqtDb(TypedSpectrogramInner<Cqt, Decibels>),
}

impl PySpectrogramInner {
    forward_methods! {
        fn data(&self) -> &Array2<f64>;
        fn n_frames(&self) -> NonZeroUsize;
        fn n_bins(&self) -> NonZeroUsize;
        fn params(&self) -> &SpectrogramParams;
        fn db_range(&self) -> Option<(f64, f64)>;
        fn times(&self) -> &[f64];
        fn frequencies(&self) -> &[f64];
        fn frequency_range(&self) -> (f64, f64);
        fn duration(&self) -> f64;
    }
}

#[derive(Debug, Clone)]
pub struct TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    spectrogram: Spectrogram<FreqScale, AmpScale>,
}

impl<FreqScale, AmpScale> TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    pub(crate) const fn new(spectrogram: Spectrogram<FreqScale, AmpScale>) -> Self {
        Self { spectrogram }
    }
}

impl<FreqScale, AmpScale> AsRef<Spectrogram<FreqScale, AmpScale>>
    for TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    fn as_ref(&self) -> &Spectrogram<FreqScale, AmpScale> {
        &self.spectrogram
    }
}

impl<FreqScale, AmpScale> AsMut<Spectrogram<FreqScale, AmpScale>>
    for TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    fn as_mut(&mut self) -> &mut Spectrogram<FreqScale, AmpScale> {
        &mut self.spectrogram
    }
}

impl<FreqScale, AmpScale> Deref for TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    type Target = Spectrogram<FreqScale, AmpScale>;

    fn deref(&self) -> &Self::Target {
        &self.spectrogram
    }
}

impl<FreqScale, AmpScale> DerefMut for TypedSpectrogramInner<FreqScale, AmpScale>
where
    FreqScale: Copy + Clone + 'static,
    AmpScale: AmpScaleSpec + 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.spectrogram
    }
}

/// Spectrogram computation result.
///
/// Contains the spectrogram data as a `NumPy` array along with frequency and time axes and the parameters used to create it.
///
#[derive(Debug, Clone)]
#[pyclass(name = "Spectrogram")]
pub struct PySpectrogram {
    spectrogram_type: PySpectrogramInner,
}

impl PySpectrogram {}

#[pymethods]
impl PySpectrogram {
    /// Get the spectrogram data as a `NumPy` array.
    ///
    /// Returns
    /// -------
    /// numpy.typing.NDArray[numpy.float64]
    ///     2D `NumPy` array with shape (`n_bins`, `n_frames`)
    #[getter]
    fn data<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let inner_data: &Array2<f64> = self.spectrogram_type.data();

        PyArray2::from_array(py, inner_data)
    }

    /// Get the frequency axis values.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     List of frequency values (Hz or scale-specific units)
    #[getter]
    fn frequencies(&self) -> &[f64] {
        self.spectrogram_type.frequencies()
    }

    /// Get the time axis values.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     List of time values in seconds
    #[getter]
    fn times(&self) -> &[f64] {
        self.spectrogram_type.times()
    }

    /// Get the number of frequency bins.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of frequency bins
    #[getter]
    fn n_bins(&self) -> usize {
        self.spectrogram_type.n_bins().get()
    }

    /// Get the number of time frames.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of time frames
    #[getter]
    fn n_frames(&self) -> usize {
        self.spectrogram_type.n_frames().get()
    }

    /// Get the shape of the spectrogram.
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    ///     Tuple of (`n_bins`, `n_frames`)
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.n_bins(), self.n_frames())
    }

    /// Get the frequency range.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     Tuple of (`f_min`, `f_max`) in Hz or scale-specific units
    fn frequency_range(&self) -> (f64, f64) {
        self.spectrogram_type.frequency_range()
    }

    /// Get the total duration.
    ///
    /// Returns
    /// -------
    /// float
    ///     Duration in seconds
    fn duration(&self) -> f64 {
        self.spectrogram_type.duration()
    }

    /// Get the decibel range if applicable.
    ///
    /// Returns
    /// -------
    /// tuple[float, float] or None
    ///     Tuple of (`min_db`, `max_db`) for decibel-scaled spectrograms, None otherwise
    fn db_range(&self) -> Option<(f64, f64)> {
        self.spectrogram_type.db_range()
    }

    /// Get the computation parameters.
    ///
    /// Returns
    /// -------
    /// SpectrogramParams
    ///     The `SpectrogramParams` used to compute this spectrogram
    #[getter]
    fn params(&self) -> PySpectrogramParams {
        (*self.spectrogram_type.params()).into()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.spectrogram_type)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.n_frames()
    }

    #[pyo3(signature = (dtype=None), text_signature = "($self, dtype=None)")]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let data = self.spectrogram_type.data();
        let arr = PyArray2::from_array(py, data);
        if let Some(dtype) = dtype {
            let casted: Bound<'py, PyAny> = arr.call_method1("astype", (dtype,))?;
            Ok(casted.unbind())
        } else {
            Ok(arr.into_any().unbind())
        }
    }

    fn __getitem__<'py>(
        &'py self,
        py: Python<'py>,
        idx: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let data = self.spectrogram_type.data();
        let arr = PyArray2::from_array(py, data);
        let sliced: Bound<'py, PyAny> = arr.get_item(idx)?;
        Ok(sliced.unbind())
    }
}

macro_rules! impl_py_spectrogram_ctor {
    (
        $(
            ($freq_scale:ty, $amp_scale:ty, $variant:ident)
        ),+ $(,)?
    ) => {
        #[allow(non_snake_case)]
        impl PySpectrogram {
            $(
                pub(crate) const fn $variant(
                    spectrogram: Spectrogram<$freq_scale, $amp_scale>
                ) -> Self {
                    Self {
                        spectrogram_type: PySpectrogramInner::$variant(
                            TypedSpectrogramInner::new(spectrogram)
                        ),
                    }
                }
            )+
        }
    };
}

impl_py_spectrogram_ctor! {
    // Linear
    (LinearHz, Power, LinearPower),
    (LinearHz, Magnitude, LinearMagnitude),
    (LinearHz, Decibels, LinearDb),
    // LogHz
    (LogHz, Power, LogHzPower),
    (LogHz, Magnitude, LogHzMagnitude),
    (LogHz, Decibels, LogHzDb),
    // Gammatone
    (Gammatone, Power, GammatonePower),
    (Gammatone, Magnitude, GammatoneMagnitude),
    (Gammatone, Decibels, GammatoneDb),
    // Mel
    (Mel, Power, MelPower),
    (Mel, Magnitude, MelMagnitude),
    (Mel, Decibels, MelDb),
    // Cqt
    (Cqt, Power, CqtPower),
    (Cqt, Magnitude, CqtMagnitude),
    (Cqt, Decibels, CqtDb),
}

/// Register the spectrogram class with the Python module.
pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PySpectrogram>()?;
    Ok(())
}
