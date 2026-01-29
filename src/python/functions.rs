//! Convenience functions for one-shot spectrogram computation.

use std::num::NonZeroUsize;

use crate::{
    Cqt, Decibels, Gammatone, LinearHz, LogHz, Magnitude, Mel, Power, Spectrogram, chromagram, fft,
    irfft, istft, magnitude_spectrum, mfcc, power_spectrum, rfft,
};
use numpy::{
    Complex64, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use super::params::{
    PyChromaParams, PyCqtParams, PyErbParams, PyLogHzParams, PyLogParams, PyMelParams,
    PyMfccParams, PySpectrogramParams, PyStftParams, PyWindowType,
};
use super::spectrogram::PySpectrogram;
use non_empty_slice::NonEmptySlice;

macro_rules! impl_py_compute_fns {
    (
        $(
            (
                freq_ty = $freq_scale:ty,
                amp_ty  = $amp_scale:ty,
                variant = $variant:ident,
                fn_name = $fn_name:ident,
                freq_desc = $freq_desc:expr,
                amp_desc  = $amp_desc:expr
            )
        ),+ $(,)?
    ) => {
        $(
            #[doc = concat!(
                "Compute a ", $freq_desc, " ", $amp_desc, " spectrogram.\n\n",
                "Parameters\n",
                "----------\n",
                "samples : numpy.typing.NDArray[numpy.float64]\n",
                "    Audio samples as a 1D NumPy array\n",
                "params : SpectrogramParams\n",
                "    Spectrogram parameters\n\n",
                "Returns\n",
                "-------\n",
                "Spectrogram\n",
                "    Spectrogram with ", $freq_desc, " frequency scale and ",
                $amp_desc, " amplitude scale"
            )]
            #[pyfunction]
            #[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float64]",
             params: "SpectrogramParams", db_params: "Optional[LogParams]"=None), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, db_params: Option[PyLogParams]=None)")]
            fn $fn_name(
                py: Python,
                samples: &Bound<'_, PyAny>,
                params: PySpectrogramParams,
                db_params: Option<PyLogParams>,
            ) -> PyResult<PySpectrogram> {

                // Import numpy once per call (cheap, cached by Python)
                let np = py.import("numpy")?;

                // Force dtype=float64 and contiguous layout using NumPy itself
                let array_any = np.call_method1(
                    "ascontiguousarray",
                    (samples, "float64"),
                )?;

                // Downcast into a concrete NumPy array
                let array = array_any.cast::<PyArray1<f64>>()?;

                let samples = array.readonly();
                let samples = samples.as_slice()?;   // &[f64]

                let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
                })?;
                let spec = py.detach(|| {
                    Spectrogram::<$freq_scale, $amp_scale>::compute(
                        samples_slice,
                        &params.inner,
                        db_params.as_ref().map(|p| &p.inner),
                    )
                })?;

                Ok(PySpectrogram::from_spectrogram(py, spec))
            }

        )+
    };
}

impl_py_compute_fns! {
    (
        freq_ty = LinearHz,
        amp_ty  = Power,
        variant = LinearPower,
        fn_name = compute_linear_power_spectrogram,
        freq_desc = "linear",
        amp_desc  = "power"
    ),
    (
        freq_ty = LinearHz,
        amp_ty  = Magnitude,
        variant = LinearMagnitude,
        fn_name = compute_linear_magnitude_spectrogram,
        freq_desc = "linear",
        amp_desc  = "magnitude"
    ),
    (
        freq_ty = LinearHz,
        amp_ty  = Decibels,
        variant = LinearDb,
        fn_name = compute_linear_db_spectrogram,
        freq_desc = "linear",
        amp_desc  = "decibel"
    ),
}

macro_rules! impl_filterbank_compute_fns {
    (
        $(
            (
                freq_ty = $freq_scale:ty,
                amp_ty  = $amp_scale:ty,
                filter_ty = $filter_ty:ty,
                py_filter_ty = $py_filter_ty:ty,
                variant = $variant:ident,
                fn_name = $fn_name:ident,
                freq_desc = $freq_desc:expr,
                amp_desc  = $amp_desc:expr
            )
        ),+ $(,)?
    ) => {
        $(
            #[doc = concat!(
                "Compute a ", $freq_desc, " ", $amp_desc, " spectrogram.\n\n",
                "Parameters\n",
                "----------\n",
                "samples : numpy.typing.NDArray[numpy.float64]\n",
                "    Audio samples as a 1D array\n",
                "params : SpectrogramParams\n",
                "    Spectrogram parameters\n",
                "filter_params : ", stringify!($py_filter_ty), "\n",
                "    Filterbank parameters\n",
                "db : typing.Optional[LogParams], optional\n",
                "    Optional decibel scaling parameters\n\n",
                "Returns\n",
                "-------\n",
                "Spectrogram\n",
                "    Spectrogram with ", $freq_desc, " frequency scale and ", $amp_desc, " amplitude scale"
            )]
            #[pyfunction]
            #[pyo3(signature = (
                samples: "numpy.typing.NDArray[numpy.float64]",
                params: "SpectrogramParams",
                filter_params,
                db: "typing.Optional[LogParams]" = None
            ), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, filter_params: FilterParams, db: typing.Optional[LogParams] = None)")]
            fn $fn_name(
                py: Python,
                samples: PyReadonlyArray1<f64>,
                params: &PySpectrogramParams,
                filter_params: &$py_filter_ty,
                db: Option<&PyLogParams>,
            ) -> PyResult<PySpectrogram> {
                let samples = samples.as_slice()?;
                let samples = NonEmptySlice::new(samples).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
                })?;
                let spec = py.detach(|| {
                    Spectrogram::<$freq_scale, $amp_scale>::compute(
                        samples,
                        &params.inner,
                        &filter_params.inner,
                        db.map(|d| &d.inner),
                    )
                })?;

                // Create PySpectrogram with Python-allocated data
                Ok(PySpectrogram::from_spectrogram(py, spec))
            }
        )+
    };
}

impl_filterbank_compute_fns! {
    // Mel variants
    (
        freq_ty = Mel,
        amp_ty  = Power,
        filter_ty = crate::MelParams,
        py_filter_ty = PyMelParams,
        variant = MelPower,
        fn_name = compute_mel_power_spectrogram,
        freq_desc = "mel",
        amp_desc  = "power"
    ),
    (
        freq_ty = Mel,
        amp_ty  = Magnitude,
        filter_ty = crate::MelParams,
        py_filter_ty = PyMelParams,
        variant = MelMagnitude,
        fn_name = compute_mel_magnitude_spectrogram,
        freq_desc = "mel",
        amp_desc  = "magnitude"
    ),
    (
        freq_ty = Mel,
        amp_ty  = Decibels,
        filter_ty = crate::MelParams,
        py_filter_ty = PyMelParams,
        variant = MelDb,
        fn_name = compute_mel_db_spectrogram,
        freq_desc = "mel",
        amp_desc  = "decibel"
    ),
    // ERB/Gammatone variants
    (
        freq_ty = Gammatone,
        amp_ty  = Power,
        filter_ty = crate::ErbParams,
        py_filter_ty = PyErbParams,
        variant = GammatonePower,
        fn_name = compute_erb_power_spectrogram,
        freq_desc = "ERB/gammatone",
        amp_desc  = "power"
    ),
    (
        freq_ty = Gammatone,
        amp_ty  = Magnitude,
        filter_ty = crate::ErbParams,
        py_filter_ty = PyErbParams,
        variant = GammatoneMagnitude,
        fn_name = compute_erb_magnitude_spectrogram,
        freq_desc = "ERB/gammatone",
        amp_desc  = "magnitude"
    ),
    (
        freq_ty = Gammatone,
        amp_ty  = Decibels,
        filter_ty = crate::ErbParams,
        py_filter_ty = PyErbParams,
        variant = GammatoneDb,
        fn_name = compute_erb_db_spectrogram,
        freq_desc = "ERB/gammatone",
        amp_desc  = "decibel"
    ),
    // LogHz variants
    (
        freq_ty = LogHz,
        amp_ty  = Power,
        filter_ty = crate::LogHzParams,
        py_filter_ty = PyLogHzParams,
        variant = LogHzPower,
        fn_name = compute_loghz_power_spectrogram,
        freq_desc = "logarithmic Hz",
        amp_desc  = "power"
    ),
    (
        freq_ty = LogHz,
        amp_ty  = Magnitude,
        filter_ty = crate::LogHzParams,
        py_filter_ty = PyLogHzParams,
        variant = LogHzMagnitude,
        fn_name = compute_loghz_magnitude_spectrogram,
        freq_desc = "logarithmic Hz",
        amp_desc  = "magnitude"
    ),
    (
        freq_ty = LogHz,
        amp_ty  = Decibels,
        filter_ty = crate::LogHzParams,
        py_filter_ty = PyLogHzParams,
        variant = LogHzDb,
        fn_name = compute_loghz_db_spectrogram,
        freq_desc = "logarithmic Hz",
        amp_desc  = "decibel"
    ),
}

// CQT variants (manual implementation since they have different API)
/// Compute a Constant-Q Transform power spectrogram.
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     Audio samples as a 1D `NumPy` array
/// params : `SpectrogramParams`
///     Spectrogram parameters
/// cqt : `CqtParams`
///     CQT parameters
/// db : typing.Optional[`LogParams`], optional
///     Optional decibel scaling parameters
///
/// Returns
/// -------
/// Spectrogram
///     CQT spectrogram with power amplitude scale
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    params: "SpectrogramParams",
    cqt: "CqtParams",
    db: "typing.Optional[LogParams]" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None)")]
pub fn compute_cqt_power_spectrogram(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
) -> PyResult<PySpectrogram> {
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let spec = py.detach(|| {
        Spectrogram::<Cqt, Power>::compute(
            samples_slice,
            &params.inner,
            &cqt.inner,
            db.map(|d| &d.inner),
        )
    })?;
    Ok(PySpectrogram::from_spectrogram(py, spec))
}

/// Compute a Constant-Q Transform magnitude spectrogram.
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     Audio samples as a 1D `NumPy` array
/// params : `SpectrogramParams`
///     Spectrogram parameters
/// cqt : `CqtParams`
///     CQT parameters
/// db : typing.Optional[`LogParams`], optional
///     Optional decibel scaling parameters
///
/// Returns
/// -------
/// Spectrogram
///     CQT spectrogram with magnitude amplitude scale
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    params: "SpectrogramParams",
    cqt: "CqtParams",
    db: "typing.Optional[LogParams]" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None)")]
pub fn compute_cqt_magnitude_spectrogram(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
) -> PyResult<PySpectrogram> {
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let spec = py.detach(|| {
        Spectrogram::<Cqt, Magnitude>::compute(
            samples_slice,
            &params.inner,
            &cqt.inner,
            db.map(|d| &d.inner),
        )
    })?;
    Ok(PySpectrogram::from_spectrogram(py, spec))
}

/// Compute a Constant-Q Transform decibel spectrogram.
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     Audio samples as a 1D `NumPy` array
/// params : `SpectrogramParams`
///     Spectrogram parameters
/// cqt : `CqtParams`
///     CQT parameters
/// db : typing.Optional[`LogParams`], optional
///     Optional decibel scaling parameters
///
/// Returns
/// -------
/// Spectrogram
///     CQT spectrogram with decibel amplitude scale
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    params: "SpectrogramParams",
    cqt: "CqtParams",
    db: "typing.Optional[LogParams]" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None)")]
pub fn compute_cqt_db_spectrogram(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
) -> PyResult<PySpectrogram> {
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let spec = py.detach(|| {
        Spectrogram::<Cqt, Decibels>::compute(
            samples_slice,
            &params.inner,
            &cqt.inner,
            db.map(|d| &d.inner),
        )
    })?;
    Ok(PySpectrogram::from_spectrogram(py, spec))
}

/// Compute a chromagram (pitch class profile).
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     Audio samples as a 1D `NumPy` array
/// `stft_params` : `StftParams`
///     STFT parameters
/// `sample_rate` : float
///     Sample rate in Hz
/// `chroma_params` : `ChromaParams`
///     Chromagram parameters
///
/// Returns
/// -------
/// numpy.ndarray
///     Chromagram as a 2D `NumPy` array (12 x `n_frames`)
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    stft_params: "StftParams",
    sample_rate: "float",
    chroma_params: "ChromaParams"
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], stft_params: StftParams, sample_rate: float, chroma_params: ChromaParams)")]
pub fn compute_chromagram(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    stft_params: &PyStftParams,
    sample_rate: f64,
    chroma_params: &PyChromaParams,
) -> PyResult<Py<PyArray2<f64>>> {
    let samples = samples.as_slice()?;

    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;

    let result = py.detach(|| {
        chromagram(
            samples_slice,
            &stft_params.inner,
            sample_rate,
            &chroma_params.inner,
        )
    })?;
    Ok(PyArray2::from_owned_array(py, result.data).unbind())
}

/// Compute MFCCs (Mel-Frequency Cepstral Coefficients).
///
/// Parameters
/// ----------
/// samples : numpy.typing.NDArray[numpy.float64]
///     Audio samples as a 1D `NumPy` array
/// `stft_params` : `StftParams`
///     STFT parameters
/// `sample_rate` : float
///     Sample rate in Hz
/// `n_mels` : int
///     Number of mel bands
/// `mfcc_params` : `MfccParams`
///     MFCC parameters
///
/// Returns
/// -------
/// numpy.ndarray
///     MFCCs as a 2D `NumPy` array (`n_mfcc` x `n_frames`)
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    stft_params: "StftParams",
    sample_rate: "float",
    n_mels: "int",
    mfcc_params: "MfccParams"
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], stft_params: StftParams, sample_rate: float, n_mels: int, mfcc_params: MfccParams)")]
pub fn compute_mfcc(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    stft_params: &PyStftParams,
    sample_rate: f64,
    n_mels: usize,
    mfcc_params: &PyMfccParams,
) -> PyResult<Py<PyArray2<f64>>> {
    let n_mels = NonZeroUsize::new(n_mels).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_mels must be a positive integer")
    })?;
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let result = py.detach(|| {
        mfcc(
            samples_slice,
            &stft_params.inner,
            sample_rate,
            n_mels,
            &mfcc_params.inner,
        )
    })?;

    Ok(PyArray2::from_owned_array(py, result.data).unbind())
}

/// Compute the raw STFT (Short-Time Fourier Transform).
///
/// Returns the complex-valued STFT matrix before any frequency mapping or amplitude scaling.
///
/// Parameters
/// -----------
///
/// :param `samples` - Audio samples as a 1D `NumPy` array
/// :param `params` - Spectrogram parameters
///
/// Returns
/// -------
///
/// Complex STFT as a 2D `NumPy` array of complex128 (`n_fft/2+1` x `n_frames`)
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    params: "SpectrogramParams"
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams)")]
pub fn compute_stft(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    params: &PySpectrogramParams,
) -> PyResult<Py<PyArray2<num_complex::Complex64>>> {
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let planner = crate::SpectrogramPlanner::new();
    let result = py.detach(|| planner.compute_stft(samples_slice, &params.inner))?;

    Ok(PyArray2::from_owned_array(py, result.data).unbind())
}

/// Compute the real-to-complex FFT of a signal.
///
/// Computes the FFT of a real-valued signal, returning only positive frequencies
/// (exploiting Hermitian symmetry).
///
/// Parameters
/// -----------
///
/// :param `samples` - Audio samples as a 1D `NumPy` array (length must equal `n_fft`)
/// :param `n_fft` - FFT size
///
/// Returns
/// -------
///
/// Complex FFT as a 1D `NumPy` array of complex128 with length `n_fft/2+1`
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "Optional[int]" = None,
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: Optional[int]=None)")]
pub fn compute_fft(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    n_fft: Option<usize>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    let n_fft = n_fft.unwrap_or_else(|| samples.len());

    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;

    let result = py.detach(|| fft(samples_slice, n_fft))?;

    Ok(numpy::PyArray1::from_owned_array(py, result).unbind())
}

/// Compute the real FFT of a signal.
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "int"
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int) -> numpy.typing.NDArray[numpy.float64]")]
pub fn compute_rfft(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    n_fft: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let result = py.detach(|| rfft(samples_slice, n_fft))?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute the power spectrum of a signal (|X|²).
///
/// Applies an optional window function and computes the power spectrum via FFT.
/// Returns only positive frequencies.
///
/// Parameters
/// -----------
///
/// :param `samples` - Audio samples as a 1D `NumPy` array (length must equal `n_fft`)
/// :param `n_fft` - FFT size
/// :param `window` - Optional window function (None for rectangular window)
///
/// Returns
/// -------
///
/// Power spectrum as a 1D `NumPy` array with length `n_fft/2+1`
///
/// Raises
/// ------
/// `DimensionMismatch` - If samples length doesn't equal `n_fft`
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "int",
    window: "typing.Optional[WindowType]" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int, window: typing.Optional[WindowType] = None)")]
pub fn compute_power_spectrum(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    n_fft: usize,
    window: Option<PyWindowType>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let window_type = window.map(|w| w.inner);

    let result = py.detach(|| power_spectrum(samples_slice, n_fft, window_type))?;

    Ok(PyArray1::from_vec(py, result.to_vec()).unbind())
}

/// Compute the magnitude spectrum of a signal (|X|).
///
/// Applies an optional window function and computes the magnitude spectrum via FFT.
/// Returns only positive frequencies.
///
/// Parameters
/// -----------
///
/// :param `samples` - Audio samples as a 1D `NumPy` array (length must equal `n_fft`)
/// :param `n_fft` - FFT size
/// :param `window` - Optional window function (None for rectangular window)
///
/// Returns
/// -------
///
/// Magnitude spectrum as a 1D `NumPy` array with length `n_fft/2+1`
///
/// Raises
/// ------
///
/// `DimensionMismatch` - If samples length doesn't equal `n_fft`
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "int",
    window: "typing.Optional[WindowType]" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int, window: typing.Optional[WindowType] = None)")]
pub fn compute_magnitude_spectrum(
    py: Python,
    samples: PyReadonlyArray1<f64>,
    n_fft: usize,
    window: Option<PyWindowType>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let samples = samples.as_slice()?;
    let samples_slice = NonEmptySlice::new(samples).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;
    let window_type = window.map(|w| w.inner);

    let result = py.detach(|| magnitude_spectrum(samples_slice, n_fft, window_type))?;

    Ok(PyArray1::from_vec(py, result.to_vec()).unbind())
}

/// Compute the inverse real FFT (complex to real).
///
/// Converts a complex frequency-domain representation back to real time-domain samples.
/// Expects only positive frequencies (Hermitian symmetry is assumed).
///
/// Parameters
/// -----------
///
/// :param `spectrum` - Complex frequency spectrum as a 1D `NumPy` array (length must equal `n_fft/2+1`)
/// :param `n_fft` - FFT size (determines output length)
///
/// Returns
/// -------
/// Real time-domain signal as a 1D `NumPy` array with length `n_fft`
///
///
/// Raises
/// ------
///
/// `DimensionMismatch` - If spectrum length doesn't equal `n_fft/2+1`
#[pyfunction]
#[inline]
#[pyo3(signature = (
    spectrum: "numpy.typing.NDArray[numpy.complex128]",
    n_fft: "int"
))]
pub fn compute_irfft(
    py: Python,
    spectrum: numpy::PyReadonlyArray1<num_complex::Complex64>,
    n_fft: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let spectrum = spectrum.as_slice()?;
    let spectrum_slice = NonEmptySlice::new(spectrum).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("spectrum array must not be empty")
    })?;
    let result = py.detach(|| irfft(spectrum_slice, n_fft))?;

    Ok(PyArray1::from_vec(py, result.to_vec()).unbind())
}

/// Compute the inverse STFT (Short-Time Fourier Transform).
///
/// Reconstructs a time-domain signal from its STFT using overlap-add synthesis.
///
/// Parameters
/// -----------
/// :param `stft_matrix` - Complex STFT as a 2D `NumPy` array (`n_fft/2+1` x `n_frames`)
/// :param `n_fft` - FFT size
/// :param `hop_size` - Number of samples between successive frames (must match forward STFT)
/// :param `window` - Window function to apply (should match forward STFT window)
/// :param `center` - If true, assume the forward STFT was centered (must match forward STFT)
///
/// Returns
/// -------
///
/// Reconstructed time-domain signal as a 1D `NumPy` array
///
/// Raises
/// ------
///
/// `DimensionMismatch` - If STFT matrix shape doesn't match parameters
#[pyfunction]
#[inline]
#[pyo3(signature = (
    stft_matrix: "numpy.typing.NDArray[numpy.complex64]",
    n_fft: "int",
    hop_size: "int",
    window: "WindowType",
    center: "bool" = true
), text_signature = "(stft_matrix: numpy.typing.NDArray[numpy.complex64], n_fft: int, hop_size: int, window: WindowType, center: bool = True)")]
pub fn compute_istft(
    py: Python,
    stft_matrix: numpy::PyReadonlyArray2<num_complex::Complex64>,
    n_fft: usize,
    hop_size: usize,
    window: PyWindowType,
    center: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let hop_size = NonZeroUsize::new(hop_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("hop_size must be non-zero positive integer")
    })?;
    let stft_array = stft_matrix.as_array().to_owned();

    let result = py.detach(|| istft(&stft_array, n_fft, hop_size, window.inner, center))?;

    Ok(PyArray1::from_vec(py, result.to_vec()).unbind())
}

/// Register all convenience functions with the Python module.
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Linear variants
    m.add_function(wrap_pyfunction!(compute_linear_power_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_linear_magnitude_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_linear_db_spectrogram, m)?)?;

    // Mel variants
    m.add_function(wrap_pyfunction!(compute_mel_power_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mel_magnitude_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mel_db_spectrogram, m)?)?;

    // ERB/Gammatone variants
    m.add_function(wrap_pyfunction!(compute_erb_power_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_erb_magnitude_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_erb_db_spectrogram, m)?)?;

    // LogHz variants
    m.add_function(wrap_pyfunction!(compute_loghz_power_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_loghz_magnitude_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_loghz_db_spectrogram, m)?)?;

    // CQT variants
    m.add_function(wrap_pyfunction!(compute_cqt_power_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cqt_magnitude_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cqt_db_spectrogram, m)?)?;

    // Additional functions
    m.add_function(wrap_pyfunction!(compute_chromagram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mfcc, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stft, m)?)?;

    // FFT functions
    m.add_function(wrap_pyfunction!(compute_fft, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rfft, m)?)?;
    m.add_function(wrap_pyfunction!(compute_irfft, m)?)?;
    m.add_function(wrap_pyfunction!(compute_power_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(compute_magnitude_spectrum, m)?)?;

    // Inverse STFT
    m.add_function(wrap_pyfunction!(compute_istft, m)?)?;

    Ok(())
}
