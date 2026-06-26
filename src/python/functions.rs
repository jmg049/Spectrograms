//! Convenience functions for one-shot spectrogram computation.

use std::num::NonZeroUsize;

use crate::{
    AmpScaleSpec, Cqt, Decibels, Gammatone, LinearHz, LogHz, Magnitude, Mel, Power, Spectrogram,
    SpectrogramPlanner, SpectrogramResult, chromagram, fft, irfft, istft, magnitude_spectrum, mfcc,
    power_spectrum, rfft,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use super::dtype::{
    Dtype, array1_to_py, complex_2d_owned, parse_dtype, vec1_to_py, with_complex_1d, with_real_1d,
};
use super::params::{
    PyChromaParams, PyCqtParams, PyErbParams, PyLogHzParams, PyLogParams, PyMelParams,
    PyMfccParams, PySpectrogramParams, PyStftParams, PyStftResult, PyWindowType,
};
use super::spectrogram::{PyScalar, PySpectrogram};
use super::{PyChromagram, PyMfcc};
use non_empty_slice::NonEmptySlice;

/// Shared body for the `dtype`-aware spectrogram compute functions.
///
/// Forces `samples` to a contiguous array of the requested precision `T`
/// (via NumPy's `ascontiguousarray`), wraps it as a [`NonEmptySlice`], runs the
/// supplied compute closure, and transfers the resulting data to Python.
fn compute_typed<F, A, T, Run>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    run: Run,
) -> PyResult<PySpectrogram>
where
    F: Copy + Clone + 'static,
    A: AmpScaleSpec + 'static,
    T: PyScalar,
    Run: FnOnce(&NonEmptySlice<T>) -> SpectrogramResult<Spectrogram<F, A, T>>,
{
    // Import numpy once per call (cheap, cached by Python)
    let np = py.import("numpy")?;

    // Force the requested dtype and contiguous layout using NumPy itself.
    let array_any = np.call_method1("ascontiguousarray", (samples, T::DTYPE))?;

    // Downcast into a concrete NumPy array of element type T.
    let array = array_any.cast::<PyArray1<T>>()?;

    let readonly = array.readonly();
    let slice = readonly.as_slice()?; // &[T]

    let samples_slice = NonEmptySlice::new(slice).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("samples array must not be empty")
    })?;

    let spec = run(samples_slice)?;

    Ok(PySpectrogram::from_spectrogram(py, spec))
}

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
                "    Spectrogram parameters\n",
                "dtype : str, optional\n",
                "    Output precision: \"float64\" (default) or \"float32\". The\n",
                "    spectrogram is computed natively at this precision.\n\n",
                "Returns\n",
                "-------\n",
                "Spectrogram\n",
                "    Spectrogram with ", $freq_desc, " frequency scale and ",
                $amp_desc, " amplitude scale"
            )]
            #[pyfunction]
            #[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float64]",
             params: "SpectrogramParams", db_params: "Optional[LogParams]"=None,
             dtype: "str"=None), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, db_params: Option[PyLogParams]=None, dtype: str = \"float64\")")]
            fn $fn_name(
                py: Python,
                samples: &Bound<'_, PyAny>,
                params: PySpectrogramParams,
                db_params: Option<PyLogParams>,
                dtype: Option<&str>,
            ) -> PyResult<PySpectrogram> {
                match parse_dtype(dtype)? {
                    Dtype::F32 => compute_typed::<$freq_scale, $amp_scale, f32, _>(py, samples, |s| {
                        py.detach(|| {
                            Spectrogram::<$freq_scale, $amp_scale, f32>::compute(
                                s,
                                &params.inner,
                                db_params.as_ref().map(|p| &p.inner),
                            )
                        })
                    }),
                    Dtype::F64 => compute_typed::<$freq_scale, $amp_scale, f64, _>(py, samples, |s| {
                        py.detach(|| {
                            Spectrogram::<$freq_scale, $amp_scale, f64>::compute(
                                s,
                                &params.inner,
                                db_params.as_ref().map(|p| &p.inner),
                            )
                        })
                    }),
                }
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
                "    Optional decibel scaling parameters\n",
                "dtype : str, optional\n",
                "    Output precision: \"float64\" (default) or \"float32\". The\n",
                "    spectrogram is computed natively at this precision.\n\n",
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
                db: "typing.Optional[LogParams]" = None,
                dtype: "str" = None
            ), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, filter_params: FilterParams, db: typing.Optional[LogParams] = None, dtype: str = \"float64\")")]
            fn $fn_name(
                py: Python,
                samples: &Bound<'_, PyAny>,
                params: &PySpectrogramParams,
                filter_params: &$py_filter_ty,
                db: Option<&PyLogParams>,
                dtype: Option<&str>,
            ) -> PyResult<PySpectrogram> {
                match parse_dtype(dtype)? {
                    Dtype::F32 => compute_typed::<$freq_scale, $amp_scale, f32, _>(py, samples, |s| {
                        py.detach(|| {
                            Spectrogram::<$freq_scale, $amp_scale, f32>::compute(
                                s,
                                &params.inner,
                                &filter_params.inner,
                                db.map(|d| &d.inner),
                            )
                        })
                    }),
                    Dtype::F64 => compute_typed::<$freq_scale, $amp_scale, f64, _>(py, samples, |s| {
                        py.detach(|| {
                            Spectrogram::<$freq_scale, $amp_scale, f64>::compute(
                                s,
                                &params.inner,
                                &filter_params.inner,
                                db.map(|d| &d.inner),
                            )
                        })
                    }),
                }
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
/// dtype : str, optional
///     Output precision: "float64" (default) or "float32".
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
    db: "typing.Optional[LogParams]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None, dtype: str = \"float64\")")]
pub fn compute_cqt_power_spectrogram(
    py: Python,
    samples: &Bound<'_, PyAny>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
    dtype: Option<&str>,
) -> PyResult<PySpectrogram> {
    match parse_dtype(dtype)? {
        Dtype::F32 => compute_typed::<Cqt, Power, f32, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Power, f32>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
        Dtype::F64 => compute_typed::<Cqt, Power, f64, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Power, f64>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
    }
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
/// dtype : str, optional
///     Output precision: "float64" (default) or "float32".
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
    db: "typing.Optional[LogParams]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None, dtype: str = \"float64\")")]
pub fn compute_cqt_magnitude_spectrogram(
    py: Python,
    samples: &Bound<'_, PyAny>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
    dtype: Option<&str>,
) -> PyResult<PySpectrogram> {
    match parse_dtype(dtype)? {
        Dtype::F32 => compute_typed::<Cqt, Magnitude, f32, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Magnitude, f32>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
        Dtype::F64 => compute_typed::<Cqt, Magnitude, f64, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Magnitude, f64>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
    }
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
/// dtype : str, optional
///     Output precision: "float64" (default) or "float32".
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
    db: "typing.Optional[LogParams]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, cqt: CqtParams, db: typing.Optional[LogParams] = None, dtype: str = \"float64\")")]
pub fn compute_cqt_db_spectrogram(
    py: Python,
    samples: &Bound<'_, PyAny>,
    params: &PySpectrogramParams,
    cqt: &PyCqtParams,
    db: Option<&PyLogParams>,
    dtype: Option<&str>,
) -> PyResult<PySpectrogram> {
    match parse_dtype(dtype)? {
        Dtype::F32 => compute_typed::<Cqt, Decibels, f32, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Decibels, f32>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
        Dtype::F64 => compute_typed::<Cqt, Decibels, f64, _>(py, samples, |s| {
            py.detach(|| {
                Spectrogram::<Cqt, Decibels, f64>::compute(
                    s,
                    &params.inner,
                    &cqt.inner,
                    db.map(|d| &d.inner),
                )
            })
        }),
    }
}

/// dtype-aware body for [`compute_chromagram`].
fn chromagram_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    stft_params: &crate::StftParams,
    sample_rate: f64,
    chroma_params: &crate::ChromaParams,
) -> PyResult<PyChromagram> {
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| chromagram::<T>(s, stft_params, sample_rate, chroma_params))?;
        Ok(PyChromagram::from_chromagram(py, result))
    })
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
/// `dtype` : str
///     Output precision: "float64" (default) or "float32".
///
/// Returns
/// -------
/// Chromagram
///     Chromagram result whose `.data` is a 2D `NumPy` array (12 x `n_frames`),
///     float64 or float32 depending on `dtype`
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    stft_params: "StftParams",
    sample_rate: "float",
    chroma_params: "ChromaParams",
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], stft_params: StftParams, sample_rate: float, chroma_params: ChromaParams, dtype: str = \"float64\")")]
pub fn compute_chromagram(
    py: Python,
    samples: &Bound<'_, PyAny>,
    stft_params: &PyStftParams,
    sample_rate: f64,
    chroma_params: &PyChromaParams,
    dtype: Option<&str>,
) -> PyResult<PyChromagram> {
    match parse_dtype(dtype)? {
        Dtype::F32 => {
            chromagram_typed::<f32>(py, samples, &stft_params.inner, sample_rate, &chroma_params.inner)
        }
        Dtype::F64 => {
            chromagram_typed::<f64>(py, samples, &stft_params.inner, sample_rate, &chroma_params.inner)
        }
    }
}

/// dtype-aware body for [`compute_mfcc`].
fn mfcc_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    stft_params: &crate::StftParams,
    sample_rate: f64,
    n_mels: NonZeroUsize,
    mfcc_params: &crate::MfccParams,
) -> PyResult<PyMfcc> {
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| mfcc::<T>(s, stft_params, sample_rate, n_mels, mfcc_params))?;
        Ok(PyMfcc::from_mfcc(py, result))
    })
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
/// `dtype` : str
///     Output precision: "float64" (default) or "float32".
///
/// Returns
/// -------
/// Mfcc
///     MFCC result whose `.data` is a 2D `NumPy` array (`n_mfcc` x `n_frames`),
///     float64 or float32 depending on `dtype`
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    stft_params: "StftParams",
    sample_rate: "float",
    n_mels: "int",
    mfcc_params: "MfccParams",
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], stft_params: StftParams, sample_rate: float, n_mels: int, mfcc_params: MfccParams, dtype: str = \"float64\")")]
pub fn compute_mfcc(
    py: Python,
    samples: &Bound<'_, PyAny>,
    stft_params: &PyStftParams,
    sample_rate: f64,
    n_mels: usize,
    mfcc_params: &PyMfccParams,
    dtype: Option<&str>,
) -> PyResult<PyMfcc> {
    let n_mels = NonZeroUsize::new(n_mels).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_mels must be a positive integer")
    })?;
    match parse_dtype(dtype)? {
        Dtype::F32 => {
            mfcc_typed::<f32>(py, samples, &stft_params.inner, sample_rate, n_mels, &mfcc_params.inner)
        }
        Dtype::F64 => {
            mfcc_typed::<f64>(py, samples, &stft_params.inner, sample_rate, n_mels, &mfcc_params.inner)
        }
    }
}

// ---------------------------------------------------------------------------
// dtype-aware generic bodies for the 1D transforms.
//
// Each public `#[pyfunction]` below parses the `dtype` string and dispatches to
// one of these monomorphic helpers. The result array carries the requested
// precision (`float32` -> f32/complex64, `float64` -> f64/complex128).
// ---------------------------------------------------------------------------

fn stft_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    params: &crate::SpectrogramParams,
) -> PyResult<PyStftResult>
where
    num_complex::Complex<T>: numpy::Element,
{
    with_real_1d::<T, _, _>(py, samples, |s| {
        let planner = SpectrogramPlanner::new();
        let result = py.detach(|| planner.compute_stft::<T>(s, params))?;
        Ok(PyStftResult::from_stft_result(py, result))
    })
}

fn fft_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
) -> PyResult<Py<PyAny>>
where
    num_complex::Complex<T>: numpy::Element,
{
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| fft::<T>(s, n_fft))?;
        Ok(array1_to_py(py, result))
    })
}

fn rfft_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
) -> PyResult<Py<PyAny>> {
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| rfft::<T>(s, n_fft))?;
        Ok(array1_to_py(py, result))
    })
}

fn power_spectrum_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
    window: Option<crate::WindowType>,
) -> PyResult<Py<PyAny>> {
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| power_spectrum::<T>(s, n_fft, window))?;
        Ok(vec1_to_py(py, result.to_vec()))
    })
}

fn magnitude_spectrum_typed<T: PyScalar>(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
    window: Option<crate::WindowType>,
) -> PyResult<Py<PyAny>> {
    with_real_1d::<T, _, _>(py, samples, |s| {
        let result = py.detach(|| magnitude_spectrum::<T>(s, n_fft, window))?;
        Ok(vec1_to_py(py, result.to_vec()))
    })
}

fn irfft_typed<T: PyScalar>(
    py: Python<'_>,
    spectrum: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
) -> PyResult<Py<PyAny>>
where
    num_complex::Complex<T>: numpy::Element,
{
    with_complex_1d::<T, _, _>(py, spectrum, |s| {
        let result = py.detach(|| irfft::<T>(s, n_fft))?;
        Ok(vec1_to_py(py, result.to_vec()))
    })
}

fn istft_typed<T: PyScalar>(
    py: Python<'_>,
    stft_matrix: &Bound<'_, PyAny>,
    n_fft: NonZeroUsize,
    hop_size: NonZeroUsize,
    window: crate::WindowType,
    center: bool,
) -> PyResult<Py<PyAny>>
where
    num_complex::Complex<T>: numpy::Element,
{
    let matrix = complex_2d_owned::<T>(py, stft_matrix)?;
    let result = py.detach(|| istft::<T>(&matrix, n_fft, hop_size, window, center))?;
    Ok(vec1_to_py(py, result.to_vec()))
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
/// :param `dtype` - Output precision: "float64" (default) or "float32". The
///     returned complex array is complex128 or complex64 to match.
///
/// Returns
/// -------
///
/// StftResult
///     STFT result whose `.data` is a complex 2D `NumPy` array
///     (`n_fft/2+1` x `n_frames`), complex128 or complex64 depending on `dtype`
#[pyfunction]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    params: "SpectrogramParams",
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], params: SpectrogramParams, dtype: str = \"float64\")")]
pub fn compute_stft(
    py: Python,
    samples: &Bound<'_, PyAny>,
    params: &PySpectrogramParams,
    dtype: Option<&str>,
) -> PyResult<PyStftResult> {
    match parse_dtype(dtype)? {
        Dtype::F32 => stft_typed::<f32>(py, samples, &params.inner),
        Dtype::F64 => stft_typed::<f64>(py, samples, &params.inner),
    }
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
/// :param `dtype` - Output precision: "float64" (default) or "float32". The
///     returned complex array is complex128 or complex64 to match.
///
/// Returns
/// -------
///
/// Complex FFT as a 1D `NumPy` array with length `n_fft/2+1`, complex128 or complex64
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "Optional[int]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: Optional[int]=None, dtype: str = \"float64\")")]
pub fn compute_fft(
    py: Python,
    samples: &Bound<'_, PyAny>,
    n_fft: Option<usize>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = match n_fft {
        Some(n) => n,
        None => samples.len()?,
    };
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    match parse_dtype(dtype)? {
        Dtype::F32 => fft_typed::<f32>(py, samples, n_fft),
        Dtype::F64 => fft_typed::<f64>(py, samples, n_fft),
    }
}

/// Compute the real FFT of a signal.
///
/// Parameters
/// -----------
///
/// :param `samples` - Audio samples as a 1D `NumPy` array
/// :param `n_fft` - FFT size
/// :param `dtype` - Output precision: "float64" (default) or "float32".
#[pyfunction]
#[inline]
#[pyo3(signature = (
    samples: "numpy.typing.NDArray[numpy.float64]",
    n_fft: "int",
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int, dtype: str = \"float64\") -> numpy.typing.NDArray[numpy.float64]")]
pub fn compute_rfft(
    py: Python,
    samples: &Bound<'_, PyAny>,
    n_fft: usize,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    match parse_dtype(dtype)? {
        Dtype::F32 => rfft_typed::<f32>(py, samples, n_fft),
        Dtype::F64 => rfft_typed::<f64>(py, samples, n_fft),
    }
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
/// :param `dtype` - Output precision: "float64" (default) or "float32".
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
    window: "typing.Optional[WindowType]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int, window: typing.Optional[WindowType] = None, dtype: str = \"float64\")")]
pub fn compute_power_spectrum(
    py: Python,
    samples: &Bound<'_, PyAny>,
    n_fft: usize,
    window: Option<PyWindowType>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let window_type = window.map(|w| w.inner);
    match parse_dtype(dtype)? {
        Dtype::F32 => power_spectrum_typed::<f32>(py, samples, n_fft, window_type),
        Dtype::F64 => power_spectrum_typed::<f64>(py, samples, n_fft, window_type),
    }
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
/// :param `dtype` - Output precision: "float64" (default) or "float32".
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
    window: "typing.Optional[WindowType]" = None,
    dtype: "str" = None
), text_signature = "(samples: numpy.typing.NDArray[numpy.float64], n_fft: int, window: typing.Optional[WindowType] = None, dtype: str = \"float64\")")]
pub fn compute_magnitude_spectrum(
    py: Python,
    samples: &Bound<'_, PyAny>,
    n_fft: usize,
    window: Option<PyWindowType>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let window_type = window.map(|w| w.inner);
    match parse_dtype(dtype)? {
        Dtype::F32 => magnitude_spectrum_typed::<f32>(py, samples, n_fft, window_type),
        Dtype::F64 => magnitude_spectrum_typed::<f64>(py, samples, n_fft, window_type),
    }
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
/// :param `dtype` - Output precision: "float64" (default) or "float32". The
///     input is read as complex128 or complex64 to match.
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
    n_fft: "int",
    dtype: "str" = None
), text_signature = "(spectrum: numpy.typing.NDArray[numpy.complex128], n_fft: int, dtype: str = \"float64\")")]
pub fn compute_irfft(
    py: Python,
    spectrum: &Bound<'_, PyAny>,
    n_fft: usize,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    match parse_dtype(dtype)? {
        Dtype::F32 => irfft_typed::<f32>(py, spectrum, n_fft),
        Dtype::F64 => irfft_typed::<f64>(py, spectrum, n_fft),
    }
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
/// :param `dtype` - Output precision: "float64" (default) or "float32". The
///     input is read as complex128 or complex64 to match.
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
    center: "bool" = true,
    dtype: "str" = None
), text_signature = "(stft_matrix: numpy.typing.NDArray[numpy.complex64], n_fft: int, hop_size: int, window: WindowType, center: bool = True, dtype: str = \"float64\")")]
pub fn compute_istft(
    py: Python,
    stft_matrix: &Bound<'_, PyAny>,
    n_fft: usize,
    hop_size: usize,
    window: PyWindowType,
    center: bool,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let n_fft = NonZeroUsize::new(n_fft).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("n_fft must be non-zero positive integer")
    })?;
    let hop_size = NonZeroUsize::new(hop_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("hop_size must be non-zero positive integer")
    })?;
    let window = window.inner;
    match parse_dtype(dtype)? {
        Dtype::F32 => istft_typed::<f32>(py, stft_matrix, n_fft, hop_size, window, center),
        Dtype::F64 => istft_typed::<f64>(py, stft_matrix, n_fft, hop_size, window, center),
    }
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
