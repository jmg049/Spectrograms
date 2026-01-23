"""Type stubs for spectrograms package."""

from typing import Optional, Tuple, List
import numpy as np
import numpy.typing as npt
from spectrograms import ErbPowerPlan

__version__: str

# ============================================================================
# Exception Types
# ============================================================================

class SpectrogramError(Exception):
    """Base exception for all spectrogram errors."""

    ...

class InvalidInputError(SpectrogramError):
    """Exception raised when invalid input is provided."""

    ...

class DimensionMismatchError(SpectrogramError):
    """Exception raised when array dimensions don't match expected values."""

    ...

class FFTBackendError(SpectrogramError):
    """Exception raised when an error occurs in the FFT backend."""

    ...

class InternalError(SpectrogramError):
    """Exception raised when an internal error occurs."""

    ...

# ============================================================================
# Parameter Classes
# ============================================================================

class WindowType:
    """Window function type for spectral analysis.

    Different window types provide different trade-offs between frequency resolution
    and spectral leakage in FFT-based analysis.
    """

    @classmethod
    def rectangular(cls) -> WindowType:
        """Create a rectangular (no) window - best frequency resolution but high leakage.

        :return: Rectangular window type
        """
        ...

    @classmethod
    def hanning(cls) -> WindowType:
        """Create a Hanning window - good general-purpose window with moderate leakage.

        :return: Hanning window type
        """
        ...

    @classmethod
    def hamming(cls) -> WindowType:
        """Create a Hamming window - similar to Hanning but slightly different coefficients.

        :return: Hamming window type
        """
        ...

    @classmethod
    def blackman(cls) -> WindowType:
        """Create a Blackman window - low leakage but wider main lobe.

        :return: Blackman window type
        """
        ...

    @classmethod
    def kaiser(cls, beta: float) -> WindowType:
        """Create a Kaiser window with the given beta parameter.

        :param beta: Beta parameter controlling trade-off between resolution and leakage
        :return: Kaiser window type
        """
        ...

    @classmethod
    def gaussian(cls, std: float) -> WindowType:
        """Create a Gaussian window with the given standard deviation.

        :param std: Standard deviation parameter controlling window width
        :return: Gaussian window type
        """
        ...

class StftParams:
    """STFT parameters for spectrogram computation."""

    def __init__(
        self, n_fft: int, hop_size: int, window: WindowType, centre: bool = True
    ) -> None:
        """Create STFT parameters.

        :param n_fft: FFT window size (must be > 0)
        :param hop_size: Number of samples between successive frames (must be > 0 and <= n_fft)
        :param window: Window function
        :param centre: Whether to center frames (pad with zeros at start/end)
        """
        ...

    @property
    def n_fft(self) -> int:
        """FFT window size."""
        ...

    @property
    def hop_size(self) -> int:
        """Number of samples between successive frames."""
        ...

    @property
    def window(self) -> WindowType:
        """Window function."""
        ...

    @property
    def centre(self) -> bool:
        """Whether to center frames."""
        ...

class LogParams:
    """Decibel conversion parameters."""

    def __init__(self, floor_db: float) -> None:
        """Create decibel conversion parameters.

        :param floor_db: Minimum power in decibels (values below this are clipped)
        """
        ...

    @property
    def floor_db(self) -> float:
        """Minimum power in decibels."""
        ...

class SpectrogramParams:
    """Spectrogram computation parameters."""

    def __init__(self, stft: StftParams, sample_rate: float) -> None:
        """Create spectrogram parameters.

        :param stft: STFT parameters
        :param sample_rate: Sample rate in Hz
        """
        ...

    @property
    def stft(self) -> StftParams:
        """STFT parameters."""
        ...

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        ...

    @classmethod
    def speech_default(cls, sample_rate: float) -> SpectrogramParams:
        """Create default parameters for speech processing.

        Uses n_fft=512, hop_size=160, Hanning window, centre=true

        :param sample_rate: Sample rate in Hz
        :return: SpectrogramParams for speech
        """
        ...

    @classmethod
    def music_default(cls, sample_rate: float) -> SpectrogramParams:
        """Create default parameters for music processing.

        Uses n_fft=2048, hop_size=512, Hanning window, centre=true

        :param sample_rate: Sample rate in Hz
        :return: SpectrogramParams for music
        """
        ...

class MelParams:
    """Mel-scale filterbank parameters."""

    def __init__(self, n_mels: int, f_min: float, f_max: float) -> None:
        """Create mel filterbank parameters.

        :param n_mels: Number of mel bands
        :param f_min: Minimum frequency in Hz
        :param f_max: Maximum frequency in Hz
        """
        ...

    @property
    def n_mels(self) -> int:
        """Number of mel bands."""
        ...

    @property
    def f_min(self) -> float:
        """Minimum frequency in Hz."""
        ...

    @property
    def f_max(self) -> float:
        """Maximum frequency in Hz."""
        ...

class ErbParams:
    """ERB-scale (Equivalent Rectangular Bandwidth) filterbank parameters."""

    def __init__(self, n_filters: int, f_min: float, f_max: float) -> None:
        """Create ERB filterbank parameters.

        :param n_filters: Number of ERB filters
        :param f_min: Minimum frequency in Hz
        :param f_max: Maximum frequency in Hz
        """
        ...

    @property
    def n_filters(self) -> int:
        """Number of ERB filters."""
        ...

    @property
    def f_min(self) -> float:
        """Minimum frequency in Hz."""
        ...

    @property
    def f_max(self) -> float:
        """Maximum frequency in Hz."""
        ...

class LogHzParams:
    """Logarithmic frequency scale parameters."""

    def __init__(self, n_bins: int, f_min: float, f_max: float) -> None:
        """Create logarithmic Hz filterbank parameters.

        :param n_bins: Number of logarithmically-spaced frequency bins
        :param f_min: Minimum frequency in Hz
        :param f_max: Maximum frequency in Hz
        """
        ...

    @property
    def n_bins(self) -> int:
        """Number of frequency bins."""
        ...

    @property
    def f_min(self) -> float:
        """Minimum frequency in Hz."""
        ...

    @property
    def f_max(self) -> float:
        """Maximum frequency in Hz."""
        ...

class CqtParams:
    """Constant-Q Transform parameters."""

    def __init__(self, bins_per_octave: int, n_octaves: int, f_min: float) -> None:
        """Create CQT parameters.

        :param bins_per_octave: Number of bins per octave (e.g., 12 for semitones)
        :param n_octaves: Number of octaves to span
        :param f_min: Minimum frequency in Hz
        """
        ...

    @property
    def num_bins(self) -> int:
        """Total number of CQT bins."""
        ...

class ChromaParams:
    """Chromagram (pitch class profile) parameters."""

    def __init__(
        self,
        tuning: float = 440.0,
        f_min: float = 32.7,
        f_max: float = 4186.0,
        norm: Optional[str] = "l2",
    ) -> None:
        """Create chromagram parameters.

        :param tuning: Reference tuning frequency in Hz (default: 440.0 for A4)
        :param f_min: Minimum frequency in Hz (default: 32.7, C1)
        :param f_max: Maximum frequency in Hz (default: 4186.0, C8)
        :param norm: Normalization method: "l1", "l2", "max", or None (default: "l2")
        """
        ...

    @classmethod
    def music_standard(cls) -> ChromaParams:
        """Create standard chroma parameters for music analysis.

        :return: ChromaParams with standard settings
        """
        ...

    @property
    def tuning(self) -> float:
        """Reference tuning frequency in Hz."""
        ...

    @property
    def f_min(self) -> float:
        """Minimum frequency in Hz."""
        ...

    @property
    def f_max(self) -> float:
        """Maximum frequency in Hz."""
        ...

class MfccParams:
    """MFCC (Mel-Frequency Cepstral Coefficients) parameters."""

    def __init__(self, n_mfcc: int = 13) -> None:
        """Create MFCC parameters.

        :param n_mfcc: Number of MFCC coefficients to compute (default: 13)
        """
        ...

    @classmethod
    def speech_standard(cls) -> MfccParams:
        """Standard MFCC parameters for speech recognition (13 coefficients).

        :return: MfccParams with standard settings
        """
        ...

    @property
    def n_mfcc(self) -> int:
        """Number of MFCC coefficients."""
        ...

class Spectrogram:
    """Spectrogram computation result.

    Contains the spectrogram data as a NumPy array along with frequency and time axes.
    """

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Get the spectrogram data as a 2D NumPy array with shape (n_bins, n_frames)."""
        ...

    @property
    def frequencies(self) -> list[float]:
        """Get the frequency axis values (Hz or scale-specific units)."""
        ...

    @property
    def times(self) -> list[float]:
        """Get the time axis values in seconds."""
        ...

    @property
    def n_bins(self) -> int:
        """Get the number of frequency bins."""
        ...

    @property
    def n_frames(self) -> int:
        """Get the number of time frames."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the spectrogram as (n_bins, n_frames)."""
        ...

    def frequency_range(self) -> tuple[float, float]:
        """Get the frequency range as (f_min, f_max) in Hz or scale-specific units."""
        ...

    def duration(self) -> float:
        """Get the total duration in seconds."""
        ...

    def db_range(self) -> Optional[tuple[float, float]]:
        """Get the decibel range if applicable.

        :return: Optional tuple (min_db, max_db) for decibel-scaled spectrograms, None otherwise
        """
        ...

    @property
    def params(self) -> SpectrogramParams:
        """Get the computation parameters."""
        ...

# ============================================================================
# Planner and Plan Classes
# ============================================================================

class SpectrogramPlanner:
    """Spectrogram planner for creating reusable computation plans.

    Creating a plan is more expensive than a single computation, but plans can be
    reused for multiple signals with the same parameters, providing significant
    performance benefits for batch processing.
    """

    def __init__(self) -> None: ...
    def linear_power_plan(self, params: SpectrogramParams) -> LinearPowerPlan:
        """Create a plan for computing linear power spectrograms.
        :param params: Spectrogram parameters

        :return: LinearPowerPlan
        """
        ...

    def linear_magnitude_plan(self, params: SpectrogramParams) -> LinearMagnitudePlan:
        """Create a plan for computing linear magnitude spectrograms.

        :param params: Spectrogram parameters

        :return: LinearMagnitudePlan
        """
        ...

    def linear_db_plan(
        self, params: SpectrogramParams, db_params: LogParams
    ) -> LinearDbPlan:
        """Create a plan for computing linear decibel spectrograms.
        :param params: Spectrogram parameters
        :param db_params: Decibel conversion parameters
        :return: LinearDbPlan
        """
        ...

    def mel_power_plan(
        self, params: SpectrogramParams, mel_params: MelParams
    ) -> MelPowerPlan:
        """Create a plan for computing mel power spectrograms.

        :param params: Spectrogram parameters
        :param mel_params: Mel filterbank parameters

        :return: MelPowerPlan
        """
        ...

    def mel_magnitude_plan(
        self, params: SpectrogramParams, mel_params: MelParams
    ) -> MelMagnitudePlan:
        """Create a plan for computing mel magnitude spectrograms.

        :param params: Spectrogram parameters
        :param mel_params: Mel filterbank parameters

        :return: MelMagnitudePlan
        """
        ...

    def mel_db_plan(
        self, params: SpectrogramParams, mel_params: MelParams, db_params: LogParams
    ) -> MelDbPlan:
        """Create a plan for computing mel decibel spectrograms.

        :param params: Spectrogram parameters
        :param mel_params: Mel filterbank parameters
        :param db_params: Decibel conversion parameters

        :return: MelDbPlan
        """
        ...

    def erb_power_plan(
        self, params: SpectrogramParams, erb_params: ErbParams
    ) -> ErbPowerPlan:
        """Create a plan for computing ERB power spectrograms.
        :param params: Spectrogram parameters
        :param erb_params: ERB filterbank parameters

        :return: ErbPowerPlan
        """
        ...

    def erb_magnitude_plan(
        self, params: SpectrogramParams, erb_params: ErbParams
    ) -> ErbMagnitudePlan:
        """Create a plan for computing ERB magnitude spectrograms.

        :param params: Spectrogram parameters
        :param erb_params: ERB filterbank parameters

        :return: ErbMagnitudePlan
        """
        ...

    def erb_db_plan(
        self, params: SpectrogramParams, erb_params: ErbParams, db_params: LogParams
    ) -> ErbDbPlan:
        """Create a plan for computing ERB decibel spectrograms.

        :param params: Spectrogram parameters
        :param erb_params: ERB filterbank parameters
        :param db_params: Decibel conversion parameters

        :return: ErbDbPlan
        """
        ...

    def loghz_power_plan(
        self, params: SpectrogramParams, loghz_params: LogHzParams
    ) -> LogHzPowerPlan:
        """Create a plan for computing logarithmic Hz power spectrograms.

        :param params: Spectrogram parameters
        :param loghz_params: Logarithmic Hz filterbank parameters

        :return: LogHzPowerPlan
        """
        ...

    def loghz_magnitude_plan(
        self, params: SpectrogramParams, loghz_params: LogHzParams
    ) -> LogHzMagnitudePlan:
        """Create a plan for computing logarithmic Hz magnitude spectrograms.

        :param params: Spectrogram parameters
        :param loghz_params: Logarithmic Hz filterbank parameters

        :return: LogHzMagnitudePlan
        """
        ...

    def loghz_db_plan(
        self, params: SpectrogramParams, loghz_params: LogHzParams, db_params: LogParams
    ) -> LogHzDbPlan:
        """Create a plan for computing logarithmic Hz decibel spectrograms.

        :param params: Spectrogram parameters
        :param loghz_params: Logarithmic Hz filterbank parameters
        :param db_params: Decibel conversion parameters

        :return: LogHzDbPlan
        """
        ...

    def cqt_power_plan(
        self, params: SpectrogramParams, cqt_params: CqtParams
    ) -> CqtPowerPlan:
        """Create a plan for computing CQT power spectrograms.

        :param params: Spectrogram parameters
        :param cqt_params: CQT parameters

        :return: CqtPowerPlan
        """
        ...

    def cqt_magnitude_plan(
        self, params: SpectrogramParams, cqt_params: CqtParams
    ) -> CqtMagnitudePlan:
        """Create a plan for computing CQT magnitude spectrograms.

        :param params: Spectrogram parameters
        :param cqt_params: CQT parameters

        :return: CqtMagnitudePlan
        """
        ...

    def cqt_db_plan(
        self, params: SpectrogramParams, cqt_params: CqtParams, db_params: LogParams
    ) -> CqtDbPlan:
        """Create a plan for computing CQT decibel spectrograms.

        :param params: Spectrogram parameters
        :param cqt_params: CQT parameters
        :param db_params: Decibel conversion parameters

        :return: CqtDbPlan
        """
        ...

class LinearPowerPlan:
    """Plan for computing linear power spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram:
        """Compute a spectrogram from audio samples.

        :param samples: Audio samples as a 1D NumPy array
        :return: Computed spectrogram
        """
        ...

    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]:
        """Compute a single frame of the spectrogram.

        :param samples: Audio samples as a 1D NumPy array
        :param frame_idx: Frame index to compute
        :return: 1D array containing the frame data
        """
        ...

    def output_shape(self, signal_length: int) -> Tuple[int, int]:
        """Get the output shape for a given signal length.

        :param signal_length: Length of input signal
        :return: Tuple (n_bins, n_frames)
        """
        ...

class LinearMagnitudePlan:
    """Plan for computing linear magnitude spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class LinearDbPlan:
    """Plan for computing linear decibel spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram:
        """
        Compute a linear decibel spectrogram from audio samples.

        :param samples: Audio samples as a 1D NumPy array
        :return: Computed spectrogram in decibels
        """
        ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Compute a single frame of the linear decibel spectrogram.

        :param samples: Audio samples as a 1D NumPy array
        :param frame_idx: Frame index to compute

        :return: 1D array containing the frame data
        """
        ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class MelPowerPlan:
    """Plan for computing mel power spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram:
        """
        Compute a mel power spectrogram from audio samples.

        :param samples: Audio samples as a 1D NumPy array

        :return: Computed mel power spectrogram
        """
        ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Compute a single frame of the mel power spectrogram.

        :param samples: Audio samples as a 1D NumPy array
        :param frame_idx: Frame index to compute
        :return: 1D array containing the frame data
        """
        ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class MelMagnitudePlan:
    """Plan for computing mel magnitude spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class MelDbPlan:
    """Plan for computing mel decibel spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class ErbPowerPlan:
    """Plan for computing ERB power spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class ErbMagnitudePlan:
    """Plan for computing ERB magnitude spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class ErbDbPlan:
    """Plan for computing ERB decibel spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class LogHzPowerPlan:
    """Plan for computing logarithmic Hz power spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class LogHzMagnitudePlan:
    """Plan for computing logarithmic Hz magnitude spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class LogHzDbPlan:
    """Plan for computing logarithmic Hz decibel spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class CqtPowerPlan:
    """Plan for computing CQT power spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class CqtMagnitudePlan:
    """Plan for computing CQT magnitude spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

class CqtDbPlan:
    """Plan for computing CQT decibel spectrograms."""

    def compute(self, samples: npt.NDArray[np.float64]) -> Spectrogram: ...
    def compute_frame(
        self, samples: npt.NDArray[np.float64], frame_idx: int
    ) -> npt.NDArray[np.float64]: ...
    def output_shape(self, signal_length: int) -> Tuple[int, int]: ...

# ============================================================================
# Spectrogram Compute Functions
# ============================================================================

def compute_linear_power_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a linear power spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with linear frequency scale and power amplitude scale
    """
    ...

def compute_linear_magnitude_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a linear magnitude spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with linear frequency scale and magnitude amplitude scale
    """
    ...

def compute_linear_db_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a linear decibel spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with linear frequency scale and decibel amplitude scale
    """
    ...

def compute_mel_power_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    mel_params: MelParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a mel power spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param mel_params: Mel filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with mel frequency scale and power amplitude scale
    """
    ...

def compute_mel_magnitude_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    mel_params: MelParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a mel magnitude spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param mel_params: Mel filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with mel frequency scale and magnitude amplitude scale
    """
    ...

def compute_mel_db_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    mel_params: MelParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a mel decibel spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param mel_params: Mel filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with mel frequency scale and decibel amplitude scale
    """
    ...

def compute_erb_power_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    erb_params: ErbParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute an ERB power spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param erb_params: ERB filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with ERB frequency scale and power amplitude scale
    """
    ...

def compute_erb_magnitude_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    erb_params: ErbParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute an ERB magnitude spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param erb_params: ERB filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with ERB frequency scale and magnitude amplitude scale
    """
    ...

def compute_erb_db_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    erb_params: ErbParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute an ERB decibel spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param erb_params: ERB filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with ERB frequency scale and decibel amplitude scale
    """
    ...

def compute_loghz_power_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    loghz_params: LogHzParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a logarithmic Hz power spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param loghz_params: LogHz filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with logarithmic Hz frequency scale and power amplitude scale
    """
    ...

def compute_loghz_magnitude_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    loghz_params: LogHzParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a logarithmic Hz magnitude spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param loghz_params: LogHz filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with logarithmic Hz frequency scale and magnitude amplitude scale
    """
    ...

def compute_loghz_db_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    loghz_params: LogHzParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a logarithmic Hz decibel spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param loghz_params: LogHz filterbank parameters
    :param db: Optional decibel scaling parameters
    :return: Spectrogram with logarithmic Hz frequency scale and decibel amplitude scale
    """
    ...

def compute_cqt_power_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    cqt_params: CqtParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a Constant-Q Transform power spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param cqt_params: CQT parameters
    :param db: Optional decibel scaling parameters
    :return: CQT spectrogram with power amplitude scale
    """
    ...

def compute_cqt_magnitude_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    cqt_params: CqtParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a Constant-Q Transform magnitude spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param cqt_params: CQT parameters
    :param db: Optional decibel scaling parameters
    :return: CQT spectrogram with magnitude amplitude scale
    """
    ...

def compute_cqt_db_spectrogram(
    samples: npt.NDArray[np.float64],
    params: SpectrogramParams,
    cqt_params: CqtParams,
    db: Optional[LogParams] = None,
) -> Spectrogram:
    """Compute a Constant-Q Transform decibel spectrogram.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :param cqt_params: CQT parameters
    :param db: Optional decibel scaling parameters
    :return: CQT spectrogram with decibel amplitude scale
    """
    ...

# ============================================================================
# Additional Compute Functions
# ============================================================================

def compute_chromagram(
    samples: npt.NDArray[np.float64],
    stft_params: StftParams,
    sample_rate: float,
    chroma_params: ChromaParams,
) -> npt.NDArray[np.float64]:
    """Compute a chromagram (pitch class profile).

    :param samples: Audio samples as a 1D NumPy array
    :param stft_params: STFT parameters
    :param sample_rate: Sample rate in Hz
    :param chroma_params: Chromagram parameters
    :return: Chromagram as a 2D NumPy array (12 × n_frames)
    """
    ...

def compute_mfcc(
    samples: npt.NDArray[np.float64],
    stft_params: StftParams,
    sample_rate: float,
    n_mels: int,
    mfcc_params: MfccParams,
) -> npt.NDArray[np.float64]:
    """Compute MFCCs (Mel-Frequency Cepstral Coefficients).

    :param samples: Audio samples as a 1D NumPy array
    :param stft_params: STFT parameters
    :param sample_rate: Sample rate in Hz
    :param n_mels: Number of mel bands
    :param mfcc_params: MFCC parameters
    :return: MFCCs as a 2D NumPy array (n_mfcc × n_frames)
    """
    ...

def compute_stft(
    samples: npt.NDArray[np.float64], params: SpectrogramParams
) -> npt.NDArray[np.complex128]:
    """Compute the raw STFT (Short-Time Fourier Transform).

    Returns the complex-valued STFT matrix before any frequency mapping or amplitude scaling.

    :param samples: Audio samples as a 1D NumPy array
    :param params: Spectrogram parameters
    :return: Complex STFT as a 2D NumPy array of complex128 (n_fft/2+1 × n_frames)
    """
    ...

# ============================================================================
# FFT Functions
# ============================================================================

def compute_rfft(
    samples: npt.NDArray[np.float64], n_fft: int
) -> npt.NDArray[np.complex128]:
    """Compute the real-to-complex FFT of a signal.

    Computes the FFT of a real-valued signal, returning only positive frequencies
    (exploiting Hermitian symmetry).

    :param samples: Audio samples as a 1D NumPy array (length must equal n_fft)
    :param n_fft: FFT size
    :return: Complex FFT as a 1D NumPy array of complex128 with length n_fft/2+1
    :raises DimensionMismatchError: If samples length doesn't equal n_fft
    """
    ...

def compute_irfft(
    spectrum: npt.NDArray[np.complex128], n_fft: int
) -> npt.NDArray[np.float64]:
    """Compute the inverse real FFT (complex to real).

    Converts a complex frequency-domain representation back to real time-domain samples.
    Expects only positive frequencies (Hermitian symmetry is assumed).

    :param spectrum: Complex frequency spectrum as a 1D NumPy array (length must equal n_fft/2+1)
    :param n_fft: FFT size (determines output length)
    :return: Real time-domain signal as a 1D NumPy array with length n_fft
    :raises DimensionMismatchError: If spectrum length doesn't equal n_fft/2+1
    """
    ...

def compute_power_spectrum(
    samples: npt.NDArray[np.float64], n_fft: int, window: Optional[WindowType] = None
) -> npt.NDArray[np.float64]:
    """Compute the power spectrum of a signal (|X|²).

    Applies an optional window function and computes the power spectrum via FFT.
    Returns only positive frequencies.

    :param samples: Audio samples as a 1D NumPy array (length must equal n_fft)
    :param n_fft: FFT size
    :param window: Optional window function (None for rectangular window)
    :return: Power spectrum as a 1D NumPy array with length n_fft/2+1
    :raises DimensionMismatchError: If samples length doesn't equal n_fft
    """
    ...

def compute_magnitude_spectrum(
    samples: npt.NDArray[np.float64], n_fft: int, window: Optional[WindowType] = None
) -> npt.NDArray[np.float64]:
    """Compute the magnitude spectrum of a signal (|X|).

    Applies an optional window function and computes the magnitude spectrum via FFT.
    Returns only positive frequencies.

    :param samples: Audio samples as a 1D NumPy array (length must equal n_fft)
    :param n_fft: FFT size
    :param window: Optional window function (None for rectangular window)
    :return: Magnitude spectrum as a 1D NumPy array with length n_fft/2+1
    :raises DimensionMismatchError: If samples length doesn't equal n_fft
    """
    ...

def compute_istft(
    stft_matrix: npt.NDArray[np.complex128],
    n_fft: int,
    hop_size: int,
    window: WindowType,
    center: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute the inverse STFT (Short-Time Fourier Transform).

    Reconstructs a time-domain signal from its STFT using overlap-add synthesis.

    :param stft_matrix: Complex STFT as a 2D NumPy array (n_fft/2+1 × n_frames)
    :param n_fft: FFT size
    :param hop_size: Number of samples between successive frames (must match forward STFT)
    :param window: Window function to apply (should match forward STFT window)
    :param center: If true, assume the forward STFT was centered (must match forward STFT)
    :return: Reconstructed time-domain signal as a 1D NumPy array
    :raises DimensionMismatchError: If STFT matrix shape doesn't match parameters
    """
    ...
