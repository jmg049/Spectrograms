"""
JAX convenience functions for spectrograms.

This module provides high-level convenience methods for converting spectrograms
to JAX arrays with optional metadata preservation and device support.

Usage:
    import spectrograms as sg
    import spectrograms.jax  # Adds .to_jax() method

    spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

    # Simple conversion
    array = spec.to_jax(device='gpu')

    # With metadata
    result = spec.to_jax(device='gpu', with_metadata=True)
    result.array         # jax.Array
    result.frequencies   # np.ndarray
    result.times         # np.ndarray
    result.params        # SpectrogramParams

    # Batching
    batch = sg.jax.batch([spec1, spec2, spec3], device='gpu')
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.dlpack
except ImportError:
    raise ImportError(
        "JAX is not installed. Install it with:\n"
        "  pip install jax jaxlib\n"
        "or visit https://jax.readthedocs.io for installation instructions."
    )

from . import _spectrograms  # type: ignore


@dataclass
class JaxSpectrogram:
    """
    A spectrogram converted to JAX with optional metadata preserved.

    Attributes:
        array: JAX array containing the spectrogram data
        frequencies: Frequency axis values (Hz or scale-specific)
        times: Time axis values (seconds)
        params: Original computation parameters
        shape: Shape of the data (n_bins, n_frames)
        db_range: Decibel range if applicable (min_db, max_db)
    """

    array: jax.Array
    frequencies: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    params: Optional[object] = None
    shape: Optional[tuple[int, int]] = None
    db_range: Optional[tuple[float, float]] = None

    def to_device(self, device: str | jax.Device) -> "JaxSpectrogram":  # type: ignore
        """Move array to a different device."""
        if isinstance(device, str):
            device = jax.devices(device)[0]
        return JaxSpectrogram(
            array=jax.device_put(self.array, device),
            frequencies=self.frequencies,
            times=self.times,
            params=self.params,
            shape=self.shape,
            db_range=self.db_range,
        )

    def cpu(self) -> "JaxSpectrogram":
        """Move array to CPU."""
        return self.to_device("cpu")

    def gpu(self, index: int = 0) -> "JaxSpectrogram":
        """Move array to GPU."""
        return self.to_device("gpu")


@dataclass
class JaxChromagram:
    """
    A chromagram converted to JAX with optional metadata preserved.

    Attributes:
        array: JAX array containing the chromagram data (12 x n_frames)
        labels: Pitch class labels (C, C#, D, ...)
        params: Original computation parameters
        shape: Shape of the data (12, n_frames)
        n_frames: Number of time frames
    """

    array: jax.Array
    labels: Optional[list[str]] = None
    params: Optional[object] = None
    shape: Optional[tuple[int, int]] = None
    n_frames: Optional[int] = None

    def to_device(self, device: str | jax.Device) -> JaxChromagram:  # type: ignore
        """Move array to a different device."""
        if isinstance(device, str):
            device = jax.devices(device)[0]
        return JaxChromagram(
            array=jax.device_put(self.array, device),
            labels=self.labels,
            params=self.params,
            shape=self.shape,
            n_frames=self.n_frames,
        )

    def cpu(self) -> JaxChromagram:
        """Move array to CPU."""
        return self.to_device("cpu")

    def gpu(self, index: int = 0) -> JaxChromagram:
        """Move array to GPU."""
        return self.to_device("gpu")


def _spectrogram_to_jax(
    self,
    device: Optional[str | jax.Device] = None,  # type: ignore
    with_metadata: bool = False,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array | JaxSpectrogram:
    """
    Convert spectrogram to JAX array with optional metadata preservation.

    This uses the DLPack protocol for conversion when possible.

    Parameters
    ----------
    device : str or jax.Device, optional
        Target device ('cpu', 'gpu', etc.). If None, uses default device.
    with_metadata : bool, default=False
        If True, return JaxSpectrogram with metadata preserved.
        If False, return just the array.
    dtype : jnp.dtype, optional
        Target dtype. If None, uses the original dtype (float64).

    Returns
    -------
    jax.Array or JaxSpectrogram
        If with_metadata=False: jax.Array on the specified device
        If with_metadata=True: JaxSpectrogram with array and metadata

    Examples
    --------
    >>> import spectrograms as sg
    >>> import spectrograms.jax
    >>> import numpy as np
    >>>
    >>> samples = np.random.randn(16000)
    >>> stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    >>> params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    >>> mel_params = sg.MelParams(n_mels=128, f_min=0.0, f_max=8000.0)
    >>> spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
    >>>
    >>> # Simple conversion
    >>> array = spec.to_jax(device='gpu')
    >>>
    >>> # With metadata
    >>> result = spec.to_jax(device='gpu', with_metadata=True)
    >>> print(result.array.shape, result.array.device())
    >>> print(f"Frequency range: {result.frequencies[0]:.1f} - {result.frequencies[-1]:.1f} Hz")
    """
    array = jax.dlpack.from_dlpack(self)

    # Move to device if needed
    if device is not None:
        if isinstance(device, str):
            device = jax.devices(device)[0]
        array = jax.device_put(array, device)

    # Convert dtype if needed
    if dtype is not None:
        array = array.astype(dtype)

    if not with_metadata:
        return array

    # Preserve metadata
    return JaxSpectrogram(
        array=array,
        frequencies=np.array(self.frequencies),
        times=np.array(self.times),
        params=self.params,
        shape=self.shape,
        db_range=self.db_range()
        if hasattr(self, "db_range") and callable(self.db_range)
        else None,  # type: ignore
    )


def batch(
    spectrograms: list,
    device: Optional[str | jax.Device] = None,  # type: ignore
    dtype: Optional[jnp.dtype] = None,
    pad: bool = False,
) -> jax.Array:
    """
    Batch multiple spectrograms into a single JAX array.

    Parameters
    ----------
    spectrograms : list of Spectrogram or Chromagram
        list of spectrograms to batch
    device : str or jax.Device, optional
        Target device for the batch array
    dtype : jnp.dtype, optional
        Target dtype. If None, uses float64.
    pad : bool, default=False
        If True, pad spectrograms to the same size (max n_frames).
        If False, all spectrograms must have the same shape.

    Returns
    -------
    jax.Array
        Batched array of shape (batch_size, n_bins, n_frames)

    Raises
    ------
    ValueError
        If spectrograms have different shapes and pad=False

    Examples
    --------
    >>> specs = [compute_mel_power_spectrogram(s, params, mel) for s in signals]
    >>> batch = sg.jax.batch(specs, device='gpu')
    >>> print(batch.shape)  # (batch_size, n_mels, n_frames)
    """
    if not spectrograms:
        raise ValueError("Cannot batch empty list of spectrograms")

    # Convert all to arrays (CPU first)
    arrays = []
    for spec in spectrograms:
        if hasattr(spec, "to_jax"):
            array = spec.to_jax(device=None, dtype=dtype)
        else:
            array = jax.dlpack.from_dlpack(spec)
            if dtype is not None:
                array = array.astype(dtype)
        arrays.append(np.array(array))  # Convert to numpy for easier manipulation

    if pad:
        # Find max n_frames
        max_frames = max(a.shape[1] for a in arrays)
        max_bins = max(a.shape[0] for a in arrays)

        # Pad all arrays
        padded = []
        for array in arrays:
            if array.shape != (max_bins, max_frames):
                # Pad to max shape
                pad_frames = max_frames - array.shape[1]
                pad_bins = max_bins - array.shape[0]
                padded_array = np.pad(
                    array, ((0, pad_bins), (0, pad_frames)), mode="constant"
                )
                padded.append(padded_array)
            else:
                padded.append(array)
        arrays = padded
    else:
        # Check all have same shape
        shape = arrays[0].shape
        if not all(a.shape == shape for a in arrays):
            raise ValueError(
                f"All spectrograms must have the same shape. "
                f"Got shapes: {[a.shape for a in arrays]}. "
                f"Use pad=True to pad to the same size."
            )

    # Stack and convert to JAX
    batched = jnp.stack(arrays)

    # Move to device if specified
    if device is not None:
        if isinstance(device, str):
            device = jax.devices(device)[0]
        batched = jax.device_put(batched, device)

    return batched


def batch_with_metadata(
    spectrograms: list,
    device: Optional[str | jax.Device] = None,  # type: ignore
    dtype: Optional[jnp.dtype] = None,
    pad: bool = False,
) -> tuple[jax.Array, list[dict]]:
    """
    Batch spectrograms and preserve their metadata separately.

    Parameters
    ----------
    spectrograms : list of Spectrogram or Chromagram
        list of spectrograms to batch
    device : str or jax.Device, optional
        Target device for the batch array
    dtype : jnp.dtype, optional
        Target dtype
    pad : bool, default=False
        Whether to pad to same size

    Returns
    -------
    array : jax.Array
        Batched array of shape (batch_size, n_bins, n_frames)
    metadata : list of dict
        list of metadata dictionaries, one per spectrogram
    """
    metadata = []
    for spec in spectrograms:
        meta = {
            "shape": spec.shape if hasattr(spec, "shape") else None,
            "frequencies": np.array(spec.frequencies)
            if hasattr(spec, "frequencies")
            else None,
            "times": np.array(spec.times) if hasattr(spec, "times") else None,
            "params": spec.params if hasattr(spec, "params") else None,
        }
        if hasattr(spec, "db_range") and callable(spec.db_range):
            meta["db_range"] = spec.db_range()
        metadata.append(meta)

    array = batch(spectrograms, device=device, dtype=dtype, pad=pad)
    return array, metadata


_spectrograms.Spectrogram.to_jax = _spectrogram_to_jax

__all__ = [
    "JaxSpectrogram",
    "JaxChromagram",
    "batch",
    "batch_with_metadata",
]
