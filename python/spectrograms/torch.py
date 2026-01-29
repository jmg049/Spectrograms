"""
PyTorch convenience functions for spectrograms.

This module provides high-level convenience methods for converting spectrograms
to PyTorch tensors with optional metadata preservation, GPU support, and batching utilities.

Usage:
    import spectrograms as sg
    import spectrograms.torch  # Adds .to_torch() method

    spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)

    # Simple conversion
    tensor = spec.to_torch(device='cuda')

    # With metadata
    result = spec.to_torch(device='cuda', with_metadata=True)
    result.tensor        # torch.Tensor
    result.frequencies   # np.ndarray
    result.times         # np.ndarray
    result.params        # SpectrogramParams
    result.shape         # tuple

    # Batching
    batch = sg.torch.batch([spec1, spec2, spec3], device='cuda')
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Install it with:\n"
        "  pip install torch\n"
        "or visit https://pytorch.org for installation instructions."
    )

from . import _spectrograms  # type: ignore


@dataclass
class TorchSpectrogram:
    """
    A spectrogram converted to PyTorch with optional metadata preserved.

    Attributes:
        tensor: PyTorch tensor containing the spectrogram data
        frequencies: Frequency axis values (Hz or scale-specific)
        times: Time axis values (seconds)
        params: Original computation parameters
        shape: Shape of the data (n_bins, n_frames)
        db_range: Decibel range if applicable (min_db, max_db)
    """

    tensor: torch.Tensor
    frequencies: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    params: Optional[object] = None
    shape: Optional[tuple[int, int]] = None
    db_range: Optional[tuple[float, float]] = None

    def to(self, device: str | torch.device) -> TorchSpectrogram:
        """Move tensor to a different device."""
        return TorchSpectrogram(
            tensor=self.tensor.to(device),
            frequencies=self.frequencies,
            times=self.times,
            params=self.params,
            shape=self.shape,
            db_range=self.db_range,
        )

    def cpu(self) -> TorchSpectrogram:
        """Move tensor to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> TorchSpectrogram:
        """Move tensor to CUDA device."""
        if device is not None:
            return self.to(f"cuda:{device}")
        return self.to("cuda")


@dataclass
class TorchChromagram:
    """
    A chromagram converted to PyTorch with optional metadata preserved.

    Attributes:
        tensor: PyTorch tensor containing the chromagram data (12 x n_frames)
        labels: Pitch class labels (C, C#, D, ...)
        params: Original computation parameters
        shape: Shape of the data (12, n_frames)
        n_frames: Number of time frames
    """

    tensor: torch.Tensor
    labels: Optional[list[str]] = None
    params: Optional[object] = None
    shape: Optional[tuple[int, int]] = None
    n_frames: Optional[int] = None

    def to(self, device: str | torch.device) -> TorchChromagram:
        """Move tensor to a different device."""
        return TorchChromagram(
            tensor=self.tensor.to(device),
            labels=self.labels,
            params=self.params,
            shape=self.shape,
            n_frames=self.n_frames,
        )

    def cpu(self) -> TorchChromagram:
        """Move tensor to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> TorchChromagram:
        """Move tensor to CUDA device."""
        if device is not None:
            return self.to(f"cuda:{device}")
        return self.to("cuda")


def _spectrogram_to_torch(
    self,
    device: str | torch.device = "cpu",
    with_metadata: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor] | TorchSpectrogram:
    """
    Convert spectrogram to PyTorch tensor with optional metadata preservation.

    This uses the DLPack protocol for conversion when possible.

    Parameters
    ----------
    device : str | torch.device, default='cpu'
        Target device ('cpu', 'cuda', 'cuda:0', etc.)
    with_metadata : bool, default=False
        If True, return TorchSpectrogram with metadata preserved.
        If False, return just the tensor.
    dtype : torch.dtype, optional
        Target dtype. If None, uses the original dtype (float64).

    Returns
    -------
    torch.Tensor or TorchSpectrogram
        If with_metadata=False: torch.Tensor on the specified device
        If with_metadata=True: TorchSpectrogram with tensor and metadata

    Examples
    --------
    >>> import spectrograms as sg
    >>> import spectrograms.torch
    >>> import numpy as np
    >>>
    >>> samples = np.random.randn(16000)
    >>> stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    >>> params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    >>> spec = sg.compute_mel_power_spectrogram(samples, params, mel_params)
    >>>
    >>> # Simple conversion
    >>> tensor = spec.to_torch(device='cuda')
    >>>
    >>> # With metadata
    >>> result = spec.to_torch(device='cuda', with_metadata=True)
    >>> print(result.tensor.shape, result.tensor.device)
    >>> print(f"Frequency range: {result.frequencies[0]:.1f} - {result.frequencies[-1]:.1f} Hz")
    """
    tensor = torch.from_dlpack(self)  # type: ignore

    # Move to device if needed
    if str(device) != "cpu" or dtype is not None:
        if dtype is None:
            dtype = tensor.dtype
        tensor = tensor.to(device=device, dtype=dtype)
    elif dtype is not None:
        tensor = tensor.to(dtype=dtype)

    if not with_metadata:
        return tensor

    # Preserve metadata
    return TorchSpectrogram(
        tensor=tensor,
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
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    pad: bool = False,
) -> torch.Tensor:
    """
    Batch multiple spectrograms into a single tensor.

    Parameters
    ----------
    spectrograms : list of Spectrogram or Chromagram
        List of spectrograms to batch
    device : str or torch.device, default='cpu'
        Target device for the batch tensor
    dtype : torch.dtype, optional
        Target dtype. If None, uses float64.
    pad : bool, default=False
        If True, pad spectrograms to the same size (max n_frames).
        If False, all spectrograms must have the same shape.

    Returns
    -------
    torch.Tensor
        Batched tensor of shape (batch_size, n_bins, n_frames)

    Raises
    ------
    ValueError
        If spectrograms have different shapes and pad=False

    Examples
    --------
    >>> specs = [compute_mel_power_spectrogram(s, params, mel) for s in signals]
    >>> batch = sg.torch.batch(specs, device='cuda')
    >>> print(batch.shape)  # (batch_size, n_mels, n_frames)
    """
    if not spectrograms:
        raise ValueError("Cannot batch empty list of spectrograms")

    # Convert all to tensors (CPU first for efficiency)
    tensors = []
    for spec in spectrograms:
        if hasattr(spec, "to_torch"):
            tensor = spec.to_torch(device="cpu", dtype=dtype)
        else:
            tensor = torch.from_dlpack(spec)  # type: ignore
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
        tensors.append(tensor)

    if pad:
        # Find max n_frames
        max_frames = max(t.shape[1] for t in tensors)
        max_bins = max(t.shape[0] for t in tensors)

        # Pad all tensors
        padded = []
        for tensor in tensors:
            if tensor.shape != (max_bins, max_frames):
                # Pad to max shape
                pad_frames = max_frames - tensor.shape[1]
                pad_bins = max_bins - tensor.shape[0]
                padded_tensor = torch.nn.functional.pad(
                    tensor, (0, pad_frames, 0, pad_bins), value=0
                )
                padded.append(padded_tensor)
            else:
                padded.append(tensor)
        tensors = padded
    else:
        # Check all have same shape
        shape = tensors[0].shape
        if not all(t.shape == shape for t in tensors):
            raise ValueError(
                f"All spectrograms must have the same shape. "
                f"Got shapes: {[t.shape for t in tensors]}. "
                f"Use pad=True to pad to the same size."
            )

    # Stack and move to device
    batched = torch.stack(tensors)
    if str(device) != "cpu":
        batched = batched.to(device)

    return batched


def batch_with_metadata(
    spectrograms: list,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    pad: bool = False,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Batch spectrograms and preserve their metadata separately.

    Parameters
    ----------
    spectrograms : list of Spectrogram or Chromagram
        List of spectrograms to batch
    device : str | torch.device, default='cpu'
        Target device for the batch tensor
    dtype : torch.dtype, optional
        Target dtype
    pad : bool, default=False
        Whether to pad to same size

    Returns
    -------
    tensor : torch.Tensor
        Batched tensor of shape (batch_size, n_bins, n_frames)
    metadata : list of dict
        List of metadata dictionaries, one per spectrogram
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

    tensor = batch(spectrograms, device=device, dtype=dtype, pad=pad)
    return tensor, metadata


# ey-patch the Spectrogram class
_spectrograms.Spectrogram.to_torch = _spectrogram_to_torch

__all__ = [
    "TorchSpectrogram",
    "TorchChromagram",
    "batch",
    "batch_with_metadata",
]
