"""
DLPack Protocol Integration with PyTorch

This example demonstrates exchange between the Spectrograms
library and PyTorch using the DLPack protocol.

Requirements:
    pip install torch

Key Features:
- Works with all spectrogram types (Linear, Mel, CQT, Gammatone, etc.)
- Supports both PySpectrogram and PyChromagram
- Compatible with PyTorch's tensor operations
"""

import numpy as np

import spectrograms as sg

try:
    import torch

    HAS_TORCH = True
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    HAS_TORCH = False
    exit(1)


def demo_basic_conversion():
    """Demonstrate basic Spectrogram to PyTorch."""
    # Create a sample spectrogram
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    spec = sg.compute_linear_power_spectrogram(samples, params)

    print("\nOriginal Spectrogram:")
    print(f"  Shape: {spec.shape}")
    print(f"  Type: {type(spec)}")
    print(f"  Memory address: {hex(id(spec.data))}")

    # Convert to PyTorch using DLPack
    tensor = torch.from_dlpack(spec)  # type: ignore

    print("\nPyTorch Tensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Memory address: {hex(tensor.data_ptr())}")


def demo_chromagram_conversion():
    """Demonstrate chromagram conversion to PyTorch."""
    print("\n" + "=" * 70)
    print("Chromagram to PyTorch")
    print("=" * 70)

    # Generate a simple test signal
    samples = np.random.randn(16000).astype(np.float64)
    stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    chroma_params = sg.ChromaParams()

    # Compute chromagram
    chroma = sg.compute_chromagram(samples, stft, 16000.0, chroma_params)

    print("\nChromagram:")
    print(f"  Shape: {chroma.shape} (12 pitch classes)")
    print(f"  Type: {type(chroma)}")

    # Convert to PyTorch
    tensor = torch.from_dlpack(chroma)  # type: ignore

    print("\nPyTorch Tensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")

    print("\nChromagram conversion successful!")


def demo_neural_network_pipeline():
    """Demonstrate using spectrograms in a PyTorch neural network pipeline."""
    print("\n" + "=" * 70)
    print("Neural Network Pipeline Example")
    print("=" * 70)

    # Create batch of spectrograms
    batch_size = 4
    samples_list = [
        np.random.randn(16000).astype(np.float64) for _ in range(batch_size)
    ]

    stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning)
    params = sg.SpectrogramParams(stft, sample_rate=16000.0)
    mel_params = sg.MelParams(
        n_mels=128, f_min=0.0, f_max=8000.0, norm=sg.MelNorm.Slaney
    )
    db_params = sg.LogParams(floor_db=-80.0)

    class NeuralNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x

    # Compute spectrograms and convert to PyTorch
    tensors = []
    for samples in samples_list:
        spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)
        spec = spec.astype(np.float32)
        tensor = torch.from_dlpack(spec)  # type: ignore
        tensors.append(tensor)

    # Stack into a batch
    batch = torch.stack(tensors)
    print(f"\nBatch shape: {batch.shape}")
    print(
        f"  [batch_size, n_mels, n_frames] = [{batch_size}, {batch.shape[1]}, {batch.shape[2]}]"
    )

    # Normalize
    mean = batch.mean()
    std = batch.std()
    normalized = (batch - mean) / std

    print("\nNormalized batch statistics:")
    print(f"  Mean: {normalized.mean():.6f}")
    print(f"  Std: {normalized.std():.6f}")

    # CPU-only
    model = NeuralNet()
    for item in batch:
        output = model(item.unsqueeze(0).unsqueeze(0))
        print(f"Output shape (CPU): {output.shape}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DLPack Protocol Integration with PyTorch")
    print("=" * 70)

    if not HAS_TORCH:
        return

    # demo_basic_conversion()
    # demo_chromagram_conversion()
    demo_neural_network_pipeline()


if __name__ == "__main__":
    main()
