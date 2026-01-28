<div align="center">

# Spectrograms

[![PyPI][pypi-img]][pypi] [![Docs][docs-img]][docs] [![License: MIT][license-img]][license]

## Fast spectrogram computation library powered by Rust.

</div>
## Features

- **Multiple Spectrogram Types**: Linear, Mel, ERB frequency scales
- **Multiple Amplitude Scales**: Power, Magnitude, Decibels
- **High Performance**: Rust implementation with Python bindings
- **Plan-based Computation**: Reuse FFT plans for efficient batch processing
- **Rich Audio Features**: MFCC, Chromagram, CQT support
- **Streaming Support**: Frame-by-frame processing for real-time applications

## Installation

```bash
pip install spectrograms
```

For the FFTW-accelerated version (requires system FFTW library) you currently must build from source:

```bash
git clone https://github.com/jmg049/Spectrograms.git
cd Spectrograms/
# In pyproject.toml under [tool.maturin], change "realfft" to `"fftw"
maturin develop --release
```
## Benchmark Results

Check out the [benchmark results](PYTHON_BENCHMARK.md) for detailed performance comparisons against NumPy and SciPy implementations across various configurations and signal types.

[![Average Speedup](../imgs/average_speedup.png)](../imgs/average_speedup.png)

## Quick Start

```python
import numpy as np
import spectrograms as sg

# Generate a test signal
sr = 16000
t = np.linspace(0, 1, sr)
samples = np.sin(2 * np.pi * 440 * t)

# Create parameters
stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning())
params = sg.SpectrogramParams(stft, sample_rate=sr)

# Compute spectrogram
spec = sg.compute_linear_power_spectrogram(samples, params)

print(f"Shape: {spec.shape}")
print(f"Frequency range: {spec.frequency_range()}")
print(f"Duration: {spec.duration():.2f}s")
```

## Mel Spectrogram Example

```python
import numpy as np
import spectrograms as sg

# Load your audio data
samples = np.random.randn(16000)  # Replace with real audio
sr = 16000

# Configure parameters
stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning())
params = sg.SpectrogramParams(stft, sample_rate=sr)
mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)
db_params = sg.LogParams(floor_db=-80.0)

# Compute mel spectrogram in dB scale
mel_spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)

# Access the data
spectrogram_data = mel_spec.data  # NumPy array (n_mels, n_frames)
frequencies = mel_spec.frequencies  # Mel frequencies
times = mel_spec.times  # Time axis in seconds
```

## Efficient Batch Processing

For processing multiple audio files, use the planner API to reuse FFT plans:

```python
import numpy as np
import spectrograms as sg

# Setup
stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning())
params = sg.SpectrogramParams(stft, sample_rate=16000)
mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)
db_params = sg.LogParams(floor_db=-80.0)

# Create plan once
planner = sg.SpectrogramPlanner()
plan = planner.mel_db_plan(params, mel_params, db_params)

# Reuse plan for multiple signals (much faster!)
signals = [np.random.randn(16000) for _ in range(100)]
spectrograms = [plan.compute(signal) for signal in signals]
```

The planner API provides 1.5-3x speedup for batch processing by reusing FFT plans.

## Advanced Features

### MFCCs (Mel-Frequency Cepstral Coefficients)

```python
stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning())
mfcc_params = sg.MfccParams(n_mfcc=13)

mfccs = sg.compute_mfcc(samples, stft, sample_rate=16000, n_mels=40, mfcc_params=mfcc_params)
# Returns shape: (n_mfcc, n_frames)
```

### Chromagram (Pitch Class Profiles)

```python
stft = sg.StftParams(n_fft=4096, hop_size=512, window=sg.WindowType.hanning())
chroma_params = sg.ChromaParams.music_standard()

chroma = sg.compute_chromagram(samples, stft, sample_rate=22050, chroma_params=chroma_params)
# Returns shape: (12, n_frames) - one row per pitch class
```

### Raw STFT

```python
params = sg.SpectrogramParams.music_default(sample_rate=44100)
stft_data = sg.compute_stft(samples, params)
# Returns complex-valued STFT matrix
```

## Window Functions

Supported window functions:
- `"hanning"` - Hann window (default)
- `"hamming"` - Hamming window
- `"blackman"` - Blackman window
- `"rectangular"` - Rectangular window (no windowing)
- `"kaiser=beta"` - Kaiser window with beta parameter (e.g., `"kaiser=5.0"`)
- `"gaussian=std"` - Gaussian window with std parameter (e.g., `"gaussian=0.4"`)

Example:
```python
stft = sg.StftParams(n_fft=512, hop_size=256, window="kaiser=8.0")
```

## Default Presets

```python
# Speech processing preset (n_fft=512, hop_size=160)
params = sg.SpectrogramParams.speech_default(sample_rate=16000)

# Music processing preset (n_fft=2048, hop_size=512)
params = sg.SpectrogramParams.music_default(sample_rate=44100)
```

## API Reference

### Parameter Classes

- `StftParams(n_fft, hop_size, window, centre=True)` - STFT configuration
- `SpectrogramParams(stft, sample_rate)` - Base spectrogram parameters
- `MelParams(n_mels, f_min, f_max)` - Mel filterbank parameters
- `ErbParams(n_filters, f_min, f_max)` - ERB filterbank parameters
- `LogParams(floor_db)` - Decibel conversion parameters
- `CqtParams(bins_per_octave, n_octaves, f_min)` - Constant-Q parameters
- `ChromaParams(tuning, f_min, f_max, norm)` - Chromagram parameters
- `MfccParams(n_mfcc)` - MFCC parameters

### Spectrogram Result

The `Spectrogram` object returned by all compute functions has:

- `.data` - NumPy array with shape (n_bins, n_frames)
- `.frequencies` - Frequency axis values (Hz or scale-specific)
- `.times` - Time axis values (seconds)
- `.n_bins` - Number of frequency bins
- `.n_frames` - Number of time frames
- `.shape` - Tuple (n_bins, n_frames)
- `.frequency_range()` - Min/max frequencies
- `.duration()` - Total duration in seconds
- `.params` - Original computation parameters

**Note: The Spectrogram object can be directly used as a NumPy array.** For example:

```python
import numpy as np
import spectrograms as sg

sine_wave = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False))

stft_params = sg.StftParams(n_fft=1024, hop_size=256, window=sg.WindowType.hanning)

spectrogram_params = sg.SpectrogramParams(stft_params, SAMPLE_RATE)

spectrogram = sg.compute_linear_power_spectrogram(sine_wave, spectrogram_params)

np.abs(spectrogram).shape  # works just fine
```

### Convenience Functions

All compute functions release the Python GIL during computation.

**Linear spectrograms:**
- `compute_linear_power_spectrogram(samples, params)`
- `compute_linear_magnitude_spectrogram(samples, params)`
- `compute_linear_db_spectrogram(samples, params, db_params)`

**Mel spectrograms:**
- `compute_mel_power_spectrogram(samples, params, mel_params)`
- `compute_mel_magnitude_spectrogram(samples, params, mel_params)`
- `compute_mel_db_spectrogram(samples, params, mel_params, db_params)`

**ERB spectrograms:**
- `compute_erb_power_spectrogram(samples, params, erb_params)`
- `compute_erb_magnitude_spectrogram(samples, params, erb_params)`
- `compute_erb_db_spectrogram(samples, params, erb_params, db_params)`

**Other features:**
- `compute_stft(samples, params)` - Raw STFT (complex output)
- `compute_cqt(samples, sample_rate, cqt_params, hop_size)` - Constant-Q Transform
- `compute_chromagram(samples, stft_params, sample_rate, chroma_params)`
- `compute_mfcc(samples, stft_params, sample_rate, n_mels, mfcc_params)`

### Planner API

Create a planner and reusable plans for batch processing:

```python
planner = sg.SpectrogramPlanner()

# Create plans (one per spectrogram type)
plan = planner.linear_power_plan(params)
plan = planner.mel_db_plan(params, mel_params, db_params)
# ... and 7 other plan types

# Use plans
spec = plan.compute(samples)
frame = plan.compute_frame(samples, frame_idx)
shape = plan.output_shape(signal_length)
```

Available plan types match the convenience functions:
- `linear_power_plan`, `linear_magnitude_plan`, `linear_db_plan`
- `mel_power_plan`, `mel_magnitude_plan`, `mel_db_plan`
- `erb_power_plan`, `erb_magnitude_plan`, `erb_db_plan`


## Performance Notes

- **Plan Reuse**: Creating FFT plans is expensive. Reuse plans via the `SpectrogramPlanner` API for 1.5-3x speedup in batch processing.
- **FFT Size**: Powers of 2 (256, 512, 1024, 2048) are significantly faster than arbitrary sizes.
- **GIL Release**: All compute functions release the Python GIL, allowing parallel processing of multiple audio files.
- **Backend**: The default `realfft` backend is pure Rust with no system dependencies. Try building from source to enable the FFTW backend. It *may* offer better performance. 

## License

MIT License

## Links

- **GitHub**: https://github.com/jmg049/Spectrograms
- **Documentation**: https://jmg049.github.io/Spectrograms
- **PyPI**: https://pypi.org/project/spectrograms/

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/jmg049/Spectrograms) for contribution guidelines.

[pypi]: https://pypi.org/project/spectrograms/
[pypi-img]: https://img.shields.io/pypi/v/spectrograms?style=for-the-badge&color=009E73&label=PyPI

[docs]: https://jmg049.github.io/Spectrograms/
[docs-img]: https://img.shields.io/pypi/v/spectrograms?style=for-the-badge&color=009E73&label=Docs
[license-img]: https://img.shields.io/crates/l/spectrograms?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/Spectrograms/blob/main/LICENSE