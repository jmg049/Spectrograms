# Spectrograms
ust
High-performance spectrogram computation with Rust and Python bindings.

## Features

- **Multiple Frequency Scales**: Linear, Mel, ERB, and CQT
- **Multiple Amplitude Scales**: Power, Magnitude, and Decibels
- **Advanced Audio Features**: MFCC, Chromagram, and raw STFT
- **Plan-Based Computation**: Reuse FFT plans for 2-10x speedup on batch processing
- **Two FFT Backends**: FFTW (fastest) or pure-Rust RealFFT
- **Streaming Support**: Frame-by-frame processing for real-time applications
- **Type-Safe Rust API**: Compile-time guarantees for spectrogram types
- **Python Bindings**: Fast computation with NumPy integration and GIL-free execution

## Why Choose Spectrograms?

- **Cross-Language**: Use from Rust or Python with consistent APIs
- **High Performance**: Rust implementation, Python bindings with minimal overhead
- **Not Limited to One Type**: Multiple frequency scales in a unified API
- **Production Ready**: Efficient batch processing and streaming support
- **Well Documented**: Comprehensive [integration guide](INTEGRATION_GUIDE.md), examples, and API docs

---

## Installation

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```toml
[dependencies]
spectrograms = "0.1"
```

For pure-Rust FFT (no system dependencies):

```toml
[dependencies]
spectrograms = {
  version = "0.1",
  default-features = false,
  features = ["realfft"]
}
```

</td>
<td>

```bash
pip install spectrograms
```

For FFTW-accelerated version (requires system FFTW library):

```bash
pip install spectrograms-fftw
```

</td>
</tr>
</table>

---

## Quick Start

### Generate a Test Signal

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use std::f64::consts::PI;

// 1 second of 440 Hz sine wave
let sample_rate = 16000.0;
let samples: Vec<f64> = (0..16000)
    .map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * PI * 440.0 * t).sin()
    })
    .collect();
```

</td>
<td>

```python
import numpy as np

# 1 second of 440 Hz sine wave
sample_rate = 16000
t = np.linspace(0, 1, sample_rate, dtype=np.float64)
samples = np.sin(2 * np.pi * 440 * t)
```

</td>
</tr>
</table>

### Compute a Basic Spectrogram

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

// Configure parameters
let stft = StftParams::new(
    512,                 // FFT size
    256,                 // hop size
    WindowType::Hanning, // window
    true                 // centre frames
)?;

let params = SpectrogramParams::new(
    stft,
    sample_rate
)?;

// Compute power spectrogram
let spec = LinearPowerSpectrogram::compute(
    &samples,
    &params,
    None
)?;

println!("Shape: {} bins × {} frames",
    spec.n_bins(), spec.n_frames());
```

</td>
<td>

```python
import spectrograms as sg

# Configure parameters
stft = sg.StftParams(
    n_fft=512,
    hop_size=256,
    window=sg.WindowType.hanning(),
    centre=True
)

params = sg.SpectrogramParams(
    stft,
    sample_rate=sample_rate
)

# Compute power spectrogram
spec = sg.compute_linear_power_spectrogram(
    samples,
    params
)

print(f"Shape: {spec.n_bins} bins × {spec.n_frames} frames")
```

</td>
</tr>
</table>

---

## Mel Spectrogram Example

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;

// Mel filterbank
let mel = MelParams::new(
    80,      // n_mels
    0.0,     // f_min
    8000.0   // f_max
)?;

// dB scaling
let db = LogParams::new(-80.0)?;

// Compute mel spectrogram in dB
let spec = MelDbSpectrogram::compute(
    &samples, &params, &mel, Some(&db)
)?;

// Access data
println!("Mel bands: {}", spec.n_bins());
println!("Frames: {}", spec.n_frames());
println!("Frequency range: {:?}",
    spec.axes().frequency_range());
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Mel filterbank
mel = sg.MelParams(
    n_mels=80,
    f_min=0.0,
    f_max=8000.0
)

# dB scaling
db = sg.LogParams(floor_db=-80.0)

# Compute mel spectrogram in dB
spec = sg.compute_mel_db_spectrogram(
    samples, params, mel, db
)

# Access data
print(f"Mel bands: {spec.n_bins}")
print(f"Frames: {spec.n_frames}")
print(f"Frequency range: {spec.frequency_range()}")
```

</td>
</tr>
</table>

---

## Efficient Batch Processing

Reuse FFT plans for **2-10x speedup** when processing multiple signals:

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let signals = vec![
    vec![0.0; 16000],
    vec![0.0; 16000],
    vec![0.0; 16000],
];

let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;
let mel = MelParams::new(80, 0.0, 8000.0)?;
let db = LogParams::new(-80.0)?;

// Create plan once
let planner = SpectrogramPlanner::new();
let mut plan = planner.mel_db_plan(
    &params, &mel, Some(&db)
)?;

// Reuse for all signals (much faster!)
for signal in signals {
    let spec = plan.compute(&signal)?;
    // Process spec...
}
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

signals = [
    np.random.randn(16000),
    np.random.randn(16000),
    np.random.randn(16000),
]

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)
mel = sg.MelParams(80, 0.0, 8000.0)
db = sg.LogParams(-80.0)

# Create plan once
planner = sg.SpectrogramPlanner()
plan = planner.mel_db_plan(params, mel, db)

# Reuse for all signals (much faster!)
for signal in signals:
    spec = plan.compute(signal)
    # Process spec...
```

</td>
</tr>
</table>

---

## Advanced Features

### MFCCs (Mel-Frequency Cepstral Coefficients)

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let stft = StftParams::new(512, 160, WindowType::Hanning, true)?;
let mfcc_params = MfccParams::new(13)?;

let mfccs = compute_mfcc(
    &samples,
    &stft,
    16000.0,
    40,  // n_mels
    &mfcc_params
)?;

// Shape: (13, n_frames)
println!("MFCCs: {} × {}", mfccs.nrows(), mfccs.ncols());
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 160, sg.WindowType.hanning(), True)
mfcc_params = sg.MfccParams(n_mfcc=13)

mfccs = sg.compute_mfcc(
    samples,
    stft,
    sample_rate=16000,
    n_mels=40,
    mfcc_params=mfcc_params
)

# Shape: (13, n_frames)
print(f"MFCCs: {mfccs.shape}")
```

</td>
</tr>
</table>

### Chromagram (Pitch Class Profiles)

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let stft = StftParams::new(4096, 512, WindowType::Hanning, true)?;
let chroma_params = ChromaParams::music_standard();

let chroma = compute_chromagram(
    &samples,
    &stft,
    22050.0,
    &chroma_params
)?;

// Shape: (12, n_frames) - one row per pitch class
println!("Chroma: {} × {}", chroma.nrows(), chroma.ncols());
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(4096, 512, sg.WindowType.hanning(), True)
chroma_params = sg.ChromaParams.music_standard()

chroma = sg.compute_chromagram(
    samples,
    stft,
    sample_rate=22050,
    chroma_params=chroma_params
)

# Shape: (12, n_frames)
print(f"Chroma: {chroma.shape}")
```

</td>
</tr>
</table>

---

## Supported Spectrogram Types

### Frequency Scales

- **Linear** (`LinearHz`): Standard FFT bins, evenly spaced in Hz
- **Mel** (`Mel`): Mel-frequency scale, perceptually motivated for speech/audio
- **ERB** (`Erb`): Equivalent Rectangular Bandwidth, models auditory perception
- **CQT**: Constant-Q Transform for music analysis
- **Log** (`LogHz`): Logarithmic frequency spacing

### Amplitude Scales

| Scale | Formula | Use Case |
|-------|---------|----------|
| **Power** | `\|X\|²` | Energy analysis, ML features |
| **Magnitude** | `\|X\|` | Spectral analysis, phase vocoder |
| **Decibels** | `10·log₁₀(power)` | Visualization, perceptual analysis |

### Type Aliases (Rust)

```rust
// Linear frequency
type LinearPowerSpectrogram = Spectrogram<LinearHz, Power>;
type LinearMagnitudeSpectrogram = Spectrogram<LinearHz, Magnitude>;
type LinearDbSpectrogram = Spectrogram<LinearHz, Decibels>;

// Mel frequency
type MelPowerSpectrogram = Spectrogram<Mel, Power>;
type MelMagnitudeSpectrogram = Spectrogram<Mel, Magnitude>;
type MelDbSpectrogram = Spectrogram<Mel, Decibels>;

// ERB frequency
type ErbPowerSpectrogram = Spectrogram<Erb, Power>;
type ErbMagnitudeSpectrogram = Spectrogram<Erb, Magnitude>;
type ErbDbSpectrogram = Spectrogram<Erb, Decibels>;
```

---

## Window Functions

Supported window functions with different frequency/time resolution trade-offs:

- **`rectangular`**: No windowing (best frequency resolution, high leakage)
- **`hanning`**: Good general-purpose window (default)
- **`hamming`**: Similar to Hanning with different coefficients
- **`blackman`**: Low sidelobes, wider main lobe
- **`bartlett`**: Triangular window
- **`kaiser=<beta>`**: Tunable trade-off (β controls shape, e.g., `kaiser=5.0`)
- **`gaussian=<std>`**: Smooth roll-off (e.g., `gaussian=0.4`)

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
// Parse from string
let window: WindowType = "hanning".parse()?;
let kaiser: WindowType = "kaiser=8.0".parse()?;

// Or use constructors
let hann = WindowType::Hanning;
let gauss = WindowType::Gaussian { std: 0.4 };
```

</td>
<td>

```python
# Use class methods
window = sg.WindowType.hanning()
kaiser = sg.WindowType.kaiser(beta=8.0)
gauss = sg.WindowType.gaussian(std=0.4)

# Or from string
stft = sg.StftParams(512, 256, "kaiser=8.0", True)
```

</td>
</tr>
</table>

---

## Default Presets

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
// Speech processing preset
// n_fft=512, hop_size=160
let params = SpectrogramParams::speech_default(16000.0)?;

// Music processing preset
// n_fft=2048, hop_size=512
let params = SpectrogramParams::music_default(44100.0)?;
```

</td>
<td>

```python
# Speech processing preset
params = sg.SpectrogramParams.speech_default(sample_rate=16000)

# Music processing preset
params = sg.SpectrogramParams.music_default(sample_rate=44100)
```

</td>
</tr>
</table>

---

## Accessing Results

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;

// Dimensions
let n_bins = spec.n_bins();
let n_frames = spec.n_frames();

// Data (ndarray::Array2<f64>)
let data = spec.data();

// Axes
let freqs = spec.axes().frequencies();
let times = spec.axes().times();
let (f_min, f_max) = spec.axes().frequency_range();
let duration = spec.axes().duration();

// Original parameters
let params = spec.params();
```

</td>
<td>

```python
spec = sg.compute_linear_power_spectrogram(samples, params)

# Dimensions
n_bins = spec.n_bins
n_frames = spec.n_frames

# Data (numpy array)
data = spec.data  # shape: (n_bins, n_frames)

# Axes
freqs = spec.frequencies
times = spec.times
f_min, f_max = spec.frequency_range()
duration = spec.duration()

# Original parameters
params = spec.params
```

</td>
</tr>
</table>

---

## Examples

Comprehensive examples in both languages:

**Rust** (`examples/`):
- [`basic_linear.rs`](examples/basic_linear.rs) - Simple linear spectrogram
- [`mel_spectrogram.rs`](examples/mel_spectrogram.rs) - Mel spectrogram with dB scaling
- [`reuse_plan.rs`](examples/reuse_plan.rs) - Batch processing with plan reuse
- [`compare_windows.rs`](examples/compare_windows.rs) - Window function comparison
- [`amplitude_scales.rs`](examples/amplitude_scales.rs) - Power, Magnitude, and dB

**Python** (`python/examples/`):
- [`basic_linear.py`](python/examples/basic_linear.py) - Linear spectrogram basics
- [`mel_spectrogram.py`](python/examples/mel_spectrogram.py) - Mel spectrograms
- [`mfcc_example.py`](python/examples/mfcc_example.py) - MFCC computation
- [`chromagram_example.py`](python/examples/chromagram_example.py) - Pitch class profiles
- [`batch_processing.py`](python/examples/batch_processing.py) - Efficient batch processing
- [`streaming.py`](python/examples/streaming.py) - Real-time frame-by-frame processing

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```bash
cargo run --example basic_linear
cargo run --example mel_spectrogram
```

</td>
<td>

```bash
python python/examples/basic_linear.py
python python/examples/mel_spectrogram.py
```

</td>
</tr>
</table>

---

## Documentation

- **[Integration Guide](INTEGRATION_GUIDE.md)**: Comprehensive walkthrough with side-by-side Rust/Python examples
- **[API Documentation](https://docs.rs/spectrograms)**: Full Rust API reference
- **[Python Documentation](docs/build/html/index.html)**: Python API reference and guides
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project

---

## Feature Flags (Rust)

The Rust library requires exactly one FFT backend:

- **`fftw`**: Uses FFTW for FFT computation
  - Fastest performance
  - Requires system FFTW library (`libfftw3-dev` on Ubuntu/Debian)
  - Not pure Rust

- **`realfft`** (default): Pure-Rust FFT implementation
  - No system dependencies
  - Slightly slower than FFTW
  - Works everywhere

Additional flags:
- **`python`** (default): Enables Python bindings
- **`serde`**: Enables serialization support

```toml
# Pure Rust, no Python
[dependencies]
spectrograms = { version = "0.1", default-features = false, features = ["realfft"] }

# FFTW backend with Python
[dependencies]
spectrograms = { version = "0.1", default-features = false, features = ["fftw", "python"] }
```

---

## Performance Tips

1. **Reuse plans**: Use `SpectrogramPlanner` for 2-10x speedup on batch processing
2. **Choose power-of-2 FFT sizes**: Best performance (512, 1024, 2048, 4096)
3. **Use FFTW backend**: Maximum speed when system dependencies are acceptable
4. **Python GIL**: All compute functions release the GIL for parallelism
5. **Streaming**: Use frame-by-frame processing for real-time applications

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{spectrograms2025,
  author = {Geraghty, Jack},
  title = {Spectrograms: High-Performance Spectrogram Computation},
  year = {2025},
  url = {https://github.com/jmg049/Spectrograms}
}
```

---

**Note**: This library focuses on spectrogram computation. For complete audio analysis pipelines, combine it with audio I/O libraries like [audio_samples](https://github.com/jmg049/audio_samples) and your preferred plotting tools.
