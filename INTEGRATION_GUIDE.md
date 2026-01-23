# Spectrograms Integration Guide

A comprehensive walkthrough for integrating spectrogram computation into your Rust and Python projects.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Building Spectrograms](#building-spectrograms)
  - [Linear Spectrograms](#linear-spectrograms)
  - [Mel Spectrograms](#mel-spectrograms)
  - [ERB Spectrograms](#erb-spectrograms)
- [Amplitude Scales](#amplitude-scales)
- [Efficient Plan Reuse](#efficient-plan-reuse)
- [Computing FFT and STFT](#computing-fft-and-stft)
- [Streaming and Online Processing](#streaming-and-online-processing)
- [Advanced Features](#advanced-features)
  - [MFCC](#mfcc)
  - [Chromagram](#chromagram)
  - [CQT](#cqt)
- [Performance Tips](#performance-tips)
- [Common Patterns](#common-patterns)

---

## Introduction

This library provides **high-performance spectrogram computation** with both Rust and Python APIs. It features:

- **Multiple frequency scales**: Linear, Mel, ERB, CQT
- **Multiple amplitude scales**: Power, Magnitude, Decibels
- **Plan-based computation**: Reuse FFT plans for 2-10x speedup on batch processing
- **Two FFT backends**: FFTW (fastest) or RealFFT (pure Rust)
- **Zero-copy design**: Minimal allocations for efficient memory usage
- **Streaming support**: Frame-by-frame processing for real-time applications

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
# Cargo.toml
[dependencies]
spectrograms = "0.1"

# Or with pure-Rust FFT
spectrograms = {
  version = "0.1",
  default-features = false,
  features = ["realfft"]
}
```

</td>
<td>

```bash
# Install from PyPI (when published)
pip install spectrograms

# Or install from source
cd python
pip install .

# Or with maturin for development
cd python
maturin develop
```

</td>
</tr>
</table>

---

## Core Concepts

### What is a Spectrogram?

A **spectrogram** is a visual representation of the spectrum of frequencies in a signal as they vary with time. It's computed using the **Short-Time Fourier Transform (STFT)**, which applies FFT to overlapping windows of the signal.

### Key Parameters

- **`n_fft`**: FFT window size (e.g., 512, 1024, 2048). Larger = better frequency resolution, worse time resolution
- **`hop_size`**: Samples between successive frames (e.g., 256). Smaller = more overlap, better time resolution
- **`window`**: Window function to reduce spectral leakage (Hanning, Hamming, Blackman, etc.)
- **`centre`**: Whether to pad signal for centered frames (recommended: true)

### The Plan Pattern

This library uses a **plan-based computation model**:

1. **One-shot**: Simple `compute()` function - creates plan internally (convenient but slower for batches)
2. **Plan reuse**: Create plan once with `SpectrogramPlanner`, reuse for multiple signals (2-10x faster)

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

// Generate 1 second of 440 Hz sine wave
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

# Generate 1 second of 440 Hz sine wave
sample_rate = 16000
t = np.linspace(0, 1, sample_rate, dtype=np.float64)
samples = np.sin(2 * np.pi * 440 * t)
```

</td>
</tr>
</table>

### Compute Your First Spectrogram

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

// Configure STFT parameters
let stft = StftParams::new(
    512,                 // n_fft
    256,                 // hop_size
    WindowType::Hanning, // window function
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

# Configure STFT parameters
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

## Building Spectrograms

### Linear Spectrograms

Linear spectrograms use **equally-spaced frequency bins** corresponding to FFT output bins.

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

// Power spectrum (|X|²)
let power = LinearPowerSpectrogram::compute(
    &samples, &params, None
)?;

// Magnitude spectrum (|X|)
let magnitude = LinearMagnitudeSpectrogram::compute(
    &samples, &params, None
)?;

// Decibel scale (10·log₁₀(power))
let db_params = LogParams::new(-80.0)?; // floor at -80 dB
let db = LinearDbSpectrogram::compute(
    &samples, &params, Some(db_params)
)?;

// Access data
let frequencies = db.axes().frequencies();
let times = db.axes().times();
let data = db.data(); // ndarray::Array2<f64>
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Power spectrum (|X|²)
power = sg.compute_linear_power_spectrogram(
    samples, params
)

# Magnitude spectrum (|X|)
magnitude = sg.compute_linear_magnitude_spectrogram(
    samples, params
)

# Decibel scale (10·log₁₀(power))
db_params = sg.LogParams(floor_db=-80.0)
db = sg.compute_linear_db_spectrogram(
    samples, params, db_params
)

# Access data
frequencies = db.frequencies  # numpy array
times = db.times              # numpy array
data = db.data                # numpy array (n_bins × n_frames)
```

</td>
</tr>
</table>

### Mel Spectrograms

Mel spectrograms apply a **mel-scale filterbank** to emphasize perceptually-relevant frequencies.

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

// Configure mel filterbank
let mel = MelParams::new(
    80,      // n_mels: number of mel bands
    0.0,     // f_min: minimum frequency (Hz)
    8000.0   // f_max: maximum frequency (Hz)
)?;

// Mel power spectrogram
let mel_power = MelPowerSpectrogram::compute(
    &samples, &params, &mel, None
)?;

// Mel dB spectrogram (most common for ML)
let db = LogParams::new(-80.0)?;
let mel_db = MelDbSpectrogram::compute(
    &samples, &params, &mel, Some(&db)
)?;

// Access mel frequencies
let mel_freqs = mel_db.axes().frequencies();
println!("Mel bands: {} bins", mel_db.n_bins());
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Configure mel filterbank
mel = sg.MelParams(
    n_mels=80,
    f_min=0.0,
    f_max=8000.0
)

# Mel power spectrogram
mel_power = sg.compute_mel_power_spectrogram(
    samples, params, mel
)

# Mel dB spectrogram (most common for ML)
db = sg.LogParams(floor_db=-80.0)
mel_db = sg.compute_mel_db_spectrogram(
    samples, params, mel, db
)

# Access mel frequencies
mel_freqs = mel_db.frequencies
print(f"Mel bands: {mel_db.n_bins} bins")
```

</td>
</tr>
</table>

### ERB Spectrograms

ERB (Equivalent Rectangular Bandwidth) spectrograms use a **gammatone filterbank** modeling auditory perception.

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

// Configure ERB filterbank
let erb = ErbParams::new(
    64,      // n_bands: number of ERB bands
    50.0,    // f_min: minimum frequency (Hz)
    8000.0   // f_max: maximum frequency (Hz)
)?;

// ERB power spectrogram
let erb_power = ErbPowerSpectrogram::compute(
    &samples, &params, &erb, None
)?;

// ERB dB spectrogram
let db = LogParams::new(-80.0)?;
let erb_db = ErbDbSpectrogram::compute(
    &samples, &params, &erb, Some(&db)
)?;
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Configure ERB filterbank
erb = sg.ErbParams(
    n_bands=64,
    f_min=50.0,
    f_max=8000.0
)

# ERB power spectrogram
erb_power = sg.compute_erb_power_spectrogram(
    samples, params, erb
)

# ERB dB spectrogram
db = sg.LogParams(floor_db=-80.0)
erb_db = sg.compute_erb_db_spectrogram(
    samples, params, erb, db
)
```

</td>
</tr>
</table>

---

## Amplitude Scales

The library supports three amplitude scales for converting complex FFT output to real values:

| Scale | Formula | Use Case |
|-------|---------|----------|
| **Power** | `\|X\|²` | Energy analysis, ML features |
| **Magnitude** | `\|X\|` | Spectral analysis, phase vocoder |
| **Decibels** | `10·log₁₀(power)` | Visualization, perceptual analysis |

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

// Power: raw energy values
let power = LinearPowerSpectrogram::compute(
    &samples, &params, None
)?;

// Magnitude: sqrt of power
let magnitude = LinearMagnitudeSpectrogram::compute(
    &samples, &params, None
)?;

// Decibels: logarithmic scale
let db_params = LogParams::new(-80.0)?; // clip below -80 dB
let db = LinearDbSpectrogram::compute(
    &samples, &params, Some(db_params)
)?;

// Verify relationship: magnitude = sqrt(power)
let power_val = power.data()[[10, 5]];
let mag_val = magnitude.data()[[10, 5]];
assert!((mag_val - power_val.sqrt()).abs() < 1e-10);
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Power: raw energy values
power = sg.compute_linear_power_spectrogram(
    samples, params
)

# Magnitude: sqrt of power
magnitude = sg.compute_linear_magnitude_spectrogram(
    samples, params
)

# Decibels: logarithmic scale
db_params = sg.LogParams(floor_db=-80.0)
db = sg.compute_linear_db_spectrogram(
    samples, params, db_params
)

# Verify relationship: magnitude = sqrt(power)
power_val = power.data[10, 5]
mag_val = magnitude.data[10, 5]
assert np.isclose(mag_val, np.sqrt(power_val))
```

</td>
</tr>
</table>

---

## Efficient Plan Reuse

For **batch processing** or **repeated computations**, creating a plan once and reusing it is **2-10x faster** than calling `compute()` repeatedly.

### Why Plan Reuse Matters

FFT plans are expensive to create but cheap to execute. Plan creation involves:
- Allocating FFT workspace buffers
- Computing optimal FFT algorithm
- Creating window function samples
- Building frequency mapping matrices (for Mel/ERB)

### Batch Processing Example

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

// Generate multiple test signals
let signals: Vec<Vec<f64>> = vec![
    generate_sine(220.0),  // A3
    generate_sine(440.0),  // A4
    generate_sine(880.0),  // A5
];

let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;
let mel = MelParams::new(80, 0.0, 8000.0)?;
let db = LogParams::new(-80.0)?;

// ❌ SLOW: Creates plan for each signal
for signal in &signals {
    let spec = MelDbSpectrogram::compute(
        signal, &params, &mel, Some(&db)
    )?;
    // Process spec...
}

// ✅ FAST: Create plan once, reuse
let planner = SpectrogramPlanner::new();
let mut plan = planner.mel_db_plan(
    &params, &mel, Some(&db)
)?;

for signal in &signals {
    let spec = plan.compute(signal)?;
    // Process spec... (2-10x faster!)
}
```

</td>
<td>

```python
import spectrograms as sg

# Generate multiple test signals
signals = [
    generate_sine(220.0),  # A3
    generate_sine(440.0),  # A4
    generate_sine(880.0),  # A5
]

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)
mel = sg.MelParams(80, 0.0, 8000.0)
db = sg.LogParams(-80.0)

# ❌ SLOW: Creates plan for each signal
for signal in signals:
    spec = sg.compute_mel_db_spectrogram(
        signal, params, mel, db
    )
    # Process spec...

# ✅ FAST: Create plan once, reuse
planner = sg.SpectrogramPlanner()
plan = planner.mel_db_plan(params, mel, db)

for signal in signals:
    spec = plan.compute(signal)
    # Process spec... (2-10x faster!)
```

</td>
</tr>
</table>

### Plan Creation Methods

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let planner = SpectrogramPlanner::new();

// Linear spectrograms
let linear_power = planner.linear_plan::<Power>(&params, None)?;
let linear_mag = planner.linear_plan::<Magnitude>(&params, None)?;
let linear_db = planner.linear_plan::<Decibels>(&params, Some(&db))?;

// Mel spectrograms
let mel_power = planner.mel_power_plan(&params, &mel)?;
let mel_mag = planner.mel_magnitude_plan(&params, &mel)?;
let mel_db = planner.mel_db_plan(&params, &mel, Some(&db))?;

// ERB spectrograms
let erb_power = planner.erb_power_plan(&params, &erb)?;
let erb_mag = planner.erb_magnitude_plan(&params, &erb)?;
let erb_db = planner.erb_db_plan(&params, &erb, Some(&db))?;
```

</td>
<td>

```python
import spectrograms as sg

planner = sg.SpectrogramPlanner()

# Linear spectrograms
linear_power = planner.linear_power_plan(params)
linear_mag = planner.linear_magnitude_plan(params)
linear_db = planner.linear_db_plan(params, db)

# Mel spectrograms
mel_power = planner.mel_power_plan(params, mel)
mel_mag = planner.mel_magnitude_plan(params, mel)
mel_db = planner.mel_db_plan(params, mel, db)

# ERB spectrograms
erb_power = planner.erb_power_plan(params, erb)
erb_mag = planner.erb_magnitude_plan(params, erb)
erb_db = planner.erb_db_plan(params, erb, db)
```

</td>
</tr>
</table>

---

## Computing FFT and STFT

### Direct STFT Computation

For advanced use cases, you can compute the raw STFT (complex-valued spectrogram):

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;
use num_complex::Complex;

let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;

// Compute STFT using planner
let planner = SpectrogramPlanner::new();
let plan = planner.stft_plan(&params)?;

// Get complex STFT output
let stft_result = plan.compute_stft(&samples)?;

// stft_result contains:
// - Complex spectrum: Array2<Complex<f64>>
// - Shape: (n_fft/2 + 1, n_frames)

// Extract magnitude and phase
for (bin_idx, frame_idx) in [(10, 5)] {
    let c = stft_result[[bin_idx, frame_idx]];
    let magnitude = c.norm();
    let phase = c.arg();
    println!("Bin {} Frame {}: mag={:.2e}, phase={:.2}",
        bin_idx, frame_idx, magnitude, phase);
}
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Compute STFT (returns complex array)
stft_result = sg.compute_stft(samples, params)

# stft_result is a complex numpy array
# Shape: (n_fft/2 + 1, n_frames)

# Extract magnitude and phase
magnitude = np.abs(stft_result)
phase = np.angle(stft_result)

# Access specific bin/frame
c = stft_result[10, 5]
print(f"Bin 10 Frame 5: mag={np.abs(c):.2e}, phase={np.angle(c):.2f}")
```

</td>
</tr>
</table>

### Computing Power Spectrum from STFT

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

// Compute STFT
let planner = SpectrogramPlanner::new();
let stft_plan = planner.stft_plan(&params)?;
let stft = stft_plan.compute_stft(&samples)?;

// Convert to power spectrum manually
let power: Vec<Vec<f64>> = stft
    .axis_iter(ndarray::Axis(1))  // iterate over frames
    .map(|frame| {
        frame.iter()
            .map(|c| c.norm_sqr())  // |X|²
            .collect()
    })
    .collect();

// Or use the built-in method
let spec = LinearPowerSpectrogram::compute(
    &samples, &params, None
)?;
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

# Compute STFT
stft = sg.compute_stft(samples, params)

# Convert to power spectrum manually
power = np.abs(stft) ** 2  # |X|²

# Or use the built-in method
spec = sg.compute_linear_power_spectrogram(
    samples, params
)

# Verify they match
assert np.allclose(power, spec.data)
```

</td>
</tr>
</table>

---

## Streaming and Online Processing

For **real-time applications**, you can process audio **frame-by-frame** without computing the entire spectrogram.

### Frame-by-Frame Processing

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
let mel = MelParams::new(40, 0.0, 8000.0)?;

// Create plan (do this once at startup)
let planner = SpectrogramPlanner::new();
let mut plan = planner.mel_power_plan(&params, &mel)?;

// Buffer for incoming audio
let mut buffer = Vec::new();

// As audio chunks arrive...
for chunk in audio_stream {
    buffer.extend_from_slice(&chunk);

    // Compute as many frames as possible
    let n_fft = stft.n_fft();
    let hop_size = stft.hop_size();
    let frames_available =
        (buffer.len() - n_fft) / hop_size + 1;

    for frame_idx in 0..frames_available {
        // Compute single frame
        let frame_data = plan.compute_frame(
            &buffer, frame_idx
        )?;

        // Process frame in real-time
        process_frame(&frame_data);
    }

    // Keep samples needed for next frames
    let samples_to_keep = n_fft +
        (frames_available - 1) * hop_size;
    buffer.drain(0..buffer.len() - samples_to_keep);
}
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)
mel = sg.MelParams(40, 0.0, 8000.0)

# Create plan (do this once at startup)
planner = sg.SpectrogramPlanner()
plan = planner.mel_power_plan(params, mel)

# Buffer for incoming audio
buffer = np.array([], dtype=np.float64)

# As audio chunks arrive...
for chunk in audio_stream:
    buffer = np.concatenate([buffer, chunk])

    # Compute as many frames as possible
    frames_available = (
        (len(buffer) - stft.n_fft) // stft.hop_size + 1
    )

    for frame_idx in range(frames_available):
        # Compute single frame
        frame_data = plan.compute_frame(buffer, frame_idx)

        # Process frame in real-time
        process_frame(frame_data)

    # Keep samples needed for next frames
    samples_to_keep = (
        stft.n_fft + (frames_available - 1) * stft.hop_size
    )
    buffer = buffer[-samples_to_keep:]
```

</td>
</tr>
</table>

### Real-Time Feature Extraction

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let mut plan = planner.mel_power_plan(&params, &mel)?;
let mut buffer = Vec::new();

for chunk in audio_stream {
    buffer.extend_from_slice(&chunk);

    let frames_available =
        (buffer.len() - stft.n_fft()) / stft.hop_size() + 1;

    if frames_available > 0 {
        // Get latest frame
        let frame = plan.compute_frame(
            &buffer, frames_available - 1
        )?;

        // Extract features on the fly
        let energy: f64 = frame.iter().sum();
        let max_bin = frame.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)|
                a.partial_cmp(b).unwrap()
            )
            .map(|(i, _)| i)
            .unwrap();

        println!("Energy: {:.2e}, Peak bin: {}",
            energy, max_bin);

        // Update buffer
        let keep = stft.n_fft() +
            (frames_available - 1) * stft.hop_size();
        buffer.drain(0..buffer.len() - keep);
    }
}
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

plan = planner.mel_power_plan(params, mel)
buffer = np.array([], dtype=np.float64)

for chunk in audio_stream:
    buffer = np.concatenate([buffer, chunk])

    frames_available = (
        (len(buffer) - stft.n_fft) // stft.hop_size + 1
    )

    if frames_available > 0:
        # Get latest frame
        frame = plan.compute_frame(
            buffer, frames_available - 1
        )

        # Extract features on the fly
        energy = np.sum(frame)
        max_bin = np.argmax(frame)

        print(f"Energy: {energy:.2e}, Peak bin: {max_bin}")

        # Update buffer
        keep = stft.n_fft + (frames_available - 1) * stft.hop_size
        buffer = buffer[-keep:]
```

</td>
</tr>
</table>

---

## Advanced Features

### MFCC

**Mel-Frequency Cepstral Coefficients** are widely used in speech recognition and audio classification.

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

// Configure MFCC parameters
let mfcc_params = MfccParams::new(
    13,    // n_mfcc: number of coefficients
    80,    // n_mels: mel bands
    0.0,   // f_min
    8000.0 // f_max
)?;

// Compute MFCCs
let mfccs = mfcc(&samples, &params, &mfcc_params)?;

// mfccs.data is Array2<f64> with shape (n_mfcc, n_frames)
println!("MFCCs: {} coefficients × {} frames",
    mfccs.n_mfcc(), mfccs.n_frames());

// Access first frame MFCCs
let first_frame = mfccs.data().column(0);
for (i, &coef) in first_frame.iter().enumerate() {
    println!("  MFCC {}: {:.3}", i, coef);
}
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Configure MFCC parameters
mfcc_params = sg.MfccParams(
    n_mfcc=13,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0
)

# Compute MFCCs
mfccs = sg.compute_mfcc(samples, params, mfcc_params)

# mfccs is a numpy array with shape (n_mfcc, n_frames)
print(f"MFCCs: {mfccs.shape[0]} coefficients × {mfccs.shape[1]} frames")

# Access first frame MFCCs
first_frame = mfccs[:, 0]
for i, coef in enumerate(first_frame):
    print(f"  MFCC {i}: {coef:.3f}")
```

</td>
</tr>
</table>

### Chromagram

**Chromagrams** (pitch class profiles) map frequencies to 12 semitone bins (C, C#, D, ..., B).

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let stft = StftParams::new(2048, 512, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;

// Configure chromagram parameters
let chroma_params = ChromaParams::new(
    12,     // n_chroma: always 12 for standard chromagram
    ChromaNorm::L2  // normalization method
)?;

// Compute chromagram
let chroma = chromagram(&samples, &params, &chroma_params)?;

// chroma.data is Array2<f64> with shape (12, n_frames)
println!("Chromagram: {} pitch classes × {} frames",
    chroma.n_chroma(), chroma.n_frames());

// Find dominant pitch class in first frame
let first_frame = chroma.data().column(0);
let (pitch_class, energy) = first_frame.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();

let pitch_names = ["C", "C#", "D", "D#", "E", "F",
                   "F#", "G", "G#", "A", "A#", "B"];
println!("Dominant pitch: {} (energy: {:.3})",
    pitch_names[pitch_class], energy);
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(2048, 512, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Configure chromagram parameters
chroma_params = sg.ChromaParams(
    n_chroma=12,
    norm="l2"
)

# Compute chromagram
chroma = sg.compute_chromagram(
    samples, params, chroma_params
)

# chroma is a numpy array with shape (12, n_frames)
print(f"Chromagram: {chroma.shape[0]} pitch classes × {chroma.shape[1]} frames")

# Find dominant pitch class in first frame
first_frame = chroma[:, 0]
pitch_class = np.argmax(first_frame)
energy = first_frame[pitch_class]

pitch_names = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]
print(f"Dominant pitch: {pitch_names[pitch_class]} (energy: {energy:.3f})")
```

</td>
</tr>
</table>

### CQT

**Constant-Q Transform** provides logarithmic frequency spacing ideal for music analysis.

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

let stft = StftParams::new(2048, 512, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, 16000.0)?;

// Configure CQT parameters
let cqt_params = CqtParams::new(
    84,     // n_bins: typically 7 octaves × 12 bins/octave
    12,     // bins_per_octave
    55.0,   // f_min: A1 note
    None    // f_max: computed from n_bins
)?;

// Compute CQT
let cqt_result = cqt(&samples, &params, &cqt_params)?;

// cqt_result.spectrogram is a CqtPowerSpectrogram
let spec = &cqt_result.spectrogram;
println!("CQT: {} bins × {} frames",
    spec.n_bins(), spec.n_frames());

// Access CQT frequencies (logarithmically spaced)
let freqs = spec.axes().frequencies();
println!("Frequency range: {:.1} Hz to {:.1} Hz",
    freqs[0], freqs[freqs.len() - 1]);
```

</td>
<td>

```python
import spectrograms as sg

stft = sg.StftParams(2048, 512, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, 16000)

# Configure CQT parameters
cqt_params = sg.CqtParams(
    n_bins=84,
    bins_per_octave=12,
    f_min=55.0,
    f_max=None  # computed from n_bins
)

# Compute CQT
cqt_result = sg.compute_cqt(samples, params, cqt_params)

# cqt_result is a Spectrogram object
print(f"CQT: {cqt_result.n_bins} bins × {cqt_result.n_frames} frames")

# Access CQT frequencies (logarithmically spaced)
freqs = cqt_result.frequencies
print(f"Frequency range: {freqs[0]:.1f} Hz to {freqs[-1]:.1f} Hz")
```

</td>
</tr>
</table>

---

## Performance Tips

### 1. Always Reuse Plans for Batch Processing

<table>
<tr>
<th>❌ Slow</th>
<th>✅ Fast</th>
</tr>
<tr>
<td>

```rust
// Creates FFT plan for EACH signal
for signal in signals {
    let spec = MelDbSpectrogram::compute(
        signal, &params, &mel, Some(&db)
    )?;
}
```

</td>
<td>

```rust
// Creates FFT plan ONCE
let mut plan = planner.mel_db_plan(&params, &mel, Some(&db))?;
for signal in signals {
    let spec = plan.compute(signal)?;
}
```

</td>
</tr>
</table>

### 2. Use Power-of-2 FFT Sizes

FFT algorithms are significantly faster when `n_fft` is a power of 2:

```rust
// ✅ Good: powers of 2
let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let stft = StftParams::new(1024, 512, WindowType::Hanning, true)?;
let stft = StftParams::new(2048, 1024, WindowType::Hanning, true)?;

// ❌ Slower: non-power-of-2
let stft = StftParams::new(1000, 500, WindowType::Hanning, true)?;
```

### 3. Choose the Right FFT Backend

- **FFTW** (default): Fastest, but requires C library dependency
- **RealFFT**: Pure Rust, slower but no external dependencies

```toml
# Use FFTW for maximum performance (default)
[dependencies]
spectrograms = "0.1"

# Use RealFFT for portability
[dependencies]
spectrograms = { version = "0.1", default-features = false, features = ["realfft"] }
```

### 4. Optimize Hop Size

Smaller `hop_size` = more frames = slower computation:

```rust
// High time resolution (slow)
let stft = StftParams::new(512, 128, WindowType::Hanning, true)?;  // 75% overlap

// Balanced (common)
let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;  // 50% overlap

// Lower time resolution (fast)
let stft = StftParams::new(512, 512, WindowType::Hanning, true)?;  // 0% overlap
```

### 5. Preallocate Output Buffers (Rust Only)

For maximum performance in Rust, you can preallocate output buffers:

```rust
let mut plan = planner.mel_db_plan(&params, &mel, Some(&db))?;

// Preallocate output buffer
let mut output = Array2::<f64>::zeros((plan.freq_axis().n_bins(), 100));

for signal in signals {
    // Compute into preallocated buffer (avoids allocation)
    plan.compute_into(signal, &mut output)?;
    // Process output...
}
```

---

## Common Patterns

### Pattern 1: Audio File to Mel Spectrogram

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;

// Load audio (using hound crate)
let reader = hound::WavReader::open("audio.wav")?;
let sample_rate = reader.spec().sample_rate as f64;
let samples: Vec<f64> = reader
    .into_samples::<i16>()
    .map(|s| s.unwrap() as f64 / 32768.0)
    .collect();

// Configure parameters
let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
let params = SpectrogramParams::new(stft, sample_rate)?;
let mel = MelParams::new(80, 0.0, sample_rate / 2.0)?;
let db = LogParams::new(-80.0)?;

// Compute mel spectrogram
let spec = MelDbSpectrogram::compute(
    &samples, &params, &mel, Some(&db)
)?;

// Save as numpy array (using ndarray-npy)
ndarray_npy::write_npy("spectrogram.npy", spec.data())?;
```

</td>
<td>

```python
import spectrograms as sg
import soundfile as sf
import numpy as np

# Load audio
samples, sample_rate = sf.read("audio.wav", dtype='float64')

# If stereo, convert to mono
if samples.ndim > 1:
    samples = samples.mean(axis=1)

# Configure parameters
stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
params = sg.SpectrogramParams(stft, sample_rate)
mel = sg.MelParams(80, 0.0, sample_rate / 2)
db = sg.LogParams(-80.0)

# Compute mel spectrogram
spec = sg.compute_mel_db_spectrogram(
    samples, params, mel, db
)

# Save as numpy array
np.save("spectrogram.npy", spec.data)
```

</td>
</tr>
</table>

### Pattern 2: Process Directory of Audio Files

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;
use std::path::Path;

fn process_directory(
    dir: &Path,
    pattern: &str
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure parameters
    let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, 16000.0)?;
    let mel = MelParams::new(80, 0.0, 8000.0)?;
    let db = LogParams::new(-80.0)?;

    // Create plan ONCE
    let planner = SpectrogramPlanner::new();
    let mut plan = planner.mel_db_plan(&params, &mel, Some(&db))?;

    // Process all files
    for entry in glob::glob(pattern)? {
        let path = entry?;
        let samples = load_audio(&path)?;

        let spec = plan.compute(&samples)?;

        // Save or process spec
        let output = path.with_extension("npy");
        ndarray_npy::write_npy(&output, spec.data())?;

        println!("Processed: {}", path.display());
    }

    Ok(())
}
```

</td>
<td>

```python
import spectrograms as sg
import soundfile as sf
import numpy as np
from pathlib import Path

def process_directory(dir_path, pattern="*.wav"):
    # Configure parameters
    stft = sg.StftParams(512, 256, sg.WindowType.hanning(), True)
    params = sg.SpectrogramParams(stft, 16000)
    mel = sg.MelParams(80, 0.0, 8000.0)
    db = sg.LogParams(-80.0)

    # Create plan ONCE
    planner = sg.SpectrogramPlanner()
    plan = planner.mel_db_plan(params, mel, db)

    # Process all files
    for audio_file in Path(dir_path).glob(pattern):
        samples, sr = sf.read(audio_file, dtype='float64')

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        spec = plan.compute(samples)

        # Save or process spec
        output = audio_file.with_suffix('.npy')
        np.save(output, spec.data)

        print(f"Processed: {audio_file}")
```

</td>
</tr>
</table>

### Pattern 3: Extract Spectral Features

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;
use ndarray::s;

// Compute power spectrogram
let spec = LinearPowerSpectrogram::compute(
    &samples, &params, None
)?;

// Spectral centroid (weighted mean frequency)
let freqs = spec.axes().frequencies();
let mut centroids = Vec::new();

for frame in spec.data().axis_iter(ndarray::Axis(1)) {
    let sum_power: f64 = frame.iter().sum();
    let weighted_sum: f64 = frame.iter()
        .enumerate()
        .map(|(i, &p)| freqs[i] * p)
        .sum();

    let centroid = weighted_sum / (sum_power + 1e-10);
    centroids.push(centroid);
}

// Spectral rolloff (95% of energy)
let mut rolloffs = Vec::new();

for frame in spec.data().axis_iter(ndarray::Axis(1)) {
    let total: f64 = frame.iter().sum();
    let threshold = 0.95 * total;

    let mut cumsum = 0.0;
    let rolloff_idx = frame.iter()
        .position(|&p| {
            cumsum += p;
            cumsum >= threshold
        })
        .unwrap_or(0);

    rolloffs.push(freqs[rolloff_idx]);
}
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

# Compute power spectrogram
spec = sg.compute_linear_power_spectrogram(
    samples, params
)

freqs = spec.frequencies

# Spectral centroid (weighted mean frequency)
centroids = np.sum(
    freqs[:, np.newaxis] * spec.data, axis=0
) / (np.sum(spec.data, axis=0) + 1e-10)

# Spectral rolloff (95% of energy)
cumsum = np.cumsum(spec.data, axis=0)
total = cumsum[-1, :]
threshold = 0.95 * total

rolloff_indices = np.argmax(
    cumsum >= threshold[np.newaxis, :], axis=0
)
rolloffs = freqs[rolloff_indices]

# Spectral bandwidth
# (standard deviation of spectrum around centroid)
bandwidths = np.sqrt(
    np.sum(
        ((freqs[:, np.newaxis] - centroids[np.newaxis, :]) ** 2) * spec.data,
        axis=0
    ) / (np.sum(spec.data, axis=0) + 1e-10)
)
```

</td>
</tr>
</table>

### Pattern 4: Spectrogram Augmentation for ML

<table>
<tr>
<th>Rust</th>
<th>Python</th>
</tr>
<tr>
<td>

```rust
use spectrograms::*;
use ndarray::{Array2, Axis};
use rand::Rng;

// Compute base spectrogram
let spec = MelDbSpectrogram::compute(
    &samples, &params, &mel, Some(&db)
)?;

// Time masking (SpecAugment)
fn time_mask(
    data: &mut Array2<f64>,
    mask_width: usize
) {
    let n_frames = data.ncols();
    let mut rng = rand::thread_rng();
    let start = rng.gen_range(0..n_frames - mask_width);

    data.slice_mut(s![.., start..start + mask_width])
        .fill(-80.0);  // Fill with floor value
}

// Frequency masking
fn freq_mask(
    data: &mut Array2<f64>,
    mask_height: usize
) {
    let n_bins = data.nrows();
    let mut rng = rand::thread_rng();
    let start = rng.gen_range(0..n_bins - mask_height);

    data.slice_mut(s![start..start + mask_height, ..])
        .fill(-80.0);
}

let mut augmented = spec.data().clone();
time_mask(&mut augmented, 10);
freq_mask(&mut augmented, 5);
```

</td>
<td>

```python
import spectrograms as sg
import numpy as np

# Compute base spectrogram
spec = sg.compute_mel_db_spectrogram(
    samples, params, mel, db
)

# Time masking (SpecAugment)
def time_mask(data, mask_width):
    n_frames = data.shape[1]
    start = np.random.randint(0, n_frames - mask_width)
    data[:, start:start + mask_width] = -80.0
    return data

# Frequency masking
def freq_mask(data, mask_height):
    n_bins = data.shape[0]
    start = np.random.randint(0, n_bins - mask_height)
    data[start:start + mask_height, :] = -80.0
    return data

# Apply augmentation
augmented = spec.data.copy()
augmented = time_mask(augmented, mask_width=10)
augmented = freq_mask(augmented, mask_height=5)
```

</td>
</tr>
</table>

---

## Summary

This guide covered:

- ✅ **Installation** for both Rust and Python
- ✅ **Basic spectrogram computation** with linear, mel, and ERB scales
- ✅ **Amplitude scales** (power, magnitude, decibels)
- ✅ **Efficient plan reuse** for 2-10x speedup on batch processing
- ✅ **FFT and STFT computation** for advanced use cases
- ✅ **Streaming processing** for real-time applications
- ✅ **Advanced features** (MFCC, chromagram, CQT)
- ✅ **Performance optimization** tips
- ✅ **Common patterns** for real-world applications

### Key Takeaways

1. **For single computations**: Use convenience functions (`compute_*`)
2. **For batch processing**: Use `SpectrogramPlanner` and reuse plans (2-10x faster)
3. **For real-time**: Use `compute_frame()` for streaming processing
4. **For performance**: Use power-of-2 FFT sizes and FFTW backend

### Next Steps

- Check out the [examples/](../examples/) directory for more code examples
- Read the [API documentation](https://docs.rs/spectrograms) for detailed reference
- See [CLAUDE.md](../CLAUDE.md) for architecture details and development guidelines

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/user/spectrograms/issues)!
