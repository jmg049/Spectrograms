# Python Examples

This directory contains comprehensive examples demonstrating the spectrograms library's Python API.

## Running the Examples

All examples are standalone Python scripts that can be run directly:

```bash
python basic_linear.py
python mel_spectrogram.py
python batch_processing.py
# ... etc
```

Make sure you have the spectrograms library installed:

```bash
pip install spectrograms
```

## Examples Overview

### 1. basic_linear.py
**Basic Linear Spectrogram**

A simple introduction to computing linear-frequency power spectrograms.

- Generates a test signal (440 Hz sine wave)
- Configures STFT parameters
- Computes and analyzes a linear power spectrogram
- Demonstrates peak frequency detection

**Topics covered:**
- `StftParams` configuration
- `SpectrogramParams` setup
- `compute_linear_power_spectrogram()` usage
- Accessing spectrogram data, frequencies, and times

**Recommended for:** First-time users

---

### 2. mel_spectrogram.py
**Mel-Scale Spectrograms**

Demonstrates mel-scale spectrograms with different amplitude scales.

- Generates a frequency chirp (sweep from 200 Hz to 4000 Hz)
- Computes mel spectrograms in three amplitude scales:
  - Power (`|X|²`)
  - Magnitude (`|X|`)
  - Decibels (dB)
- Compares the different representations

**Topics covered:**
- `MelParams` configuration
- Mel filterbank parameters
- Different amplitude scales
- dB conversion with `LogParams`

**Recommended for:** Audio analysis, speech processing

---

### 3. batch_processing.py
**Efficient Batch Processing with Plan Reuse**

Performance comparison between convenience API and planner API.

- Processes 50 random signals two ways:
  1. Convenience API (creates new plan each time)
  2. Planner API (reuses a single plan)
- Measures and compares performance
- Demonstrates 1.5-3x speedup from plan reuse

**Topics covered:**
- `SpectrogramPlanner` usage
- Creating and reusing plans
- Performance optimization
- When to use each API

**Recommended for:** Batch processing, production applications

---

### 4. compare_windows.py
**Window Function Comparison**

Compares different window functions and their effects on spectrograms.

- Tests 8 different window functions
- Analyzes spectral leakage for each
- Demonstrates frequency peak detection
- Provides guidelines for choosing windows

**Window functions tested:**
- Hann (hanning)
- Hamming
- Blackman
- Bartlett (triangular)
- Rectangular
- Kaiser (multiple β values)
- Gaussian

**Topics covered:**
- Window function selection
- Spectral leakage
- Trade-offs between windows
- Application-specific recommendations

**Recommended for:** Understanding window functions, signal processing

---

### 5. streaming.py
**Streaming/Frame-by-Frame Processing**

Demonstrates real-time processing using frame-by-frame computation.

- Simulates streaming audio in chunks
- Processes audio as it arrives (online processing)
- Compares streaming vs. batch results
- Shows buffer management for frame computation
- Demonstrates real-time feature extraction

**Topics covered:**
- `compute_frame()` method
- Online processing patterns
- Buffer management
- Real-time feature extraction (spectral centroid, rolloff)

**Recommended for:** Real-time applications, audio streaming

---

### 6. mfcc_example.py
**MFCCs (Mel-Frequency Cepstral Coefficients)**

Comprehensive guide to computing and understanding MFCCs.

- Generates a speech-like test signal
- Computes MFCCs with different parameter settings
- Analyzes coefficient statistics and interpretation
- Demonstrates feature normalization
- Shows common applications

**Topics covered:**
- `MfccParams` configuration
- Speech standard preset
- MFCC coefficient interpretation (C0, C1, C2, etc.)
- Feature normalization for ML
- Applications in speech recognition

**Recommended for:** Speech recognition, speaker identification, audio classification

---

### 7. chromagram_example.py
**Chromagrams (Pitch Class Profiles)**

Music analysis using chromagrams for harmonic content.

- Generates musical chords (C major, A minor)
- Computes chromagrams (12 pitch classes)
- Analyzes pitch class distributions
- Implements simple chord recognition
- Demonstrates template matching

**Topics covered:**
- `ChromaParams` configuration
- Music standard preset
- Pitch class representation
- Chord recognition
- Applications in music analysis

**Recommended for:** Music analysis, chord recognition, harmonic analysis

---

## Example Progression

We recommend going through the examples in this order:

1. **basic_linear.py** - Start here to learn the basics
2. **mel_spectrogram.py** - Learn about frequency scales
3. **compare_windows.py** - Understand window functions
4. **batch_processing.py** - Optimize for production
5. **streaming.py** - Real-time processing patterns
6. **mfcc_example.py** or **chromagram_example.py** - Application-specific features

## Common Patterns

### Creating Parameters

```python
import spectrograms as sg

# STFT configuration
stft = sg.StftParams(
    n_fft=512,
    hop_size=256,
    window="hanning",
    centre=True
)

# Base spectrogram parameters
params = sg.SpectrogramParams(stft, sample_rate=16000)

# Or use presets
params = sg.SpectrogramParams.speech_default(16000)
params = sg.SpectrogramParams.music_default(44100)
```

### Computing Spectrograms

```python
# One-shot computation (convenience API)
spec = sg.compute_mel_db_spectrogram(samples, params, mel_params, db_params)

# Batch processing (planner API)
planner = sg.SpectrogramPlanner()
plan = planner.mel_db_plan(params, mel_params, db_params)

for signal in signals:
    spec = plan.compute(signal)
    # Process spec...
```

### Accessing Results

```python
# Spectrogram has these properties/methods:
spec.data           # NumPy array (n_bins, n_frames)
spec.frequencies    # List of frequency values
spec.times          # List of time values
spec.n_bins         # Number of frequency bins
spec.n_frames       # Number of time frames
spec.shape          # (n_bins, n_frames)
spec.frequency_range()  # (f_min, f_max)
spec.duration()     # Total duration in seconds
spec.params         # Original parameters
```

## Parameter Guidelines

### Sample Rates

- **Speech**: 8000 Hz or 16000 Hz
- **Music**: 22050 Hz or 44100 Hz
- **High-quality audio**: 48000 Hz

### FFT Sizes (powers of 2 for efficiency)

- **Speech**: 512 (32ms at 16kHz)
- **Music**: 2048-4096 (46-93ms at 44.1kHz)
- **Chromagram**: 4096-8192 (better frequency resolution)

### Hop Sizes

- **Speech**: 160 (10ms at 16kHz) - 50% overlap with n_fft=512
- **Music**: 512 (11.6ms at 44.1kHz) - 75% overlap with n_fft=2048
- **General**: 50-75% overlap is common

### Mel Parameters

- **Speech**: 40-80 mel bands, 0 Hz to Nyquist
- **Music**: 128 mel bands, 0 Hz to Nyquist

### MFCC Parameters

- **Speech recognition**: 13 coefficients
- **Speaker recognition**: 13-20 coefficients
- **Music**: 13-20 coefficients

## Performance Tips

1. **Use plan reuse** for batch processing (1.5-3x faster)
2. **Choose FFT sizes** that are powers of 2
3. **Release GIL**: All compute functions release the GIL for parallelism
4. **Memory**: Pre-allocate arrays when processing many files

## Further Reading

- Main package README: `../README.md`
- Type stubs for IDE support: `../spectrograms/__init__.pyi`
- Rust documentation: https://github.com/jmg049/Spectrograms

## Questions or Issues?

- GitHub Issues: https://github.com/jmg049/Spectrograms/issues
- Package documentation: https://github.com/jmg049/Spectrograms
