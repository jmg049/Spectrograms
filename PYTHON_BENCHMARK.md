# Benchmark: Spectrograms vs NumPy

## What We're Comparing

**spectrograms** (Rust implementation with Python bindings) against **straightforward NumPy reference implementations** and **SciPy implementations** for common audio spectrogram operations.

**Key insight**: While the likes of the NumPy implementations *can* be optimized to match spectrograms' performance, **spectrograms provides these optimizations out-of-the-box**. Users get high-performance implementations without needing to understand pre-computation, filterbank caching, or memory layout optimizations --- they just call the function and get optimal performance automatically. 

![[Mean performance speedup across all parameter configurations and signal types](./imgs/spectrograms_vs_numpy_avg_speedup.png)](./imgs/spectrograms_vs_numpy_avg_speedup.png)

|Operator |Rust (ms)|Rust Std|Numpy (ms)|Numpy Std|Scipy (ms)|Scipy Std|Avg Speedup vs NumPy|Avg Speedup vs SciPy|
|---------|---------|--------|----------|---------|----------|---------|--------------------|--------------------|
|db       |0.257    |0.165   |0.350     |0.251    |0.451     |0.366    |1.363               |1.755               |
|erb      |0.601    |0.437   |3.713     |2.703    |3.714     |2.723    |6.178               |6.181               |
|loghz    |0.178    |0.149   |0.547     |0.998    |0.534     |0.965    |3.068               |2.996               |
|magnitude|0.140    |0.089   |0.198     |0.133    |0.319     |0.277    |1.419               |2.287               |
|mel      |0.180    |0.139   |0.630     |0.851    |0.612     |0.801    |3.506               |3.406               |
|power    |0.126    |0.082   |0.205     |0.141    |0.327     |0.288    |1.630               |2.603               |

|Operator |Fixture |Rust (ms)|Rust Std|Numpy (ms)|Numpy Std|Scipy (ms)|Scipy Std|Speedup vs NumPy|Speedup vs SciPy|
|---------|--------|---------|--------|----------|---------|----------|---------|----------------|----------------|
|db       |chirp   |0.260    |0.167   |0.353     |0.254    |0.454     |0.369    |1.357           |1.743           |
|db       |impulse |0.251    |0.160   |0.337     |0.243    |0.437     |0.357    |1.345           |1.745           |
|db       |noise   |0.266    |0.172   |0.361     |0.259    |0.463     |0.375    |1.359           |1.743           |
|db       |sine_3k |0.252    |0.162   |0.348     |0.250    |0.448     |0.365    |1.383           |1.779           |
|db       |sine_440|0.256    |0.164   |0.350     |0.251    |0.451     |0.366    |1.369           |1.764           |
|erb      |chirp   |0.604    |0.448   |3.746     |2.717    |3.721     |2.693    |6.198           |6.158           |
|erb      |impulse |0.600    |0.438   |3.593     |2.673    |3.581     |2.681    |5.986           |5.967           |
|erb      |noise   |0.599    |0.432   |3.747     |2.721    |3.746     |2.709    |6.251           |6.248           |
|erb      |sine_3k |0.604    |0.439   |3.744     |2.676    |3.755     |2.755    |6.202           |6.219           |
|erb      |sine_440|0.597    |0.428   |3.733     |2.734    |3.768     |2.782    |6.254           |6.313           |
|loghz    |chirp   |0.177    |0.148   |0.513     |0.882    |0.514     |0.892    |2.892           |2.903           |
|loghz    |impulse |0.177    |0.149   |0.520     |1.045    |0.503     |0.932    |2.933           |2.834           |
|loghz    |noise   |0.179    |0.151   |0.532     |0.964    |0.538     |0.963    |2.964           |3.000           |
|loghz    |sine_3k |0.178    |0.148   |0.531     |0.881    |0.535     |0.944    |2.977           |2.998           |
|loghz    |sine_440|0.178    |0.151   |0.638     |1.182    |0.579     |1.083    |3.574           |3.245           |
|magnitude|chirp   |0.140    |0.090   |0.198     |0.133    |0.320     |0.278    |1.414           |2.283           |
|magnitude|impulse |0.137    |0.087   |0.196     |0.132    |0.313     |0.273    |1.435           |2.293           |
|magnitude|noise   |0.140    |0.090   |0.198     |0.133    |0.321     |0.278    |1.414           |2.287           |
|magnitude|sine_3k |0.140    |0.090   |0.198     |0.133    |0.320     |0.278    |1.414           |2.281           |
|magnitude|sine_440|0.140    |0.090   |0.199     |0.134    |0.321     |0.279    |1.420           |2.291           |
|mel      |chirp   |0.179    |0.137   |0.584     |0.681    |0.581     |0.622    |3.265           |3.248           |
|mel      |impulse |0.180    |0.138   |0.601     |0.787    |0.586     |0.774    |3.338           |3.254           |
|mel      |noise   |0.180    |0.140   |0.582     |0.663    |0.578     |0.609    |3.231           |3.206           |
|mel      |sine_3k |0.182    |0.143   |0.617     |0.698    |0.607     |0.727    |3.397           |3.343           |
|mel      |sine_440|0.178    |0.135   |0.766     |1.263    |0.708     |1.146    |4.312           |3.988           |
|power    |chirp   |0.127    |0.085   |0.205     |0.140    |0.329     |0.292    |1.616           |2.595           |
|power    |impulse |0.125    |0.080   |0.202     |0.138    |0.321     |0.285    |1.617           |2.567           |
|power    |noise   |0.125    |0.080   |0.204     |0.138    |0.329     |0.286    |1.631           |2.633           |
|power    |sine_3k |0.125    |0.080   |0.205     |0.142    |0.326     |0.283    |1.639           |2.607           |
|power    |sine_440|0.127    |0.086   |0.208     |0.149    |0.331     |0.294    |1.646           |2.616           |

## Operations Tested

- **Power spectrogram**: Linear frequency, power scale (|X|²)
- **Magnitude spectrogram**: Linear frequency, magnitude scale (|X|)
- **Decibel spectrogram**: Linear frequency, dB scale (10·log₁₀)
- **Mel spectrogram**: Mel-frequency scale, power
- **LogHz spectrogram**: Logarithmic frequency scale, power
- **ERB spectrogram**: ERB/gammatone scale, power

## Test Setup

**Signal fixtures** (5 types):
- 440 Hz sine wave (1 second)
- 3 kHz sine wave (1 second)
- White noise (1 second)
- Chirp (100-3000 Hz sweep)
- Unit impulse


**Benchmark protocol**:
- Configurable parameter sweeps
- 10 warmup iterations
- 100 timed iterations per operation
- Timing: `time.perf_counter()` (high-resolution monotonic clock)
- Results: mean and standard deviation across all fixtures

## Running the Benchmark

```bash
# Ensure you have the latest build
maturin develop --release

# Launch Jupyter (recommend for rendering HTML elements, but functional in the likes of VSCode too)
jupyter lab notebook python/examples/notebook.ipynb

# Run all cells to reproduce results (given your machine specs)
```

## Implementation Notes

**spectrograms optimizations** (applied automatically):
- Pre-computed filterbanks for Mel/ERB/LogHz during plan creation
- Sparse matrix operations for filterbank application
- Minimal memory allocation (pre-allocated workspace buffers)
- Native Rust implementation with zero-copy Python bindings

**NumPy reference implementations** (naive approach):
- Runtime filterbank computation
- Dense matrix operations
- Per-call memory allocation
- Pure Python with NumPy/SciPy

The point: Users shouldn't need to know about these optimizations. **spectrograms provides them by default.**
