# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A high-performance Rust library (`spectrograms`) with Python bindings (via PyO3/maturin) for FFT-based audio and image processing. Supports linear, mel, ERB, CQT, chromagram, MFCC, binaural, and MDCT transforms. Offers RealFFT (default, pure Rust) and FFTW (C library, faster) backends.

## Build Commands

```bash
# Rust
cargo build                                        # default features: realfft + python
cargo build --release
cargo build --no-default-features --features realfft   # no Python bindings
cargo build --no-default-features --features fftw,python  # FFTW backend

# Python bindings (requires maturin)
maturin develop                  # install into current virtualenv (debug)
maturin develop --release        # optimized
maturin build                    # build wheel

# Docs
cargo docs                       # alias for: cargo doc --no-deps --open
```

## Test Commands

```bash
# Rust
cargo test                             # all tests
cargo test --test builder_tests        # individual test file
cargo test my_fn_name                  # single test by name

# Python (run after maturin develop)
pytest python/tests/                   # all Python tests
pytest python/tests/test_mdct.py      # single test file
pytest -k "test_name"                 # single test by name
```

## Lint / Format

```bash
cargo fmt
cargo clippy
```

## Benchmarks

```bash
cargo bench
cargo bench --bench stft_benchmarks   # specific bench file
```

## Architecture

### Rust crate (`src/`)

- **`fft_backend.rs`** — Abstraction over RealFFT / FFTW; selected at compile time via feature flags.
- **`spectrogram.rs`** — Core STFT, window application, and the generic `Spectrogram<FreqScale, AmpScale>` type with type-safe compile-time combinations (e.g. `MelPowerSpectrogram`, `ErbDbSpectrogram`). This is the largest module.
- **`window.rs`** — Window functions (Hanning, Hamming, Blackman, Kaiser, Gaussian).
- **`erb.rs`, `cqt.rs`** — ERB and Constant-Q Transform frequency scales.
- **`mfcc.rs`, `chroma.rs`** — Audio feature extraction.
- **`binaural.rs`** — ITD/IPD/ILD/ILR spatial audio cues.
- **`fft2d.rs`, `image_ops.rs`** — 2D FFT and convolution/filtering for image processing.
- **`mdct.rs`** — Modified Discrete Cosine Transform (audio codec use).
- **`error.rs`** — Crate-level error types.

### Python bindings (`src/python/`)

- **`mod.rs`** — PyO3 module root; registers all submodules and classes.
- **`params.rs`** — Python wrappers for `StftParams`, `SpectrogramParams`, `MelParams`, `CqtParams`, etc.
- **`spectrogram.rs`** — `PySpectrogram` class (holds `py_data: Option<Py<PyArray2<f64>>>` for zero-copy DLPack).
- **`functions.rs`** — Compute functions exposed to Python; macros generate wrappers.
- **`planner.rs`** — Plan-based batch API (`SpectrogramPlanner`).
- **`binaural.rs`, `mdct.rs`, `fft2d.rs`** — Domain-specific Python wrappers.
- **`dlpack.rs`** — DLPack protocol implementation for zero-copy PyTorch/JAX tensor exchange.

### Python package (`python/spectrograms/`)

Pure-Python layer on top of the compiled extension. `torch.py` and `jax.py` provide convenience wrappers using DLPack.

## Key Design Patterns

**Type-safe spectrograms**: `Spectrogram<F, A>` enforces frequency scale and amplitude scale at compile time. Use the type aliases (`MelPowerSpectrogram`, etc.) rather than the generic form directly.

**Plan-based API**: `SpectrogramPlanner` and `Fft2dPlanner` reuse FFT plans across calls — essential for batch workloads.

**Non-empty input**: Uses `NonEmptyVec` / `NonZeroUsize` (macro `nzu!`) to enforce valid inputs at compile time.

**Zero-copy DLPack**: `PySpectrogram` stores a `Py<PyArray2<f64>>` created during computation; `__dlpack__` returns a view of it with no copy. Modifying the returned tensor modifies the spectrogram data.

**GIL-free Python**: Computation functions release the GIL, enabling true parallelism from Python threads.

## Feature Flags

| Flag | Notes |
|------|-------|
| `realfft` (default) | Pure Rust FFT — no system deps |
| `fftw` | Faster; requires `libfftw3-dev` installed |
| `python` (default) | PyO3 bindings |
| `serde` | Serialize/deserialize spectrograms |
| `rayon` | Parallel computation (used in binaural) |

## Python API Conventions

- Window types: `sg.WindowType.hanning` (enum), not a string.
- Mel spectrograms: pass `sg.MelParams(n_mels, f_min, f_max)`.
- CQT spectrograms: pass `sg.CqtParams(bins_per_octave, n_octaves, f_min)`.
- Chromagram: `sg.compute_chromagram(samples, stft, sample_rate, chroma_params)`.

## Investigation Rules

- Never read more than 3 files without pausing and reporting findings first
- When investigating performance: state your hypothesis BEFORE reading any code
- Do not recurse into dependencies or generated files
- If you are about to read a file, say which file and why, then stop and wait for confirmation
- Prefer `grep`/`rg` over full file reads to locate relevant code first