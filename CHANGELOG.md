# Changelog

> Date format = YYYY-MM-DD

## [2.0.0] - 2026-06-26

### Added

- **Precision-generic API (f32/f64)**: all core computations are now generic over the floating-point scalar type via a new sealed `Sample` trait (implemented for `f32` and `f64`), exported at the crate root. The scalar type **defaults to `f64`**, so existing code is source-compatible and unchanged; `f32` is opt-in (inferred from inputs or via turbofish, e.g. `stft::<f32>(..)`). Coverage spans STFT, Mel/ERB/CQT, MFCC, chroma, MDCT, convolution, minimum-phase, binaural, 2D FFT, and image operations. Added `tests/f32_smoke_tests.rs` exercising the f32 path end-to-end.
- **End-to-end native-`T` pipeline**: the high-level typed spectrogram pipeline (`Spectrogram<_, _, T>`, `Workspace`, frequency mapping, amplitude scaling, and the ERB/mel filterbank application) now computes natively in `T` rather than via f64 intermediates, so `f32` spectrograms are genuinely single-precision throughout (filterbank coefficients are still built in f64 and converted at apply time; f64 results are bit-identical).
- **Python `dtype` parameter (everywhere)**: every Python function whose Rust counterpart is generic over the scalar accepts `dtype="float32"` / `"float64"` (default `"float64"`) and computes natively at that precision, returning arrays of the matching dtype. This covers: all `compute_*_spectrogram` functions (linear/mel/erb/log-Hz/CQT × power/magnitude/dB); the 1D `compute_stft`/`compute_fft`/`compute_rfft`/`compute_irfft`/`compute_power_spectrum`/`compute_magnitude_spectrum`/`compute_istft`; the 2D `fft2d`/`ifft2d`/`power_spectrum_2d`/`magnitude_spectrum_2d`, the `fftshift` family, and the image ops (`convolve_fft`, low/high/band-pass filters, `detect_edges_fft`, `sharpen_fft`); `mdct`/`imdct`; the binaural `compute_itd/ipd/ild/ilr_spectrogram` (+ the two `*_diff`); and the `SpectrogramPlanner` plan builders (the plan carries its precision; plans gained a `.dtype` property); `compute_chromagram`, `compute_mfcc`, `gaussian_kernel_2d`, `fftfreq`, `rfftfreq`, the `Fft2dPlanner` class (`.dtype` at construction), and the `WindowType.make_*` window generators. `Spectrogram` gained a `.dtype` property, and `.data` / `.astype` / the DLPack export carry the chosen dtype (zero-copy preserved). The only Python functions that remain `f64` are those whose result is inherently `f64` (scalar/axis metadata getters). Enabled by making the `SpectrogramPlanner` builders and `Spectrogram::compute` convenience constructors generic over the scalar, and by making the Rust `chromagram`, `mfcc`, `gaussian_kernel_2d`, `fftfreq`, `rfftfreq` functions generic over the scalar (`chromagram`/`mfcc` infer it from their input; non-breaking for f64).

### Changed

- The FFT backend plan traits (`R2cPlan`, `C2rPlan`, `C2cPlan`, and the 2D/planner variants) and the `realfft`/`fftw` plan structs (`RealFftPlan`, `RealFftC2cPlan`, `RealFftInversePlan`, …) are now generic over the scalar type with a `T = f64` default, so bare names continue to mean the `f64` versions.
- **BREAKING (Python)**: `compute_stft`, `compute_chromagram`, `compute_mfcc`, and the binaural `compute_itd/ipd/ild/ilr_spectrogram` now return rich result **objects** (`StftResult`, `Chromagram`, `Mfcc`, `ItdSpectrogram`/`IpdSpectrogram`/`IldSpectrogram`/`IlrSpectrogram`) instead of bare numpy arrays — consistent with `compute_*_spectrogram` returning `Spectrogram`. The objects stay array-compatible (`np.asarray(x)`, `__array__`, `__dlpack__`) and expose `.data`, `.dtype`, metadata, and (binaural) the previously-unreachable `.histogram(...)`. Access the raw array via `.data`. `Chromagram`/`Mfcc` and the four binaural classes are newly registered; `Chromagram` was previously dead code and `StftResult` was orphaned.
- The Rust `chromagram`, `mfcc`, `gaussian_kernel_2d`, `fftfreq`, `rfftfreq` functions are now generic over the scalar (`chromagram`/`mfcc` infer it from their input slice; non-breaking for `f64` callers).

### Removed

- **BREAKING**: removed the legacy single-precision names, now fully covered by the generic API — the `R2cPlanF32` / `C2cPlanF32` traits, the `RealFftPlanF32` / `RealFftC2cPlanF32` type aliases, and the `mdct_f32` / `imdct_f32` functions. Migrate to `RealFftPlan<f32>`, `RealFftC2cPlan<f32>`, the generic `R2cPlan<f32>` / `C2cPlan<f32>` traits, and `mdct::<f32>` / `imdct::<f32>`. This is the change that takes the crate to **2.0.0**.

## [1.4.4] - 2026-06-26

### Added

- **FFT convolution module** (`convolution`): `fft_convolve` and `fft_deconvolve` for FFT-based 1D convolution/deconvolution, plus a streaming `OverlapSaveConvolver` for block-based (real-time) convolution with a fixed impulse response.
- **Minimum-phase**: `minimum_phase` and `minimum_phase_with` compute the minimum-phase version of an impulse response via the real-cepstrum method.

## [1.4.3] - 2026-06-12

### Added

- Re-export the complex-to-complex FFT plans (`RealFftC2cPlan`, `RealFftC2cPlanF32`, behind the `realfft` feature) and `num_complex::Complex`, so downstream crates can drive the planned C2C FFTs directly (used by `opus_native`'s MDCT backend).

### Removed

- Excluded the internal `CLAUDE.md` from the published crate.

## [1.4.2] - 2026-06-12

### Changed

- ERB / gammatone IIR filterbank improvements in `erb.rs`, plus assorted fixes across `fft_backend.rs` and `spectrogram.rs`.

### Fixed

- 43 Clippy warnings across `mdct.rs`, `binaural.rs`, `spectrogram.rs`, and `fft_backend.rs`:
  - Added `# Errors`, `# Panics`, and `# Safety` doc sections where required.
  - Removed redundant `#[must_use]` from functions whose return types are already `#[must_use]`.
  - Added `// SAFETY:` comments to all undocumented `unsafe` blocks.
  - Hoisted nested `pow_mag` / `np_mod` helper functions to module scope in `binaural.rs`.
  - Simplified histogram weight branches (both branches were `1.0`).
  - Fixed needless range loop in `imdct_frame`.
- 6 rustdoc "unresolved link" warnings in `mdct.rs` doc comments (bracket notation in math expressions).
- Empty line between doc comment sections in `spectrogram.rs` (`LogParams::new_unchecked`).

## [1.4.1] - 2026-05-27

### Changed

- Dependency bump.
- Packaging: include `.rs` sources in the published package and sync the Python package version.

## [1.4.0] - 2026-04-28

### Added

- **MDCT/IMDCT**: new `mdct` module with `MdctParams`, `mdct`, `imdct`, and the single-precision `mdct_f32` / `imdct_f32`. Computed in O(N log N) via a single C2c(N) FFT (packing trick). `MdctParams::sine_window` produces parameters satisfying the TDAC condition for perfect reconstruction at 50% hop; `mdct`/`imdct` take `&NonEmptySlice<T>` and the plan constructors are infallible.
  - Python bindings: `sg.MdctParams`, `sg.mdct`, `sg.imdct` (GIL-free).
  - Benchmarks `mdct_benchmarks.rs` and `mdct_vs_vorbis.rs`; a Rust example in `readme_snippets.rs`, a Python `python/examples/mdct_example.py`, and a README MDCT section.

## [1.1.0] - 2026-02-10

### Added

- Binaural spectrograms: ITD, IPD, ILD, ILR with histogram support.
- `compute_itd_spectrogram_diff` and `compute_ilr_spectrogram_diff` comparison functions.
- Rayon parallelisation for binaural computation (feature flag `rayon`).

## [1.0.2] - 2026-02-10

- Added `PyStftResult` wrapper.
