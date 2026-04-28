# Changelog

> Date format = YYYY-MM-DD

## [Unreleased]

### Added

- **MDCT/IMDCT**: New `mdct` module with `MdctParams`, `mdct`, `imdct`, `mdct_f32`, and `imdct_f32`.
  - Generic over f32/f64 via sealed trait; computed in O(N log N) via a single C2c(N) FFT (packing trick).
  - `MdctParams::sine_window` constructor produces parameters that satisfy the TDAC condition for perfect reconstruction at 50% hop.
  - Python bindings: `sg.MdctParams`, `sg.mdct`, `sg.imdct` with GIL-free computation.
  - Benchmarks: `mdct_benchmarks.rs` and `mdct_vs_vorbis.rs`.
- **MDCT examples**: Rust example in `readme_snippets.rs` and Python `python/examples/mdct_example.py`.
- **README**: Added MDCT section with Rust and Python usage examples.

### Changed

- `compute_mdct` renamed to `mdct`; `compute_imdct` renamed to `imdct`; likewise for f32 variants.
- `mdct` and `mdct_f32` now accept `&NonEmptySlice<T>` instead of `&[T]`, consistent with the rest of the crate API.
- `MdctFwdPlan::new` and `MdctInvPlan::new` no longer return `SpectrogramResult` (infallible).

### Fixed

- 43 Clippy warnings resolved across `mdct.rs`, `binaural.rs`, `spectrogram.rs`, and `fft_backend.rs`:
  - Added `# Errors`, `# Panics`, and `# Safety` doc sections where required.
  - Removed redundant `#[must_use]` from functions whose return types are already `#[must_use]`.
  - Added `// SAFETY:` comments to all undocumented `unsafe` blocks.
  - Hoisted nested `pow_mag` / `np_mod` helper functions to module scope in `binaural.rs`.
  - Simplified histogram weight branches (both branches were `1.0`).
  - Fixed needless range loop in `imdct_frame`.
- 6 rustdoc "unresolved link" warnings in `mdct.rs` doc comments (bracket notation in math expressions).
- Empty line between doc comment sections in `spectrogram.rs` (`LogParams::new_unchecked`).

## [1.1.0] - 2026-02-10

### Added

- Binaural spectrograms: ITD, IPD, ILD, ILR with histogram support.
- `compute_itd_spectrogram_diff` and `compute_ilr_spectrogram_diff` comparison functions.
- Rayon parallelisation for binaural computation (feature flag `rayon`).

## [1.0.2] - 2026-02-10

- Added `PyStftResult` wrapper.
