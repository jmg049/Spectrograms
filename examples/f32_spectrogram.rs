//! Computing spectral features in single precision (`f32`).
//!
//! The crate defaults to `f64`, but every core operation is generic over the
//! scalar type via the `Sample` trait. Feeding `f32` input computes in `f32` —
//! half the memory footprint, and a natural match for ML pipelines that train
//! in `f32`. The scalar is inferred from the input, so no turbofish is needed.
//!
//! Run with: `cargo run --example f32_spectrogram`

use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use spectrograms::{Complex, WindowType, nzu, power_spectrum, stft};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1 second of a 440 Hz tone at 16 kHz, in f32.
    let sample_rate = 16_000.0_f32;
    let samples: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
        .collect();
    let samples = NonEmptyVec::new(samples).unwrap();

    // STFT — `T = f32` is inferred from the input slice.
    let spec: Array2<Complex<f32>> =
        stft(&samples, nzu!(1024), nzu!(256), WindowType::Hanning, true)?;
    println!("STFT: {} bins x {} frames (f32)", spec.nrows(), spec.ncols());

    // Single-frame power spectrum, also computed in f32.
    let frame = NonEmptyVec::new(samples.as_slice()[..1024].to_vec()).unwrap();
    let power = power_spectrum(&frame, nzu!(1024), Some(WindowType::Hanning))?;
    let peak = power
        .as_slice()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let peak_hz = peak as f32 * sample_rate / 1024.0;
    println!("Power-spectrum peak near {peak_hz:.1} Hz (expected ~440 Hz)");

    // Each f32 element is 4 bytes vs 8 for f64 — half the footprint.
    let elems = spec.len();
    println!(
        "STFT buffer: {} bytes (f32) vs {} bytes (f64 would be)",
        elems * std::mem::size_of::<Complex<f32>>(),
        elems * std::mem::size_of::<Complex<f64>>(),
    );

    Ok(())
}
