//! STFT streaming example
//!
//! Demonstrates frame-by-frame processing for real-time applications.

use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::{f64::consts::PI, num::NonZeroUsize};

fn generate_chirp(sample_rate: f64, duration_s: f64) -> NonEmptyVec<f64> {
    let n_samples = (sample_rate * duration_s) as usize;
    let v = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let freq = 200.0 + 1000.0 * t; // Chirp from 200 to 1200 Hz
            (2.0 * PI * freq * t).sin()
        })
        .collect();
    NonEmptyVec::new(v).unwrap()
}

fn find_peak_frequency(
    spectrum: &[num_complex::Complex<f64>],
    sample_rate: f64,
    n_fft: NonZeroUsize,
) -> f64 {
    let (max_idx, _) = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
        .unwrap();

    max_idx as f64 * sample_rate / n_fft.get() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 16000.0;
    let samples = generate_chirp(sample_rate, 1.0);

    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft_params, sample_rate)?;

    let mut plan = StftPlan::new(&params)?;

    let (n_bins, n_frames) = plan.output_shape(samples.len())?;
    println!("Processing {} frames ({} bins each)...\n", n_frames, n_bins);

    // Process every 10th frame
    for frame_idx in (0..n_frames.get()).step_by(10) {
        let spectrum = plan.compute_frame_simple(&samples, frame_idx)?;
        let peak_freq = find_peak_frequency(&spectrum, sample_rate, params.stft().n_fft());
        let time_s = frame_idx as f64 * params.frame_period_seconds();
        println!(
            "Frame {:3} (t={:.3}s): peak at {:.1} Hz",
            frame_idx, time_s, peak_freq
        );
    }

    Ok(())
}
