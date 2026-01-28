//! Multi-channel STFT processing example
//!
//! Demonstrates processing each channel of stereo audio separately
//! using a single reusable STFT plan.

use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::f64::consts::PI;

fn generate_stereo_signal(
    sample_rate: f64,
    duration_s: f64,
) -> (NonEmptyVec<f64>, NonEmptyVec<f64>) {
    let n_samples = (sample_rate * duration_s) as usize;

    let left: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // 440 Hz sine wave (A4 note)
            (2.0 * PI * 440.0 * t).sin()
        })
        .collect();

    let right: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // 554.37 Hz sine wave (C#5 note)
            (2.0 * PI * 554.37 * t).sin()
        })
        .collect();

    let left = NonEmptyVec::new(left).unwrap();
    let right = NonEmptyVec::new(right).unwrap();
    (left, right)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 16000.0;
    let (left_channel, right_channel) = generate_stereo_signal(sample_rate, 1.0);

    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft_params, sample_rate)?;

    // Create a single plan and reuse it for both channels
    let mut plan = StftPlan::new(&params)?;

    println!("Processing stereo audio...\n");

    // Process left channel
    let stft_left = plan.compute(&left_channel, &params)?;
    println!(
        "Left channel:  {} bins x {} frames",
        stft_left.n_bins(),
        stft_left.n_frames()
    );

    // Reuse same plan for right channel
    let stft_right = plan.compute(&right_channel, &params)?;
    println!(
        "Right channel: {} bins x {} frames",
        stft_right.n_bins(),
        stft_right.n_frames()
    );

    println!("\nThis approach is efficient for multi-channel audio:");
    println!("- Single FFT plan creation (expensive operation done once)");
    println!("- Reused for all channels");
    println!("- Minimal memory overhead");

    Ok(())
}
