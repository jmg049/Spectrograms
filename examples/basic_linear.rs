use non_empty_slice::NonEmptyVec;
/// Basic linear spectrogram example
///
/// This example demonstrates:
/// - Creating a simple sine wave
/// - Computing a linear-frequency power spectrogram
/// - Accessing spectrogram data and axes
use spectrograms::{LinearPowerSpectrogram, SpectrogramParams, StftParams, WindowType, nzu};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a 1-second sine wave at 440 Hz
    let sample_rate = 16000.0;
    let duration = 1.0;
    let frequency = 440.0;

    let samples: Vec<f64> = (0..(duration * sample_rate) as usize)
        .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin())
        .collect();
    let samples = NonEmptyVec::new(samples).unwrap();

    println!("Generated {} samples at {} Hz", samples.len(), sample_rate);

    // Set up spectrogram parameters
    let stft = StftParams::new(
        nzu!(512),           // n_fft: FFT window size
        nzu!(256),           // hop_size: samples between frames
        WindowType::Hanning, // window function
        true,                // centre: pad signal for centered frames
    )?;

    let params = SpectrogramParams::new(stft, sample_rate)?;

    // Compute the spectrogram (this is the convenience API)
    let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;

    // Access spectrogram properties
    println!("\nSpectrogram properties:");
    println!("  Frequency bins: {}", spec.n_bins());
    println!("  Time frames: {}", spec.n_frames());
    println!("  Shape: {} x {}", spec.n_bins(), spec.n_frames());

    // Access frequency and time axes
    let (f_min, f_max) = spec.axes().frequency_range();
    println!("\nFrequency range: {:.1} Hz to {:.1} Hz", f_min, f_max);

    let duration_computed = spec.axes().duration();
    println!("Duration: {:.3} seconds", duration_computed);

    // Find the peak frequency in the first frame
    let first_frame_data = spec.data().column(0);
    let (peak_bin, peak_power) = first_frame_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let peak_freq = spec.axes().frequencies()[peak_bin];
    println!(
        "\nPeak in first frame: {:.1} Hz (power: {:.2e})",
        peak_freq, peak_power
    );

    Ok(())
}
