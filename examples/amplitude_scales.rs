/// Amplitude scale comparison example
///
/// This example demonstrates:
/// - Power, Magnitude, and dB amplitude scales
/// - When to use each scale
/// - Conversion relationships
use spectrograms::{
    LinearDbSpectrogram, LinearMagnitudeSpectrogram, LinearPowerSpectrogram, LogParams,
    SpectrogramParams, StftParams, WindowType,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a simple sine wave
    let sample_rate = 16000.0;
    let samples: Vec<f64> = (0..1600)
        .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();

    // Set up parameters
    let stft = StftParams::new(512, 256, WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, sample_rate)?;

    println!("Comparing amplitude scales for a 440 Hz sine wave:\n");

    // Compute power spectrogram
    let power_spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;
    let power_frame = power_spec.data().column(0);
    let peak_power_idx = power_frame
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let peak_power = power_frame[peak_power_idx];

    println!("Power scale (|X|²):");
    println!("  Peak power: {:.6e}", peak_power);
    println!("  Use case: Energy analysis, signal power measurements");
    println!();

    // Compute magnitude spectrogram
    let mag_spec = LinearMagnitudeSpectrogram::compute(&samples, &params, None)?;
    let mag_frame = mag_spec.data().column(0);
    let peak_mag = mag_frame[peak_power_idx];

    println!("Magnitude scale (|X|):");
    println!("  Peak magnitude: {:.6e}", peak_mag);
    println!(
        "  Relationship: magnitude = sqrt(power) = {:.6e}",
        peak_power.sqrt()
    );
    println!("  Use case: Amplitude analysis, linear perception");
    println!();

    // Compute dB spectrogram
    let db_params = LogParams::new(-80.0)?;
    let db_spec = LinearDbSpectrogram::compute(&samples, &params, Some(&db_params))?;
    let db_frame = db_spec.data().column(0);
    let peak_db = db_frame[peak_power_idx];

    println!("Decibel scale (10·log₁₀(power)):");
    println!("  Peak level: {:.2} dB", peak_db);
    println!(
        "  Relationship: dB = 10*log10(power) = {:.2} dB",
        10.0 * peak_power.log10()
    );
    println!("  Use case: Audio analysis, dynamic range visualization, human perception");
    println!();

    // Demonstrate dynamic range
    let power_min = power_frame.iter().cloned().fold(f64::INFINITY, f64::min);
    let power_max = peak_power;
    let power_range = power_max / power_min;

    let db_min = db_frame.iter().cloned().fold(f64::INFINITY, f64::min);
    let db_max = peak_db;
    let db_range = db_max - db_min;

    println!("Dynamic range comparison:");
    println!("  Power scale: {:.2e} (ratio)", power_range);
    println!("  dB scale: {:.2} dB (difference)", db_range);
    println!("  → dB scale compresses large dynamic ranges for better visualization");
    println!();

    println!("Scale selection guide:");
    println!("  • Power: Best for energy-based analysis, preserves additivity");
    println!("  • Magnitude: Closer to linear amplitude perception");
    println!("  • Decibels: Best for visualization, matches human hearing, wide dynamic range");

    Ok(())
}
