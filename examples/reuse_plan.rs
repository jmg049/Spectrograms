use non_empty_slice::{NonEmptyVec, non_empty_vec};
/// Plan reuse example
///
/// This example demonstrates:
/// - Creating a reusable spectrogram plan
/// - Processing multiple signals efficiently
/// - Performance benefits of plan reuse
use spectrograms::{
    LinearPowerSpectrogram, SpectrogramParams, SpectrogramPlanner, StftParams, WindowType, nzu,
};

use std::f64::consts::PI;
use std::time::Instant;

fn generate_sine(frequency: f64, sample_rate: f64, duration: f64) -> NonEmptyVec<f64> {
    let v = (0..(duration * sample_rate) as usize)
        .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin())
        .collect();
    NonEmptyVec::new(v).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        eprintln!("Warning: Running in debug mode may affect performance measurements.");
        eprintln!("Build with release mode, otherwise do not be surprised by the results.\n");
    }

    let sample_rate = 16000.0;
    let duration = 1.0;

    // Generate multiple test signals at different frequencies
    let frequencies = non_empty_vec![220.0, 440.0, 880.0, 1760.0]; // A3, A4, A5, A6
    let signals: Vec<NonEmptyVec<f64>> = frequencies
        .iter()
        .map(|&freq| generate_sine(freq, sample_rate, duration))
        .collect();

    println!("Generated {} signals", signals.len());

    // Set up parameters
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, sample_rate)?;

    // Method 1: Using convenience API (creates plan each time)
    println!("\n--- Method 1: Convenience API (no plan reuse) ---");
    let start = Instant::now();
    for (i, signal) in signals.iter().enumerate() {
        let _spec = LinearPowerSpectrogram::compute(signal, &params, None)?;
        println!("  Processed signal {} ({:.1} Hz)", i + 1, frequencies[i]);
    }
    let elapsed_no_reuse = start.elapsed();
    println!("Time: {:?}", elapsed_no_reuse);

    // Method 2: Using planner (reuses plan)
    println!("\n--- Method 2: Planner API (with plan reuse) ---");
    let start = Instant::now();

    // Create the plan once
    let planner = SpectrogramPlanner::new();
    let mut plan = planner.linear_plan::<spectrograms::Power>(&params, None)?;

    // Reuse it for all signals
    for (i, signal) in signals.iter().enumerate() {
        let _spec = plan.compute(signal)?;
        println!("  Processed signal {} ({:.1} Hz)", i + 1, frequencies[i]);
    }
    let elapsed_with_reuse = start.elapsed();
    println!("Time: {:?}", elapsed_with_reuse);

    // Compare performance
    println!("\n--- Performance comparison ---");
    println!("Without plan reuse: {:?}", elapsed_no_reuse);
    println!("With plan reuse:    {:?}", elapsed_with_reuse);

    let speedup = elapsed_no_reuse.as_secs_f64() / elapsed_with_reuse.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    Ok(())
}
