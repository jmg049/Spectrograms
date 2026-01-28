//! STFT batch processing example
//!
//! Demonstrates the performance benefit of reusable STFT plans.

use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::time::Instant;

fn generate_test_signals(n: usize, len: usize) -> NonEmptyVec<NonEmptyVec<f64>> {
    let v = (0..n)
        .map(|_| NonEmptyVec::new(vec![0.5; len]).unwrap())
        .collect();
    NonEmptyVec::new(v).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signals = generate_test_signals(100, 16000);

    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft_params, 16000.0)?;

    println!(
        "Processing {} signals of length {}...\n",
        signals.len(),
        signals[0].len()
    );

    // Method 1: One-shot API (creates new plan each time)
    let start = Instant::now();
    let planner = SpectrogramPlanner::new();
    for signal in &signals {
        let _stft = planner.compute_stft(signal, &params)?;
    }
    let time_oneshot = start.elapsed();
    println!("One-shot API:     {:?}", time_oneshot);

    // Method 2: Reusable plan (efficient)
    let start = Instant::now();
    let mut plan = StftPlan::new(&params)?;
    for signal in &signals {
        let _stft = plan.compute(signal, &params)?;
    }
    let time_reuse = start.elapsed();
    println!("Reusable plan:    {:?}", time_reuse);

    let speedup = time_oneshot.as_secs_f64() / time_reuse.as_secs_f64();
    println!("\nSpeedup: {:.2}x faster", speedup);

    Ok(())
}
