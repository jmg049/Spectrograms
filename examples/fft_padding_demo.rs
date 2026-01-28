use non_empty_slice::non_empty_vec;
use spectrograms::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FFT Automatic Zero-Padding Demo ===\n");

    // Test 1: Short signal with padding
    println!("Test 1: Short signal (3 samples) padded to 8");
    let short = non_empty_vec![1.0, 2.0, 3.0];
    let padded = fft(&short, nzu!(8))?;
    println!("  Input length: {}", short.len());
    println!("  FFT size: 8");
    println!("  Output bins: {} (expected: 5)\n", padded.len());

    // Test 2: Exact length (no padding needed)
    println!("Test 2: Exact length (8 samples)");
    let exact = non_empty_vec![1.0; nzu!(8)];
    let result = fft(&exact, nzu!(8))?;
    println!("  Input length: {}", exact.len());
    println!("  FFT size: 8");
    println!("  Output bins: {} (expected: 5)\n", result.len());

    // Test 3: Power spectrum with windowing and padding
    println!("Test 3: Power spectrum with Hanning window and padding");
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let power = power_spectrum(&signal, nzu!(8), Some(WindowType::Hanning))?;
    println!("  Input length: {}", signal.len());
    println!("  FFT size: 8");
    println!("  Output bins: {} (expected: 5)\n", power.len());

    // Test 4: Error case - input too long
    println!("Test 4: Error case - input longer than n_fft");
    let long = non_empty_vec![1.0; nzu!(10)];
    match fft(&long, nzu!(8)) {
        Err(e) => println!("  Expected error: {}\n", e),
        Ok(_) => println!("  ERROR: Should have failed!\n"),
    }

    // Test 5: Frequency resolution preservation
    println!("Test 5: Verify frequency resolution is preserved");
    let fs = 1000.0;
    let n_fft = 256;
    let signal = non_empty_vec![1.0; nzu!(128)]; // Half length
    let spectrum = fft(&signal, nzu!(256))?;
    let df = fs / n_fft as f64;
    println!("  Sample rate: {} Hz", fs);
    println!("  FFT size: {}", n_fft);
    println!("  Input length: {} (padded to {})", signal.len(), n_fft);
    println!("  Frequency resolution: {:.5} Hz", df);
    println!(
        "  Output bins: {} (expected: {})\n",
        spectrum.len(),
        n_fft / 2 + 1
    );

    // Test 6: Planner with variable-length inputs
    println!("Test 6: FftPlanner with variable-length inputs");
    let mut planner = FftPlanner::new();
    let signals = non_empty_vec![
        non_empty_vec![1.0; nzu!(100)],
        non_empty_vec![1.0; nzu!(128)],
        non_empty_vec![1.0; nzu!(50)]
    ];
    for (i, signal) in signals.iter().enumerate() {
        let result = planner.fft(signal, nzu!(128))?;
        println!(
            "  Signal {}: {} samples -> {} FFT bins",
            i + 1,
            signal.len(),
            result.len()
        );
    }

    Ok(())
}
