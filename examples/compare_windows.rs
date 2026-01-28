use non_empty_slice::non_empty_vec;
/// Window function comparison example
///
/// This example demonstrates:
/// - Using different window functions
/// - Comparing their effects on spectrograms
/// - Understanding window trade-offs
use spectrograms::{LinearPowerSpectrogram, SpectrogramParams, StftParams, WindowType, nzu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a test signal: impulse (delta function)
    let sample_rate = 16000.0;
    let mut samples = non_empty_vec![0.0; nzu!(1600)];
    samples[800] = 1.0; // Impulse in the middle

    println!("Generated impulse signal");
    println!("An impulse has equal energy at all frequencies,");
    println!("so the spectrogram shows the window's frequency response.\n");

    // Define window functions to compare
    let windows = vec![
        ("Rectangular", WindowType::Rectangular),
        ("Hanning", WindowType::Hanning),
        ("Hamming", WindowType::Hamming),
        ("Blackman", WindowType::Blackman),
        ("Kaiser (β=5)", WindowType::Kaiser { beta: 5.0 }),
        ("Gaussian (σ=0.5)", WindowType::Gaussian { std: 0.5 }),
    ];

    println!("Comparing {} window functions:\n", windows.len());

    for (name, window) in windows {
        // Set up parameters with this window
        let stft = StftParams::new(nzu!(512), nzu!(256), window, true)?;
        let params = SpectrogramParams::new(stft, sample_rate)?;

        // Compute spectrogram
        let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;

        // Analyze the spectrum at the impulse frame
        let impulse_frame = spec.n_frames().get() / 2; // Middle frame
        let frame_data = spec.data().column(impulse_frame);

        // Find peak and measure spectral spread
        let peak_power = frame_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let half_power = peak_power / 2.0;

        // Count bins above half power (main lobe width indicator)
        let bins_above_half = frame_data.iter().filter(|&&p| p > half_power).count();

        // Calculate total power spread
        let total_power: f64 = frame_data.iter().sum();
        let mean_bin = frame_data
            .iter()
            .enumerate()
            .map(|(i, &p)| i as f64 * p)
            .sum::<f64>()
            / total_power;

        let variance = frame_data
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let diff = i as f64 - mean_bin;
                diff * diff * p
            })
            .sum::<f64>()
            / total_power;

        let std_dev = variance.sqrt();

        println!("{}:", name);
        println!("  Bins above half-power: {}", bins_above_half);
        println!("  Spectral spread (std): {:.2} bins", std_dev);
        println!("  Peak power: {:.2e}", peak_power);
        println!(
            "  Trade-off: {} main lobe, {} sidelobes\n",
            if bins_above_half > 15 {
                "wide"
            } else {
                "narrow"
            },
            if std_dev > 20.0 { "high" } else { "low" }
        );
    }

    println!("Window selection guide:");
    println!("  * Rectangular: Best frequency resolution, worst spectral leakage");
    println!("  * Hanning/Hamming: Good all-around choice, balanced trade-off");
    println!("  * Blackman: Low sidelobes, wider main lobe");
    println!("  * Kaiser: Tunable (β controls main lobe vs sidelobes)");
    println!("  * Gaussian: Smooth roll-off, optimal time-frequency localization");

    Ok(())
}
