//! Edge detection using frequency-domain filtering
//!
//! Demonstrates edge detection by applying high-pass filters in the frequency
//! domain, which emphasizes high-frequency components (edges and details).

use ndarray::Array2;
use spectrograms::SpectrogramResult;
use spectrograms::image_ops::{detect_edges_fft, highpass_filter, lowpass_filter};

fn main() -> SpectrogramResult<()> {
    println!("=== Edge Detection via Frequency Domain Filtering ===\n");

    // Create a test image with geometric shapes
    let size = 128;
    let image = create_test_image(size);

    println!("Created {}x{} test image with geometric shapes", size, size);

    // Compute image statistics
    let mean: f64 = image.iter().sum::<f64>() / (size * size) as f64;
    let variance: f64 =
        image.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (size * size) as f64;

    println!("Original image:");
    println!("  Mean: {:.2}, Std dev: {:.2}", mean, variance.sqrt());

    // Apply high-pass filters with different cutoffs
    println!("\n=== High-Pass Filtering ===\n");

    let cutoffs = vec![0.05, 0.1, 0.2];

    for cutoff in cutoffs {
        let filtered = highpass_filter(&image.view().view(), cutoff)?;

        let filtered_mean: f64 = filtered.iter().sum::<f64>() / (size * size) as f64;
        let filtered_max = filtered.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        println!("High-pass filter (cutoff: {}):", cutoff);
        println!("  Mean: {:.2e} (should be ~0)", filtered_mean);
        println!("  Max magnitude: {:.2}", filtered_max);
        println!("  Edge emphasis: {:.1}%", (filtered_max / 100.0) * 100.0);
    }

    // Compare high-pass vs low-pass
    println!("\n=== High-Pass vs Low-Pass Comparison ===\n");

    let cutoff = 0.15;
    let highpass = highpass_filter(&image.view(), cutoff)?;
    let lowpass = lowpass_filter(&image.view(), cutoff)?;

    println!("With cutoff fraction = {}:", cutoff);

    let hp_energy: f64 = highpass.iter().map(|&x| x.powi(2)).sum();
    let lp_energy: f64 = lowpass.iter().map(|&x| x.powi(2)).sum();
    let total_energy = hp_energy + lp_energy;

    println!(
        "  High-pass energy: {:.2e} ({:.1}%)",
        hp_energy,
        (hp_energy / total_energy) * 100.0
    );
    println!(
        "  Low-pass energy: {:.2e} ({:.1}%)",
        lp_energy,
        (lp_energy / total_energy) * 100.0
    );

    // Use convenience function for edge detection
    println!("\n=== Edge Detection (Convenience Function) ===\n");

    let edges = detect_edges_fft(&image.view())?;

    let edge_mean: f64 = edges.iter().sum::<f64>() / (size * size) as f64;
    let edge_max = edges.iter().map(|&x| x.abs()).fold(0.0, f64::max);

    println!("Detected edges:");
    println!("  Mean: {:.2e}", edge_mean);
    println!("  Max edge strength: {:.2}", edge_max);

    // Find strongest edges (top 1% of pixels)
    let threshold = edge_max * 0.5;
    let strong_edges = edges
        .iter()
        .filter(|&&x: &&f64| x.abs() > threshold)
        .count();

    println!(
        "  Strong edges (> {}): {} pixels ({:.1}%)",
        threshold,
        strong_edges,
        (strong_edges as f64 / (size * size) as f64) * 100.0
    );

    println!("\n=== Frequency Domain Advantages ===");
    println!("Frequency-domain edge detection:");
    println!("  * Processes entire image in parallel");
    println!("  * No directional bias (unlike Sobel, Prewitt)");
    println!("  * Adjustable frequency response");
    println!("  * Efficient for large images with FFT");

    Ok(())
}

/// Create a test image with geometric shapes (circles and rectangles)
fn create_test_image(size: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
        let center = size as f64 / 2.0;
        let i_f = i as f64;
        let j_f = j as f64;

        // Circle in the center
        let dist_sq = (i_f - center).powi(2) + (j_f - center).powi(2);
        let circle_radius = size as f64 / 4.0;

        if dist_sq < circle_radius.powi(2) {
            return 100.0; // Inside circle
        }

        // Rectangle in top-left
        if i < size / 4 && j < size / 4 {
            return 80.0;
        }

        // Diagonal gradient
        let gradient = ((i + j) as f64 / (2.0 * size as f64)) * 50.0;

        // Add some noise
        let noise = ((i as f64 * 12.9898 + j as f64 * 78.233).sin() * 43758.5453).fract();

        gradient + noise * 5.0
    })
}
