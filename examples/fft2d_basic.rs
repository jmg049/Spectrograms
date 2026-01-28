//! Basic 2D FFT example
//!
//! Demonstrates computing 2D FFT on a simple synthetic image and analyzing
//! the frequency content.

use ndarray::Array2;
use spectrograms::SpectrogramResult;
use spectrograms::fft2d::{fft2d, ifft2d, power_spectrum_2d};

fn main() -> SpectrogramResult<()> {
    println!("=== Basic 2D FFT Example ===\n");

    // Create a simple 64x64 image with a vertical stripe pattern
    let size = 64;
    let image = Array2::<f64>::from_shape_fn((size, size), |(_, j)| {
        // Create vertical stripes (high frequency in horizontal direction)
        if j % 8 < 4 { 1.0 } else { 0.0 }
    });

    println!("Created {}x{} image with vertical stripes", size, size);

    // Compute 2D FFT
    let spectrum = fft2d(&image.view())?;
    println!("FFT spectrum shape: {:?}", spectrum.shape());
    println!(
        "  (Note: {} = {}/2 + 1 due to Hermitian symmetry)",
        spectrum.shape()[1],
        size
    );

    // Analyze frequency content
    let power = power_spectrum_2d(&image.view())?;

    // Find peak frequencies (excluding DC component)
    let mut max_power = 0.0;
    let mut max_pos = (0, 0);

    for i in 0..power.nrows() {
        for j in 1..power.ncols() {
            // Skip DC (j=0)
            if power[[i, j]] > max_power {
                max_power = power[[i, j]];
                max_pos = (i, j);
            }
        }
    }

    println!("\nFrequency analysis:");
    println!("  DC component power: {:.2e}", power[[0, 0]]);
    println!("  Peak frequency at: {:?}", max_pos);
    println!("  Peak power: {:.2e}", max_power);

    // Verify roundtrip accuracy
    let reconstructed = ifft2d(&spectrum, size)?;

    let mut max_error = 0.0;
    for i in 0..size {
        for j in 0..size {
            let error = (reconstructed[[i, j]] - image[[i, j]]).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    println!("\nRoundtrip accuracy:");
    println!("  Maximum reconstruction error: {:.2e}", max_error);

    if max_error < 1e-10 {
        println!("  ✓ Excellent accuracy!");
    }

    // Demonstrate batch processing with planner
    println!("\n=== Batch Processing with Planner ===\n");

    use spectrograms::fft2d::Fft2dPlanner;

    let mut planner = Fft2dPlanner::new();

    let images = vec![
        Array2::<f64>::zeros((64, 64)),
        Array2::<f64>::ones((64, 64)),
        Array2::<f64>::from_shape_fn((64, 64), |(i, j)| (i + j) as f64),
    ];

    println!("Processing {} images with cached plans...", images.len());

    for (idx, img) in images.iter().enumerate() {
        let spectrum = planner.fft2d(&img.view())?;
        let power = spectrum.mapv(|c: num_complex::Complex<f64>| c.norm_sqr());
        let total_power: f64 = power.iter().sum();
        println!("  Image {}: total power = {:.2e}", idx, total_power);
    }

    Ok(())
}
