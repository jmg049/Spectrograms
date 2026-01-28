//! Gaussian blur using FFT convolution
//!
//! Demonstrates applying Gaussian blur to an image using FFT-based convolution,
//! which is faster than spatial convolution for large kernels.

use std::num::NonZeroUsize;

use ndarray::Array2;
use spectrograms::image_ops::{convolve_fft, gaussian_kernel_2d};
use spectrograms::{SpectrogramResult, nzu};

fn main() -> SpectrogramResult<()> {
    println!("=== Gaussian Blur via FFT Convolution ===\n");

    // Create a test image with a bright square in the center
    let size = 128;
    let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
        let center_i = size as f64 / 2.0;
        let center_j = size as f64 / 2.0;

        // Create a 40x40 square in the center
        if (i as f64 - center_i).abs() < 20.0 && (j as f64 - center_j).abs() < 20.0 {
            100.0 // Bright square
        } else {
            0.0 // Dark background
        }
    });

    println!("Created {}x{} test image with bright square", size, size);

    // Compute statistics on original image
    let original_mean: f64 = image.iter().sum::<f64>() / (size * size) as f64;
    let original_min = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let original_max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\nOriginal image statistics:");
    println!(
        "  Min: {:.2}, Max: {:.2}, Mean: {:.2}",
        original_min, original_max, original_mean
    );

    // Apply different levels of blur
    let blur_configs = vec![(5, 1.0, "Light"), (9, 2.0, "Medium"), (15, 3.0, "Heavy")];

    for (kernel_size, sigma, description) in blur_configs {
        println!(
            "\n{} blur (kernel size: {}, sigma: {}):",
            description, kernel_size, sigma
        );

        // Create Gaussian kernel
        let kernel_size = NonZeroUsize::new(kernel_size).unwrap();
        let kernel = gaussian_kernel_2d(kernel_size, sigma)?;

        // Verify kernel is normalized
        let kernel_sum: f64 = kernel.iter().sum();
        println!("  Kernel sum: {:.6} (should be 1.0)", kernel_sum);

        // Apply blur via FFT convolution
        let blurred = convolve_fft(&image.view(), &kernel.view())?;

        // Compute statistics on blurred image
        let blurred_mean: f64 = blurred.iter().sum::<f64>() / (size * size) as f64;
        let blurred_min = blurred.iter().cloned().fold(f64::INFINITY, f64::min);
        let blurred_max = blurred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("  Blurred image:");
        println!(
            "    Min: {:.2}, Max: {:.2}, Mean: {:.2}",
            blurred_min, blurred_max, blurred_mean
        );
        println!(
            "    Peak reduction: {:.1}%",
            (original_max - blurred_max) / original_max * 100.0
        );
    }

    // Compare with edge statistics
    println!("\n=== Edge Analysis ===\n");

    let light_kernel = gaussian_kernel_2d(nzu!(5), 1.0)?;
    let blurred = convolve_fft(&image.view(), &light_kernel.view())?;

    // Sample values at center and edge
    let center_val = blurred[[size / 2, size / 2]];
    let edge_val = blurred[[size / 2, 0]];

    println!("Sample values:");
    println!("  Center (inside square): {:.2}", center_val);
    println!("  Edge (outside square): {:.2}", edge_val);
    println!("  Ratio: {:.1}:1", center_val / edge_val.max(1e-10));

    println!("\nExample completed successfully!");

    Ok(())
}
