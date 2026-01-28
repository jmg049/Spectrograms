//! Comprehensive image operations tests covering convolution, filtering, and edge detection

use std::num::NonZeroUsize;

use ndarray::Array2;
use spectrograms::{image_ops::*, nzu};

const EPSILON: f64 = 1e-10;
const LOOSE_EPSILON: f64 = 1e-6;

/// Helper to assert arrays are close
fn assert_arrays_close(a: &Array2<f64>, b: &Array2<f64>, epsilon: f64, msg: &str) {
    assert_eq!(a.shape(), b.shape(), "{}: Shape mismatch", msg);
    for ((i, j), &val_a) in a.indexed_iter() {
        let val_b = b[[i, j]];
        assert!(
            (val_a - val_b).abs() < epsilon,
            "{}: Mismatch at [{}, {}]: {} vs {}, diff = {}",
            msg,
            i,
            j,
            val_a,
            val_b,
            (val_a - val_b).abs()
        );
    }
}

// ============================================================================
// Gaussian Kernel Tests
// ============================================================================

#[test]
fn test_gaussian_kernel_normalized() {
    for &size in &[3, 5, 7, 9, 11, 15] {
        let size = NonZeroUsize::new(size).unwrap();
        for &sigma in &[0.5, 1.0, 2.0, 3.0] {
            let kernel = gaussian_kernel_2d(size, sigma).unwrap();

            // Sum should be very close to 1.0
            let sum: f64 = kernel.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Kernel size {} sigma {} not normalized: sum = {}",
                size,
                sigma,
                sum
            );
        }
    }
}

#[test]
fn test_gaussian_kernel_symmetric() {
    let size = nzu!(7);
    let sigma = 2.0;
    let kernel = gaussian_kernel_2d(size, sigma).unwrap();

    let center = size.get() / 2;

    // Check symmetry around center
    for i in 0..size.get() {
        for j in 0..size.get() {
            let mirror_i = size.get() - 1 - i;
            let mirror_j = size.get() - 1 - j;

            assert!(
                (kernel[[i, j]] - kernel[[mirror_i, mirror_j]]).abs() < EPSILON,
                "Kernel not symmetric at [{}, {}]",
                i,
                j
            );
        }
    }

    // Check center is maximum
    let center_val = kernel[[center, center]];
    for val in kernel.iter() {
        assert!(
            *val <= center_val + EPSILON,
            "Center should be maximum value"
        );
    }
}

#[test]
fn test_gaussian_kernel_sigma_effect() {
    let size = nzu!(9);

    let narrow = gaussian_kernel_2d(size, 0.5).unwrap();
    let wide = gaussian_kernel_2d(size, 3.0).unwrap();

    let center = size.get() / 2;

    // Narrow kernel should have higher peak
    assert!(
        narrow[[center, center]] > wide[[center, center]],
        "Narrow kernel should have higher peak"
    );

    // Wide kernel should have flatter distribution
    let corner = 0;
    assert!(
        wide[[corner, corner]] > narrow[[corner, corner]],
        "Wide kernel should spread more to corners"
    );
}

#[test]
fn test_gaussian_kernel_odd_sizes() {
    // All odd sizes should work
    for &size in &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21] {
        let size = NonZeroUsize::new(size).unwrap();
        let kernel = gaussian_kernel_2d(size, 1.0);
        assert!(kernel.is_ok(), "Failed for odd size {}", size);
        let k = kernel.unwrap();
        assert_eq!(k.shape(), &[size.get(), size.get()]);
    }
}

#[test]
fn test_gaussian_kernel_even_sizes() {
    // Even sizes should return an error (kernels must have odd size for clear center)
    for &size in &[2, 4, 6, 8, 10, 12, 14] {
        let size = NonZeroUsize::new(size).unwrap();
        let kernel = gaussian_kernel_2d(size, 1.0);
        assert!(kernel.is_err(), "Should fail for even size {}", size);
    }
}

#[test]
fn test_gaussian_kernel_extreme_sigma() {
    let size = nzu!(11);

    // Very small sigma (sharp peak)
    let sharp = gaussian_kernel_2d(size, 0.1).unwrap();
    let center = size.get() / 2;
    assert!(sharp[[center, center]] > 0.9); // Most weight at center

    // Very large sigma (flat)
    let flat = gaussian_kernel_2d(size, 10.0).unwrap();
    // Should be more uniform
    let min_val = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(max_val - min_val < 0.01); // Nearly flat
}

// ============================================================================
// Convolution Tests
// ============================================================================

#[test]
fn test_convolve_identity_kernel() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
    });

    // Identity kernel
    let mut kernel = Array2::<f64>::zeros((3, 3));
    kernel[[1, 1]] = 1.0;

    let result = convolve_fft(&image.view(), &kernel.view()).unwrap();

    // Should be approximately equal (boundary effects on edges)
    for i in 2..(image.nrows() - 2) {
        for j in 2..(image.ncols() - 2) {
            assert!(
                (result[[i, j]] - image[[i, j]]).abs() < LOOSE_EPSILON,
                "Identity convolution failed at [{}, {}]",
                i,
                j
            );
        }
    }
}

#[test]
fn test_convolve_box_filter() {
    // Box filter = averaging
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        if i >= 28 && i < 36 && j >= 28 && j < 36 {
            1.0
        } else {
            0.0
        }
    });

    // 3x3 box filter (normalized)
    let kernel = Array2::<f64>::from_elem((3, 3), 1.0 / 9.0);

    let result = convolve_fft(&image.view(), &kernel.view()).unwrap();

    // Result should be smoothed version
    // Center of square should still be high
    let center = result[[32, 32]];
    assert!(center > 0.5, "Center should still be high after smoothing");

    // Edges should be blurred (lower than center)
    let edge = result[[28, 32]];
    assert!(edge < center, "Edges should be smoothed");
}

#[test]
fn test_convolve_separability() {
    // Gaussian convolution should be separable
    let image = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| i as f64 + j as f64);

    let kernel_2d = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();
    let result_2d = convolve_fft(&image.view(), &kernel_2d.view()).unwrap();

    // Result should be reasonable
    assert!(result_2d.nrows() == 32 && result_2d.ncols() == 32);
}

#[test]
fn test_convolve_preserves_energy() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
    });

    // Normalized kernel (sum = 1)
    let kernel = gaussian_kernel_2d(nzu!(5), 1.0).unwrap();

    let result = convolve_fft(&image.view(), &kernel.view()).unwrap();

    let original_mean: f64 = image.iter().sum::<f64>() / (64.0 * 64.0);
    let result_mean: f64 = result.iter().sum::<f64>() / (64.0 * 64.0);

    // Mean should be approximately preserved
    assert!(
        (original_mean - result_mean).abs() < 0.1,
        "Mean not preserved: {} vs {}",
        original_mean,
        result_mean
    );
}

#[test]
fn test_convolve_different_kernel_sizes() {
    let image = Array2::<f64>::ones((64, 64));

    for &kernel_size in &[3, 5, 7, 9, 11, 15] {
        let kernel_size = NonZeroUsize::new(kernel_size).unwrap();
        let kernel = gaussian_kernel_2d(kernel_size, 1.0).unwrap();
        let result = convolve_fft(&image.view(), &kernel.view()).unwrap();
        assert_eq!(result.shape(), &[64, 64]);
    }
}

#[test]
fn test_convolve_large_kernel() {
    // Large kernels should still work (this is where FFT method shines)
    let image = Array2::<f64>::from_shape_fn((128, 128), |(i, j)| {
        ((i as f64 - 64.0).powi(2) + (j as f64 - 64.0).powi(2)).sqrt()
    });

    let kernel = gaussian_kernel_2d(nzu!(31), 5.0).unwrap();
    let result = convolve_fft(&image.view(), &kernel.view()).unwrap();

    assert_eq!(result.shape(), &[128, 128]);
}

// ============================================================================
// Low-Pass Filter Tests
// ============================================================================

#[test]
fn test_lowpass_constant_image() {
    // Low-pass of constant should be constant
    let constant = Array2::<f64>::from_elem((64, 64), 50.0);
    let filtered = lowpass_filter(&constant.view(), 0.3).unwrap();

    for &val in filtered.iter() {
        assert!(
            (val - 50.0).abs() < LOOSE_EPSILON,
            "Constant should pass through low-pass"
        );
    }
}

#[test]
fn test_lowpass_removes_high_freq() {
    // High-frequency pattern
    let high_freq = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        ((i as f64 * 0.8).sin() + (j as f64 * 0.8).cos()) * 10.0
    });

    let filtered = lowpass_filter(&high_freq.view(), 0.2).unwrap();

    // Variance should be reduced
    let original_var: f64 = high_freq.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);
    let filtered_var: f64 = filtered.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);

    assert!(
        filtered_var < original_var,
        "Low-pass should reduce variance of high-freq signal"
    );
}

#[test]
fn test_lowpass_cutoff_effect() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.3).sin() + (j as f64 * 0.3).cos()
    });

    // Lower cutoff = more smoothing
    let smooth_heavy = lowpass_filter(&image.view(), 0.1).unwrap();
    let smooth_light = lowpass_filter(&image.view(), 0.5).unwrap();

    let heavy_var: f64 = smooth_heavy.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);
    let light_var: f64 = smooth_light.iter().map(|&x| x * x).sum::<f64>() / (64.0 * 64.0);

    assert!(heavy_var < light_var, "Lower cutoff should smooth more");
}

#[test]
fn test_lowpass_invalid_cutoff() {
    let image = Array2::<f64>::ones((32, 32));

    // Cutoff must be between 0 and 1
    assert!(lowpass_filter(&image.view(), -0.1).is_err());
    assert!(lowpass_filter(&image.view(), 1.5).is_err());
}

// ============================================================================
// High-Pass Filter Tests
// ============================================================================

#[test]
fn test_highpass_constant_image() {
    // High-pass of constant should be ~zero
    let constant = Array2::<f64>::from_elem((64, 64), 100.0);
    let filtered = highpass_filter(&constant.view(), 0.1).unwrap();

    let max_abs = filtered.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    assert!(max_abs < 1.0, "High-pass of constant should be ~zero");
}

#[test]
fn test_highpass_preserves_high_freq() {
    // High-frequency checkerboard
    let checkerboard =
        Array2::<f64>::from_shape_fn((64, 64), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { -1.0 });

    let filtered = highpass_filter(&checkerboard.view(), 0.1).unwrap();

    // Should preserve most of the pattern
    let original_energy: f64 = checkerboard.iter().map(|&x| x * x).sum();
    let filtered_energy: f64 = filtered.iter().map(|&x| x * x).sum();

    assert!(
        filtered_energy / original_energy > 0.5,
        "High-pass should preserve high-frequency content"
    );
}

#[test]
fn test_highpass_cutoff_effect() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, _)| {
        (i as f64 * 0.2).sin() + 10.0 // Low freq + DC
    });

    let filtered_low_cutoff = highpass_filter(&image.view(), 0.1).unwrap();
    let filtered_high_cutoff = highpass_filter(&image.view(), 0.3).unwrap();

    // Higher cutoff should remove more
    let energy_low: f64 = filtered_low_cutoff.iter().map(|&x| x * x).sum();
    let energy_high: f64 = filtered_high_cutoff.iter().map(|&x| x * x).sum();

    assert!(
        energy_high < energy_low,
        "Higher cutoff should remove more energy"
    );
}

#[test]
fn test_highpass_lowpass_complement() {
    // Highpass + lowpass should approximately equal original (ignoring transition band)
    let image = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos() + 5.0
    });

    let cutoff = 0.3;
    let low = lowpass_filter(&image.view(), cutoff).unwrap();
    let high = highpass_filter(&image.view(), cutoff).unwrap();

    let combined = &low + &high;

    // Should be approximately equal to original (within reason)
    for i in 0..32 {
        for j in 0..32 {
            let diff = (combined[[i, j]] - image[[i, j]]).abs();
            // Loose tolerance due to filter transition band
            assert!(
                diff < 1.0,
                "Low+High should approximate original at [{}, {}]",
                i,
                j
            );
        }
    }
}

// ============================================================================
// Band-Pass Filter Tests
// ============================================================================

#[test]
fn test_bandpass_valid_range() {
    let image = Array2::<f64>::ones((64, 64));

    let result = bandpass_filter(&image.view(), 0.2, 0.5);
    assert!(result.is_ok(), "Valid bandpass range should work");
}

#[test]
fn test_bandpass_invalid_range() {
    let image = Array2::<f64>::ones((32, 32));

    // Low >= high is invalid
    assert!(bandpass_filter(&image.view(), 0.5, 0.3).is_err());
    assert!(bandpass_filter(&image.view(), 0.5, 0.5).is_err());

    // Out of bounds
    assert!(bandpass_filter(&image.view(), -0.1, 0.5).is_err());
    assert!(bandpass_filter(&image.view(), 0.2, 1.5).is_err());
}

#[test]
fn test_bandpass_filters_extremes() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.1).sin() + // Low freq
        (j as f64 * 2.0).sin() + // High freq
        10.0 // DC
    });

    // Band that excludes both DC and very high frequencies
    let filtered = bandpass_filter(&image.view(), 0.15, 0.4).unwrap();

    // Should have removed DC (mean close to 0)
    let mean: f64 = filtered.iter().sum::<f64>() / (64.0 * 64.0);
    assert!(mean.abs() < 1.0, "Bandpass should remove DC component");
}

#[test]
fn test_bandpass_narrow_band() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.3).sin() + (j as f64 * 0.3).cos()
    });

    // Very narrow band
    let filtered = bandpass_filter(&image.view(), 0.25, 0.35).unwrap();

    // Should produce a result
    assert_eq!(filtered.shape(), &[64, 64]);
}

// ============================================================================
// Edge Detection Tests
// ============================================================================

#[test]
fn test_edge_detection_constant() {
    // No edges in constant image
    let constant = Array2::<f64>::from_elem((64, 64), 50.0);
    let edges = detect_edges_fft(&constant.view()).unwrap();

    let max_edge = edges.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    assert!(max_edge < 1.0, "Constant image should have no edges");
}

#[test]
fn test_edge_detection_step() {
    // Step function has strong edge
    let mut step = Array2::<f64>::zeros((64, 64));
    for i in 0..64 {
        for j in 32..64 {
            step[[i, j]] = 1.0;
        }
    }

    let edges = detect_edges_fft(&step.view()).unwrap();

    // Should detect strong vertical edge around column 32
    let mut found_edge = false;
    for i in 0..64 {
        for j in 28..36 {
            if edges[[i, j]].abs() > 0.01 {
                found_edge = true;
            }
        }
    }

    assert!(found_edge, "Should detect step edge");
}

#[test]
fn test_edge_detection_rectangle() {
    // Rectangle has 4 edges
    let mut rect = Array2::<f64>::zeros((64, 64));
    for i in 20..44 {
        for j in 20..44 {
            rect[[i, j]] = 1.0;
        }
    }

    let edges = detect_edges_fft(&rect.view()).unwrap();

    // Should have non-zero values around perimeter
    let edge_strength: f64 = edges.iter().map(|&x| x.abs()).sum();
    assert!(edge_strength > 0.1, "Should detect rectangle edges");
}

// ============================================================================
// Sharpening Tests
// ============================================================================

#[test]
fn test_sharpen_zero_amount() {
    let image = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| i as f64 + j as f64);

    // Zero sharpening = identity
    let sharpened = sharpen_fft(&image.view(), 0.0).unwrap();

    assert_arrays_close(
        &image,
        &sharpened,
        LOOSE_EPSILON,
        "Zero sharpening should be identity",
    );
}

#[test]
fn test_sharpen_increases_contrast() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        ((i as f64 - 32.0).powi(2) + (j as f64 - 32.0).powi(2)).sqrt()
    });

    let sharpened = sharpen_fft(&image.view(), 1.0).unwrap();

    // Sharpening should increase variance
    let original_std = {
        let mean = image.iter().sum::<f64>() / (64.0 * 64.0);
        let variance: f64 = image.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (64.0 * 64.0);
        variance.sqrt()
    };

    let sharpened_std = {
        let mean = sharpened.iter().sum::<f64>() / (64.0 * 64.0);
        let variance: f64 =
            sharpened.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (64.0 * 64.0);
        variance.sqrt()
    };

    assert!(
        sharpened_std > original_std,
        "Sharpening should increase standard deviation"
    );
}

#[test]
fn test_sharpen_different_amounts() {
    let image = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| {
        (i as f64 * 0.2).sin() + (j as f64 * 0.2).cos()
    });

    let sharp1 = sharpen_fft(&image.view(), 0.5).unwrap();
    let sharp2 = sharpen_fft(&image.view(), 2.0).unwrap();

    // Higher amount = more sharpening = higher variance
    let var1: f64 = sharp1.iter().map(|&x| x * x).sum::<f64>() / (32.0 * 32.0);
    let var2: f64 = sharp2.iter().map(|&x| x * x).sum::<f64>() / (32.0 * 32.0);

    assert!(
        var2 > var1,
        "Higher sharpening amount should increase variance more"
    );
}

#[test]
fn test_sharpen_preserves_mean() {
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, _)| (i as f64 * 0.1).sin() + 10.0);

    let sharpened = sharpen_fft(&image.view(), 1.5).unwrap();

    let original_mean: f64 = image.iter().sum::<f64>() / (64.0 * 64.0);
    let sharpened_mean: f64 = sharpened.iter().sum::<f64>() / (64.0 * 64.0);

    // Mean should be approximately preserved
    assert!(
        (original_mean - sharpened_mean).abs() < 0.5,
        "Sharpening should preserve mean: {} vs {}",
        original_mean,
        sharpened_mean
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_complete_pipeline() {
    // Simulating a complete image processing pipeline
    let original = Array2::<f64>::from_shape_fn((128, 128), |(i, j)| {
        let r = ((i as f64 - 64.0).powi(2) + (j as f64 - 64.0).powi(2)).sqrt();
        (-r / 20.0).exp() * 100.0
    });

    // Step 1: Denoise with low-pass
    let denoised = lowpass_filter(&original.view(), 0.4).unwrap();
    assert_eq!(denoised.shape(), &[128, 128]);

    // Step 2: Detect edges
    let edges = detect_edges_fft(&denoised.view()).unwrap();
    assert_eq!(edges.shape(), &[128, 128]);

    // Step 3: Sharpen original
    let sharpened = sharpen_fft(&original.view(), 1.0).unwrap();
    assert_eq!(sharpened.shape(), &[128, 128]);

    // Step 4: Apply Gaussian blur
    let kernel = gaussian_kernel_2d(nzu!(9), 2.0).unwrap();
    let blurred = convolve_fft(&original.view(), &kernel.view()).unwrap();
    assert_eq!(blurred.shape(), &[128, 128]);

    // All operations completed successfully
}

#[test]
fn test_cascade_filters() {
    // Cascading multiple filters
    let image = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.2).sin() + (j as f64 * 0.3).cos() + 5.0
    });

    // Remove DC
    let step1 = highpass_filter(&image.view(), 0.05).unwrap();

    // Remove very high frequencies
    let step2 = lowpass_filter(&step1.view(), 0.7).unwrap();

    // Effectively a bandpass
    assert_eq!(step2.shape(), &[64, 64]);

    // Compare with direct bandpass
    let direct = bandpass_filter(&image.view(), 0.05, 0.7).unwrap();

    // Should be similar (not exactly equal due to filter design)
    let mut total_diff = 0.0;
    for i in 0..64 {
        for j in 0..64 {
            total_diff += (step2[[i, j]] - direct[[i, j]]).abs();
        }
    }
    let avg_diff = total_diff / (64.0 * 64.0);

    assert!(
        avg_diff < 1.0,
        "Cascaded filters should approximate bandpass"
    );
}

#[test]
fn test_error_handling_invalid_inputs() {
    // Test that invalid inputs are properly rejected
    let image = Array2::<f64>::ones((32, 32));

    // Invalid cutoff values
    assert!(lowpass_filter(&image.view(), -0.1).is_err());
    assert!(lowpass_filter(&image.view(), 1.5).is_err());
    assert!(highpass_filter(&image.view(), -0.1).is_err());
    assert!(highpass_filter(&image.view(), 1.5).is_err());

    // Invalid bandpass ranges
    assert!(bandpass_filter(&image.view(), 0.5, 0.3).is_err());
    assert!(bandpass_filter(&image.view(), -0.1, 0.5).is_err());
    assert!(bandpass_filter(&image.view(), 0.2, 1.5).is_err());

    // Invalid Gaussian kernel parameters
    assert!(gaussian_kernel_2d(nzu!(5), 0.0).is_err()); // Sigma 0
    assert!(gaussian_kernel_2d(nzu!(5), -1.0).is_err()); // Negative sigma
}
