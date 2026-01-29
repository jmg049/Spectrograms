//! Comprehensive 2D FFT tests covering edge cases, numerical accuracy, and backend behavior

use ndarray::Array2;
use num_complex::Complex;
use spectrograms::fft2d::*;

const EPSILON: f64 = 1e-10;

/// Helper to check if two complex arrays are approximately equal
fn assert_complex_arrays_close(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>, epsilon: f64) {
    assert_eq!(a.shape(), b.shape(), "Shapes must match");
    for ((i, j), &val_a) in a.indexed_iter() {
        let val_b = b[[i, j]];
        let diff = (val_a - val_b).norm();
        assert!(
            diff < epsilon,
            "Mismatch at [{}, {}]: {:?} vs {:?}, diff = {}",
            i,
            j,
            val_a,
            val_b,
            diff
        );
    }
}

/// Helper to check if two real arrays are approximately equal
fn assert_real_arrays_close(a: &Array2<f64>, b: &Array2<f64>, epsilon: f64) {
    assert_eq!(a.shape(), b.shape(), "Shapes must match");
    for ((i, j), &val_a) in a.indexed_iter() {
        let val_b = b[[i, j]];
        let diff = (val_a - val_b).abs();
        assert!(
            diff < epsilon,
            "Mismatch at [{}, {}]: {} vs {}, diff = {}",
            i,
            j,
            val_a,
            val_b,
            diff
        );
    }
}

// ============================================================================
// Edge Cases: Various Dimensions
// ============================================================================

#[test]
fn test_power_of_2_sizes() {
    // Test common power-of-2 sizes (most efficient)
    for &size in &[8, 16, 32, 64, 128, 256] {
        let data = Array2::<f64>::ones((size, size));
        let result = fft2d(&data.view());
        assert!(result.is_ok(), "Failed for size {}x{}", size, size);

        let spectrum = result.unwrap();
        assert_eq!(spectrum.nrows(), size);
        assert_eq!(spectrum.ncols(), size / 2 + 1);

        // Roundtrip
        let reconstructed = ifft2d(&spectrum, size).unwrap();
        assert_real_arrays_close(&data, &reconstructed, EPSILON);
    }
}

#[test]
fn test_non_power_of_2_sizes() {
    // Test non-power-of-2 sizes (should still work)
    for &size in &[10, 15, 20, 31, 50, 63, 100] {
        let data = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            (i as f64).sin() + (j as f64).cos()
        });

        let spectrum = fft2d(&data.view()).unwrap();
        assert_eq!(spectrum.nrows(), size);
        assert_eq!(spectrum.ncols(), size / 2 + 1);

        let reconstructed = ifft2d(&spectrum, size).unwrap();
        assert_real_arrays_close(&data, &reconstructed, EPSILON);
    }
}

#[test]
fn test_odd_dimensions() {
    // Test various odd dimensions
    for &nrows in &[17, 31, 63] {
        for &ncols in &[19, 33, 65] {
            let data = Array2::<f64>::zeros((nrows, ncols));
            let spectrum = fft2d(&data.view()).unwrap();
            assert_eq!(spectrum.shape(), &[nrows, ncols / 2 + 1]);

            let reconstructed = ifft2d(&spectrum, ncols).unwrap();
            assert_real_arrays_close(&data, &reconstructed, EPSILON);
        }
    }
}

#[test]
fn test_even_dimensions() {
    // Test various even dimensions
    for &nrows in &[16, 32, 64] {
        for &ncols in &[18, 34, 66] {
            let data = Array2::<f64>::ones((nrows, ncols));
            let spectrum = fft2d(&data.view()).unwrap();
            assert_eq!(spectrum.shape(), &[nrows, ncols / 2 + 1]);

            let reconstructed = ifft2d(&spectrum, ncols).unwrap();
            assert_real_arrays_close(&data, &reconstructed, EPSILON);
        }
    }
}

#[test]
fn test_rectangular_images() {
    // Test non-square images
    let cases = vec![(32, 64), (64, 32), (16, 128), (128, 16)];

    for (nrows, ncols) in cases {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        let spectrum = fft2d(&data.view()).unwrap();
        assert_eq!(spectrum.shape(), &[nrows, ncols / 2 + 1]);

        let reconstructed = ifft2d(&spectrum, ncols).unwrap();
        assert_real_arrays_close(&data, &reconstructed, EPSILON);
    }
}

#[test]
fn test_very_small_images() {
    // Test edge case: very small images
    for &size in &[1, 2, 3, 4, 5] {
        let data = Array2::<f64>::ones((size, size));
        let spectrum = fft2d(&data.view()).unwrap();
        assert_eq!(spectrum.shape(), &[size, size / 2 + 1]);

        let reconstructed = ifft2d(&spectrum, size).unwrap();
        assert_real_arrays_close(&data, &reconstructed, EPSILON);
    }
}

// ============================================================================
// Special Input Patterns
// ============================================================================

#[test]
fn test_all_zeros() {
    let data = Array2::<f64>::zeros((32, 32));
    let spectrum = fft2d(&data.view()).unwrap();

    // All spectrum components should be zero
    for &val in spectrum.iter() {
        assert!(val.norm() < EPSILON, "Spectrum of zeros should be zero");
    }
}

#[test]
fn test_all_ones() {
    let size = 32;
    let data = Array2::<f64>::ones((size, size));
    let spectrum = fft2d(&data.view()).unwrap();

    // Only DC component should be non-zero
    let dc_magnitude = spectrum[[0, 0]].norm();
    let expected_dc = (size * size) as f64;
    assert!((dc_magnitude - expected_dc).abs() < EPSILON);

    // All other components should be ~zero
    for i in 0..size {
        for j in 0..(size / 2 + 1) {
            if i == 0 && j == 0 {
                continue;
            }
            assert!(
                spectrum[[i, j]].norm() < EPSILON,
                "Non-DC component at [{}, {}] should be ~zero",
                i,
                j
            );
        }
    }
}

#[test]
fn test_delta_function() {
    // Delta at origin
    let mut data = Array2::<f64>::zeros((32, 32));
    data[[0, 0]] = 1.0;

    let spectrum = fft2d(&data.view()).unwrap();

    // Spectrum of delta should be all ones (flat spectrum)
    for &val in spectrum.iter() {
        assert!(
            (val.re - 1.0).abs() < EPSILON && val.im.abs() < EPSILON,
            "Spectrum of delta should be 1+0i, got {:?}",
            val
        );
    }
}

#[test]
fn test_delta_at_center() {
    // Delta at center
    let size = 32;
    let mut data = Array2::<f64>::zeros((size, size));
    data[[size / 2, size / 2]] = 1.0;

    let spectrum = fft2d(&data.view()).unwrap();
    let reconstructed = ifft2d(&spectrum, size).unwrap();

    assert_real_arrays_close(&data, &reconstructed, EPSILON);
}

#[test]
fn test_horizontal_stripes() {
    // Horizontal stripes (only horizontal frequencies)
    let data = Array2::<f64>::from_shape_fn((64, 64), |(i, _j)| (i as f64 * 0.5).sin());

    let spectrum = fft2d(&data.view()).unwrap();

    // Only vertical frequencies (constant in x) should be non-zero
    // This means spectrum[:, 0] should dominate
    let col0_power: f64 = (0..64).map(|i| spectrum[[i, 0]].norm_sqr()).sum();
    let total_power: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();

    assert!(
        col0_power / total_power > 0.99,
        "Horizontal stripes should have most power in DC column"
    );
}

#[test]
fn test_vertical_stripes() {
    // Vertical stripes (only vertical frequencies)
    let data = Array2::<f64>::from_shape_fn((64, 64), |(_i, j)| (j as f64 * 0.5).sin());

    let spectrum = fft2d(&data.view()).unwrap();
    let reconstructed = ifft2d(&spectrum, 64).unwrap();

    assert_real_arrays_close(&data, &reconstructed, EPSILON);
}

#[test]
fn test_checkerboard_pattern() {
    // Checkerboard (high-frequency in both directions)
    let data =
        Array2::<f64>::from_shape_fn((32, 32), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { -1.0 });

    let spectrum = fft2d(&data.view()).unwrap();
    let reconstructed = ifft2d(&spectrum, 32).unwrap();

    assert_real_arrays_close(&data, &reconstructed, EPSILON);
}

// ============================================================================
// Mathematical Properties
// ============================================================================

#[test]
fn test_parsevals_theorem() {
    // Parseval's theorem: Energy in spatial domain equals energy in frequency domain
    let data = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
    });

    let spatial_energy: f64 = data.iter().map(|&x| x * x).sum();

    let spectrum = fft2d(&data.view()).unwrap();

    // Account for Hermitian symmetry: most frequencies appear twice
    let mut freq_energy = 0.0;
    let ncols = spectrum.ncols();
    for i in 0..spectrum.nrows() {
        for j in 0..ncols {
            let power = spectrum[[i, j]].norm_sqr();
            // DC and Nyquist (if even) columns appear once, others twice
            if j == 0 || (j == ncols - 1 && (data.ncols() % 2 == 0)) {
                freq_energy += power;
            } else {
                freq_energy += 2.0 * power;
            }
        }
    }

    // Normalize by array size
    freq_energy /= (data.nrows() * data.ncols()) as f64;

    let relative_error = (spatial_energy - freq_energy).abs() / spatial_energy;
    assert!(
        relative_error < 1e-6,
        "Parseval's theorem violated: spatial={}, freq={}, error={}",
        spatial_energy,
        freq_energy,
        relative_error
    );
}

#[test]
fn test_linearity() {
    // FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
    let x = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| i as f64 + j as f64);
    let y = Array2::<f64>::from_shape_fn((32, 32), |(i, _j)| (i as f64).sin());

    let a = 2.0;
    let b = 3.0;

    let combined = &x * a + &y * b;

    let fft_combined = fft2d(&combined.view()).unwrap();
    let fft_x = fft2d(&x.view()).unwrap();
    let fft_y = fft2d(&y.view()).unwrap();
    let fft_linear = &fft_x * Complex::new(a, 0.0) + &fft_y * Complex::new(b, 0.0);

    assert_complex_arrays_close(&fft_combined, &fft_linear, EPSILON);
}

// ============================================================================
// Planner Tests
// ============================================================================

#[test]
fn test_planner_different_sizes() {
    let mut planner = Fft2dPlanner::new();

    // Plan should handle different sizes
    let sizes = vec![(16, 16), (32, 32), (64, 64), (16, 32), (32, 16)];

    for (nrows, ncols) in sizes {
        let data = Array2::<f64>::ones((nrows, ncols));
        let result = planner.fft2d(&data.view());
        assert!(result.is_ok(), "Failed for size {}x{}", nrows, ncols);

        let spectrum = result.unwrap();
        assert_eq!(spectrum.shape(), &[nrows, ncols / 2 + 1]);
    }
}

#[test]
fn test_planner_repeated_calls() {
    // Planner should give consistent results across repeated calls
    let mut planner = Fft2dPlanner::new();

    let data = Array2::<f64>::from_shape_fn((32, 32), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
    });

    let first = planner.fft2d(&data.view()).unwrap();

    for _ in 0..10 {
        let result = planner.fft2d(&data.view()).unwrap();
        assert_complex_arrays_close(&first, &result, EPSILON);
    }
}

#[test]
fn test_planner_power_spectrum_consistency() {
    let mut planner = Fft2dPlanner::new();

    let data = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| i as f64 + j as f64);

    // Power spectrum via planner
    let power1 = planner.power_spectrum_2d(&data.view()).unwrap();

    // Power spectrum computed manually
    let spectrum = planner.fft2d(&data.view()).unwrap();
    let power2 = spectrum.mapv(|c| c.norm_sqr());

    assert_real_arrays_close(&power1, &power2, EPSILON);
}

#[test]
fn test_planner_magnitude_spectrum_consistency() {
    let mut planner = Fft2dPlanner::new();

    let data = Array2::<f64>::from_shape_fn((64, 64), |(i, j)| (i as f64).sin() + (j as f64).cos());

    let magnitude1 = planner.magnitude_spectrum_2d(&data.view()).unwrap();

    let spectrum = planner.fft2d(&data.view()).unwrap();
    let magnitude2 = spectrum.mapv(|c| c.norm());

    assert_real_arrays_close(&magnitude1, &magnitude2, EPSILON);
}

// ============================================================================
// FFTShift Tests
// ============================================================================

#[test]
fn test_fftshift_real() {
    let data = Array2::<f64>::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f64);

    let shifted = fftshift(data.clone());

    // Check that DC (0,0) moved to center
    let center = (4, 4);
    assert_eq!(shifted[[center.0, center.1]], data[[0, 0]]);
}

#[test]
fn test_ifftshift_real() {
    let data = Array2::<f64>::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f64);

    let shifted = fftshift(data.clone());
    let restored = ifftshift(shifted);

    assert_real_arrays_close(&data, &restored, EPSILON);
}

#[test]
fn test_fftshift_complex() {
    let data = Array2::<Complex<f64>>::from_shape_fn((8, 8), |(i, j)| {
        Complex::new((i * 8 + j) as f64, 0.0)
    });

    let shifted = fftshift(data.clone());
    let restored = ifftshift(shifted);

    assert_complex_arrays_close(&data, &restored, EPSILON);
}

#[test]
fn test_fftshift_roundtrip_odd_size() {
    let data = Array2::<f64>::from_shape_fn((7, 5), |(i, j)| (i * 5 + j) as f64);

    let shifted = fftshift(data.clone());
    let restored = ifftshift(shifted);

    assert_real_arrays_close(&data, &restored, EPSILON);
}

#[test]
fn test_fftshift_1d_roundtrip_odd_length() {
    let data = vec![0, 1, 2, 3, 4];
    let shifted = fftshift_1d(data.clone());
    assert_eq!(shifted, vec![2, 3, 4, 0, 1]);

    let restored = ifftshift_1d(shifted);
    assert_eq!(restored, data);
}

#[test]
fn test_fftshift_1d_roundtrip_even_length() {
    let data = vec![0, 1, 2, 3, 4, 5];
    let shifted = fftshift_1d(data.clone());
    assert_eq!(shifted, vec![3, 4, 5, 0, 1, 2]);

    let restored = ifftshift_1d(shifted);
    assert_eq!(restored, data);
}

// TODO: fftshift/ifftshift roundtrip for odd sizes needs investigation
// The roundtrip may not be perfect for odd-sized arrays due to center calculation
// #[test]
// fn test_fftshift_odd_size() {
//     let data = Array2::<f64>::from_shape_fn((7, 7), |(i, j)| (i * 7 + j) as f64);

//     let shifted = fftshift(data.clone());
//     let restored = ifftshift(shifted);

//     assert_real_arrays_close(&data, &restored, EPSILON);
// }

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_complete_workflow() {
    // Simulate complete image processing workflow
    let original = Array2::<f64>::from_shape_fn((128, 128), |(i, j)| {
        ((i as f64 - 64.0).powi(2) + (j as f64 - 64.0).powi(2)).sqrt() / 10.0
    });

    // Compute FFT
    let spectrum = fft2d(&original.view()).unwrap();

    // Compute power spectrum
    let power = power_spectrum_2d(&original.view()).unwrap();

    // Verify power matches spectrum
    let power_from_spectrum = spectrum.mapv(|c| c.norm_sqr());
    assert_real_arrays_close(&power, &power_from_spectrum, EPSILON);

    // Reconstruct
    let reconstructed = ifft2d(&spectrum, 128).unwrap();
    assert_real_arrays_close(&original, &reconstructed, EPSILON);
}

#[test]
fn test_multiple_operations_chain() {
    let mut planner = Fft2dPlanner::new();

    let data1 = Array2::<f64>::ones((32, 32));
    let data2 = Array2::<f64>::zeros((64, 64));
    let data3 = Array2::<f64>::from_shape_fn((48, 48), |(i, j)| i as f64 + j as f64);

    // Chain multiple operations
    let spec1 = planner.fft2d(&data1.view()).unwrap();
    let power1 = planner.power_spectrum_2d(&data1.view()).unwrap();

    let spec2 = planner.fft2d(&data2.view()).unwrap();
    let mag2 = planner.magnitude_spectrum_2d(&data2.view()).unwrap();
    let spec3 = planner.fft2d(&data3.view()).unwrap();
    let recon3 = planner.ifft2d(&spec3.view(), 48).unwrap();

    // Verify results
    assert_eq!(spec1.shape(), &[32, 17]);
    assert_eq!(power1.shape(), &[32, 17]);
    assert_eq!(spec2.shape(), &[64, 33]);
    assert_eq!(mag2.shape(), &[64, 33]);
    assert_real_arrays_close(&data3, &recon3, EPSILON);
}
