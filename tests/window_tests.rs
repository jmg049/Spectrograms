use spectrograms::{StftParams, WindowType, make_window, nzu};
use std::str::FromStr;

#[test]
fn test_window_from_str_rectangular() {
    assert_eq!(
        WindowType::from_str("rectangle").unwrap(),
        WindowType::Rectangular
    );
    assert_eq!(
        WindowType::from_str("rect").unwrap(),
        WindowType::Rectangular
    );
    assert_eq!(
        WindowType::from_str("RECT").unwrap(),
        WindowType::Rectangular
    );
}

#[test]
fn test_window_from_str_hanning() {
    assert_eq!(
        WindowType::from_str("hanning").unwrap(),
        WindowType::Hanning
    );
    assert_eq!(WindowType::from_str("hann").unwrap(), WindowType::Hanning);
    assert_eq!(WindowType::from_str("HANN").unwrap(), WindowType::Hanning);
}

#[test]
fn test_window_from_str_hamming() {
    assert_eq!(
        WindowType::from_str("hamming").unwrap(),
        WindowType::Hamming
    );
    assert_eq!(WindowType::from_str("hamm").unwrap(), WindowType::Hamming);
}

#[test]
fn test_window_from_str_blackman() {
    assert_eq!(
        WindowType::from_str("blackman").unwrap(),
        WindowType::Blackman
    );
    assert_eq!(
        WindowType::from_str("BLACKMAN").unwrap(),
        WindowType::Blackman
    );
}

#[test]
fn test_window_from_str_kaiser() {
    let kaiser = WindowType::from_str("kaiser=5.0").unwrap();
    match kaiser {
        WindowType::Kaiser { beta } => {
            assert!((beta - 5.0).abs() < 1e-10);
        }
        _ => panic!("Expected Kaiser window"),
    }

    let kaiser2 = WindowType::from_str("KAISER=10.5").unwrap();
    match kaiser2 {
        WindowType::Kaiser { beta } => {
            assert!((beta - 10.5).abs() < 1e-10);
        }
        _ => panic!("Expected Kaiser window"),
    }
}

#[test]
fn test_window_from_str_gaussian() {
    let gaussian = WindowType::from_str("gaussian=2.5").unwrap();
    match gaussian {
        WindowType::Gaussian { std } => {
            assert!((std - 2.5).abs() < 1e-10);
        }
        _ => panic!("Expected Gaussian window"),
    }
}

#[test]
fn test_window_from_str_invalid() {
    assert!(WindowType::from_str("").is_err());
    assert!(WindowType::from_str("invalid").is_err());
    assert!(WindowType::from_str("kaiser").is_err()); // Missing parameter
    assert!(WindowType::from_str("gaussian").is_err()); // Missing parameter
    assert!(WindowType::from_str("kaiser=").is_err()); // Empty parameter
    assert!(WindowType::from_str("kaiser=abc").is_err()); // Invalid number
}

#[test]
fn test_window_display() {
    assert_eq!(WindowType::Rectangular.to_string(), "Rectangular");
    assert_eq!(WindowType::Hanning.to_string(), "Hanning");
    assert_eq!(WindowType::Hamming.to_string(), "Hamming");
    assert_eq!(WindowType::Blackman.to_string(), "Blackman");
    assert_eq!(
        WindowType::Kaiser { beta: 5.0 }.to_string(),
        "Kaiser(beta=5)"
    );
    assert_eq!(
        WindowType::Gaussian { std: 2.5 }.to_string(),
        "Gaussian(std=2.5)"
    );
}

#[test]
fn test_window_default() {
    assert_eq!(WindowType::default(), WindowType::Hanning);
}

#[test]
fn test_window_clone_eq() {
    let w1 = WindowType::Hanning;
    let w2 = w1.clone();
    assert_eq!(w1, w2);

    let w3 = WindowType::Kaiser { beta: 5.0 };
    let w4 = w3.clone();
    assert_eq!(w3, w4);
}

// ============================================================================
// Custom Window Tests
// ============================================================================

#[test]
fn test_custom_window_creation() {
    let coeffs = vec![0.0, 0.5, 1.0, 0.5, 0.0];
    let window = WindowType::custom(coeffs.clone()).unwrap();

    match window {
        WindowType::Custom { coefficients, size } => {
            assert_eq!(size.get(), 5);
            assert_eq!(**coefficients, coeffs);
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_empty_error() {
    let result = WindowType::custom(vec![]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("cannot be empty"));
}

#[test]
fn test_custom_window_nan_error() {
    let result = WindowType::custom(vec![1.0, 2.0, f64::NAN, 4.0]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("not finite"));
    assert!(err_msg.contains("index 2"));
}

#[test]
fn test_custom_window_infinity_error() {
    let result = WindowType::custom(vec![1.0, f64::INFINITY, 3.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not finite"));
}

#[test]
fn test_custom_window_in_make_window() {
    let coeffs = vec![0.1, 0.5, 1.0, 0.5, 0.1];
    let window = WindowType::custom(coeffs.clone()).unwrap();
    let result = make_window(window, nzu!(5));

    assert_eq!(result.len().get(), 5);
    for (i, &val) in result.iter().enumerate() {
        assert!((val - coeffs[i]).abs() < 1e-10);
    }
}

#[test]
#[should_panic(expected = "Custom window size mismatch")]
fn test_custom_window_size_mismatch_panic() {
    let coeffs = vec![0.1, 0.5, 1.0, 0.5, 0.1];
    let window = WindowType::custom(coeffs).unwrap();
    // Try to use 5-element window with n_fft=10
    let _ = make_window(window, nzu!(10));
}

#[test]
fn test_custom_window_in_stft_params() {
    let coeffs = vec![1.0; 512];
    let window = WindowType::custom(coeffs).unwrap();

    // Should succeed with matching size
    let result = StftParams::new(nzu!(512), nzu!(256), window, true);
    assert!(result.is_ok());
}

#[test]
fn test_custom_window_stft_size_mismatch_error() {
    let coeffs = vec![1.0; 256];
    let window = WindowType::custom(coeffs).unwrap();

    // Should fail with mismatched size
    let result = StftParams::new(nzu!(512), nzu!(256), window, true);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("256"));
    assert!(err_msg.contains("512"));
}

#[test]
fn test_custom_window_clone() {
    let coeffs = vec![0.1, 0.5, 1.0, 0.5, 0.1];
    let window1 = WindowType::custom(coeffs).unwrap();
    let window2 = window1.clone();

    assert_eq!(window1, window2);

    // Verify Arc sharing works correctly
    match (window1, window2) {
        (
            WindowType::Custom {
                coefficients: c1, ..
            },
            WindowType::Custom {
                coefficients: c2, ..
            },
        ) => {
            assert_eq!(c1, c2);
        }
        _ => panic!("Expected Custom windows"),
    }
}

#[test]
fn test_custom_window_display() {
    let coeffs = vec![1.0; 128];
    let window = WindowType::custom(coeffs).unwrap();
    assert_eq!(window.to_string(), "Custom(n=128)");
}

#[test]
fn test_custom_window_is_parameterized() {
    let coeffs = vec![1.0; 10];
    let window = WindowType::custom(coeffs).unwrap();
    assert!(!window.is_parameterized());
    assert_eq!(window.parameter_value(), None);
}

// ============================================================================
// Window Normalization Tests
// ============================================================================

#[test]
fn test_custom_window_sum_normalization() {
    let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let window = WindowType::custom_with_normalization(coeffs, Some("sum")).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let sum: f64 = coefficients.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Sum should be 1.0, got {}", sum);
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_peak_normalization() {
    let coeffs = vec![0.5, 1.0, 2.0, 1.0, 0.5];
    let window = WindowType::custom_with_normalization(coeffs, Some("peak")).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let max = coefficients
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((max - 1.0).abs() < 1e-10, "Max should be 1.0, got {}", max);
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_max_alias() {
    // "max" should work as alias for "peak"
    let coeffs = vec![0.5, 1.0, 2.0, 1.0, 0.5];
    let window = WindowType::custom_with_normalization(coeffs, Some("max")).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let max = coefficients
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((max - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_energy_normalization() {
    let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let window = WindowType::custom_with_normalization(coeffs, Some("energy")).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let energy: f64 = coefficients.iter().map(|x| x * x).sum();
            assert!(
                (energy - 1.0).abs() < 1e-10,
                "Energy should be 1.0, got {}",
                energy
            );
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_rms_alias() {
    // "rms" should work as alias for "energy"
    let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let window = WindowType::custom_with_normalization(coeffs, Some("rms")).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let energy: f64 = coefficients.iter().map(|x| x * x).sum();
            assert!((energy - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_no_normalization() {
    let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let original_sum: f64 = coeffs.iter().sum();

    let window = WindowType::custom_with_normalization(coeffs.clone(), None).unwrap();

    match window {
        WindowType::Custom { coefficients, .. } => {
            let sum: f64 = coefficients.iter().sum();
            assert!(
                (sum - original_sum).abs() < 1e-10,
                "Sum should be unchanged"
            );
        }
        _ => panic!("Expected Custom window"),
    }
}

#[test]
fn test_custom_window_invalid_normalization() {
    let coeffs = vec![1.0, 2.0, 3.0];
    let result = WindowType::custom_with_normalization(coeffs, Some("invalid"));

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unknown normalization"));
    assert!(err_msg.contains("invalid"));
}

#[test]
fn test_custom_window_zero_sum_error() {
    let coeffs = vec![0.0, 0.0, 0.0];
    let result = WindowType::custom_with_normalization(coeffs, Some("sum"));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("sum is zero"));
}

#[test]
fn test_custom_window_zero_peak_error() {
    let coeffs = vec![0.0, 0.0, 0.0];
    let result = WindowType::custom_with_normalization(coeffs, Some("peak"));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("maximum is zero"));
}

#[test]
fn test_custom_window_zero_energy_error() {
    let coeffs = vec![0.0, 0.0, 0.0];
    let result = WindowType::custom_with_normalization(coeffs, Some("energy"));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("energy is zero"));
}

#[cfg(feature = "serde")]
#[test]
fn test_custom_window_serde() {
    use serde_json;

    let coeffs = vec![0.1, 0.5, 1.0, 0.5, 0.1];
    let window = WindowType::custom(coeffs.clone()).unwrap();

    // Serialize
    let json = serde_json::to_string(&window).unwrap();

    // Deserialize
    let deserialized: WindowType = serde_json::from_str(&json).unwrap();

    // Verify
    match deserialized {
        WindowType::Custom { coefficients, size } => {
            assert_eq!(size.get(), 5);
            assert_eq!(**coefficients, coeffs);
        }
        _ => panic!("Expected Custom window"),
    }
}
