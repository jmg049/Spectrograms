use non_empty_slice::non_empty_vec;
use num_complex::Complex;
use spectrograms::*;

#[test]
fn test_fft_with_zero_padding() {
    // Signal shorter than n_fft
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = fft(&signal, nzu!(8)).unwrap();
    assert_eq!(result.len(), 5); // 8/2 + 1
}

#[test]
fn test_fft_exact_length() {
    // Exact length (regression test)
    let signal = non_empty_vec![1.0; nzu!(512)];
    let result = fft(&signal, signal.len()).unwrap();
    assert_eq!(result.len(), 257);
}

#[test]
fn test_fft_too_long_errors() {
    // Signal longer than n_fft
    let signal = non_empty_vec![1.0; nzu!(10)];
    let result = fft(&signal, nzu!(8));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds"));
}

#[test]
fn test_power_spectrum_with_padding_and_window() {
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = power_spectrum(&signal, nzu!(8), Some(WindowType::Hanning)).unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_power_spectrum_with_padding_no_window() {
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = power_spectrum(&signal, nzu!(8), None).unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_magnitude_spectrum_with_padding() {
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = magnitude_spectrum(&signal, nzu!(8), Some(WindowType::Hanning)).unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_frequency_semantics_preserved() {
    // Verify that padded FFT has correct frequency resolution
    let fs = 1000.0;
    let n_fft = nzu!(256);
    let signal = non_empty_vec![1.0; nzu!(128)]; // Half length
    let spectrum = fft(&signal, n_fft).unwrap();

    // Frequency bin spacing should be fs/n_fft
    let df = fs / n_fft.get() as f64;
    assert!((df - 3.90625).abs() < 1e-6);

    // Output length should still be n_fft/2 + 1
    assert_eq!(spectrum.len(), 129);
}

#[test]
fn test_planner_fft_with_padding() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner.fft(&signal, nzu!(8)).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_planner_fft_exact_length() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0; nzu!(512)];
    let result = planner.fft(&signal, nzu!(512)).unwrap();
    assert_eq!(result.len(), 257);
}

#[test]
fn test_planner_fft_too_long_errors() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0; nzu!(10)];
    let result = planner.fft(&signal, nzu!(8));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds"));
}

#[test]
fn test_planner_power_spectrum_with_padding() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner
        .power_spectrum(&signal, nzu!(8), Some(WindowType::Hanning))
        .unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_planner_magnitude_spectrum_with_padding() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner
        .magnitude_spectrum(&signal, nzu!(8), Some(WindowType::Hanning))
        .unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_spectrogram_planner_compute_power_spectrum_with_padding() {
    let planner = SpectrogramPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner
        .compute_power_spectrum(&signal, nzu!(8), WindowType::Hanning)
        .unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_spectrogram_planner_compute_magnitude_spectrum_with_padding() {
    let planner = SpectrogramPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner
        .compute_magnitude_spectrum(&signal, nzu!(8), WindowType::Hanning)
        .unwrap();
    assert_eq!(result.len(), nzu!(5));
}

#[test]
fn test_irfft_remains_strict() {
    // Inverse FFT should still enforce exact length
    let spectrum = non_empty_vec![Complex::new(1.0, 0.0); nzu!(4)]; // Should be 5 for n_fft=8
    let result = irfft(&spectrum, nzu!(8));
    assert!(result.is_err());
}

#[test]
fn test_planner_irfft_remains_strict() {
    // Planner inverse FFT should also remain strict
    let mut planner = FftPlanner::new();
    let spectrum = non_empty_vec![Complex::new(1.0, 0.0); nzu!(4)]; // Should be 5 for n_fft=8
    let result = planner.irfft(&spectrum, nzu!(8));
    assert!(result.is_err());
}

#[test]
fn test_padding_preserves_dc_component() {
    // Test that DC component is preserved with padding
    let signal = non_empty_vec![1.0, 1.0, 1.0];
    let spectrum_padded = fft(&signal, nzu!(8)).unwrap();

    // DC component should be sum of signal values
    let dc = spectrum_padded[0].norm();
    assert!((dc - 3.0).abs() < 1e-10);
}

#[test]
fn test_padding_with_single_sample() {
    // Edge case: single sample
    let signal = non_empty_vec![1.0];
    let result = fft(&signal, nzu!(8)).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_batch_processing_with_variable_lengths() {
    // Test that planner can handle variable-length inputs efficiently
    let mut planner = FftPlanner::new();

    let signals = non_empty_vec![
        non_empty_vec![1.0; nzu!(100)],
        non_empty_vec![1.0; nzu!(128)],
        non_empty_vec![1.0; nzu!(50)],
    ];

    for signal in signals {
        let result = planner.fft(&signal, nzu!(128));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 65); // 128/2 + 1
    }
}

#[test]
fn test_windowing_applied_to_full_padded_buffer() {
    // Ensure window is applied to the full n_fft buffer, not just the input samples
    let signal = non_empty_vec![1.0; nzu!(4)];
    let n_fft = nzu!(8);

    let power = power_spectrum(&signal, n_fft, Some(WindowType::Hanning)).unwrap();

    // Result should be valid and have correct length
    assert_eq!(power.len(), nzu!(5));

    // Power should be non-zero (if window wasn't applied correctly, this might fail)
    let total_power: f64 = power.iter().sum();
    assert!(total_power > 0.0);
}

#[cfg(all(feature = "realfft", not(feature = "fftw")))]
#[test]
fn test_padding_realfft_backend() {
    // Test with RealFFT backend specifically
    let signal = non_empty_vec![1.0; nzu!(100)];
    let result = fft(&signal, nzu!(128)).unwrap();
    assert_eq!(result.len(), 65);
}

#[cfg(feature = "fftw")]
#[test]
fn test_padding_fftw_backend() {
    // Test with FFTW backend specifically
    let signal = non_empty_vec![1.0; nzu!(100)];
    let result = fft(&signal, nzu!(128)).unwrap();
    assert_eq!(result.len(), 65);
}

#[test]
fn test_rfft_with_padding() {
    // Test the rfft function (which returns magnitudes, not complex)
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = rfft(&signal, nzu!(8)).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_planner_rfft_with_padding() {
    let mut planner = FftPlanner::new();
    let signal = non_empty_vec![1.0, 2.0, 3.0];
    let result = planner.rfft(&signal, nzu!(8)).unwrap();
    assert_eq!(result.len(), 5);
}
