use spectrograms::*;
use std::f64::consts::PI;

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect()
}

#[test]
fn test_cqt_integration_basic() {
    let sample_rate = 16000.0;
    let duration = 3.0;
    let n_samples = (sample_rate * duration) as usize;
    let frequency = 440.0; // A4

    let samples = sine_wave(frequency, sample_rate, n_samples);

    // Create CQT parameters: 12 bins/octave, 7 octaves from C1
    let cqt_params = CqtParams::new(12, 7, 32.7).unwrap();

    // Create STFT parameters with large window for good low-frequency resolution
    let stft = StftParams::new(16384, 4096, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    // Create CQT spectrogram plan
    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Power>(&params, &cqt_params, None)
        .unwrap();

    // Compute CQT spectrogram
    let spectrogram = plan.compute(&samples).unwrap();

    // Basic checks
    assert_eq!(spectrogram.n_bins(), 84); // 12 * 7 = 84 bins
    assert!(spectrogram.n_frames() > 0);

    // Find bin with maximum energy
    let data = spectrogram.data();
    let mut max_energy = 0.0;
    let mut max_bin = 0;

    for bin in 0..spectrogram.n_bins() {
        let energy: f64 = (0..spectrogram.n_frames())
            .map(|frame| data[[bin, frame]])
            .sum();
        if energy > max_energy {
            max_energy = energy;
            max_bin = bin;
        }
    }

    // Check that detected frequency is close to input frequency
    let detected_freq = cqt_params.bin_frequency(max_bin);
    let freq_error = (detected_freq - frequency).abs();
    let freq_error_pct = (freq_error / frequency) * 100.0;

    // Should detect within 5% error (CQT has coarser resolution than FFT)
    assert!(
        freq_error_pct < 5.0,
        "Detected {} Hz, expected {} Hz (error: {:.2}%)",
        detected_freq,
        frequency,
        freq_error_pct
    );
}

#[test]
fn test_cqt_params_validation() {
    // Valid parameters
    assert!(CqtParams::new(12, 7, 32.7).is_ok());

    // Invalid: zero bins per octave
    assert!(CqtParams::new(0, 7, 32.7).is_err());

    // Invalid: zero octaves
    assert!(CqtParams::new(12, 0, 32.7).is_err());

    // Invalid: negative f_min
    assert!(CqtParams::new(12, 7, -10.0).is_err());

    // Invalid: zero f_min
    assert!(CqtParams::new(12, 7, 0.0).is_err());

    // Invalid: infinite f_min
    assert!(CqtParams::new(12, 7, f64::INFINITY).is_err());
}

#[test]
fn test_cqt_num_bins() {
    let cqt = CqtParams::new(12, 7, 32.7).unwrap();
    assert_eq!(cqt.num_bins(), 84); // 12 * 7

    let cqt = CqtParams::new(24, 5, 20.0).unwrap();
    assert_eq!(cqt.num_bins(), 120); // 24 * 5
}

#[test]
fn test_cqt_frequencies() {
    let cqt = CqtParams::new(12, 1, 100.0).unwrap();

    // Should have 12 bins (one octave)
    assert_eq!(cqt.num_bins(), 12);

    // First frequency should be f_min
    assert!((cqt.bin_frequency(0) - 100.0).abs() < 1e-6);

    // Last frequency in this octave should be just below 2 * f_min
    // Specifically: 100 * 2^(11/12) ≈ 189.0 Hz
    let last_freq = cqt.bin_frequency(11);
    let expected_last = 100.0 * 2.0f64.powf(11.0 / 12.0);
    assert!((last_freq - expected_last).abs() < 1e-6);

    // Frequencies should be logarithmically spaced
    let freqs = cqt.frequencies();
    assert_eq!(freqs.len(), 12);

    for i in 1..freqs.len() {
        let ratio = freqs[i] / freqs[i - 1];
        // Ratio between adjacent bins should be constant (semitone = 2^(1/12))
        assert!((ratio - 2.0f64.powf(1.0 / 12.0)).abs() < 1e-6);
    }
}

#[test]
fn test_cqt_with_power_amplitude() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let cqt_params = CqtParams::new(12, 5, 50.0).unwrap();
    let stft = StftParams::new(8192, 2048, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Power>(&params, &cqt_params, None)
        .unwrap();
    let spec = plan.compute(&samples).unwrap();

    // Power values should be non-negative
    for val in spec.data().iter() {
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_cqt_with_magnitude_amplitude() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let cqt_params = CqtParams::new(12, 5, 50.0).unwrap();
    let stft = StftParams::new(8192, 2048, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Magnitude>(&params, &cqt_params, None)
        .unwrap();
    let spec = plan.compute(&samples).unwrap();

    // Magnitude values should be non-negative and generally smaller than power
    for val in spec.data().iter() {
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_cqt_with_decibel_amplitude() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let cqt_params = CqtParams::new(12, 5, 50.0).unwrap();
    let stft = StftParams::new(8192, 2048, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let log_params = LogParams::new(-80.0).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Decibels>(&params, &cqt_params, Some(&log_params))
        .unwrap();
    let spec = plan.compute(&samples).unwrap();

    // Decibel values should be floored at -80 dB
    for val in spec.data().iter() {
        assert!(*val >= -80.0);
    }
}

#[test]
fn test_cqt_frame_computation() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let cqt_params = CqtParams::new(12, 5, 50.0).unwrap();
    let stft = StftParams::new(8192, 2048, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Power>(&params, &cqt_params, None)
        .unwrap();

    // Compute single frame
    let frame = plan.compute_frame(&samples, 0).unwrap();

    assert_eq!(frame.len(), 60); // 12 * 5 = 60 bins

    // Frame values should be non-negative
    for &val in &frame {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_cqt_output_shape() {
    let sample_rate = 16000.0;
    let signal_length = 16000;

    let cqt_params = CqtParams::new(12, 6, 40.0).unwrap();
    let stft = StftParams::new(8192, 2048, WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let plan = planner
        .cqt_plan::<Power>(&params, &cqt_params, None)
        .unwrap();

    let (n_bins, n_frames) = plan.output_shape(signal_length).unwrap();

    assert_eq!(n_bins, 72); // 12 * 6
    assert!(n_frames > 0);
}
