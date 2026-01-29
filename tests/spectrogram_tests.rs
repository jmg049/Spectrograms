use non_empty_slice::{NonEmptyVec, non_empty_vec};
use spectrograms::{
    LinearDbSpectrogram, LinearMagnitudeSpectrogram, LinearPowerSpectrogram, LogParams,
    MelDbSpectrogram, MelMagnitudeSpectrogram, MelParams, MelPowerSpectrogram, SpectrogramParams,
    SpectrogramPlanner, StftParams, WindowType, nzu,
};
use std::f64::consts::PI;

/// Generate a sine wave at a specific frequency
fn sine_wave(freq: f64, sample_rate: f64, duration: f64) -> NonEmptyVec<f64> {
    let n_samples = (duration * sample_rate) as usize;
    let v = (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect();
    NonEmptyVec::new(v).unwrap()
}

#[test]
fn test_linear_power_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let spec = LinearPowerSpectrogram::compute(&samples, &params, None).unwrap();

    assert_eq!(spec.n_bins(), nzu!(257)); // n_fft/2 + 1
}

#[test]
fn test_linear_magnitude_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let spec = LinearMagnitudeSpectrogram::compute(&samples, &params, None).unwrap();

    assert_eq!(spec.n_bins(), nzu!(257));
}

#[test]
fn test_linear_db_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let db = LogParams::new(-80.0).unwrap();

    let spec = LinearDbSpectrogram::compute(&samples, &params, Some(&db)).unwrap();

    assert_eq!(spec.n_bins(), nzu!(257));

    // Check that all values are >= floor_db
    for &val in spec.data().iter() {
        assert!(val >= -80.0, "Value {} is below floor", val);
    }
}

#[test]
fn test_mel_power_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let mel = MelParams::new(nzu!(80), 0.0, 8000.0).unwrap();

    let spec = MelPowerSpectrogram::compute(&samples, &params, &mel, None).unwrap();

    assert_eq!(spec.n_bins(), nzu!(80));
}

#[test]
fn test_mel_magnitude_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let mel = MelParams::new(nzu!(80), 0.0, 8000.0).unwrap();

    let spec = MelMagnitudeSpectrogram::compute(&samples, &params, &mel, None).unwrap();

    assert_eq!(spec.n_bins(), nzu!(80));
}

#[test]
fn test_mel_db_spectrogram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let mel = MelParams::new(nzu!(80), 0.0, 8000.0).unwrap();
    let db = LogParams::new(-80.0).unwrap();

    let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db)).unwrap();

    assert_eq!(spec.n_bins(), nzu!(80));

    // Check that all values are >= floor_db
    for &val in spec.data().iter() {
        assert!(val >= -80.0, "Value {} is below floor", val);
    }
}

#[test]
fn test_spectrogram_short_input() {
    let samples: NonEmptyVec<f64> = non_empty_vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    // Should still work even with very short input (produces 1 frame)
    let spec = LinearPowerSpectrogram::compute(&samples, &params, None).unwrap();
    assert_eq!(spec.n_frames(), nzu!(1));
}

#[test]
fn test_spectrogram_plan_reuse() {
    let sample_rate = 16000.0;
    let samples1 = sine_wave(440.0, sample_rate, 0.5);
    let samples2 = sine_wave(880.0, sample_rate, 0.5);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    let spec1 = plan.compute(&samples1).unwrap();
    let spec2 = plan.compute(&samples2).unwrap();

    // Both should have same structure
    assert_eq!(spec1.n_bins(), spec2.n_bins());

    // But different data
    assert_ne!(spec1.data(), spec2.data());
}

#[test]
fn test_mel_f_max_exceeds_nyquist() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();
    let mel = MelParams::new(nzu!(80), 0.0, 10000.0).unwrap(); // 10kHz > 8kHz Nyquist

    let result = MelPowerSpectrogram::compute(&samples, &params, &mel, None);
    assert!(result.is_err());
}

#[test]
fn test_different_windows() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let windows = non_empty_vec![
        WindowType::Rectangular,
        WindowType::Hanning,
        WindowType::Hamming,
        WindowType::Blackman,
        WindowType::Kaiser { beta: 5.0 },
        WindowType::Gaussian { std: 0.5 },
    ];

    for window in windows {
        let stft = StftParams::new(nzu!(512), nzu!(256), window.clone(), true).unwrap();
        let params = SpectrogramParams::new(stft, sample_rate).unwrap();

        let spec = LinearPowerSpectrogram::compute(&samples, &params, None);
        assert!(spec.is_ok(), "Failed with window: {}", window.to_string());
    }
}

#[test]
fn test_frequency_axis() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let spec = LinearPowerSpectrogram::compute(&samples, &params, None).unwrap();

    let freqs = spec.axes().frequencies();
    assert_eq!(freqs.len(), spec.n_bins());

    // First frequency should be 0
    assert!((freqs[0] - 0.0).abs() < 1e-6);

    // Last frequency should be close to Nyquist
    let nyquist = sample_rate / 2.0;
    assert!((freqs[freqs.len().get() - 1] - nyquist).abs() < 1e-3);

    // Frequencies should be monotonically increasing
    for i in 1..freqs.len().get() {
        assert!(freqs[i] > freqs[i - 1]);
    }
}

#[test]
fn test_time_axis() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let spec = LinearPowerSpectrogram::compute(&samples, &params, None).unwrap();

    let times = spec.axes().times();
    assert_eq!(times.len(), spec.n_frames());

    // First time should be 0
    assert!((times[0] - 0.0).abs() < 1e-6);

    // Times should be monotonically increasing
    for i in 1..times.len().get() {
        assert!(times[i] > times[i - 1]);
    }

    // Time step should be hop_size / sample_rate
    let expected_dt = 256.0 / sample_rate;
    for i in 1..times.len().get() {
        let dt = times[i] - times[i - 1];
        assert!((dt - expected_dt).abs() < 1e-6);
    }
}
