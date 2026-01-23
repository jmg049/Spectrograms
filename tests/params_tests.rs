use spectrograms::{LogParams, MelParams, SpectrogramParams, StftParams, WindowType};

#[test]
fn test_stft_params_valid() {
    let params = StftParams::new(512, 256, WindowType::Hanning, true);
    assert!(params.is_ok());

    let p = params.unwrap();
    assert_eq!(p.n_fft(), 512);
    assert_eq!(p.hop_size(), 256);
    assert_eq!(p.window(), WindowType::Hanning);
    assert!(p.centre());
}

#[test]
fn test_stft_params_zero_n_fft() {
    let result = StftParams::new(0, 256, WindowType::Hanning, true);
    assert!(result.is_err());
}

#[test]
fn test_stft_params_zero_hop_size() {
    let result = StftParams::new(512, 0, WindowType::Hanning, true);
    assert!(result.is_err());
}

#[test]
fn test_stft_params_hop_larger_than_n_fft() {
    let result = StftParams::new(512, 1024, WindowType::Hanning, true);
    assert!(result.is_err());
}

#[test]
fn test_stft_params_equal_hop_and_n_fft() {
    let result = StftParams::new(512, 512, WindowType::Hanning, true);
    assert!(result.is_ok());
}

#[test]
fn test_spectrogram_params_valid() {
    let stft = StftParams::new(512, 256, WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0);
    assert!(params.is_ok());

    let p = params.unwrap();
    assert_eq!(p.sample_rate_hz(), 16000.0);
    assert_eq!(p.nyquist_hz(), 8000.0);
    assert_eq!(p.frame_period_seconds(), 256.0 / 16000.0);
}

#[test]
fn test_spectrogram_params_zero_sample_rate() {
    let stft = StftParams::new(512, 256, WindowType::Hanning, true).unwrap();
    let result = SpectrogramParams::new(stft, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_spectrogram_params_negative_sample_rate() {
    let stft = StftParams::new(512, 256, WindowType::Hanning, true).unwrap();
    let result = SpectrogramParams::new(stft, -16000.0);
    assert!(result.is_err());
}

#[test]
fn test_spectrogram_params_infinite_sample_rate() {
    let stft = StftParams::new(512, 256, WindowType::Hanning, true).unwrap();
    let result = SpectrogramParams::new(stft, f64::INFINITY);
    assert!(result.is_err());
}

#[test]
fn test_mel_params_valid() {
    let params = MelParams::new(80, 0.0, 8000.0);
    assert!(params.is_ok());

    let p = params.unwrap();
    assert_eq!(p.n_mels(), 80);
    assert_eq!(p.f_min(), 0.0);
    assert_eq!(p.f_max(), 8000.0);
}

#[test]
fn test_mel_params_zero_n_mels() {
    let result = MelParams::new(0, 0.0, 8000.0);
    assert!(result.is_err());
}

#[test]
fn test_mel_params_negative_f_min() {
    let result = MelParams::new(80, -100.0, 8000.0);
    assert!(result.is_err());
}

#[test]
fn test_mel_params_f_max_less_than_f_min() {
    let result = MelParams::new(80, 8000.0, 100.0);
    assert!(result.is_err());
}

#[test]
fn test_mel_params_equal_f_min_f_max() {
    let result = MelParams::new(80, 8000.0, 8000.0);
    assert!(result.is_err());
}

#[test]
fn test_log_params_valid() {
    let params = LogParams::new(-80.0);
    assert!(params.is_ok());
    assert_eq!(params.unwrap().floor_db(), -80.0);

    let params2 = LogParams::new(0.0);
    assert!(params2.is_ok());
    assert_eq!(params2.unwrap().floor_db(), 0.0);
}

#[test]
fn test_log_params_infinite() {
    let result = LogParams::new(f64::INFINITY);
    assert!(result.is_err());

    let result2 = LogParams::new(f64::NEG_INFINITY);
    assert!(result2.is_err());
}

#[test]
fn test_log_params_nan() {
    let result = LogParams::new(f64::NAN);
    assert!(result.is_err());
}
