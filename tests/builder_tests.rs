use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};

#[test]
fn test_stft_builder() {
    let stft = StftParams::builder()
        .n_fft(1024)
        .hop_size(256)
        .window(WindowType::Hamming)
        .centre(false)
        .build()
        .unwrap();

    assert_eq!(stft.n_fft(), 1024);
    assert_eq!(stft.hop_size(), 256);
    assert_eq!(stft.window(), WindowType::Hamming);
    assert!(!stft.centre());
}

#[test]
fn test_stft_builder_defaults() {
    let stft = StftParams::builder()
        .n_fft(512)
        .hop_size(128)
        .build()
        .unwrap();

    assert_eq!(stft.n_fft(), 512);
    assert_eq!(stft.hop_size(), 128);
    assert_eq!(stft.window(), WindowType::Hanning); // default
    assert!(stft.centre()); // default
}

#[test]
fn test_stft_builder_missing_n_fft() {
    let result = StftParams::builder().hop_size(256).build();
    assert!(result.is_err());
}

#[test]
fn test_stft_builder_missing_hop_size() {
    let result = StftParams::builder().n_fft(512).build();
    assert!(result.is_err());
}

#[test]
fn test_spectrogram_params_builder() {
    let params = SpectrogramParams::builder()
        .sample_rate(44100.0)
        .n_fft(2048)
        .hop_size(512)
        .window(WindowType::Blackman)
        .centre(true)
        .build()
        .unwrap();

    assert_eq!(params.sample_rate_hz(), 44100.0);
    assert_eq!(params.stft().n_fft(), 2048);
    assert_eq!(params.stft().hop_size(), 512);
}

#[test]
fn test_spectrogram_params_builder_missing_sample_rate() {
    let result = SpectrogramParams::builder()
        .n_fft(512)
        .hop_size(256)
        .build();
    assert!(result.is_err());
}

#[test]
fn test_spectrogram_params_speech_default() {
    let params = SpectrogramParams::speech_default(16000.0).unwrap();

    assert_eq!(params.sample_rate_hz(), 16000.0);
    assert_eq!(params.stft().n_fft(), 512);
    assert_eq!(params.stft().hop_size(), 160);
}

#[test]
fn test_spectrogram_params_music_default() {
    let params = SpectrogramParams::music_default(44100.0).unwrap();

    assert_eq!(params.sample_rate_hz(), 44100.0);
    assert_eq!(params.stft().n_fft(), 2048);
    assert_eq!(params.stft().hop_size(), 512);
}

#[test]
fn test_mel_params_standard() {
    let mel = MelParams::standard(16000.0).unwrap();

    assert_eq!(mel.n_mels(), 128);
    assert_eq!(mel.f_min(), 0.0);
    assert_eq!(mel.f_max(), 8000.0);
}

#[test]
fn test_mel_params_speech_standard() {
    let mel = MelParams::speech_standard().unwrap();

    assert_eq!(mel.n_mels(), 40);
    assert_eq!(mel.f_min(), 0.0);
    assert_eq!(mel.f_max(), 8000.0);
}
