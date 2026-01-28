#![cfg(feature = "serde")]

use ndarray::Array2;
use non_empty_slice::{NonEmptyVec, non_empty_vec};
use num_complex::Complex;
use spectrograms::*;

/// Helper to generate a test signal (440 Hz sine wave)
fn test_signal(sample_rate: f64, duration: f64) -> NonEmptyVec<f64> {
    let n_samples = (sample_rate * duration) as usize;
    let v = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();
    NonEmptyVec::new(v).unwrap()
}

/// Helper to verify array equality within tolerance
fn arrays_equal(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tolerance)
}

/// Helper to verify complex array equality within tolerance
fn complex_arrays_equal(
    a: &Array2<Complex<f64>>,
    b: &Array2<Complex<f64>>,
    tolerance: f64,
) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x.re - y.re).abs() < tolerance && (x.im - y.im).abs() < tolerance)
}

#[test]
fn test_linear_power_spectrogram_json() {
    let signal = test_signal(16000.0, 1.0);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    let spec = LinearPowerSpectrogram::compute(&signal, &params, None).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();

    // Deserialize back
    let deserialized: LinearPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
    assert_eq!(spec.frequencies().len(), deserialized.frequencies().len());
    assert_eq!(spec.times().len(), deserialized.times().len());
}

#[test]
fn test_linear_magnitude_spectrogram_bincode() {
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    let spec = LinearMagnitudeSpectrogram::compute(&signal, &params, None).unwrap();

    // Serialize to bincode
    let bytes = bincode2::serialize(&spec).unwrap();

    // Deserialize back
    let deserialized: LinearMagnitudeSpectrogram = bincode2::deserialize(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_linear_decibel_spectrogram_messagepack() {
    let signal = test_signal(8000.0, 0.3);
    let stft = StftParams::new(nzu!(128), nzu!(64), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 8000.0).unwrap();
    let log_params = LogParams::new(-80.0).unwrap();

    let spec = LinearDbSpectrogram::compute(&signal, &params, Some(&log_params)).unwrap();

    // Serialize to MessagePack
    let bytes = rmp_serde::to_vec(&spec).unwrap();

    // Deserialize back
    let deserialized: LinearDbSpectrogram = rmp_serde::from_slice(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_mel_spectrogram_json() {
    let signal = test_signal(22050.0, 0.5);
    let stft = StftParams::new(nzu!(1024), nzu!(512), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 22050.0).unwrap();
    let mel_params = MelParams::new(nzu!(80), 0.0, 11025.0).unwrap();

    let spec = MelPowerSpectrogram::compute(&signal, &params, &mel_params, None).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();

    // Deserialize back
    let deserialized: MelPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
    assert_eq!(spec.frequencies().len(), deserialized.frequencies().len());
}

#[test]
fn test_erb_spectrogram_bincode() {
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();
    let erb_params = ErbParams::new(nzu!(40), 50.0, 8000.0).unwrap();

    let spec = ErbPowerSpectrogram::compute(&signal, &params, &erb_params, None).unwrap();

    // Serialize to bincode
    let bytes = bincode2::serialize(&spec).unwrap();

    // Deserialize back
    let deserialized: ErbPowerSpectrogram = bincode2::deserialize(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_loghz_spectrogram_json() {
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();
    let log_params = LogHzParams::new(nzu!(60), 50.0, 8000.0).unwrap();

    let spec = LogHzPowerSpectrogram::compute(&signal, &params, &log_params, None).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();

    // Deserialize back
    let deserialized: LogHzPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_cqt_spectrogram_json() {
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();
    let cqt_params = CqtParams::new(nzu!(12), nzu!(5), 55.0).unwrap();

    let spec = CqtPowerSpectrogram::compute(&signal, &params, &cqt_params, None).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();

    // Deserialize back
    let deserialized: CqtPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_mfcc_json() {
    let signal = test_signal(16000.0, 1.0);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    let mfcc = mfcc(&signal, &stft, 16000.0, nzu!(40), &mfcc_params).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&mfcc).unwrap();

    // Deserialize back
    let deserialized: Mfcc = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(mfcc.data.shape(), deserialized.data.shape());
    assert!(arrays_equal(&mfcc.data, &deserialized.data, 1e-10));
    assert_eq!(mfcc.n_coefficients(), deserialized.n_coefficients());
    assert_eq!(mfcc.n_frames(), deserialized.n_frames());
}

#[test]
fn test_mfcc_bincode() {
    let signal = test_signal(22050.0, 0.5);
    let stft = StftParams::new(nzu!(1024), nzu!(512), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(20));

    let mfcc = mfcc(&signal, &stft, 22050.0, nzu!(80), &mfcc_params).unwrap();

    // Serialize to bincode
    let bytes = bincode2::serialize(&mfcc).unwrap();

    // Deserialize back
    let deserialized: Mfcc = bincode2::deserialize(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(mfcc.data.shape(), deserialized.data.shape());
    assert!(arrays_equal(&mfcc.data, &deserialized.data, 1e-10));
}

#[test]
fn test_chromagram_json() {
    let signal = test_signal(22050.0, 1.0);
    let stft = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::new(440.0, 55.0, 4200.0, ChromaNorm::None).unwrap();

    let chroma = chromagram(&signal, &stft, 22050.0, &chroma_params).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&chroma).unwrap();

    // Deserialize back
    let deserialized: Chromagram = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(chroma.data.shape(), deserialized.data.shape());
    assert!(arrays_equal(&chroma.data, &deserialized.data, 1e-10));
    assert_eq!(chroma.n_frames(), deserialized.n_frames());
    assert_eq!(chroma.n_bins(), deserialized.n_bins());
}

#[test]
fn test_chromagram_messagepack() {
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::new(440.0, 55.0, 8000.0, ChromaNorm::L2).unwrap();

    let chroma = chromagram(&signal, &stft, 16000.0, &chroma_params).unwrap();

    // Serialize to MessagePack
    let bytes = rmp_serde::to_vec(&chroma).unwrap();

    // Deserialize back
    let deserialized: Chromagram = rmp_serde::from_slice(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(chroma.data.shape(), deserialized.data.shape());
    assert!(arrays_equal(&chroma.data, &deserialized.data, 1e-10));
}

#[test]
fn test_cqt_result_json() {
    let signal = test_signal(16000.0, 0.5);
    let params = CqtParams::new(nzu!(12), nzu!(6), 55.0).unwrap();

    let cqt = cqt(&signal, 16000.0, &params, nzu!(256)).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&cqt).unwrap();

    // Deserialize back
    let deserialized: CqtResult = serde_json::from_str(&json).unwrap();

    // Verify data integrity
    assert_eq!(cqt.data.shape(), deserialized.data.shape());
    assert!(complex_arrays_equal(&cqt.data, &deserialized.data, 1e-10));
    assert_eq!(cqt.frequencies.len(), deserialized.frequencies.len());
    assert_eq!(cqt.sample_rate, deserialized.sample_rate);
    assert_eq!(cqt.hop_size, deserialized.hop_size);
}

#[test]
fn test_cqt_result_bincode() {
    let signal = test_signal(22050.0, 0.3);
    let params = CqtParams::new(nzu!(12), nzu!(7), 55.0).unwrap();

    let cqt = cqt(&signal, 22050.0, &params, nzu!(512)).unwrap();

    // Serialize to bincode
    let bytes = bincode2::serialize(&cqt).unwrap();

    // Deserialize back
    let deserialized: CqtResult = bincode2::deserialize(&bytes).unwrap();

    // Verify data integrity
    assert_eq!(cqt.data.shape(), deserialized.data.shape());
    assert!(complex_arrays_equal(&cqt.data, &deserialized.data, 1e-10));
}

#[test]
fn test_array_shape_preservation() {
    // Test that ndarray serialization preserves shape metadata
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    let spec = LinearPowerSpectrogram::compute(&signal, &params, None).unwrap();
    let original_bins = spec.n_bins();
    let original_frames = spec.n_frames();

    // Serialize and deserialize
    let json = serde_json::to_string(&spec).unwrap();
    let deserialized: LinearPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify shape is exactly preserved
    assert_eq!(original_bins, deserialized.n_bins());
    assert_eq!(original_frames, deserialized.n_frames());
    assert_eq!(spec.data().nrows(), deserialized.data().nrows());
    assert_eq!(spec.data().ncols(), deserialized.data().ncols());
}

#[test]
fn test_empty_edge_case() {
    // Test serialization of minimal data (very short signal)
    let signal = non_empty_vec![0.0; nzu!(256)]; // Minimal signal
    let stft = StftParams::new(nzu!(128), nzu!(64), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    let spec = LinearPowerSpectrogram::compute(&signal, &params, None).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();

    // Deserialize back
    let deserialized: LinearPowerSpectrogram = serde_json::from_str(&json).unwrap();

    // Verify
    assert_eq!(spec.n_bins(), deserialized.n_bins());
    assert_eq!(spec.n_frames(), deserialized.n_frames());
    assert!(arrays_equal(spec.data(), deserialized.data(), 1e-10));
}

#[test]
fn test_format_size_comparison() {
    // Compare serialization sizes for different formats
    let signal = test_signal(16000.0, 1.0);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    let spec = LinearPowerSpectrogram::compute(&signal, &params, None).unwrap();

    // JSON
    let json = serde_json::to_string(&spec).unwrap();
    let json_size = json.len();

    // Bincode
    let bincode_bytes = bincode2::serialize(&spec).unwrap();
    let bincode_size = bincode_bytes.len();

    // MessagePack
    let msgpack_bytes = rmp_serde::to_vec(&spec).unwrap();
    let msgpack_size = msgpack_bytes.len();

    // Bincode should be smallest, JSON should be largest
    assert!(bincode_size < json_size);
    assert!(msgpack_size < json_size);

    // Print for informational purposes (visible with --nocapture)
    println!("JSON size: {} bytes", json_size);
    println!("Bincode size: {} bytes", bincode_size);
    println!("MessagePack size: {} bytes", msgpack_size);
}

#[test]
fn test_complex_number_serialization() {
    // Verify Complex<f64> serializes correctly
    let signal = test_signal(16000.0, 0.5);
    let params = CqtParams::new(nzu!(12), nzu!(6), 55.0).unwrap();

    let cqt = cqt(&signal, 16000.0, &params, nzu!(256)).unwrap();

    // Get a sample complex value
    let sample_value = cqt.data[[0, 0]];

    // Serialize to JSON
    let json = serde_json::to_string(&cqt).unwrap();

    // JSON should contain complex number data (format may vary, so we don't check specifics)
    // The important part is that deserialization works correctly

    // Deserialize and verify the complex value
    let deserialized: CqtResult = serde_json::from_str(&json).unwrap();
    let deserialized_value = deserialized.data[[0, 0]];

    assert!((sample_value.re - deserialized_value.re).abs() < 1e-10);
    assert!((sample_value.im - deserialized_value.im).abs() < 1e-10);
}

#[test]
fn test_all_frequency_scales_roundtrip() {
    // Test that all frequency scale markers serialize correctly
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let base_params = SpectrogramParams::new(stft, 16000.0).unwrap();

    // Linear
    let linear = LinearPowerSpectrogram::compute(&signal, &base_params, None).unwrap();
    let linear_json = serde_json::to_string(&linear).unwrap();
    let _: LinearPowerSpectrogram = serde_json::from_str(&linear_json).unwrap();

    // Mel
    let mel_params = MelParams::new(nzu!(40), 0.0, 8000.0).unwrap();
    let mel = MelPowerSpectrogram::compute(&signal, &base_params, &mel_params, None).unwrap();
    let mel_json = serde_json::to_string(&mel).unwrap();
    let _: MelPowerSpectrogram = serde_json::from_str(&mel_json).unwrap();

    // ERB
    let erb_params = ErbParams::new(nzu!(30), 50.0, 8000.0).unwrap();
    let erb = ErbPowerSpectrogram::compute(&signal, &base_params, &erb_params, None).unwrap();
    let erb_json = serde_json::to_string(&erb).unwrap();
    let _: ErbPowerSpectrogram = serde_json::from_str(&erb_json).unwrap();

    // LogHz
    let log_params = LogHzParams::new(nzu!(50), 50.0, 8000.0).unwrap();
    let log = LogHzPowerSpectrogram::compute(&signal, &base_params, &log_params, None).unwrap();
    let log_json = serde_json::to_string(&log).unwrap();
    let _: LogHzPowerSpectrogram = serde_json::from_str(&log_json).unwrap();

    // CQT
    let cqt_params = CqtParams::new(nzu!(12), nzu!(5), 55.0).unwrap();
    let cqt = CqtPowerSpectrogram::compute(&signal, &base_params, &cqt_params, None).unwrap();
    let cqt_json = serde_json::to_string(&cqt).unwrap();
    let _: CqtPowerSpectrogram = serde_json::from_str(&cqt_json).unwrap();
}

#[test]
fn test_all_amplitude_scales_roundtrip() {
    // Test that all amplitude scale markers serialize correctly
    let signal = test_signal(16000.0, 0.5);
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, 16000.0).unwrap();

    // Power
    let power = LinearPowerSpectrogram::compute(&signal, &params, None).unwrap();
    let power_json = serde_json::to_string(&power).unwrap();
    let _: LinearPowerSpectrogram = serde_json::from_str(&power_json).unwrap();

    // Magnitude
    let mag = LinearMagnitudeSpectrogram::compute(&signal, &params, None).unwrap();
    let mag_json = serde_json::to_string(&mag).unwrap();
    let _: LinearMagnitudeSpectrogram = serde_json::from_str(&mag_json).unwrap();

    // Decibels
    let log_params = LogParams::new(-80.0).unwrap();
    let db = LinearDbSpectrogram::compute(&signal, &params, Some(&log_params)).unwrap();
    let db_json = serde_json::to_string(&db).unwrap();
    let _: LinearDbSpectrogram = serde_json::from_str(&db_json).unwrap();
}
