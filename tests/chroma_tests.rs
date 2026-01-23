use spectrograms::*;
use std::f64::consts::PI;

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect()
}

#[test]
fn test_chromagram_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard().unwrap();

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    // Chromagram should have 12 pitch classes
    assert_eq!(result.n_bins(), 12);
    assert!(result.n_frames() > 0);

    // All values should be non-negative and finite
    for val in result.data.iter() {
        assert!(val.is_finite());
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_chromagram_a440() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard().unwrap();

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    // 440 Hz is A4, which corresponds to pitch class 9 (A) in standard tuning
    let data = &result.data;

    // Sum energy across all frames for each pitch class
    let mut pitch_energies = vec![0.0; 12];
    for pitch in 0..12 {
        for frame in 0..result.n_frames() {
            pitch_energies[pitch] += data[[pitch, frame]];
        }
    }

    // Find pitch class with maximum energy
    let max_pitch = pitch_energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // Should detect A (pitch class 9)
    assert_eq!(
        max_pitch, 9,
        "Expected A (9), got pitch class {}",
        max_pitch
    );
}

#[test]
fn test_chromagram_c_note() {
    let sample_rate = 16000.0;
    let c_freq = 261.63; // C4
    let samples = sine_wave(c_freq, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard().unwrap();

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    let data = &result.data;
    let mut pitch_energies = vec![0.0; 12];
    for pitch in 0..12 {
        for frame in 0..result.n_frames() {
            pitch_energies[pitch] += data[[pitch, frame]];
        }
    }

    let max_pitch = pitch_energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // Should detect C (pitch class 0)
    assert_eq!(
        max_pitch, 0,
        "Expected C (0), got pitch class {}",
        max_pitch
    );
}

#[test]
fn test_chroma_params_validation() {
    // Valid parameters
    assert!(ChromaParams::music_standard().is_ok());

    // Valid with custom tuning
    assert!(ChromaParams::new(442.0, 50.0, 8000.0, ChromaNorm::L2).is_ok());

    // Invalid: zero tuning frequency
    assert!(ChromaParams::new(0.0, 50.0, 8000.0, ChromaNorm::L2).is_err());

    // Invalid: negative tuning frequency
    assert!(ChromaParams::new(-440.0, 50.0, 8000.0, ChromaNorm::L2).is_err());

    // Invalid: f_min >= f_max
    assert!(ChromaParams::new(440.0, 1000.0, 500.0, ChromaNorm::L2).is_err());
}

#[test]
fn test_chroma_normalization_none() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard()
        .unwrap()
        .with_norm(ChromaNorm::None);

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    // Without normalization, values can be any non-negative number
    for val in result.data.iter() {
        assert!(val.is_finite());
        assert!(*val >= 0.0);
    }
}

#[test]
fn test_chroma_normalization_l1() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard()
        .unwrap()
        .with_norm(ChromaNorm::L1);

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    let data = &result.data;

    // Check L1 normalization: sum of each frame should be close to 1.0 (or 0 if all zeros)
    for frame in 0..result.n_frames() {
        let frame_sum: f64 = (0..12).map(|pitch| data[[pitch, frame]]).sum();

        if frame_sum > 1e-10 {
            // If frame has energy, sum should be close to 1.0
            assert!(
                (frame_sum - 1.0).abs() < 1e-6,
                "L1 norm should be 1.0, got {}",
                frame_sum
            );
        }
    }
}

#[test]
fn test_chroma_normalization_l2() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard()
        .unwrap()
        .with_norm(ChromaNorm::L2);

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    let data = &result.data;

    // Check L2 normalization: sqrt of sum of squares should be close to 1.0
    for frame in 0..result.n_frames() {
        let frame_energy: f64 = (0..12).map(|pitch| data[[pitch, frame]].powi(2)).sum();
        let frame_norm = frame_energy.sqrt();

        if frame_energy > 1e-10 {
            assert!(
                (frame_norm - 1.0).abs() < 1e-6,
                "L2 norm should be 1.0, got {}",
                frame_norm
            );
        }
    }
}

#[test]
fn test_chroma_normalization_max() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard()
        .unwrap()
        .with_norm(ChromaNorm::Max);

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    let data = &result.data;

    // Check Max normalization: maximum value in each frame should be 1.0
    for frame in 0..result.n_frames() {
        let frame_max = (0..12)
            .map(|pitch| data[[pitch, frame]])
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if frame_max > 1e-10 {
            assert!(
                (frame_max - 1.0).abs() < 1e-6,
                "Max value should be 1.0, got {}",
                frame_max
            );
        }
    }
}

#[test]
fn test_chroma_silence() {
    let sample_rate = 16000.0;
    let samples = vec![0.0; 16000];

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard().unwrap();

    let result = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    // Chromagram of silence should be all zeros (or very small values)
    for val in result.data.iter() {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_chroma_consistency() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let stft = StftParams::new(2048, 512, WindowType::Hanning, true).unwrap();
    let chroma_params = ChromaParams::music_standard().unwrap();

    // Compute twice with same parameters
    let result1 = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();
    let result2 = chromagram(&samples, &stft, sample_rate, &chroma_params).unwrap();

    // Results should be identical
    assert_eq!(result1.data.dim(), result2.data.dim());

    for (val1, val2) in result1.data.iter().zip(result2.data.iter()) {
        assert!((val1 - val2).abs() < 1e-10);
    }
}
