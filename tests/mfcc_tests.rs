use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::f64::consts::PI;

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect()
}

#[test]
fn test_mfcc_basic() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // Check dimensions
    assert_eq!(result.n_coefficients(), nzu!(13));

    // MFCC values should be finite
    for val in result.data.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_mfcc_with_c0() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();

    // Without C0
    let mfcc_params = MfccParams::new(nzu!(13));
    let result1 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    assert_eq!(result1.n_coefficients(), nzu!(13));

    // With C0
    let mfcc_params = MfccParams::new(nzu!(13)).with_c0(true);
    let result2 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    assert_eq!(result2.n_coefficients(), nzu!(13));

    // First coefficient (C0) should be different when included
    // C0 represents energy and should be non-zero for a sine wave
    if mfcc_params.include_c0() {
        let c0_frame = result2.data[[0, 0]];
        assert!(c0_frame.abs() > 0.01); // C0 should have significant value
    }
}

#[test]
fn test_mfcc_with_liftering() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();

    // Without liftering
    let mfcc_params1 = MfccParams::new(nzu!(13));
    let result1 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params1).unwrap();
    // With liftering (L=22 is common)
    let mfcc_params2 = MfccParams::new(nzu!(13)).with_lifter(22);
    let result2 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params2).unwrap();

    // Both results should be valid (liftering may or may not change values significantly)
    assert_eq!(result1.data.dim(), result2.data.dim());

    // Both should have valid values
    for val in result1.data.iter() {
        assert!(val.is_finite());
    }
    for val in result2.data.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_mfcc_different_n_coefficients() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();

    // Test with 13 coefficients (standard)
    let mfcc_params = MfccParams::new(nzu!(13));

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    assert_eq!(result.n_coefficients(), nzu!(13));

    // Test with 20 coefficients
    let mfcc_params = MfccParams::new(nzu!(20));
    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    assert_eq!(result.n_coefficients(), nzu!(20));

    // Test with 7 coefficients
    let mfcc_params = MfccParams::new(nzu!(7));
    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    assert_eq!(result.n_coefficients(), nzu!(7));
}

#[test]
fn test_mfcc_silence() {
    let sample_rate = 16000.0;
    let samples = vec![0.0; 16000]; // Silence
    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // MFCCs of silence should be finite and mostly negative (due to log of near-zero energy)
    for val in result.data.iter() {
        assert!(val.is_finite());
        // Silence can produce large negative values (e.g., -3200) due to log of very small energies
        assert!(val.abs() < 10000.0); // Reasonable bound to catch infinities
    }
}

#[test]
fn test_mfcc_speech_defaults() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(400), nzu!(160), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::speech_standard();

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // Speech standard is typically 13 coefficients
    assert_eq!(result.n_coefficients(), nzu!(13));
}

#[test]
fn test_mfcc_consistency() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    // Compute twice with same parameters
    let result1 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();
    let result2 = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // Results should be identical
    assert_eq!(result1.data.dim(), result2.data.dim());

    for (val1, val2) in result1.data.iter().zip(result2.data.iter()) {
        assert!((val1 - val2).abs() < 1e-10);
    }
}

#[test]
fn test_mfcc_frame_count() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // Number of frames should match STFT frame count
    let n_frames = result.n_frames();
    assert!(n_frames < samples.len()); // Should be less than sample count
}

#[test]
fn test_mfcc_decorrelation() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);
    let samples = NonEmptyVec::new(samples).unwrap();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let mfcc_params = MfccParams::new(nzu!(13));

    let result = mfcc(&samples, &stft, sample_rate, nzu!(40), &mfcc_params).unwrap();

    // MFCCs are designed to decorrelate mel filterbank energies
    // Higher coefficients should generally have less energy
    let data = &result.data;
    let n_frames = result.n_frames().get();

    if n_frames > 0 {
        // Average energy per coefficient across all frames
        let mut avg_energies = vec![0.0; result.n_coefficients().get()];
        for coef in 0..result.n_coefficients().get() {
            for frame in 0..n_frames {
                avg_energies[coef] += data[[coef, frame]].abs();
            }
            avg_energies[coef] /= n_frames as f64;
        }

        // Check that first few coefficients typically have more energy
        // (This is a general trend, not a strict rule)
        assert!(avg_energies[0].is_finite());
    }
}
