/// Test to verify that CQT doesn't suffer from double-windowing.
///
/// Before the fix, CQT would apply both the STFT window AND the CQT kernel window,
/// resulting in incorrect frequency response. After the fix, CQT receives unwindowed
/// frames and only the CQT kernel windowing is applied.
///
/// This test verifies that changing the STFT window type has minimal impact on
/// CQT results, since the STFT window is now bypassed for CQT.
use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::f64::consts::PI;

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> NonEmptyVec<f64> {
    let v = (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect();
    NonEmptyVec::new(v).unwrap()
}

#[test]
fn test_cqt_stft_window_independence() {
    // Test that CQT results are independent of STFT window choice,
    // proving that the STFT window is properly bypassed for CQT.

    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;
    let frequency = 440.0; // A4

    let samples = sine_wave(frequency, sample_rate, n_samples);

    // Create CQT parameters
    let cqt_params = CqtParams::new(nzu!(12), nzu!(5), 100.0).unwrap();

    // Test with different STFT window types
    // If CQT is correctly implemented, the results should be nearly identical
    // regardless of STFT window choice, since CQT uses its own windowing.

    let window_types = vec![
        WindowType::Rectangular,
        WindowType::Hanning,
        WindowType::Hamming,
        WindowType::Blackman,
    ];

    let mut spectrograms = Vec::new();

    for window_type in &window_types {
        let stft = StftParams::new(nzu!(8192), nzu!(2048), window_type.clone(), false).unwrap();
        let params = SpectrogramParams::new(stft, sample_rate).unwrap();

        let planner = SpectrogramPlanner::new();
        let mut plan = planner
            .cqt_plan::<Power>(&params, &cqt_params, None)
            .unwrap();

        let spec = plan.compute(&samples).unwrap();
        spectrograms.push(spec);
    }

    // Compare all spectrograms - they should be nearly identical
    // since the STFT window should not affect CQT results
    let reference = &spectrograms[0];

    for (i, spec) in spectrograms.iter().enumerate().skip(1) {
        assert_eq!(spec.n_bins(), reference.n_bins());
        assert_eq!(spec.n_frames(), reference.n_frames());

        // Calculate relative difference
        let ref_data = reference.data();
        let spec_data = spec.data();

        let mut max_rel_diff: f64 = 0.0;
        let mut total_rel_diff: f64 = 0.0;
        let mut count: usize = 0;

        for bin in 0..spec.n_bins().get() {
            for frame in 0..spec.n_frames().get() {
                let ref_val = ref_data[[bin, frame]];
                let spec_val = spec_data[[bin, frame]];

                // Only compare where there's significant energy
                if ref_val > 1e-10 {
                    let rel_diff: f64 = ((spec_val - ref_val) / ref_val).abs();
                    max_rel_diff = max_rel_diff.max(rel_diff);
                    total_rel_diff += rel_diff;
                    count += 1;
                }
            }
        }

        let avg_rel_diff = if count > 0 {
            total_rel_diff / count as f64
        } else {
            0.0
        };

        // With proper implementation, different STFT windows should produce
        // nearly identical CQT results (< 1% average difference)
        assert!(
            avg_rel_diff < 0.01,
            "CQT with {} window differs from Rectangular by {:.2}% (max: {:.2}%)",
            window_types[i],
            avg_rel_diff * 100.0,
            max_rel_diff * 100.0
        );
    }
}

#[test]
fn test_cqt_energy_preservation() {
    // Test that CQT energy is reasonable (not doubled or halved due to windowing issues)

    let sample_rate = 16000.0;
    let n_samples = 16000;
    let frequency = 440.0;

    let samples = sine_wave(frequency, sample_rate, n_samples);

    // Compute input signal energy
    let input_energy: f64 = samples.iter().map(|x| x * x).sum();

    // Create CQT parameters
    let cqt_params = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    let stft = StftParams::new(nzu!(8192), nzu!(2048), WindowType::Hanning, false).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .cqt_plan::<Power>(&params, &cqt_params, None)
        .unwrap();

    let spec = plan.compute(&samples).unwrap();

    // Sum all CQT energy
    let cqt_energy: f64 = spec.data().iter().sum();

    // CQT energy should be proportional to input energy (not zero, not excessive)
    // This is a sanity check that windowing is applied correctly
    assert!(cqt_energy > 0.0, "CQT energy should be positive");

    // The ratio should be reasonable (accounting for time-frequency transform properties)
    // This ensures we're not doubling energy through double-windowing
    let energy_ratio = cqt_energy / input_energy;

    // Acceptable range based on transform properties and windowing
    assert!(
        energy_ratio > 0.01 && energy_ratio < 100.0,
        "Energy ratio {:.2} seems unreasonable (possible windowing issue)",
        energy_ratio
    );
}
