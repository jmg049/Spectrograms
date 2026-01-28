use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use spectrograms::{
    LinearPowerSpectrogram, SpectrogramParams, SpectrogramPlanner, WindowType, nzu,
};
use std::f64::consts::PI;

fn sine_wave(freq: f64, sample_rate: f64, n_samples: usize) -> NonEmptyVec<f64> {
    let v = (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect();
    NonEmptyVec::new(v).unwrap()
}

#[test]
fn test_compute_frame() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let params = SpectrogramParams::builder()
        .sample_rate(sample_rate)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .window(WindowType::Hanning)
        .centre(true)
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    // Compute single frame
    let frame = plan.compute_frame(&samples, 0).unwrap();

    // Check frame size (n_fft/2 + 1)
    assert_eq!(frame.len(), nzu!(257));

    // All values should be non-negative (power)
    for &val in &frame {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_compute_frame_multiple() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let params = SpectrogramParams::builder()
        .sample_rate(sample_rate)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    // Compute multiple frames
    let frame0 = plan.compute_frame(&samples, 0).unwrap();
    let frame1 = plan.compute_frame(&samples, 1).unwrap();

    // Frames should have same length
    assert_eq!(frame0.len(), frame1.len());

    // Frames should be different (different time windows)
    assert_ne!(frame0, frame1);
}

#[test]
fn test_compute_into() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let params = SpectrogramParams::builder()
        .sample_rate(sample_rate)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    // Get expected shape
    let (n_bins, n_frames) = plan.output_shape(samples.len()).unwrap();

    // Pre-allocate output
    let mut output = Array2::<f64>::zeros((n_bins.get(), n_frames.get()));

    // Compute into pre-allocated buffer
    plan.compute_into(&samples, &mut output).unwrap();

    // Check dimensions
    assert_eq!(output.nrows(), n_bins.get());
    assert_eq!(output.ncols(), n_frames.get());

    // Check that data was written
    let sum: f64 = output.iter().sum();
    assert!(sum > 0.0);
}

#[test]
fn test_compute_into_wrong_size() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let params = SpectrogramParams::builder()
        .sample_rate(sample_rate)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    // Wrong size buffer
    let mut output = Array2::<f64>::zeros((100, 50));

    // Should error
    let result = plan.compute_into(&samples, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_output_shape() {
    let params = SpectrogramParams::builder()
        .sample_rate(16000.0)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    let (n_bins, _) = plan.output_shape(nzu!(16000)).unwrap();

    // n_bins should be n_fft/2 + 1
    assert_eq!(n_bins, nzu!(257));
}

#[test]
fn test_compute_into_matches_compute() {
    let sample_rate = 16000.0;
    let samples = sine_wave(440.0, sample_rate, 16000);

    let params = SpectrogramParams::builder()
        .sample_rate(sample_rate)
        .n_fft(nzu!(512))
        .hop_size(nzu!(256))
        .build()
        .unwrap();

    let planner = SpectrogramPlanner::new();
    let mut plan = planner
        .linear_plan::<spectrograms::Power>(&params, None)
        .unwrap();

    // Compute using regular method
    let spec1 = LinearPowerSpectrogram::compute(&samples, &params, None).unwrap();

    // Compute using compute_into
    let (n_bins, n_frames) = plan.output_shape(samples.len()).unwrap();
    let mut output = Array2::<f64>::zeros((n_bins.get(), n_frames.get()));
    plan.compute_into(&samples, &mut output).unwrap();

    // Results should match
    assert_eq!(spec1.data().dim(), output.dim());

    // Check values are close
    for i in 0..n_bins.get() {
        for j in 0..n_frames.get() {
            let diff = (spec1.data()[[i, j]] - output[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Mismatch at ({}, {}): {} vs {}",
                i,
                j,
                spec1.data()[[i, j]],
                output[[i, j]]
            );
        }
    }
}
