use ndarray::Array2;
use non_empty_slice::non_empty_vec;
use num_complex::Complex;
use spectrograms::*;

#[test]
fn test_stft_plan_reuse() {
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let signals = non_empty_vec![
        non_empty_vec![0.0; nzu!(16000)],
        non_empty_vec![1.0; nzu!(16000)],
        non_empty_vec![0.5; nzu!(16000)]
    ];

    let mut plan = StftPlan::new(&params).unwrap();

    for signal in signals {
        let stft = plan.compute(&signal, &params).unwrap();
        assert_eq!(stft.n_bins(), nzu!(257));
    }
}

#[test]
fn test_stft_plan_frame_by_frame() {
    let samples = non_empty_vec![0.0; nzu!(16000)];
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let mut plan = StftPlan::new(&params).unwrap();

    let (n_bins, n_frames) = plan.output_shape(samples.len()).unwrap();

    for frame_idx in 0..n_frames.get() {
        let frame = plan.compute_frame_simple(&samples, frame_idx).unwrap();
        assert_eq!(frame.len(), n_bins);
    }
}

#[test]
fn test_stft_plan_compute_into() {
    let samples = non_empty_vec![0.0; nzu!(16000)];

    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let mut plan = StftPlan::new(&params).unwrap();

    let (n_bins, n_frames) = plan.output_shape(samples.len()).unwrap();
    let mut output = Array2::<Complex<f64>>::zeros((n_bins.get(), n_frames.get()));

    plan.compute_into(&samples, &mut output).unwrap();

    assert_eq!(output.nrows(), n_bins.get());
    assert_eq!(output.ncols(), n_frames.get());
}

#[test]
fn test_stft_plan_matches_compute_stft() {
    let samples = non_empty_vec![0.5; nzu!(16000)];
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let planner = SpectrogramPlanner::new();

    // One-shot API
    let stft1 = planner.compute_stft(&samples, &params).unwrap();

    // Reusable plan API
    let mut plan = StftPlan::new(&params).unwrap();
    let stft2 = plan.compute(&samples, &params).unwrap();

    // Both should produce identical results
    assert_eq!(stft1.data.shape(), stft2.data.shape());
    for i in 0..stft1.data.len() {
        let v1 = stft1.data.as_slice().unwrap()[i];
        let v2 = stft2.data.as_slice().unwrap()[i];
        assert!((v1.re - v2.re).abs() < 1e-10);
        assert!((v1.im - v2.im).abs() < 1e-10);
    }
}

#[test]
fn test_stft_plan_dimension_mismatch() {
    let samples = non_empty_vec![0.0; nzu!(16000)];
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let mut plan = StftPlan::new(&params).unwrap();

    let mut wrong_size = Array2::<Complex<f64>>::zeros((100, 50));

    let result = plan.compute_into(&samples, &mut wrong_size);
    assert!(result.is_err());
}

#[test]
fn test_stft_plan_multichannel() {
    // Test case: Process each channel of stereo audio separately
    let left_channel = non_empty_vec![0.5; nzu!(16000)];
    let right_channel = non_empty_vec![0.3; nzu!(16000)];

    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let mut plan = StftPlan::new(&params).unwrap();

    let stft_left = plan.compute(&left_channel, &params).unwrap();
    let stft_right = plan.compute(&right_channel, &params).unwrap();

    assert_eq!(stft_left.data.shape(), stft_right.data.shape());
    assert_eq!(stft_left.n_bins(), nzu!(257));
}

#[test]
fn test_stft_plan_getters() {
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let plan = StftPlan::new(&params).unwrap();

    assert_eq!(plan.n_fft(), nzu!(512));
    assert_eq!(plan.hop_size(), nzu!(256));
    assert_eq!(plan.n_bins(), nzu!(257));
}

#[test]
fn test_stft_plan_output_shape() {
    let stft_params = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft_params, 16000.0).unwrap();

    let plan = StftPlan::new(&params).unwrap();

    let (n_bins, _) = plan.output_shape(nzu!(16000)).unwrap();
    assert_eq!(n_bins, nzu!(257));
}
