//! End-to-end smoke tests for the f32 instantiation of the precision-generic API.
//!
//! The crate defaults to f64; these tests lock in that the same public entry
//! points work when instantiated at `f32` (the precision the future no_std /
//! embedded build will use).

use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use spectrograms::{
    WindowType, fft, fft2d, fft_convolve, istft, make_window, minimum_phase, power_spectrum, rfft,
    stft,
};

fn nz(n: usize) -> std::num::NonZeroUsize {
    std::num::NonZeroUsize::new(n).unwrap()
}

#[test]
fn f32_window_is_finite() {
    let w = make_window::<f32>(WindowType::Hanning, nz(1024));
    assert_eq!(w.len().get(), 1024);
    assert!(w.as_slice().iter().all(|x: &f32| x.is_finite()));
    let sum: f32 = w.as_slice().iter().copied().sum();
    assert!(sum > 0.0);
}

#[test]
fn f32_power_spectrum_finds_tone() {
    // Pure tone with an exact 8-sample period -> bin 1024/8 = 128.
    let n_fft = 1024usize;
    let sig: Vec<f32> = (0..n_fft)
        .map(|i| (std::f32::consts::TAU * i as f32 / 8.0).sin())
        .collect();
    let sig = NonEmptyVec::new(sig).unwrap();

    let p = power_spectrum::<f32>(&sig, nz(n_fft), None).unwrap();
    assert!(p.as_slice().iter().all(|x: &f32| x.is_finite() && *x >= 0.0));

    let (argmax, _) = p
        .as_slice()
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    assert!(
        (argmax as isize - 128).abs() <= 1,
        "expected spectral peak near bin 128, got {argmax}"
    );
}

#[test]
fn f32_fft_rfft_finite() {
    let sig: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
    let sig = NonEmptyVec::new(sig).unwrap();

    let spec = fft::<f32>(&sig, nz(512)).unwrap();
    assert_eq!(spec.len(), 512 / 2 + 1); // real-FFT half spectrum
    assert!(spec.iter().all(|c| c.re.is_finite() && c.im.is_finite()));

    let r = rfft::<f32>(&sig, nz(512)).unwrap();
    assert!(r.iter().all(|x: &f32| x.is_finite()));
}

#[test]
fn f32_stft_istft_roundtrip_finite() {
    let sig: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.02).sin()).collect();
    let sig = NonEmptyVec::new(sig).unwrap();

    let s: Array2<spectrograms::Complex<f32>> =
        stft::<f32>(&sig, nz(256), nz(128), WindowType::Hanning, true).unwrap();
    assert_eq!(s.nrows(), 256 / 2 + 1);
    assert!(s.iter().all(|c| c.re.is_finite() && c.im.is_finite()));

    let recon = istft::<f32>(&s, nz(256), nz(128), WindowType::Hanning, true).unwrap();
    assert!(recon.as_slice().iter().all(|x: &f32| x.is_finite()));
}

#[test]
fn f32_convolve_with_impulse_is_identity() {
    let a = NonEmptyVec::new(vec![1.0f32, -2.0, 3.0, 0.5, 4.0]).unwrap();
    let impulse = NonEmptyVec::new(vec![1.0f32]).unwrap();

    let out = fft_convolve::<f32>(&a, &impulse).unwrap();
    assert_eq!(out.len().get(), a.len().get());
    for (o, e) in out.as_slice().iter().zip(a.as_slice().iter()) {
        assert!((o - e).abs() < 1e-4, "convolution with impulse should be identity");
    }
}

#[test]
fn f32_minimum_phase_finite() {
    let ir: Vec<f32> = (0..64).map(|i| (-(i as f32) * 0.1).exp()).collect();
    let mp = minimum_phase::<f32>(&ir).unwrap();
    assert!(!mp.is_empty());
    assert!(mp.iter().all(|x: &f32| x.is_finite()));
}

#[test]
fn f32_fft2d_finite() {
    let img = Array2::<f32>::from_shape_fn((16, 16), |(i, j)| (i as f32 - j as f32).sin());
    let spec = fft2d::<f32>(&img.view()).unwrap();
    assert_eq!(spec.dim(), (16, 16 / 2 + 1));
    assert!(spec.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
}
