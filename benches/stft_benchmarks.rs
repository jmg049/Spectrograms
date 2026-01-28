//! Benchmarks for STFT (Short-Time Fourier Transform) operations
//!
//! Run with: cargo bench --bench stft_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::hint::black_box;

/// Benchmark STFT at various FFT sizes
fn bench_stft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_sizes");

    let sample_rate = 16000.0;
    let duration = 1.0; // 1 second
    let n_samples = (sample_rate * duration) as usize;

    // Generate test signal
    let signal: NonEmptyVec<f64> = NonEmptyVec::new(
        (0..n_samples)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / sample_rate).sin())
            .collect(),
    )
    .unwrap();

    let configs = vec![
        (nzu!(256), nzu!(128), "256_hop128"),
        (nzu!(512), nzu!(256), "512_hop256"),
        (nzu!(1024), nzu!(512), "1024_hop512"),
        (nzu!(2048), nzu!(1024), "2048_hop1024"),
    ];

    for (n_fft, hop_size, label) in configs {
        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, n_fft, hop_size),
            |b, &(sig, nfft, hop)| {
                b.iter(|| {
                    let result =
                        stft(black_box(sig), nfft, hop, WindowType::Hanning, true).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark STFT with different hop sizes (overlap)
fn bench_stft_hop_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_hop_sizes");

    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal: NonEmptyVec<f64> =
        NonEmptyVec::new((0..n_samples).map(|i| (i as f64 * 0.01).sin()).collect()).unwrap();

    let n_fft = nzu!(512);

    // Different hop sizes = different overlap percentages
    let hop_sizes = vec![
        (nzu!(512), "0%_overlap"),   // No overlap
        (nzu!(384), "25%_overlap"),  // 25% overlap
        (nzu!(256), "50%_overlap"),  // 50% overlap
        (nzu!(128), "75%_overlap"),  // 75% overlap
        (nzu!(64), "87.5%_overlap"), // 87.5% overlap
    ];

    for (hop_size, label) in hop_sizes {
        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, n_fft, hop_size),
            |b, &(sig, nfft, hop)| {
                b.iter(|| {
                    let result =
                        stft(black_box(sig), nfft, hop, WindowType::Hanning, true).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark STFT with different window functions
fn bench_stft_windows(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_window_functions");

    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal: NonEmptyVec<f64> =
        NonEmptyVec::new((0..n_samples).map(|i| (i as f64 * 0.01).sin()).collect()).unwrap();

    let windows = vec![
        (WindowType::Hanning, "Hanning"),
        (WindowType::Hamming, "Hamming"),
        (WindowType::Blackman, "Blackman"),
        (WindowType::Rectangular, "Rectangular"),
        (WindowType::Kaiser { beta: 8.6 }, "Kaiser_beta8.6"),
    ];

    for (window, label) in windows {
        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, window),
            |b, &(sig, ref win)| {
                b.iter(|| {
                    let result =
                        stft(black_box(sig), nzu!(512), nzu!(256), win.clone(), true).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ISTFT (inverse STFT)
fn bench_istft(c: &mut Criterion) {
    let mut group = c.benchmark_group("istft_sizes");

    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal: NonEmptyVec<f64> = NonEmptyVec::new(
        (0..n_samples)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / sample_rate).sin())
            .collect(),
    )
    .unwrap();

    let configs = vec![
        (nzu!(512), nzu!(256), "512_hop256"),
        (nzu!(1024), nzu!(512), "1024_hop512"),
        (nzu!(2048), nzu!(1024), "2048_hop1024"),
    ];

    for (n_fft, hop_size, label) in configs {
        // Compute STFT first
        let spectrum = stft(&signal, n_fft, hop_size, WindowType::Hanning, true).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&spectrum, n_fft, hop_size),
            |b, &(spec, nfft, hop)| {
                b.iter(|| {
                    let result =
                        istft(black_box(spec), nfft, hop, WindowType::Hanning, true).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark STFT roundtrip (STFT -> ISTFT)
fn bench_stft_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_roundtrip");

    let sample_rate = 16000.0;
    let duration = 1.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal: NonEmptyVec<f64> = NonEmptyVec::new(
        (0..n_samples)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / sample_rate).sin())
            .collect(),
    )
    .unwrap();

    let configs = vec![
        (nzu!(512), nzu!(256), "512_hop256"),
        (nzu!(1024), nzu!(512), "1024_hop512"),
        (nzu!(2048), nzu!(1024), "2048_hop1024"),
    ];

    for (n_fft, hop_size, label) in configs {
        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, n_fft, hop_size),
            |b, &(sig, nfft, hop)| {
                b.iter(|| {
                    let spectrum =
                        stft(black_box(sig), nfft, hop, WindowType::Hanning, true).unwrap();
                    let reconstructed =
                        istft(&spectrum, nfft, hop, WindowType::Hanning, true).unwrap();
                    black_box(reconstructed);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark STFT with different signal lengths
fn bench_stft_signal_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_signal_lengths");

    let sample_rate = 16000.0;

    // Different audio durations
    let durations = vec![
        (0.1, "100ms"),
        (0.5, "500ms"),
        (1.0, "1s"),
        (2.0, "2s"),
        (5.0, "5s"),
    ];

    for (duration, label) in durations {
        let n_samples = (sample_rate * duration) as usize;
        let signal: NonEmptyVec<f64> = NonEmptyVec::new(
            (0..n_samples)
                .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / sample_rate).sin())
                .collect(),
        )
        .unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result = stft(
                    black_box(sig),
                    nzu!(512),
                    nzu!(256),
                    WindowType::Hanning,
                    true,
                )
                .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark typical speech vs music STFT configurations
fn bench_stft_use_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_use_cases");

    // Speech: 16kHz, 512 FFT (32ms), 50% overlap
    let speech_signal: NonEmptyVec<f64> = NonEmptyVec::new(
        (0..16000)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 16000.0).sin())
            .collect(),
    )
    .unwrap();

    group.throughput(Throughput::Elements(16000));

    group.bench_function("speech_16kHz_512fft", |b| {
        b.iter(|| {
            let result = stft(
                black_box(&speech_signal),
                nzu!(512),
                nzu!(256),
                WindowType::Hanning,
                true,
            )
            .unwrap();
            black_box(result);
        });
    });

    // Music: 44.1kHz, 2048 FFT (46ms), 75% overlap
    let music_signal: NonEmptyVec<f64> = NonEmptyVec::new(
        (0..44100)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 440.0 / 44100.0).sin())
            .collect(),
    )
    .unwrap();

    group.throughput(Throughput::Elements(44100));

    group.bench_function("music_44.1kHz_2048fft", |b| {
        b.iter(|| {
            let result = stft(
                black_box(&music_signal),
                nzu!(2048),
                nzu!(512),
                WindowType::Hanning,
                true,
            )
            .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_stft_sizes,
    bench_stft_hop_sizes,
    bench_stft_windows,
    bench_istft,
    bench_stft_roundtrip,
    bench_stft_signal_lengths,
    bench_stft_use_cases,
);
criterion_main!(benches);
