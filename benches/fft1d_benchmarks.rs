//! Benchmarks for 1D FFT operations
//!
//! Run with: cargo bench --bench fft1d_benchmarks --no-default-features --features realfft
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use non_empty_slice::{NonEmptyVec, non_empty_vec};
use spectrograms::*;
use std::hint::black_box;

/// Benchmark real-to-complex FFT (fft) at various sizes
fn bench_fft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_sizes");

    // Common FFT sizes in audio processing
    let sizes = vec![
        (nzu!(128), "128"),
        (nzu!(256), "256"),
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
        (nzu!(8192), "8192"),
    ];

    for (n_fft, label) in sizes {
        let signal = non_empty_vec![0.5; n_fft];

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result = fft(black_box(sig), n_fft).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark inverse real FFT (irfft) at various sizes
fn bench_irfft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("irfft_sizes");

    let sizes = vec![
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (n_fft, label) in sizes {
        // Create spectrum by doing forward FFT
        let signal = non_empty_vec![0.5; n_fft];

        let spectrum = fft(&signal, n_fft).unwrap();

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &spectrum, |b, spec| {
            b.iter(|| {
                let result = irfft(
                    black_box(non_empty_slice::non_empty_slice!(spec.as_slice().unwrap())),
                    n_fft,
                )
                .unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark power spectrum computation
fn bench_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_spectrum");

    let sizes = vec![
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (n_fft, label) in sizes {
        let signal = non_empty_vec![0.5; n_fft];

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result =
                    power_spectrum(black_box(sig), n_fft, Some(WindowType::Hanning)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark magnitude spectrum computation
fn bench_magnitude_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("magnitude_spectrum");

    let sizes = vec![
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (n_fft, label) in sizes {
        let signal = non_empty_vec![0.5; n_fft];

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result =
                    magnitude_spectrum(black_box(sig), n_fft, Some(WindowType::Hanning)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark FFT roundtrip (fft -> irfft)
fn bench_fft_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_roundtrip");

    let sizes = vec![
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (n_fft, label) in sizes {
        let signal = non_empty_vec![0.5; n_fft];

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let spectrum = fft(black_box(sig), n_fft).unwrap();
                let reconstructed = irfft(
                    non_empty_slice::non_empty_slice!(spectrum.as_slice().unwrap()),
                    n_fft,
                )
                .unwrap();
                black_box(reconstructed);
            });
        });
    }

    group.finish();
}

/// Benchmark non-power-of-2 FFT sizes
fn bench_non_power_of_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_non_power_of_2");

    let sizes = vec![
        (nzu!(500), "500"),
        (nzu!(750), "750"),
        (nzu!(1000), "1000"),
        (nzu!(1500), "1500"),
        (nzu!(3000), "3000"),
    ];

    for (n_fft, label) in sizes {
        let signal = non_empty_vec![0.5; n_fft];
        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result = fft(black_box(sig), n_fft).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark FftPlanner for 1D FFT (plan reuse)
fn bench_fft_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_planner_vs_oneshot");

    let n_fft = nzu!(2048);
    let signal = non_empty_vec![0.5; n_fft];

    group.throughput(Throughput::Elements(n_fft.get() as u64));

    // One-shot: creates plan each time
    group.bench_function("oneshot", |b| {
        b.iter(|| {
            let result = fft(black_box(&signal), n_fft).unwrap();
            black_box(result);
        });
    });

    // Planner: reuses plan
    group.bench_function("planner", |b| {
        let mut planner = FftPlanner::new();
        b.iter(|| {
            let result = planner.fft(black_box(&signal), n_fft).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark batch processing with FftPlanner
fn bench_batch_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_fft_processing");

    let n_fft = nzu!(2048);
    let n_signals = 100;

    // Generate batch of signals
    let signals: Vec<NonEmptyVec<f64>> = (0..n_signals)
        .map(|i| {
            let v = (0..n_fft.get())
                .map(|j| ((i + j) as f64 * 0.01).sin())
                .collect();
            NonEmptyVec::new(v).unwrap()
        })
        .collect();

    group.throughput(Throughput::Elements((n_signals * n_fft.get()) as u64));

    // One-shot: creates plan for each signal
    group.bench_function("oneshot_100_signals", |b| {
        b.iter(|| {
            for sig in &signals {
                let result = fft(black_box(sig), n_fft).unwrap();
                black_box(result);
            }
        });
    });

    // Planner: reuses single plan
    group.bench_function("planner_100_signals", |b| {
        b.iter(|| {
            let mut planner = FftPlanner::new();
            for sig in &signals {
                let result = planner.fft(black_box(sig), n_fft).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark typical audio processing FFT sizes
fn bench_audio_fft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_fft_sizes");

    // Typical audio FFT sizes and their use cases
    let configs = vec![
        (nzu!(512), "512_speech"),     // 32ms at 16kHz
        (nzu!(1024), "1024_music"),    // 23ms at 44.1kHz
        (nzu!(2048), "2048_hq_music"), // 46ms at 44.1kHz
        (nzu!(4096), "4096_analysis"), // High-res frequency analysis
    ];

    for (n_fft, label) in configs {
        let signal = non_empty_vec![0.5; n_fft];

        group.throughput(Throughput::Elements(n_fft.get() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &signal, |b, sig| {
            b.iter(|| {
                let result =
                    power_spectrum(black_box(sig), n_fft, Some(WindowType::Hanning)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fft_sizes,
    bench_irfft_sizes,
    bench_power_spectrum,
    bench_magnitude_spectrum,
    bench_fft_roundtrip,
    bench_non_power_of_2,
    bench_fft_planner,
    bench_batch_fft,
    bench_audio_fft_sizes,
);
criterion_main!(benches);
