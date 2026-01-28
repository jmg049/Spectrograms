//! Benchmarks for 2D FFT operations
//!
//! Run with: cargo bench --bench fft2d_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use spectrograms::fft2d::*;
use std::hint::black_box;

/// Benchmark 2D FFT at various image sizes
fn bench_fft2d_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d_sizes");

    // Test a range of common image sizes
    let sizes = vec![
        (32, 32, "32x32"),
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (1024, 1024, "1024x1024"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        // Set throughput to number of elements
        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = fft2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark inverse 2D FFT (IFFT) at various sizes
fn bench_ifft2d_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft2d_sizes");

    let sizes = vec![
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        // Create spectrum by doing forward FFT
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });
        let spectrum = fft2d(&data.view()).unwrap();

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &spectrum, |b, spec| {
            b.iter(|| {
                let result = ifft2d(black_box(spec), ncols).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark power spectrum computation
fn bench_power_spectrum_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_spectrum_2d");

    let sizes = vec![
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = power_spectrum_2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark magnitude spectrum computation
fn bench_magnitude_spectrum_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("magnitude_spectrum_2d");

    let sizes = vec![
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = magnitude_spectrum_2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark fftshift operation
fn bench_fftshift(c: &mut Criterion) {
    let mut group = c.benchmark_group("fftshift");

    let sizes = vec![
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| (i * nrows + j) as f64);

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = fftshift(black_box(data.clone()));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark roundtrip (FFT -> IFFT)
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d_roundtrip");

    let sizes = vec![
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let spectrum = fft2d(black_box(&data.view())).unwrap();
                let reconstructed = ifft2d(&spectrum, ncols).unwrap();
                black_box(reconstructed);
            });
        });
    }

    group.finish();
}

/// Benchmark non-power-of-2 sizes
fn bench_non_power_of_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d_non_power_of_2");

    let sizes = vec![
        (100, 100, "100x100"),
        (150, 150, "150x150"),
        (200, 200, "200x200"),
        (300, 300, "300x300"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = fft2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark rectangular (non-square) images
fn bench_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d_rectangular");

    let sizes = vec![
        (128, 256, "128x256"),
        (256, 128, "256x128"),
        (256, 512, "256x512"),
        (512, 256, "512x256"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let result = fft2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fft2d_sizes,
    bench_ifft2d_sizes,
    bench_power_spectrum_2d,
    bench_magnitude_spectrum_2d,
    bench_fftshift,
    bench_roundtrip,
    bench_non_power_of_2,
    bench_rectangular,
);
criterion_main!(benches);
