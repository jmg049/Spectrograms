//! Benchmarks comparing plan reuse vs one-shot operations
//!
//! These benchmarks demonstrate the performance benefits of using Fft2dPlanner
//! for batch processing compared to one-shot function calls.
//!
//! Run with: cargo bench --bench planner_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use spectrograms::fft2d::*;
use std::hint::black_box;

/// Benchmark: One-shot FFT2D (creates new plan each time)
fn bench_oneshot_fft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneshot_vs_planner/fft2d");

    let sizes = vec![
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::from_shape_fn((nrows, ncols), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((nrows * ncols) as u64));

        // One-shot: creates plan every time
        group.bench_with_input(BenchmarkId::new("oneshot", label), &data, |b, data| {
            b.iter(|| {
                let result = fft2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });

        // Planner: reuses plan
        group.bench_with_input(BenchmarkId::new("planner", label), &data, |b, data| {
            let mut planner = Fft2dPlanner::new();
            b.iter(|| {
                let result = planner.fft2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Power spectrum computation (one-shot vs planner)
fn bench_oneshot_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneshot_vs_planner/power_spectrum");

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

        // One-shot
        group.bench_with_input(BenchmarkId::new("oneshot", label), &data, |b, data| {
            b.iter(|| {
                let result = power_spectrum_2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });

        // Planner
        group.bench_with_input(BenchmarkId::new("planner", label), &data, |b, data| {
            let mut planner = Fft2dPlanner::new();
            b.iter(|| {
                let result = planner.power_spectrum_2d(black_box(&data.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Batch processing (10 images)
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let n_images = 10;
    let size = 256;

    // Generate batch of images
    let images: Vec<Array2<f64>> = (0..n_images)
        .map(|i| {
            Array2::<f64>::from_shape_fn((size, size), |(row, col)| {
                (row as f64 * 0.1 * (i + 1) as f64).sin()
                    + (col as f64 * 0.2 * (i + 1) as f64).cos()
            })
        })
        .collect();

    group.throughput(Throughput::Elements((n_images * size * size) as u64));

    // One-shot: creates plan for each image
    group.bench_function("oneshot_10_images", |b| {
        b.iter(|| {
            for img in &images {
                let result = fft2d(black_box(&img.view())).unwrap();
                black_box(result);
            }
        });
    });

    // Planner: reuses single plan for all images
    group.bench_function("planner_10_images", |b| {
        b.iter(|| {
            let mut planner = Fft2dPlanner::new();
            for img in &images {
                let result = planner.fft2d(black_box(&img.view())).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark: Batch processing with multiple sizes
fn bench_batch_mixed_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing_mixed_sizes");

    // Create images of different sizes
    let images = vec![
        Array2::<f64>::from_shape_fn((64, 64), |(i, _j)| (i as f64).sin()),
        Array2::<f64>::from_shape_fn((128, 128), |(i, _j)| (i as f64).sin()),
        Array2::<f64>::from_shape_fn((256, 256), |(i, _j)| (i as f64).sin()),
        Array2::<f64>::from_shape_fn((64, 64), |(_i, j)| (j as f64).cos()),
        Array2::<f64>::from_shape_fn((128, 128), |(_i, j)| (j as f64).cos()),
        Array2::<f64>::from_shape_fn((256, 256), |(_i, j)| (j as f64).cos()),
    ];

    let total_elements: usize = images.iter().map(|img| img.len()).sum();
    group.throughput(Throughput::Elements(total_elements as u64));

    // One-shot
    group.bench_function("oneshot_mixed", |b| {
        b.iter(|| {
            for img in &images {
                let result = fft2d(black_box(&img.view())).unwrap();
                black_box(result);
            }
        });
    });

    // Planner (should handle multiple sizes efficiently)
    group.bench_function("planner_mixed", |b| {
        b.iter(|| {
            let mut planner = Fft2dPlanner::new();
            for img in &images {
                let result = planner.fft2d(black_box(&img.view())).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark: Roundtrip processing (FFT -> IFFT) with planner
fn bench_roundtrip_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_planner");

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

        // One-shot roundtrip
        group.bench_with_input(BenchmarkId::new("oneshot", label), &data, |b, data| {
            b.iter(|| {
                let spectrum = fft2d(black_box(&data.view())).unwrap();
                let reconstructed = ifft2d(&spectrum, ncols).unwrap();
                black_box(reconstructed);
            });
        });

        // Planner roundtrip
        group.bench_with_input(BenchmarkId::new("planner", label), &data, |b, data| {
            let mut planner = Fft2dPlanner::new();
            b.iter(|| {
                let spectrum = planner.fft2d(black_box(&data.view())).unwrap();
                let reconstructed = planner.ifft2d(&spectrum.view(), ncols).unwrap();
                black_box(reconstructed);
            });
        });
    }

    group.finish();
}

/// Benchmark: Same-size batch processing (realistic video frame scenario)
fn bench_video_frames(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_frames");

    // Simulate 30 frames of 512x512 video
    let n_frames = 30;
    let size = 512;

    let frames: Vec<Array2<f64>> = (0..n_frames)
        .map(|i| {
            Array2::<f64>::from_shape_fn((size, size), |(row, col)| {
                // Simulate changing content across frames
                ((row as f64 + i as f64 * 5.0) * 0.1).sin()
                    + ((col as f64 + i as f64 * 3.0) * 0.2).cos()
            })
        })
        .collect();

    group.throughput(Throughput::Elements((n_frames * size * size) as u64));

    // One-shot: typical naive approach
    group.bench_function("oneshot_30_frames", |b| {
        b.iter(|| {
            for frame in &frames {
                let spectrum = fft2d(black_box(&frame.view())).unwrap();
                let power = spectrum.mapv(|c| c.norm_sqr());
                black_box(power);
            }
        });
    });

    // Planner: optimized approach
    group.bench_function("planner_30_frames", |b| {
        b.iter(|| {
            let mut planner = Fft2dPlanner::new();
            for frame in &frames {
                let power = planner.power_spectrum_2d(black_box(&frame.view())).unwrap();
                black_box(power);
            }
        });
    });

    group.finish();
}

/// Benchmark: Plan creation overhead
fn bench_plan_creation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_creation_overhead");

    let sizes = vec![
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    for (nrows, ncols, label) in sizes {
        let data = Array2::<f64>::zeros((nrows, ncols));

        // Measure time for single FFT with plan creation
        group.bench_with_input(
            BenchmarkId::new("with_plan_creation", label),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = fft2d(black_box(&data.view())).unwrap();
                    black_box(result);
                });
            },
        );

        // Measure time for FFT with pre-existing plan (amortized)
        group.bench_with_input(
            BenchmarkId::new("plan_already_cached", label),
            &data,
            |b, data| {
                let mut planner = Fft2dPlanner::new();
                // Warm up: create plan
                let _ = planner.fft2d(&data.view()).unwrap();

                b.iter(|| {
                    let result = planner.fft2d(black_box(&data.view())).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_oneshot_fft2d,
    bench_oneshot_power_spectrum,
    bench_batch_processing,
    bench_batch_mixed_sizes,
    bench_roundtrip_planner,
    bench_video_frames,
    bench_plan_creation_overhead,
);
criterion_main!(benches);
