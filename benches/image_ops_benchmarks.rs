//! Benchmarks for image processing operations
//!
//! Run with: cargo bench --bench image_ops_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use spectrograms::{image_ops::*, nzu};
use std::hint::black_box;

/// Benchmark Gaussian kernel generation
fn bench_gaussian_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_kernel_2d");

    let sizes = vec![
        (nzu!(3), 1.0, "3x3_sigma1.0"),
        (nzu!(5), 1.0, "5x5_sigma1.0"),
        (nzu!(7), 2.0, "7x7_sigma2.0"),
        (nzu!(9), 2.0, "9x9_sigma2.0"),
        (nzu!(15), 3.0, "15x15_sigma3.0"),
        (nzu!(21), 3.0, "21x21_sigma3.0"),
    ];

    for (size, sigma, label) in sizes {
        group.throughput(Throughput::Elements((size.get() * size.get()) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(size, sigma),
            |b, &(s, sig)| {
                b.iter(|| {
                    let result = gaussian_kernel_2d(black_box(s), black_box(sig)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FFT-based convolution with different kernel sizes
fn bench_convolve_fft_kernel_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve_fft_kernel_sizes");

    let image_size = 256;
    let image = Array2::<f64>::from_shape_fn((image_size, image_size), |(i, j)| {
        ((i as f64 - 128.0).powi(2) + (j as f64 - 128.0).powi(2)).sqrt()
    });

    let kernel_sizes = vec![
        (nzu!(3), "3x3"),
        (nzu!(5), "5x5"),
        (nzu!(7), "7x7"),
        (nzu!(9), "9x9"),
        (nzu!(15), "15x15"),
        (nzu!(21), "21x21"),
        (nzu!(31), "31x31"),
    ];

    for (kernel_size, label) in kernel_sizes {
        let kernel = gaussian_kernel_2d(kernel_size, 2.0).unwrap();

        group.throughput(Throughput::Elements((image_size * image_size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&image, &kernel),
            |b, &(img, kern)| {
                b.iter(|| {
                    let result =
                        convolve_fft(black_box(&img.view()), black_box(&kern.view())).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FFT-based convolution at different image sizes
fn bench_convolve_fft_image_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve_fft_image_sizes");

    let kernel = gaussian_kernel_2d(nzu!(9), 2.0).unwrap();

    let sizes = vec![
        (64, "64x64"),
        (128, "128x128"),
        (256, "256x256"),
        (512, "512x512"),
    ];

    for (size, label) in sizes {
        let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i as f64 - size as f64 / 2.0).powi(2) + (j as f64 - size as f64 / 2.0).powi(2)).sqrt()
        });

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&image, &kernel),
            |b, &(img, kern)| {
                b.iter(|| {
                    let result =
                        convolve_fft(black_box(&img.view()), black_box(&kern.view())).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark low-pass filtering
fn bench_lowpass_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("lowpass_filter");

    let sizes = vec![(128, "128x128"), (256, "256x256"), (512, "512x512")];

    let cutoff = 0.3;

    for (size, label) in sizes {
        let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &image, |b, img| {
            b.iter(|| {
                let result = lowpass_filter(black_box(&img.view()), cutoff).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark high-pass filtering
fn bench_highpass_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("highpass_filter");

    let sizes = vec![(128, "128x128"), (256, "256x256"), (512, "512x512")];

    let cutoff = 0.2;

    for (size, label) in sizes {
        let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
        });

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &image, |b, img| {
            b.iter(|| {
                let result = highpass_filter(black_box(&img.view()), cutoff).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark band-pass filtering
fn bench_bandpass_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandpass_filter");

    let sizes = vec![(128, "128x128"), (256, "256x256"), (512, "512x512")];

    let low_cutoff = 0.1;
    let high_cutoff = 0.4;

    for (size, label) in sizes {
        let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos() + 10.0
        });

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &image, |b, img| {
            b.iter(|| {
                let result =
                    bandpass_filter(black_box(&img.view()), low_cutoff, high_cutoff).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark edge detection
fn bench_detect_edges_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_edges_fft");

    let sizes = vec![(128, "128x128"), (256, "256x256"), (512, "512x512")];

    for (size, label) in sizes {
        // Create image with edges
        let mut image = Array2::<f64>::zeros((size, size));
        for i in (size / 4)..(3 * size / 4) {
            for j in (size / 4)..(3 * size / 4) {
                image[[i, j]] = 1.0;
            }
        }

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &image, |b, img| {
            b.iter(|| {
                let result = detect_edges_fft(black_box(&img.view())).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark image sharpening
fn bench_sharpen_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpen_fft");

    let sizes = vec![(128, "128x128"), (256, "256x256"), (512, "512x512")];

    let amount = 1.5;

    for (size, label) in sizes {
        let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i as f64 - size as f64 / 2.0).powi(2) + (j as f64 - size as f64 / 2.0).powi(2)).sqrt()
        });

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &image, |b, img| {
            b.iter(|| {
                let result = sharpen_fft(black_box(&img.view()), amount).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark complete image processing pipeline
fn bench_processing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("processing_pipeline");

    let size = 256;
    let image = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
        let r = ((i as f64 - 128.0).powi(2) + (j as f64 - 128.0).powi(2)).sqrt();
        (-r / 20.0).exp() * 100.0 + (i as f64 * 0.1).sin() * 10.0
    });

    group.throughput(Throughput::Elements((size * size) as u64));

    group.bench_function("denoise_detect_sharpen", |b| {
        b.iter(|| {
            // Step 1: Denoise with low-pass
            let denoised = lowpass_filter(black_box(&image.view()), 0.4).unwrap();

            // Step 2: Detect edges
            let edges = detect_edges_fft(black_box(&denoised.view())).unwrap();

            // Step 3: Sharpen original
            let sharpened = sharpen_fft(black_box(&image.view()), 1.0).unwrap();

            black_box((edges, sharpened));
        });
    });

    group.finish();
}

/// Benchmark different cutoff values for low-pass filter
fn bench_lowpass_cutoffs(c: &mut Criterion) {
    let mut group = c.benchmark_group("lowpass_filter_cutoffs");

    let image = Array2::<f64>::from_shape_fn((256, 256), |(i, j)| {
        (i as f64 * 0.1).sin() + (j as f64 * 0.2).cos()
    });

    let cutoffs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    for cutoff in cutoffs {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cutoff_{:.1}", cutoff)),
            &cutoff,
            |b, &c| {
                b.iter(|| {
                    let result = lowpass_filter(black_box(&image.view()), c).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian_kernel,
    bench_convolve_fft_kernel_sizes,
    bench_convolve_fft_image_sizes,
    bench_lowpass_filter,
    bench_highpass_filter,
    bench_bandpass_filter,
    bench_detect_edges_fft,
    bench_sharpen_fft,
    bench_processing_pipeline,
    bench_lowpass_cutoffs,
);
criterion_main!(benches);
