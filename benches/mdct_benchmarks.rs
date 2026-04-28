//! Benchmarks for MDCT/IMDCT computation.
//!
//! Run with: cargo bench --bench mdct_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use spectrograms::*;
use std::hint::black_box;

/// Generate a test audio signal (mix of sinusoids)
fn generate_signal(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n_samples = (sample_rate * duration) as usize;
    (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (440.0 * 2.0 * std::f64::consts::PI * t).sin()
                + 0.5 * (880.0 * 2.0 * std::f64::consts::PI * t).sin()
                + 0.25 * (1760.0 * 2.0 * std::f64::consts::PI * t).sin()
        })
        .collect()
}

/// Benchmark forward MDCT at various window sizes (sine window, 50% hop)
fn bench_mdct_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_window_sizes");

    let signal = generate_signal(44100.0, 1.0);

    let configs = [
        (nzu!(256), "256"),
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (window_size, label) in configs {
        let params = MdctParams::sine_window(window_size).unwrap();
        group.throughput(Throughput::Elements(signal.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params),
            |b, &(sig, par)| {
                b.iter(|| black_box(compute_mdct(black_box(sig), par).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark inverse MDCT (IMDCT) at various window sizes
fn bench_imdct_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("imdct_window_sizes");

    let signal = generate_signal(44100.0, 1.0);

    let configs = [
        (nzu!(256), "256"),
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (window_size, label) in configs {
        let params = MdctParams::sine_window(window_size).unwrap();
        let coeffs = compute_mdct(&signal, &params).unwrap();
        group.throughput(Throughput::Elements(signal.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&coeffs, &params),
            |b, &(co, par)| {
                b.iter(|| black_box(compute_imdct(black_box(co), par, None).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark full MDCT → IMDCT roundtrip
fn bench_mdct_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_roundtrip");

    let signal = generate_signal(44100.0, 1.0);
    let orig_len = signal.len();

    let configs = [
        (nzu!(512), "512"),
        (nzu!(1024), "1024"),
        (nzu!(2048), "2048"),
        (nzu!(4096), "4096"),
    ];

    for (window_size, label) in configs {
        let params = MdctParams::sine_window(window_size).unwrap();
        group.throughput(Throughput::Elements(signal.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params),
            |b, &(sig, par)| {
                b.iter(|| {
                    let coeffs = compute_mdct(black_box(sig), par).unwrap();
                    black_box(compute_imdct(&coeffs, par, Some(orig_len)).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark effect of hop size on throughput (window_size=2048)
fn bench_mdct_hop_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_hop_sizes");

    let signal = generate_signal(44100.0, 1.0);
    let window_size = nzu!(2048);

    let hop_configs = [
        (nzu!(512), "25pct"),
        (nzu!(1024), "50pct"),
        (nzu!(1536), "75pct"),
    ];

    for (hop_size, label) in hop_configs {
        let params = MdctParams::new(window_size, hop_size, WindowType::Hanning).unwrap();
        group.throughput(Throughput::Elements(signal.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params),
            |b, &(sig, par)| {
                b.iter(|| black_box(compute_mdct(black_box(sig), par).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark different window types (sine vs Hanning vs Blackman), window_size=2048
fn bench_mdct_window_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_window_types");

    let signal = generate_signal(44100.0, 1.0);
    let window_size = nzu!(2048);
    let hop_size = nzu!(1024);

    group.throughput(Throughput::Elements(signal.len() as u64));

    // Sine window (perfect-reconstruction)
    let sine_params = MdctParams::sine_window(window_size).unwrap();
    group.bench_function("sine", |b| {
        b.iter(|| black_box(compute_mdct(black_box(&signal), &sine_params).unwrap()));
    });

    // Hanning window
    let hanning_params = MdctParams::new(window_size, hop_size, WindowType::Hanning).unwrap();
    group.bench_function("hanning", |b| {
        b.iter(|| black_box(compute_mdct(black_box(&signal), &hanning_params).unwrap()));
    });

    // Blackman window
    let blackman_params = MdctParams::new(window_size, hop_size, WindowType::Blackman).unwrap();
    group.bench_function("blackman", |b| {
        b.iter(|| black_box(compute_mdct(black_box(&signal), &blackman_params).unwrap()));
    });

    group.finish();
}

/// Benchmark at typical audio codec window sizes (AAC=2048, MP3=1152, Opus=480)
fn bench_mdct_codec_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_codec_sizes");

    let signal = generate_signal(44100.0, 1.0);

    // MP3: window_size=1152 (N=576 MDCT coefficients per frame)
    // AAC: window_size=2048 (N=1024)
    // Opus: window_size=480 (N=240)
    // Vorbis: window_size=2048 (N=1024)
    let codec_configs = [
        (nzu!(480), nzu!(240), "opus_480"),
        (nzu!(1152), nzu!(576), "mp3_1152"),
        (nzu!(2048), nzu!(1024), "aac_2048"),
    ];

    for (window_size, hop_size, label) in codec_configs {
        let params = MdctParams::sine_window(window_size).unwrap();
        group.throughput(Throughput::Elements(signal.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params),
            |b, &(sig, par)| {
                b.iter(|| black_box(compute_mdct(black_box(sig), par).unwrap()));
            },
        );
        let _ = hop_size; // used only for documentation
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mdct_window_sizes,
    bench_imdct_window_sizes,
    bench_mdct_roundtrip,
    bench_mdct_hop_sizes,
    bench_mdct_window_types,
    bench_mdct_codec_sizes,
);
criterion_main!(benches);
