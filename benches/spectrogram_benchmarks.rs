//! Benchmarks for spectrogram computation (all frequency scales and amplitude scales)
//!
//! Run with: cargo bench --bench spectrogram_benchmarks --no-default-features --features realfft

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::hint::black_box;

/// Helper to generate test audio signal
fn generate_signal(sample_rate: f64, duration: f64) -> NonEmptyVec<f64> {
    let n_samples = (sample_rate * duration) as usize;
    let v = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Mix of frequencies
            (440.0 * 2.0 * std::f64::consts::PI * t).sin()
                + 0.5 * (880.0 * 2.0 * std::f64::consts::PI * t).sin()
        })
        .collect();
    NonEmptyVec::new(v).unwrap()
}

/// Benchmark linear power spectrogram at different FFT sizes
fn bench_linear_power_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_power_spectrogram");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let configs = vec![
        (nzu!(256), nzu!(128), "256_hop128"),
        (nzu!(512), nzu!(256), "512_hop256"),
        (nzu!(1024), nzu!(512), "1024_hop512"),
        (nzu!(2048), nzu!(1024), "2048_hop1024"),
    ];

    for (n_fft, hop_size, label) in configs {
        let stft = StftParams::new(n_fft, hop_size, WindowType::Hanning, true).unwrap();
        let params = SpectrogramParams::new(stft, sample_rate).unwrap();

        group.throughput(Throughput::Elements(signal.len().get() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params),
            |b, &(sig, par)| {
                b.iter(|| {
                    let result =
                        LinearPowerSpectrogram::compute(black_box(sig), par, None).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different amplitude scales (power, magnitude, dB)
fn bench_amplitude_scales(c: &mut Criterion) {
    let mut group = c.benchmark_group("amplitude_scales");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    // Power
    group.bench_function("power", |b| {
        b.iter(|| {
            let result =
                LinearPowerSpectrogram::compute(black_box(&signal), &params, None).unwrap();
            black_box(result);
        });
    });

    // Magnitude
    group.bench_function("magnitude", |b| {
        b.iter(|| {
            let result =
                LinearMagnitudeSpectrogram::compute(black_box(&signal), &params, None).unwrap();
            black_box(result);
        });
    });

    // Decibels
    group.bench_function("decibels", |b| {
        let db_params = LogParams::new(-80.0).unwrap();
        b.iter(|| {
            let result =
                LinearDbSpectrogram::compute(black_box(&signal), &params, Some(&db_params))
                    .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark Mel-scale spectrograms
fn bench_mel_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_spectrogram");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    // Different numbers of mel bands
    let mel_configs = vec![
        (nzu!(40), "40_mels"),
        (nzu!(80), "80_mels"),
        (nzu!(128), "128_mels"),
    ];

    for (n_mels, label) in mel_configs {
        let mel_params = MelParams::new(n_mels, 0.0, sample_rate / 2.0).unwrap();

        group.throughput(Throughput::Elements(signal.len().get() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&signal, &params, &mel_params),
            |b, &(sig, par, mel)| {
                b.iter(|| {
                    let result =
                        MelPowerSpectrogram::compute(black_box(sig), par, mel, None).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ERB-scale spectrograms
fn bench_erb_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("erb_spectrogram");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let erb_params = ErbParams::new(nzu!(64), 50.0, sample_rate / 2.0).unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    group.bench_function("erb_64_filters", |b| {
        b.iter(|| {
            let result =
                ErbPowerSpectrogram::compute(black_box(&signal), &params, &erb_params, None)
                    .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark CQT (Constant-Q Transform)
fn bench_cqt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cqt");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 0.5); // Shorter for CQT (expensive)

    let stft = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let cqt_params = CqtParams::new(
        nzu!(12), // bins_per_octave
        nzu!(7),  // n_octaves
        32.7,     // f_min (C1)
    )
    .unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    group.bench_function("cqt_84_bins", |b| {
        b.iter(|| {
            let result =
                CqtPowerSpectrogram::compute(black_box(&signal), &params, &cqt_params, None)
                    .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark MFCC computation
fn bench_mfcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let mel_params = MelParams::new(nzu!(40), 0.0, sample_rate / 2.0).unwrap();

    // Different numbers of MFCC coefficients
    let mfcc_configs = vec![
        (nzu!(13), "13_coeffs"),
        (nzu!(20), "20_coeffs"),
        (nzu!(40), "40_coeffs"),
    ];

    for (n_mfcc, label) in mfcc_configs {
        let mfcc_params = MfccParams::new(n_mfcc);

        group.throughput(Throughput::Elements(signal.len().get() as u64));

        let n_mels = mel_params.n_mels();

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(
                &signal,
                params.stft(),
                params.sample_rate_hz(),
                n_mels,
                &mfcc_params,
            ),
            |b, &(sig, stft, sr, mels, mfcc_par)| {
                b.iter(|| {
                    let result = mfcc(black_box(sig), stft, sr, mels, mfcc_par).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark chromagram computation
fn bench_chromagram(c: &mut Criterion) {
    let mut group = c.benchmark_group("chromagram");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let chroma_params = ChromaParams::new(
        440.0,  // tuning (A4)
        32.7,   // f_min (C1)
        4186.0, // f_max (C8)
        ChromaNorm::L2,
    )
    .unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    group.bench_function("chromagram_12_bins", |b| {
        b.iter(|| {
            let result = chromagram(
                black_box(&signal),
                params.stft(),
                params.sample_rate_hz(),
                &chroma_params,
            )
            .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark plan reuse for spectrograms
fn bench_spectrogram_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram_planner");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let mel_params = MelParams::new(nzu!(80), 0.0, sample_rate / 2.0).unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    // One-shot computation
    group.bench_function("oneshot_mel", |b| {
        b.iter(|| {
            let result =
                MelPowerSpectrogram::compute(black_box(&signal), &params, &mel_params, None)
                    .unwrap();
            black_box(result);
        });
    });

    // Planner computation
    group.bench_function("planner_mel", |b| {
        let planner = SpectrogramPlanner::new();
        let mut plan = planner
            .mel_plan::<Power>(&params, &mel_params, None)
            .unwrap();

        b.iter(|| {
            let result = plan.compute(black_box(&signal)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark batch spectrogram computation
fn bench_batch_spectrograms(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_spectrograms");

    let sample_rate = 16000.0;
    let n_signals = 50;

    // Generate batch of signals
    let signals: Vec<NonEmptyVec<f64>> = (0..n_signals)
        .map(|_| generate_signal(sample_rate, 0.5))
        .collect();

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let mel_params = MelParams::new(nzu!(80), 0.0, sample_rate / 2.0).unwrap();

    let total_samples: usize = signals.iter().map(|s| s.len().get()).sum();
    group.throughput(Throughput::Elements(total_samples as u64));

    // One-shot: creates plan for each signal
    group.bench_function("oneshot_50_signals", |b| {
        b.iter(|| {
            for sig in &signals {
                let result =
                    MelPowerSpectrogram::compute(black_box(sig), &params, &mel_params, None)
                        .unwrap();
                black_box(result);
            }
        });
    });

    // Planner: reuses plan for all signals
    group.bench_function("planner_50_signals", |b| {
        b.iter(|| {
            let planner = SpectrogramPlanner::new();
            let mut plan = planner
                .mel_plan::<Power>(&params, &mel_params, None)
                .unwrap();

            for sig in &signals {
                let result = plan.compute(black_box(sig)).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark typical speech processing pipeline
fn bench_speech_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("speech_pipeline");

    let sample_rate = 16000.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true).unwrap(); // 10ms at 16kHz
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let mel_params = MelParams::new(nzu!(40), 0.0, 8000.0).unwrap();

    let mfcc_params = MfccParams::new(nzu!(13));

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    let n_mels = mel_params.n_mels();

    group.bench_function("mel_to_mfcc", |b| {
        b.iter(|| {
            let result = mfcc(
                black_box(&signal),
                params.stft(),
                params.sample_rate_hz(),
                n_mels,
                &mfcc_params,
            )
            .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark typical music processing pipeline
fn bench_music_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("music_pipeline");

    let sample_rate = 44100.0;
    let signal = generate_signal(sample_rate, 1.0);

    let stft = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true).unwrap();
    let params = SpectrogramParams::new(stft, sample_rate).unwrap();

    let chroma_params = ChromaParams::new(440.0, 32.7, 4186.0, ChromaNorm::L2).unwrap();

    group.throughput(Throughput::Elements(signal.len().get() as u64));

    group.bench_function("chromagram", |b| {
        b.iter(|| {
            let result = chromagram(
                black_box(&signal),
                params.stft(),
                params.sample_rate_hz(),
                &chroma_params,
            )
            .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_power_spectrogram,
    bench_amplitude_scales,
    bench_mel_spectrogram,
    bench_erb_spectrogram,
    bench_cqt,
    bench_mfcc,
    bench_chromagram,
    bench_spectrogram_planner,
    bench_batch_spectrograms,
    bench_speech_pipeline,
    bench_music_pipeline,
);
criterion_main!(benches);
