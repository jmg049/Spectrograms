//! MDCT comparison: our Rust implementation vs libvorbis MDCT via FFI.
//!
//! libvorbis exports mdct_init/mdct_forward/mdct_backward/mdct_clear as public
//! symbols. Their MDCT uses f32 and does NOT apply windowing (caller's
//! responsibility), so we include windowing time explicitly for a fair comparison.
//!
//! NOTE: libvorbis MDCT requires power-of-2 window sizes (it computes log2n via
//! round(log(n)/log(2)) and builds bit-reversal tables assuming n = 2^log2n).
//! Non-power-of-2 sizes (e.g. 480, 1152) cause out-of-bounds access and SIGSEGV.
//! Benchmark sizes are therefore chosen as powers of 2.
//!
//! Run with:
//!   cargo bench --bench mdct_vs_vorbis --no-default-features --features realfft
//!
//! For FFTW3 backend comparison also run:
//!   cargo bench --bench mdct_vs_vorbis --no-default-features --features fftw

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use spectrograms::*;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::os::raw::{c_float, c_int};

// ── libvorbis MDCT FFI ────────────────────────────────────────────────────────
//
// Internal struct from lib/mdct.h (libvorbis 1.3.7, unchanged since 1.0).
// Layout on 64-bit: n(4) log2n(4) trig*(8) bitrev*(8) scale(4) _pad(4) = 32 B
#[repr(C)]
struct VorbisMdctLookup {
    n: c_int,
    log2n: c_int,
    trig: *mut c_float,
    bitrev: *mut c_int,
    scale: c_float,
}

// SAFETY: libvorbis is thread-safe for distinct lookup objects.
unsafe impl Send for VorbisMdctLookup {}

#[link(name = "vorbis")]
unsafe extern "C" {
    /// Allocate and initialise MDCT tables for a window of size `n` (= 2N).
    fn mdct_init(lookup: *mut VorbisMdctLookup, n: c_int);
    /// Release MDCT tables.
    fn mdct_clear(lookup: *mut VorbisMdctLookup);
    /// Forward MDCT: pre-windowed 2N f32 samples → N f32 coefficients.
    fn mdct_forward(lookup: *mut VorbisMdctLookup, input: *const c_float, output: *mut c_float);
    /// Inverse MDCT: N f32 coefficients → 2N f32 samples (NOT windowed).
    fn mdct_backward(lookup: *mut VorbisMdctLookup, input: *const c_float, output: *mut c_float);
}

/// RAII wrapper around `VorbisMdctLookup`.
struct VorbisMdct {
    lookup: VorbisMdctLookup,
    n: usize, // window size (2N)
}

impl VorbisMdct {
    fn new(window_size: usize) -> Self {
        let mut lookup = VorbisMdctLookup {
            n: 0,
            log2n: 0,
            trig: std::ptr::null_mut(),
            bitrev: std::ptr::null_mut(),
            scale: 0.0,
        };
        unsafe { mdct_init(&mut lookup, window_size as c_int) };
        Self {
            lookup,
            n: window_size,
        }
    }

    /// Forward MDCT of a pre-windowed frame.
    fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n / 2);
        unsafe {
            mdct_forward(&mut self.lookup, input.as_ptr(), output.as_mut_ptr());
        }
    }

    /// Inverse MDCT.
    fn backward(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.n / 2);
        debug_assert_eq!(output.len(), self.n);
        unsafe {
            mdct_backward(&mut self.lookup, input.as_ptr(), output.as_mut_ptr());
        }
    }
}

impl Drop for VorbisMdct {
    fn drop(&mut self) {
        unsafe { mdct_clear(&mut self.lookup) };
    }
}

// ── Signal generation ─────────────────────────────────────────────────────────

fn generate_signal_f64(sample_rate: f64, duration: f64) -> Vec<f64> {
    let n = (sample_rate * duration) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (440.0 * 2.0 * std::f64::consts::PI * t).sin()
                + 0.5 * (880.0 * 2.0 * std::f64::consts::PI * t).sin()
        })
        .collect()
}

fn generate_signal_f32(sample_rate: f32, duration: f32) -> Vec<f32> {
    let n = (sample_rate * duration) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (440.0f32 * 2.0 * std::f32::consts::PI * t).sin()
                + 0.5 * (880.0f32 * 2.0 * std::f32::consts::PI * t).sin()
        })
        .collect()
}

fn make_sine_window_f32(window_size: usize) -> Vec<f32> {
    (0..window_size)
        .map(|n| (std::f32::consts::PI * (n as f32 + 0.5) / window_size as f32).sin())
        .collect()
}

// ── Benchmark: forward MDCT comparison ───────────────────────────────────────

fn bench_mdct_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_forward_comparison");

    // Power-of-2 window sizes comparable to real codec usage:
    //   512  ≈ Opus CELT short frame (nearest power-of-2 to 480 at 48 kHz)
    //   1024   Vorbis short / AAC LD / generic
    //   2048   Vorbis long / AAC LC standard
    // (libvorbis requires power-of-2 sizes; 480 and 1152 cause SIGSEGV)
    let configs: &[(usize, &str)] = &[(512, "512"), (1024, "1024"), (2048, "2048")];

    let signal_f64 = generate_signal_f64(44100.0, 1.0);
    let signal_f32 = generate_signal_f32(44100.0, 1.0);

    for &(window_size, label) in configs {
        let hop_size = window_size / 2;
        let n_coeff = window_size / 2;
        let n_frames = (signal_f64.len() - window_size) / hop_size + 1;

        group.throughput(Throughput::Elements(signal_f64.len() as u64));

        // ── Our MDCT (f64, sine window, realfft or fftw backend) ────────────
        let rust_params = MdctParams::sine_window(NonZeroUsize::new(window_size).unwrap()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rust_f64", label),
            &(&signal_f64, &rust_params),
            |b, &(sig, par)| {
                b.iter(|| black_box(compute_mdct(black_box(sig), par).unwrap()));
            },
        );

        // ── Our MDCT (f32, sine window) ─────────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("rust_f32", label),
            &(&signal_f32, &rust_params),
            |b, &(sig, par)| {
                b.iter(|| black_box(compute_mdct_f32(black_box(sig), par).unwrap()));
            },
        );

        // ── libvorbis MDCT (f32, pre-windowing included) ────────────────────
        //
        // libvorbis mdct_forward expects a pre-windowed frame.
        // We include windowing time to match the semantics of our compute_mdct
        // (which also applies the window as part of its pre-twiddle step).
        let window_f32 = make_sine_window_f32(window_size);
        let mut vorbis = VorbisMdct::new(window_size);
        let mut framed = vec![0.0f32; window_size];
        let mut out_buf = vec![0.0f32; n_coeff];

        group.bench_with_input(
            BenchmarkId::new("vorbis_f32", label),
            &(&signal_f32, &window_f32),
            |b, &(sig, win)| {
                b.iter(|| {
                    for frame_idx in 0..n_frames {
                        let start = frame_idx * hop_size;
                        // Window the frame (f32 multiply)
                        for (i, v) in framed.iter_mut().enumerate() {
                            *v = sig[start + i] * win[i];
                        }
                        vorbis.forward(black_box(&framed), &mut out_buf);
                    }
                    black_box(&out_buf);
                });
            },
        );
    }

    group.finish();
}

// ── Benchmark: inverse MDCT comparison ────────────────────────────────────────

fn bench_mdct_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct_inverse_comparison");

    // Power-of-2 sizes only — see module-level note on libvorbis constraint.
    let configs: &[(usize, &str)] = &[(512, "512"), (1024, "1024"), (2048, "2048")];

    let signal_f64 = generate_signal_f64(44100.0, 1.0);
    let signal_f32 = generate_signal_f32(44100.0, 1.0);

    for &(window_size, label) in configs {
        let hop_size = window_size / 2;
        let n_coeff = window_size / 2;
        let n_frames = (signal_f64.len() - window_size) / hop_size + 1;

        group.throughput(Throughput::Elements(signal_f64.len() as u64));

        // ── Our IMDCT (f64) ─────────────────────────────────────────────────
        let rust_params = MdctParams::sine_window(NonZeroUsize::new(window_size).unwrap()).unwrap();
        let coeffs_f64 = compute_mdct(&signal_f64, &rust_params).unwrap();
        let coeffs_f32_bench = compute_mdct_f32(&signal_f32, &rust_params).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rust_f64", label),
            &(&coeffs_f64, &rust_params),
            |b, &(co, par)| {
                b.iter(|| black_box(compute_imdct(black_box(co), par, None).unwrap()));
            },
        );

        // ── Our IMDCT (f32) ──────────────────────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("rust_f32", label),
            &(&coeffs_f32_bench, &rust_params),
            |b, &(co, par)| {
                b.iter(|| black_box(compute_imdct_f32(black_box(co), par, None).unwrap()));
            },
        );

        // ── libvorbis IMDCT (f32) ────────────────────────────────────────────
        // We build the coefficient array by doing a forward pass first.
        let window_f32 = make_sine_window_f32(window_size);
        let mut vorbis = VorbisMdct::new(window_size);
        let mut framed = vec![0.0f32; window_size];
        let coeffs_f32: Vec<Vec<f32>> = (0..n_frames)
            .map(|frame_idx| {
                let start = frame_idx * hop_size;
                for (i, v) in framed.iter_mut().enumerate() {
                    v.clone_from(&(signal_f32[start + i] * window_f32[i]));
                }
                let mut out = vec![0.0f32; n_coeff];
                vorbis.forward(&framed, &mut out);
                out
            })
            .collect();

        let mut imdct_out = vec![0.0f32; window_size];

        group.bench_with_input(
            BenchmarkId::new("vorbis_f32", label),
            &coeffs_f32,
            |b, co| {
                b.iter(|| {
                    for frame in co.iter() {
                        vorbis.backward(black_box(frame), &mut imdct_out);
                    }
                    black_box(&imdct_out);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mdct_forward, bench_mdct_inverse);
criterion_main!(benches);
