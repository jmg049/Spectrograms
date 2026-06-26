//! FFT-based 1D linear convolution and deconvolution for real signals.
//!
//! Built on the crate's cached `fft`/`irfft` plans. Used for impulse-response
//! extraction (deconvolution) and synthetic-signal construction (convolution).

use std::num::NonZeroUsize;

use ndarray::Array1;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use num_complex::Complex;

use crate::error::{SpectrogramError, SpectrogramResult};
use crate::fft_backend::C2cPlanF32;
use crate::fft_backend::realfft_backend::RealFftC2cPlanF32;
use crate::spectrogram::{fft, irfft};

/// Linear convolution of two real signals via FFT.
///
/// Both signals are zero-padded to `next_power_of_two(a.len() + b.len() - 1)`,
/// multiplied in the frequency domain, and inverse-transformed. The output has
/// length `a.len() + b.len() - 1`.
///
/// # Errors
/// Propagates any FFT/IFFT failure from the underlying plan.
pub fn fft_convolve(
    a: &NonEmptySlice<f64>,
    b: &NonEmptySlice<f64>,
) -> SpectrogramResult<NonEmptyVec<f64>> {
    let out_len = a.len().get() + b.len().get() - 1;
    let n_fft = out_len.next_power_of_two();
    // SAFETY: out_len >= 1, so n_fft >= 1.
    let n = unsafe { NonZeroUsize::new_unchecked(n_fft) };

    let a_spec = fft(a, n)?;
    let b_spec = fft(b, n)?;
    let product: Array1<Complex<f64>> = &a_spec * &b_spec;

    let product_slice = product.as_slice().expect("fft output is contiguous");
    // SAFETY: r2c output size is >= 1 for n >= 1.
    let product_ne = unsafe { NonEmptySlice::new_unchecked(product_slice) };

    let full = irfft(product_ne, n)?;
    let mut v = full.into_vec();
    v.truncate(out_len);
    // SAFETY: out_len >= 1.
    Ok(unsafe { NonEmptyVec::new_unchecked(v) })
}

/// Regularised spectral-division deconvolution of two real signals.
///
/// Computes `Y(f) = N(f)·conj(D(f)) / (|D(f)|² + ε)` where `ε = regularization ·
/// max|D(f)|²`, then inverse-transforms. With `regularization = 0.0` this is a
/// pure inverse filter; small positive values stabilise division near spectral
/// nulls of the denominator. Output length is `numerator.len() -
/// denominator.len() + 1` (clamped to at least 1).
/// The numerator is expected to be the full linear-convolution output of the original signal and the denominator; passing a shorter numerator may introduce circular-aliasing artifacts.
///
/// # Errors
/// Propagates any FFT/IFFT failure from the underlying plan.
pub fn fft_deconvolve(
    numerator: &NonEmptySlice<f64>,
    denominator: &NonEmptySlice<f64>,
    regularization: f64,
) -> SpectrogramResult<NonEmptyVec<f64>> {
    let n_len = numerator.len().get();
    let d_len = denominator.len().get();
    let n_fft = n_len.max(d_len).next_power_of_two();
    // SAFETY: n_fft >= 1.
    let n = unsafe { NonZeroUsize::new_unchecked(n_fft) };

    let num_spec = fft(numerator, n)?;
    let den_spec = fft(denominator, n)?;

    let max_d2 = den_spec
        .iter()
        .map(Complex::norm_sqr)
        .fold(0.0_f64, f64::max);
    let eps = regularization * max_d2;

    let quotient: Array1<Complex<f64>> =
        Array1::from_iter(num_spec.iter().zip(den_spec.iter()).map(|(nn, dd)| {
            let denom = dd.norm_sqr() + eps;
            if denom == 0.0 {
                Complex::new(0.0, 0.0)
            } else {
                (*nn) * dd.conj() / denom
            }
        }));

    let q_slice = quotient
        .as_slice()
        .expect("Array1 from_iter is always contiguous");
    // SAFETY: r2c output size >= 1.
    let q_ne = unsafe { NonEmptySlice::new_unchecked(q_slice) };

    let full = irfft(q_ne, n)?;
    let out_len = if n_len >= d_len {
        n_len - d_len + 1
    } else {
        n_len
    };
    let mut v = full.into_vec();
    v.truncate(out_len.max(1));
    // SAFETY: length >= 1.
    Ok(unsafe { NonEmptyVec::new_unchecked(v) })
}

/// Streaming FFT convolution engine (overlap-save, single-precision).
///
/// Designed for real-time block processing such as room-correction FIR filtering
/// with long impulse responses (thousands of taps), where direct time-domain
/// convolution is O(block · taps) and becomes the dominant cost. This engine is
/// O(block · log N) per block and turns a multi-thousand-tap filter from a
/// real-time bottleneck into a negligible one.
///
/// # How it works
///
/// The impulse response is FFT'd once at construction and its spectrum cached.
/// Each call to [`process_block`](Self::process_block) transforms a window made of
/// the previous tail plus the new input block, multiplies by the cached spectrum,
/// inverse-transforms, and keeps only the alias-free tail — the standard
/// overlap-save method. After construction there are **no heap allocations** on
/// the processing path, so it is safe to call from an audio callback.
///
/// # Semantics
///
/// Computes true linear convolution `y[n] = Σ_k h[k]·x[n−k]`, producing exactly
/// as many output samples as input samples (the filter's own latency is carried
/// in `h`: ~`taps/2` for a linear-phase IR, ~0 for a minimum-phase IR). The block
/// length is fixed at construction and every input slice must match it.
///
/// # Example
///
/// ```
/// use spectrograms::OverlapSaveConvolver;
/// use std::num::NonZeroUsize;
///
/// // A trivial 3-tap moving-average impulse response.
/// let ir = [1.0f32 / 3.0; 3];
/// let block = NonZeroUsize::new(256).unwrap();
/// let mut conv = OverlapSaveConvolver::new(&ir, block).unwrap();
///
/// let input = vec![1.0f32; 256];
/// let mut output = vec![0.0f32; 256];
/// conv.process_block(&input, &mut output).unwrap();
/// // Steady-state output of a normalized 3-tap average of all-ones is ~1.0.
/// assert!((output[100] - 1.0).abs() < 1e-5);
/// ```
pub struct OverlapSaveConvolver {
    block: usize,
    n_fft: usize,
    overlap: usize,
    fft: RealFftC2cPlanF32,
    /// FFT of the zero-padded impulse response (length `n_fft`).
    h_spec: Vec<Complex<f32>>,
    /// Most recent `overlap` input samples carried between blocks.
    history: Vec<f32>,
    /// Complex work buffer reused every block (length `n_fft`).
    work: Vec<Complex<f32>>,
    inv_n: f32,
}

impl OverlapSaveConvolver {
    /// Build a convolver for impulse response `ir` and a fixed `block` size.
    ///
    /// The FFT size is `next_power_of_two(block + ir.len() − 1)`. The impulse
    /// response is transformed once here; this is the only place that allocates.
    ///
    /// # Errors
    /// Returns an error if `ir` is empty or the FFT plan cannot be built.
    pub fn new(ir: &[f32], block: NonZeroUsize) -> SpectrogramResult<Self> {
        if ir.is_empty() {
            return Err(SpectrogramError::invalid_input(
                "impulse response must not be empty",
            ));
        }
        let block = block.get();
        let n_fft = (block + ir.len() - 1).next_power_of_two();
        let overlap = n_fft - block;

        let mut fft = RealFftC2cPlanF32::new(n_fft);

        // Cache the IR spectrum: zero-pad to n_fft, forward FFT.
        let mut h_spec = vec![Complex::new(0.0f32, 0.0f32); n_fft];
        for (dst, &src) in h_spec.iter_mut().zip(ir.iter()) {
            dst.re = src;
        }
        fft.forward(&mut h_spec)?;

        Ok(Self {
            block,
            n_fft,
            overlap,
            fft,
            h_spec,
            history: vec![0.0f32; overlap],
            work: vec![Complex::new(0.0f32, 0.0f32); n_fft],
            inv_n: 1.0 / n_fft as f32,
        })
    }

    /// The fixed block size this convolver expects.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block
    }

    /// The internal FFT size.
    #[must_use]
    pub const fn fft_size(&self) -> usize {
        self.n_fft
    }

    /// Reset inter-block state to silence (clears the overlap history).
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Filter one block in place-free fashion: `output[i] = (h * x)[i]`.
    ///
    /// `input` and `output` must both have length [`block_size`](Self::block_size).
    /// No allocation occurs.
    ///
    /// # Errors
    /// Returns an error if either slice length differs from the block size, or if
    /// an FFT step fails.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> SpectrogramResult<()> {
        if input.len() != self.block || output.len() != self.block {
            return Err(SpectrogramError::invalid_input(format!(
                "process_block expects input and output of length {} (got {} and {})",
                self.block,
                input.len(),
                output.len()
            )));
        }

        // Build the time-domain window: [history (overlap) | input (block)].
        for (dst, &src) in self.work[..self.overlap]
            .iter_mut()
            .zip(self.history.iter())
        {
            dst.re = src;
            dst.im = 0.0;
        }
        for (dst, &src) in self.work[self.overlap..].iter_mut().zip(input.iter()) {
            dst.re = src;
            dst.im = 0.0;
        }

        // Save the new history = last `overlap` samples of [history | input]
        // (i.e. window[block..n_fft]) before the FFT overwrites `work`.
        for (dst, src) in self.history.iter_mut().zip(self.work[self.block..].iter()) {
            *dst = src.re;
        }

        // Circular convolution in the frequency domain.
        self.fft.forward(&mut self.work)?;
        for (w, h) in self.work.iter_mut().zip(self.h_spec.iter()) {
            *w *= *h;
        }
        self.fft.inverse(&mut self.work)?;

        // The alias-free part is the last `block` samples; rescale by 1/n_fft.
        let inv_n = self.inv_n;
        for (out, w) in output.iter_mut().zip(self.work[self.overlap..].iter()) {
            *out = w.re * inv_n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use non_empty_slice::NonEmptyVec;

    fn ne(v: Vec<f64>) -> NonEmptyVec<f64> {
        NonEmptyVec::new(v).unwrap()
    }

    #[test]
    fn convolve_with_unit_impulse_returns_input_shifted() {
        let signal = ne(vec![1.0, 2.0, 3.0, 4.0]);
        // impulse delayed by 2 samples
        let impulse = ne(vec![0.0, 0.0, 1.0]);
        let out = fft_convolve(signal.as_non_empty_slice(), impulse.as_non_empty_slice()).unwrap();
        let out = out.into_vec();
        // expect [0,0,1,2,3,4]
        assert_eq!(out.len(), 6);
        let expected = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn deconvolve_recovers_impulse_response() {
        // h is the "system"; x is the excitation; y = x * h. Recover h from (y, x).
        let x = ne(vec![1.0, 0.7, -0.3, 0.2, 0.9, -0.5, 0.1, 0.4]);
        let h = ne(vec![0.0, 0.0, 1.0, 0.5]); // impulse at index 2, plus a tap
        let y = fft_convolve(x.as_non_empty_slice(), h.as_non_empty_slice()).unwrap();

        let recovered = fft_deconvolve(y.as_non_empty_slice(), x.as_non_empty_slice(), 0.0)
            .unwrap()
            .into_vec();

        let hv = h.into_vec();
        assert!(recovered.len() >= hv.len());
        for (i, &want) in hv.iter().enumerate() {
            assert!(
                (recovered[i] - want).abs() < 1e-6,
                "tap {i}: got {}, want {want}",
                recovered[i]
            );
        }
    }

    #[test]
    fn convolve_matches_direct_convolution() {
        let a = ne(vec![1.0, -2.0, 0.5]);
        let b = ne(vec![0.25, 1.0, -0.5, 2.0]);
        let out = fft_convolve(a.as_non_empty_slice(), b.as_non_empty_slice())
            .unwrap()
            .into_vec();
        // direct convolution reference
        let av = a.into_vec();
        let bv = b.into_vec();
        let mut want = vec![0.0; av.len() + bv.len() - 1];
        for (i, &x) in av.iter().enumerate() {
            for (j, &y) in bv.iter().enumerate() {
                want[i + j] += x * y;
            }
        }
        assert_eq!(out.len(), want.len());
        for (got, w) in out.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-9, "got {got}, want {w}");
        }
    }

    #[test]
    fn overlap_save_matches_direct_streaming() {
        use crate::convolution::OverlapSaveConvolver;
        use std::num::NonZeroUsize;

        // A reasonably long, non-symmetric impulse response.
        let taps = 200;
        let ir: Vec<f32> = (0..taps)
            .map(|k| ((k as f32 * 0.13).sin()) * (-(k as f32) / 60.0).exp())
            .collect();

        // A streamed input signal.
        let total = 1024usize;
        let x: Vec<f32> = (0..total)
            .map(|n| (n as f32 * 0.05).sin() + 0.3 * (n as f32 * 0.21).cos())
            .collect();

        let block = 128usize;
        let mut conv = OverlapSaveConvolver::new(&ir, NonZeroUsize::new(block).unwrap()).unwrap();

        // Streamed output.
        let mut got = vec![0.0f32; total];
        let mut obuf = vec![0.0f32; block];
        let mut start = 0;
        while start + block <= total {
            conv.process_block(&x[start..start + block], &mut obuf)
                .unwrap();
            got[start..start + block].copy_from_slice(&obuf);
            start += block;
        }

        // Direct streaming convolution reference: y[n] = Σ_k ir[k] x[n-k].
        for n in 0..total {
            let mut acc = 0.0f32;
            for (k, &h) in ir.iter().enumerate() {
                if n >= k {
                    acc += h * x[n - k];
                }
            }
            assert!(
                (got[n] - acc).abs() < 1e-3,
                "sample {n}: overlap-save {} vs direct {acc}",
                got[n]
            );
        }
    }
}
