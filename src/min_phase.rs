//! Minimum-phase conversion of FIR impulse responses.
//!
//! Given any FIR impulse response (typically a linear-phase one), produce a
//! minimum-phase impulse response with the **same magnitude response** but whose
//! energy is concentrated at the start. For a linear-phase filter the algorithmic
//! latency is ~`taps/2` samples; the minimum-phase equivalent removes almost all
//! of that latency while preserving the magnitude response (and hence the
//! low-frequency resolution of a room-correction filter).
//!
//! ## Method
//!
//! The real-cepstrum (homomorphic) method:
//! 1. `H = FFT(h)` on a heavily oversampled grid (to avoid cepstral time-aliasing).
//! 2. `Ĥ = log|H|` (real).
//! 3. real cepstrum `c = IFFT(Ĥ)`.
//! 4. window `c` to keep only the causal part, doubling the non-DC/Nyquist taps
//!    (this is the discrete Hilbert relation that builds the minimum-phase
//!    phase from the log-magnitude).
//! 5. `H_min = exp(FFT(c))`, `h_min = real(IFFT(H_min))`.
//!
//! The result is truncated to the requested length.

use num_complex::Complex;

use crate::Sample;
use crate::error::{SpectrogramError, SpectrogramResult};
use crate::fft_backend::C2cPlan;

/// Oversampling factor applied to the IR length when choosing the FFT size.
///
/// Larger values reduce time-aliasing of the cepstrum (and thus magnitude error)
/// at the cost of a bigger one-off transform. 8x is a good default for audio FIRs.
const DEFAULT_OVERSAMPLE: usize = 8;

/// Convert an FIR impulse response to its minimum-phase equivalent.
///
/// The returned filter has the same length as `ir` and (very nearly) the same
/// magnitude response, but minimal latency. Computation is performed in the
/// sample precision `T` (`f32` or `f64`); `f64` offers more numerical headroom.
///
/// # Errors
/// Returns an error if `ir` is empty or an FFT step fails.
///
/// # Example
/// ```
/// use spectrograms::minimum_phase;
///
/// // A 5-tap linear-phase (symmetric) low-pass prototype.
/// let lin = [0.1f32, 0.2, 0.4, 0.2, 0.1];
/// let mp = minimum_phase(&lin).unwrap();
/// assert_eq!(mp.len(), lin.len());
/// // Minimum-phase energy is front-loaded: first tap is the largest.
/// assert!(mp[0].abs() >= mp[mp.len() - 1].abs());
/// ```
pub fn minimum_phase<T: Sample>(ir: &[T]) -> SpectrogramResult<Vec<T>> {
    minimum_phase_with(ir, ir.len(), DEFAULT_OVERSAMPLE)
}

/// Like [`minimum_phase`] but with explicit output length and oversampling factor.
///
/// - `out_len`: number of taps to keep from the (front-loaded) minimum-phase IR.
/// - `oversample`: FFT size is `next_power_of_two(ir.len() * oversample)`, clamped
///   to at least `next_power_of_two(ir.len())`. Higher values lower magnitude error.
///
/// # Errors
/// Returns an error if `ir` is empty, `out_len` is zero, or an FFT step fails.
pub fn minimum_phase_with<T: Sample>(
    ir: &[T],
    out_len: usize,
    oversample: usize,
) -> SpectrogramResult<Vec<T>> {
    if ir.is_empty() {
        return Err(SpectrogramError::invalid_input(
            "impulse response must not be empty",
        ));
    }
    if out_len == 0 {
        return Err(SpectrogramError::invalid_input(
            "out_len must be greater than zero",
        ));
    }

    let oversample = oversample.max(1);
    let n = (ir.len() * oversample).next_power_of_two();
    let inv_n = T::one() / T::from_usize(n);

    let mut fft = T::plan_c2c(n)?;

    // buf := zero-padded IR (complex, imag = 0)
    let mut buf = vec![Complex::new(T::zero(), T::zero()); n];
    for (dst, &src) in buf.iter_mut().zip(ir.iter()) {
        dst.re = src;
    }

    // H = FFT(h)
    fft.forward(&mut buf)?;

    // Ĥ = log|H|  (guard against log(0) at spectral nulls)
    let max_mag2 = buf.iter().map(Complex::norm_sqr).fold(T::zero(), T::max);
    let eps = if max_mag2 > T::zero() {
        max_mag2 * T::from_f64(1e-20)
    } else {
        T::from_f64(1e-300)
    };
    for x in buf.iter_mut() {
        let log_mag = T::from_f64(0.5) * (x.norm_sqr() + eps).ln();
        x.re = log_mag;
        x.im = T::zero();
    }

    // real cepstrum c = IFFT(Ĥ)
    fft.inverse(&mut buf)?;
    for x in buf.iter_mut() {
        *x *= inv_n;
    }

    // Causal-doubling window (discrete Hilbert): keep causal part, double the
    // taps strictly between DC and Nyquist, zero the anticausal part.
    // buf[0] (DC) and buf[n/2] (Nyquist, n even) keep weight 1.
    let half = n / 2;
    for x in buf.iter_mut().take(half).skip(1) {
        *x *= T::from_f64(2.0);
    }
    for x in buf.iter_mut().skip(half + 1) {
        *x = Complex::new(T::zero(), T::zero());
    }

    // H_min = exp(FFT(c))
    fft.forward(&mut buf)?;
    for x in buf.iter_mut() {
        let mag = x.re.exp();
        *x = Complex::new(mag * x.im.cos(), mag * x.im.sin());
    }

    // h_min = real(IFFT(H_min))
    fft.inverse(&mut buf)?;

    let take = out_len.min(n);
    let out: Vec<T> = buf.iter().take(take).map(|x| x.re * inv_n).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    /// DFT magnitude of a real sequence at `k/n` cycles, evaluated directly.
    fn mag_at(h: &[f32], n: usize, k: usize) -> f64 {
        let mut acc = Complex::new(0.0f64, 0.0);
        let w = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        for (idx, &v) in h.iter().enumerate() {
            acc += Complex::from_polar(f64::from(v), w * idx as f64);
        }
        acc.norm()
    }

    #[test]
    fn magnitude_response_is_preserved() {
        // Linear-phase (symmetric) windowed-sinc-ish low-pass prototype.
        let taps = 64;
        let fc = 0.15; // normalized cutoff
        let mid = (taps - 1) as f64 / 2.0;
        let lin: Vec<f32> = (0..taps)
            .map(|k| {
                let x = k as f64 - mid;
                let sinc = if x.abs() < 1e-9 {
                    2.0 * fc
                } else {
                    (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x)
                };
                // Hann window
                let w =
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (taps - 1) as f64).cos();
                (sinc * w) as f32
            })
            .collect();

        let mp = minimum_phase(&lin).unwrap();
        assert_eq!(mp.len(), lin.len());

        // Compare magnitude responses on a fine grid; they must match closely.
        let n = 512;
        for k in 0..=n / 2 {
            let a = mag_at(&lin, n, k);
            let b = mag_at(&mp, n, k);
            assert!(
                (a - b).abs() < 1e-2 + 1e-2 * a,
                "bin {k}: linear {a} vs min-phase {b}"
            );
        }
    }

    #[test]
    fn energy_is_front_loaded() {
        let taps = 64;
        let mid = (taps - 1) as f64 / 2.0;
        let lin: Vec<f32> = (0..taps)
            .map(|k| {
                let x = k as f64 - mid;
                let sinc = if x.abs() < 1e-9 {
                    0.3
                } else {
                    (0.3 * std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                sinc as f32
            })
            .collect();
        let mp = minimum_phase(&lin).unwrap();

        // Centre of energy (group-delay proxy) must move earlier.
        let centroid = |h: &[f32]| {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (i, &v) in h.iter().enumerate() {
                let e = f64::from(v) * f64::from(v);
                num += i as f64 * e;
                den += e;
            }
            num / den
        };
        let lin_c = centroid(&lin);
        let mp_c = centroid(&mp);
        assert!(
            mp_c < lin_c * 0.5,
            "min-phase energy centroid {mp_c} should be well before linear-phase {lin_c}"
        );
    }
}
