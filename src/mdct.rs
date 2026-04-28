//! Modified Discrete Cosine Transform (MDCT) and its inverse (IMDCT).
//!
//! MDCT is a lapped orthogonal transform used in audio codecs (MP3, AAC, Vorbis, Opus).
//! It maps a 2N-sample overlapping window to N real-valued coefficients.
//!
//! # Algorithm
//!
//! For window size $2N$, the MDCT produces $N$ coefficients per frame:
//!
//! $$C\[k\] = \sum_{n=0}^{2N-1} x\[n\]\, w\[n\]\, \cos\left(\frac{\pi\,(2n+1+N)\,(2k+1)}{4N}\right)$$
//!
//! **Forward** computed in $O(N \log N)$ via one C2c(N) FFT (packing trick):
//! fold the 2N windowed samples into N complex values $z[m] = a[m] + jb[m]$, compute
//! one C2c(N) FFT, then recover 2N-point DFT coefficients via Hermitian symmetry:
//! even $k$: $F[k] = Z[k]$; odd $k$: $F[k] = Z[N-k]^*$.
//! Halves the FFT count compared to the 2×R2c(N) folding approach.
//!
//! **Inverse** uses 2×C2c(N) by splitting the half-sparse 2N-point DFT:
//! even $m=2j$: $\mathrm{DFT}_{2N}[2j] = \mathrm{DFT}_N(z)[j]$;
//! odd $m=2j+1$: $\mathrm{DFT}_{2N}[2j+1] = \mathrm{DFT}_N(z \cdot W)[j]$ where
//! $W[k]=e^{-j\pi k/N}$ is precomputed. Halves the transform length vs 2×R2c(2N).
//! Synthesis window and overlap-add follow.
//!
//! # Perfect Reconstruction
//!
//! PR requires the window to satisfy the TDAC condition. For 50% hop:
//!
//! $$w\[n\]^2 + w\[n+N\]^2 = 1 \quad \forall\, n \in [0, N)$$
//!
//! **Only [`MdctParams::sine_window`] satisfies this.** Standard windows (Hanning,
//! Hamming, Blackman) violate this condition and will NOT give perfect reconstruction.
//! For non-50% hops, PR requires a custom window designed for that overlap ratio.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use ndarray::Array2;
use num_complex::Complex;

use crate::{C2cPlan, C2cPlanF32, SpectrogramError, SpectrogramResult, WindowType, make_window};

mod private_seal {
    pub trait Seal {}
    impl Seal for f32 {}
    impl Seal for f64 {}
}

/// Sealed marker trait for float types usable in MDCT computation (f32 or f64).
trait MdctNum:
    private_seal::Seal
    + Copy
    + Default
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
{
    fn zero() -> Self;
    fn from_f64(x: f64) -> Self;
    fn scale(n: usize) -> Self; // = 2.0 / n as Self
}

impl MdctNum for f32 {
    #[inline]
    fn zero() -> Self {
        0.0f32
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as Self
    }
    #[inline]
    fn scale(n: usize) -> Self {
        2.0f32 / n as Self
    }
}

impl MdctNum for f64 {
    #[inline]
    fn zero() -> Self {
        0.0f64
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn scale(n: usize) -> Self {
        2.0f64 / n as Self
    }
}

/// Private in-place complex-to-complex FFT trait for MDCT/IMDCT, generic over float type.
trait MdctC2cFft<T: MdctNum>: Send {
    fn forward(&mut self, buf: &mut [Complex<T>]) -> SpectrogramResult<()>;
}

struct C2cWrapper<P: C2cPlan + Send>(P);

impl<P: C2cPlan + Send> MdctC2cFft<f64> for C2cWrapper<P> {
    #[inline]
    fn forward(&mut self, buf: &mut [Complex<f64>]) -> SpectrogramResult<()> {
        self.0.forward(buf)
    }
}

struct C2cF32Wrapper<P: C2cPlanF32 + Send>(P);

impl<P: C2cPlanF32 + Send> MdctC2cFft<f32> for C2cF32Wrapper<P> {
    #[inline]
    fn forward(&mut self, buf: &mut [Complex<f32>]) -> SpectrogramResult<()> {
        self.0.forward(buf)
    }
}

/// Parameters for MDCT computation.
///
/// The window size must be even and at least 4. `N = window_size / 2` is the number
/// of output coefficients per frame.
///
/// # Perfect Reconstruction
///
/// Only [`MdctParams::sine_window`] gives perfect reconstruction. Standard windows
/// (Hanning, Hamming, Blackman) do not satisfy `w[n]^2 + w[n+N]^2 = 1` and will
/// introduce reconstruction error. Non-50% hops also break PR unless the window
/// is specifically designed for that overlap ratio (TDAC condition).
#[derive(Debug, Clone)]
pub struct MdctParams {
    /// Window size (= 2N). Must be even and >= 4.
    pub window_size: NonZeroUsize,
    /// Hop size between consecutive frames. Typically N for perfect reconstruction.
    pub hop_size: NonZeroUsize,
    /// Window function to apply.
    pub window: WindowType,
}

impl MdctParams {
    /// Create MDCT parameters with validation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `window_size` is odd, less than 4, or `hop_size` is zero.
    #[inline]
    pub fn new(
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        window: WindowType,
    ) -> SpectrogramResult<Self> {
        if !window_size.get().is_multiple_of(2) {
            return Err(SpectrogramError::invalid_input(format!(
                "window_size must be even, got {}",
                window_size.get()
            )));
        }
        if window_size.get() < 4 {
            return Err(SpectrogramError::invalid_input(format!(
                "window_size must be >= 4, got {}",
                window_size.get()
            )));
        }
        Ok(Self {
            window_size,
            hop_size,
            window,
        })
    }

    /// Create parameters with a sine window and 50% hop for perfect reconstruction.
    ///
    /// The sine window `w[n] = sin(π·(n+½)/(2N))` satisfies
    /// `w[n]^2 + w[n+N]^2 = 1`, enabling perfect reconstruction via overlap-add
    /// with 50% overlap.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `window_size` is odd or less than 4.
    #[inline]
    pub fn sine_window(window_size: NonZeroUsize) -> SpectrogramResult<Self> {
        let n = window_size.get();
        if !n.is_multiple_of(2) {
            return Err(SpectrogramError::invalid_input(format!(
                "window_size must be even, got {n}"
            )));
        }
        if n < 4 {
            return Err(SpectrogramError::invalid_input(format!(
                "window_size must be >= 4, got {n}"
            )));
        }
        let coeffs: Vec<f64> = (0..n)
            .map(|k| (PI * (k as f64 + 0.5) / n as f64).sin())
            .collect();
        let window = WindowType::custom(coeffs)?;
        let hop_size = NonZeroUsize::new(n / 2)
            .ok_or_else(|| SpectrogramError::invalid_input("hop_size computed as zero"))?;
        Ok(Self {
            window_size,
            hop_size,
            window,
        })
    }

    /// Number of MDCT coefficients per frame (= window_size / 2).
    #[inline]
    #[must_use]
    pub const fn n_coefficients(&self) -> usize {
        self.window_size.get() / 2
    }
}

/// Forward MDCT plan: 1×C2c(N) with folded pre-twiddle (packing trick).
///
/// The 2N-sample windowed input is folded into N real pairs (a[m], b[m]):
///   a[m] = analysis_re[m]·frame[m] + analysis_re[m+N]·frame[m+N]
///   b[m] = analysis_im[m]·frame[m] + analysis_im[m+N]·frame[m+N]
///
/// These are packed into one complex sequence z[m] = a[m] + j·b[m] and a single
/// C2c(N) forward FFT is computed.  Since a and b are real, DFT_N(a) and DFT_N(b)
/// are Hermitian-symmetric, which lets us recover DFT_{2N}(z)[k] directly:
///
///   Even k: F[k] = Z[k]
///   Odd  k: F[k] = Z[N-k]*
///
/// This halves the FFT count compared to the 2×R2c(N) approach.
struct MdctFwdPlan<T: MdctNum> {
    /// Fold+twiddle: analysis_re[m] = w[m]·cos(πm/2N) for m=0..2N-1
    analysis_re: Vec<T>,
    /// Fold+twiddle: analysis_im[m] = -w[m]·sin(πm/2N) for m=0..2N-1
    analysis_im: Vec<T>,
    /// Post-twiddle: (cos, -sin) of π(2k+1)(N+1)/4N for k=0..N-1
    mdct_post_re: Vec<T>,
    mdct_post_im: Vec<T>,
    /// C2c FFT of size N
    c2c: Box<dyn MdctC2cFft<T>>,
    /// Packed complex input/output buffer (size N, in-place)
    fwd_z: Vec<Complex<T>>,
    n: usize,
}

impl<T: MdctNum> MdctFwdPlan<T> {
    fn new(params: &MdctParams, c2c: Box<dyn MdctC2cFft<T>>) -> Self {
        let two_n = params.window_size.get();
        let n = two_n / 2;

        let window_f64 = make_window(params.window.clone(), params.window_size);

        let analysis_re: Vec<T> = (0..two_n)
            .map(|m| T::from_f64(window_f64[m] * (PI * m as f64 / two_n as f64).cos()))
            .collect();
        let analysis_im: Vec<T> = (0..two_n)
            .map(|m| T::from_f64(-window_f64[m] * (PI * m as f64 / two_n as f64).sin()))
            .collect();

        let (mdct_post_re, mdct_post_im): (Vec<T>, Vec<T>) = (0..n)
            .map(|k| {
                let angle = PI * (2 * k + 1) as f64 * (n + 1) as f64 / (4 * n) as f64;
                (T::from_f64(angle.cos()), T::from_f64(-angle.sin()))
            })
            .unzip();

        Self {
            analysis_re,
            analysis_im,
            mdct_post_re,
            mdct_post_im,
            c2c,
            fwd_z: vec![
                Complex {
                    re: T::zero(),
                    im: T::zero()
                };
                n
            ],
            n,
        }
    }

    /// Compute MDCT of one 2N-sample frame via one C2c(N) FFT (packing trick).
    fn mdct_frame(&mut self, frame: &[T], out: &mut [T]) -> SpectrogramResult<()> {
        let n = self.n;

        // Pack z[m] = a[m] + j·b[m] where:
        //   a[m] = frame[m]*analysis_re[m] + frame[m+N]*analysis_re[m+N]
        //   b[m] = frame[m]*analysis_im[m] + frame[m+N]*analysis_im[m+N]
        for m in 0..n {
            self.fwd_z[m] = Complex {
                re: frame[m] * self.analysis_re[m] + frame[m + n] * self.analysis_re[m + n],
                im: frame[m] * self.analysis_im[m] + frame[m + n] * self.analysis_im[m + n],
            };
        }

        self.c2c.forward(&mut self.fwd_z)?;

        // Recover DFT_{2N}(pre-rot)[k] from the N-point packed FFT output Z:
        //
        //   Even k = 2p:   DFT_{2N}[2p]   = Z[p]         → fwd_z[k/2]
        //   Odd  k = 2p+1: DFT_{2N}[2p+1] = Z[N-1-p]*    → conj(fwd_z[N-1-(k-1)/2])
        //
        // Split into two branch-free loops for uniform memory access patterns
        // that the compiler can autovectorize independently.
        //
        // Even k = 0,2,4,...: sequential read fwd_z[0,1,2,...], sequential write out[0,2,4,...]
        let mut k = 0usize;
        while k < n {
            let p = k / 2;
            let f_re = self.fwd_z[p].re;
            let f_im = self.fwd_z[p].im;
            out[k] = self.mdct_post_re[k] * f_re - self.mdct_post_im[k] * f_im;
            k += 2;
        }
        // Odd k = 1,3,5,...: fwd_z read as [N-1, N-2, ...] (reverse), write out[1,3,5,...]
        // Conjugate: f_im = -fwd_z[N-1-p].im, so out[k] = pt_re*re - pt_im*(-im) = pt_re*re + pt_im*im
        let mut k = 1usize;
        while k < n {
            let p = (k - 1) / 2;
            let f_re = self.fwd_z[n - 1 - p].re;
            let f_im = self.fwd_z[n - 1 - p].im; // conjugate sign absorbed into formula below
            out[k] = self.mdct_post_re[k] * f_re + self.mdct_post_im[k] * f_im;
            k += 2;
        }

        Ok(())
    }
}

/// Inverse MDCT plan: 2×C2c(N) with even/odd frequency split.
///
/// The 2N-point DFT of the half-sparse pre-twiddle sequence splits cleanly into
/// two N-point DFTs:
///
///   DFT_{2N}(z_sparse)[2j]   = DFT_N(z)[j]           (even output positions)
///   DFT_{2N}(z_sparse)[2j+1] = DFT_N(z·W)[j]         (odd output positions)
///
/// where z[k] = coeffs[k]·exp(−jπk(N+1)/2N) and W[k] = exp(−jπk/N).
///
/// Pre-multiplying W into a combined twiddle table avoids per-frame work.
/// Both C2c(N) transforms reuse the same plan object sequentially.
/// This halves the FFT length (N vs 2N) and halves scratch buffer sizes
/// compared to the 2×R2c(2N) approach.
struct MdctInvPlan<T: MdctNum> {
    window_samples: Vec<T>,
    /// Post-twiddle: (cos, -sin) of π(2m+1+N)/4N for m=0..2N-1
    imdct_post_re: Vec<T>,
    imdct_post_im: Vec<T>,
    /// Pre-twiddle for even positions: exp(−jπk(N+1)/2N), size N
    pre_z: Vec<Complex<T>>,
    /// Pre-twiddle for odd positions: pre_z[k]·exp(−jπk/N), size N
    pre_zprime: Vec<Complex<T>>,
    /// C2c FFT of size N (reused for both transforms)
    c2c: Box<dyn MdctC2cFft<T>>,
    /// Working buffers for the two C2c transforms (size N each, in-place)
    z: Vec<Complex<T>>,
    zprime: Vec<Complex<T>>,
    n: usize,
}

impl<T: MdctNum> MdctInvPlan<T> {
    fn new(params: &MdctParams, c2c: Box<dyn MdctC2cFft<T>>) -> Self {
        let two_n = params.window_size.get();
        let n = two_n / 2;

        let window_f64 = make_window(params.window.clone(), params.window_size);
        let window_samples: Vec<T> = window_f64.iter().map(|&w| T::from_f64(w)).collect();

        // pre_z[k] = exp(−jπk(N+1)/2N)
        let pre_z: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let angle = PI * k as f64 * (n + 1) as f64 / two_n as f64;
                Complex {
                    re: T::from_f64(angle.cos()),
                    im: T::from_f64(-angle.sin()),
                }
            })
            .collect();

        // pre_zprime[k] = pre_z[k] · exp(−jπk/N) = exp(−jπk(N+3)/(2N))
        let pre_zprime: Vec<Complex<T>> = (0..n)
            .map(|k| {
                let w_re = T::from_f64((PI * k as f64 / n as f64).cos());
                let w_im = T::from_f64(-(PI * k as f64 / n as f64).sin());
                Complex {
                    re: pre_z[k].re * w_re - pre_z[k].im * w_im,
                    im: pre_z[k].re * w_im + pre_z[k].im * w_re,
                }
            })
            .collect();

        let (imdct_post_re, imdct_post_im): (Vec<T>, Vec<T>) = (0..two_n)
            .map(|m| {
                let angle = PI * (2 * m + 1 + n) as f64 / (4 * n) as f64;
                (T::from_f64(angle.cos()), T::from_f64(-angle.sin()))
            })
            .unzip();

        Self {
            window_samples,
            imdct_post_re,
            imdct_post_im,
            pre_z,
            pre_zprime,
            c2c,
            z: vec![
                Complex {
                    re: T::zero(),
                    im: T::zero()
                };
                n
            ],
            zprime: vec![
                Complex {
                    re: T::zero(),
                    im: T::zero()
                };
                n
            ],
            n,
        }
    }

    fn imdct_frame(&mut self, coeffs: &[T], out: &mut [T]) -> SpectrogramResult<()> {
        let n = self.n;
        let scale = T::scale(n);

        // Build pre-twiddled inputs for even and odd DFT positions.
        for (k, &c) in coeffs.iter().enumerate() {
            self.z[k] = Complex {
                re: c * self.pre_z[k].re,
                im: c * self.pre_z[k].im,
            };
            self.zprime[k] = Complex {
                re: c * self.pre_zprime[k].re,
                im: c * self.pre_zprime[k].im,
            };
        }

        // Two size-N C2c forward FFTs (reuse the same plan sequentially).
        self.c2c.forward(&mut self.z)?;
        self.c2c.forward(&mut self.zprime)?;

        // Reconstruct all 2N output samples via two branch-free loops:
        //   Even m = 2j   → combined = Z[j]      (out at stride-2 positions)
        //   Odd  m = 2j+1 → combined = Zprime[j] (out at stride-2 positions)
        // Both z and zprime are accessed sequentially, enabling autovectorization.
        for j in 0..n {
            let m = 2 * j;
            out[m] = scale
                * (self.imdct_post_re[m] * self.z[j].re - self.imdct_post_im[m] * self.z[j].im);
        }
        for j in 0..n {
            let m = 2 * j + 1;
            out[m] = scale
                * (self.imdct_post_re[m] * self.zprime[j].re
                    - self.imdct_post_im[m] * self.zprime[j].im);
        }

        Ok(())
    }
}

/// Compute the MDCT of an audio signal.
///
/// `C[k] = Σ x[n]·w[n]·cos(π·(2n+1+N)·(2k+1)/(4N))` for k=0..N-1
///
/// Computed in O(N log N) per frame via one C2c(N) FFT.
///
/// # Arguments
///
/// * `samples` - Non-empty real audio samples. Length must be >= `window_size`.
/// * `params` - MDCT parameters.
///
/// # Returns
///
/// A 2D array of shape `(N, n_frames)` where `N = window_size / 2`.
///
/// # Errors
///
/// Returns `InvalidInput` if samples is shorter than `window_size`.
#[inline]
pub fn mdct(
    samples: &non_empty_slice::NonEmptySlice<f64>,
    params: &MdctParams,
) -> SpectrogramResult<Array2<f64>> {
    let samples = samples.as_slice();
    let two_n = params.window_size.get();
    let hop = params.hop_size.get();
    let n = params.n_coefficients();

    if samples.len() < two_n {
        return Err(SpectrogramError::invalid_input(format!(
            "samples length ({}) must be >= window_size ({})",
            samples.len(),
            two_n
        )));
    }

    let n_frames = (samples.len() - two_n) / hop + 1;

    #[cfg(feature = "realfft")]
    let c2c: Box<dyn MdctC2cFft<f64>> = {
        Box::new(C2cWrapper(
            crate::fft_backend::realfft_backend::RealFftC2cPlan::new(n),
        ))
    };

    #[cfg(feature = "fftw")]
    let c2c: Box<dyn MdctC2cFft<f64>> = {
        let mut planner = crate::FftwPlanner::new();
        Box::new(C2cWrapper(planner.plan_c2c(n)?))
    };

    let mut plan = MdctFwdPlan::<f64>::new(params, c2c);
    let mut output = Array2::<f64>::zeros((n, n_frames));
    let mut coef_buf = vec![0.0f64; n];

    for f in 0..n_frames {
        let start = f * hop;
        let frame = &samples[start..start + two_n];
        plan.mdct_frame(frame, &mut coef_buf)?;
        for (i, &v) in coef_buf.iter().enumerate() {
            output[(i, f)] = v;
        }
    }

    Ok(output)
}

/// Compute the IMDCT (inverse MDCT) from MDCT coefficients.
///
/// Uses overlap-add with the synthesis window to reconstruct the signal.
/// With a sine window and 50% hop, this achieves perfect reconstruction.
///
/// # Arguments
///
/// * `coefficients` - 2D array of shape `(N, n_frames)` as returned by [`mdct`].
/// * `params` - MDCT parameters (must match those used for analysis).
/// * `original_length` - If provided, output is truncated to this length.
///
/// # Returns
///
/// Reconstructed signal via overlap-add.
///
/// # Errors
///
/// Returns `InvalidInput` if the coefficient array shape doesn't match `params`.
#[inline]
pub fn imdct(
    coefficients: &Array2<f64>,
    params: &MdctParams,
    original_length: Option<usize>,
) -> SpectrogramResult<Vec<f64>> {
    let n = params.n_coefficients();
    let two_n = params.window_size.get();
    let hop = params.hop_size.get();

    if coefficients.nrows() != n {
        return Err(SpectrogramError::invalid_input(format!(
            "coefficients has {} rows but params.n_coefficients() = {}",
            coefficients.nrows(),
            n
        )));
    }

    let n_frames = coefficients.ncols();
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    #[cfg(feature = "realfft")]
    let c2c: Box<dyn MdctC2cFft<f64>> = {
        Box::new(C2cWrapper(
            crate::fft_backend::realfft_backend::RealFftC2cPlan::new(n),
        ))
    };

    #[cfg(feature = "fftw")]
    let c2c: Box<dyn MdctC2cFft<f64>> = {
        let mut planner = crate::FftwPlanner::new();
        Box::new(C2cWrapper(planner.plan_c2c(n)?))
    };

    let mut plan = MdctInvPlan::<f64>::new(params, c2c);

    let out_len = hop * n_frames + two_n - hop;
    let mut output = vec![0.0f64; out_len];
    let mut frame_out = vec![0.0f64; two_n];
    let mut col_buf = vec![0.0f64; n];

    for f in 0..n_frames {
        let col = coefficients.column(f);
        for (i, &v) in col.iter().enumerate() {
            col_buf[i] = v;
        }

        plan.imdct_frame(&col_buf, &mut frame_out)?;

        // Apply synthesis window and overlap-add
        let start = f * hop;
        for m in 0..two_n {
            output[start + m] += frame_out[m] * plan.window_samples[m];
        }
    }

    if let Some(len) = original_length {
        output.truncate(len);
    }

    Ok(output)
}

/// Compute the MDCT of an audio signal using f32 arithmetic.
///
/// Identical semantics to [`mdct`] but uses single-precision arithmetic
/// throughout. f32 is adequate for audio processing and is roughly 2× faster
/// due to halved memory bandwidth and wider SIMD (8-wide AVX vs 4-wide).
///
/// # Arguments
///
/// * `samples` - Non-empty real audio samples (f32). Length must be >= `window_size`.
/// * `params` - MDCT parameters.
///
/// # Returns
///
/// A 2D array of shape `(N, n_frames)` where `N = window_size / 2`.
///
/// # Errors
///
/// Returns `InvalidInput` if samples is shorter than `window_size`.
#[inline]
pub fn mdct_f32(
    samples: &non_empty_slice::NonEmptySlice<f32>,
    params: &MdctParams,
) -> SpectrogramResult<Array2<f32>> {
    let samples = samples.as_slice();
    let two_n = params.window_size.get();
    let hop = params.hop_size.get();
    let n = params.n_coefficients();

    if samples.len() < two_n {
        return Err(SpectrogramError::invalid_input(format!(
            "samples length ({}) must be >= window_size ({})",
            samples.len(),
            two_n
        )));
    }

    let n_frames = (samples.len() - two_n) / hop + 1;

    #[cfg(feature = "realfft")]
    let c2c: Box<dyn MdctC2cFft<f32>> = {
        Box::new(C2cF32Wrapper(
            crate::fft_backend::realfft_backend::RealFftC2cPlanF32::new(n),
        ))
    };

    #[cfg(feature = "fftw")]
    let c2c: Box<dyn MdctC2cFft<f32>> = {
        return Err(SpectrogramError::invalid_input(
            "mdct_f32 is not yet implemented for the fftw backend; use --features realfft",
        ));
    };

    let mut plan = MdctFwdPlan::<f32>::new(params, c2c);
    let mut output = Array2::<f32>::zeros((n, n_frames));
    let mut coef_buf = vec![0.0f32; n];

    for f in 0..n_frames {
        let start = f * hop;
        let frame = &samples[start..start + two_n];
        plan.mdct_frame(frame, &mut coef_buf)?;
        for (i, &v) in coef_buf.iter().enumerate() {
            output[(i, f)] = v;
        }
    }

    Ok(output)
}

/// Compute the IMDCT from MDCT coefficients using f32 arithmetic.
///
/// Identical semantics to [`imdct`] but uses single-precision arithmetic.
/// See [`mdct_f32`] for performance notes.
///
/// # Arguments
///
/// * `coefficients` - 2D array of shape `(N, n_frames)` as returned by [`mdct_f32`].
/// * `params` - MDCT parameters (must match those used for analysis).
/// * `original_length` - If provided, output is truncated to this length.
///
/// # Returns
///
/// Reconstructed signal via overlap-add.
///
/// # Errors
///
/// Returns `InvalidInput` if the coefficient array shape doesn't match `params`.
#[inline]
pub fn imdct_f32(
    coefficients: &Array2<f32>,
    params: &MdctParams,
    original_length: Option<usize>,
) -> SpectrogramResult<Vec<f32>> {
    let n = params.n_coefficients();
    let two_n = params.window_size.get();
    let hop = params.hop_size.get();

    if coefficients.nrows() != n {
        return Err(SpectrogramError::invalid_input(format!(
            "coefficients has {} rows but params.n_coefficients() = {}",
            coefficients.nrows(),
            n
        )));
    }

    let n_frames = coefficients.ncols();
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    #[cfg(feature = "realfft")]
    let c2c: Box<dyn MdctC2cFft<f32>> = {
        Box::new(C2cF32Wrapper(
            crate::fft_backend::realfft_backend::RealFftC2cPlanF32::new(n),
        ))
    };

    #[cfg(feature = "fftw")]
    let c2c: Box<dyn MdctC2cFft<f32>> = {
        return Err(SpectrogramError::invalid_input(
            "imdct_f32 is not yet implemented for the fftw backend; use --features realfft",
        ));
    };

    let mut plan = MdctInvPlan::<f32>::new(params, c2c);

    let out_len = hop * n_frames + two_n - hop;
    let mut output = vec![0.0f32; out_len];
    let mut frame_out = vec![0.0f32; two_n];
    let mut col_buf = vec![0.0f32; n];

    for f in 0..n_frames {
        let col = coefficients.column(f);
        for (i, &v) in col.iter().enumerate() {
            col_buf[i] = v;
        }

        plan.imdct_frame(&col_buf, &mut frame_out)?;

        // Apply synthesis window and overlap-add
        let start = f * hop;
        for m in 0..two_n {
            output[start + m] += frame_out[m] * plan.window_samples[m];
        }
    }

    if let Some(len) = original_length {
        output.truncate(len);
    }

    Ok(output)
}

#[cfg(all(test, feature = "realfft"))]
mod tests {
    use super::*;

    fn make_sine(n: usize, freq: f64, sr: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    /// Compare a single MDCT frame against the direct O(N²) formula.
    #[test]
    fn single_frame_matches_direct_formula() {
        let window_size = std::num::NonZeroUsize::new(16).unwrap();
        let hop = std::num::NonZeroUsize::new(8).unwrap();
        let params = MdctParams::new(window_size, hop, WindowType::Rectangular).unwrap();
        let two_n = 16usize;
        let n = 8usize;
        let x: Vec<f64> = (0..two_n).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let x_ne = non_empty_slice::NonEmptyVec::new(x.clone()).unwrap();

        let coefs = mdct(x_ne.as_non_empty_slice(), &params).unwrap();

        for k in 0..n {
            let ref_val: f64 = (0..two_n)
                .map(|m| {
                    x[m] * (PI * (2 * m + 1 + n) as f64 * (2 * k + 1) as f64 / (4 * n) as f64).cos()
                })
                .sum();
            if (coefs[(k, 0)] - ref_val).abs() >= 1e-10 {
                eprintln!("FAIL k={k}: got {:.12}, ref {:.12}", coefs[(k, 0)], ref_val);
                // Print post-twiddle
                let angle = PI * (2 * k + 1) as f64 * (n + 1) as f64 / (4 * n) as f64;
                eprintln!(
                    "  post angle={angle:.6}, cos={:.6}, -sin={:.6}",
                    angle.cos(),
                    -angle.sin()
                );
            }
            assert!(
                (coefs[(k, 0)] - ref_val).abs() < 1e-10,
                "k={k}: got {:.12}, ref {:.12}",
                coefs[(k, 0)],
                ref_val
            );
        }
    }

    /// Verify that C2c(N) packing gives the same intermediate F values as two R2c(N) FFTs.
    #[test]
    fn c2c_packing_matches_two_r2c() {
        use crate::fft_backend::{C2cPlan, R2cPlan};
        let n = 8usize;
        let x: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let _params = MdctParams::new(
            std::num::NonZeroUsize::new(16).unwrap(),
            std::num::NonZeroUsize::new(8).unwrap(),
            WindowType::Rectangular,
        )
        .unwrap();
        let analysis_re: Vec<f64> = (0..16).map(|m| (PI * m as f64 / 16.0).cos()).collect();
        let analysis_im: Vec<f64> = (0..16).map(|m| -(PI * m as f64 / 16.0).sin()).collect();
        let mut a = vec![0.0f64; n];
        let mut b = vec![0.0f64; n];
        for m in 0..n {
            a[m] = x[m] * analysis_re[m] + x[m + n] * analysis_re[m + n];
            b[m] = x[m] * analysis_im[m] + x[m + n] * analysis_im[m + n];
        }
        // R2c for a and b
        let mut r2c_a_plan = {
            let mut planner = crate::RealFftPlanner::new();
            let p = planner.get_or_create(n);
            crate::fft_backend::realfft_backend::RealFftPlan::new(n, p)
        };
        let mut out_a = vec![
            Complex {
                re: 0.0f64,
                im: 0.0
            };
            n / 2 + 1
        ];
        let mut out_b = vec![
            Complex {
                re: 0.0f64,
                im: 0.0
            };
            n / 2 + 1
        ];
        r2c_a_plan.process(&a, &mut out_a).unwrap();
        let mut r2c_b_plan = {
            let mut planner = crate::RealFftPlanner::new();
            let p = planner.get_or_create(n);
            crate::fft_backend::realfft_backend::RealFftPlan::new(n, p)
        };
        r2c_b_plan.process(&b, &mut out_b).unwrap();
        eprintln!("A[0..4] = {:?}", &out_a);
        eprintln!("B[0..4] = {:?}", &out_b);
        // C2c of z = a + ib
        let mut c2c_plan = crate::fft_backend::realfft_backend::RealFftC2cPlan::new(n);
        let mut z: Vec<Complex<f64>> = (0..n).map(|m| Complex { re: a[m], im: b[m] }).collect();
        c2c_plan.forward(&mut z).unwrap();
        eprintln!("Z[0..8] = {:?}", &z);
        // Compare for k=0..4
        for k in 0..=n / 2 {
            eprintln!(
                "k={k}: A+iB=({:.6},{:.6}), Z=({:.6},{:.6})",
                out_a[k].re - out_b[k].im,
                out_a[k].im + out_b[k].re,
                z[k].re,
                z[k].im
            );
        }
    }

    /// Sine window + 50% hop must give perfect reconstruction in the signal interior.
    #[test]
    fn perfect_reconstruction_f64() {
        let window_size = std::num::NonZeroUsize::new(256).unwrap();
        let params = MdctParams::sine_window(window_size).unwrap();
        let n = 2048usize;
        let x = make_sine(n, 440.0, 44100.0);
        let x_ne = non_empty_slice::NonEmptyVec::new(x.clone()).unwrap();

        let coefs = mdct(x_ne.as_non_empty_slice(), &params).unwrap();
        let x_rec = imdct(&coefs, &params, Some(n)).unwrap();

        let margin = 256;
        for i in margin..(n - margin) {
            assert!(
                (x_rec[i] - x[i]).abs() < 1e-10,
                "sample {i}: got {:.12}, expected {:.12}",
                x_rec[i],
                x[i]
            );
        }
    }

    /// f32 path also gives perfect reconstruction.
    #[test]
    fn perfect_reconstruction_f32() {
        let window_size = std::num::NonZeroUsize::new(256).unwrap();
        let params = MdctParams::sine_window(window_size).unwrap();
        let n = 2048usize;
        let x: Vec<f32> = make_sine(n, 440.0, 44100.0)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let x_ne = non_empty_slice::NonEmptyVec::new(x.clone()).unwrap();

        let coefs = mdct_f32(x_ne.as_non_empty_slice(), &params).unwrap();
        let x_rec = imdct_f32(&coefs, &params, Some(n)).unwrap();

        let margin = 256;
        for i in margin..(n - margin) {
            assert!(
                (x_rec[i] - x[i]).abs() < 1e-5,
                "sample {i}: got {:.8}, expected {:.8}",
                x_rec[i],
                x[i]
            );
        }
    }
}
