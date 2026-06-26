use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult};

/// Output size for a real-to-complex FFT of length `n`.
///
/// # Arguments
///
/// * `n` - Length of the real input
///
/// # Returns
///
/// Length of the complex output slice required for the FFT.
#[inline]
#[must_use]
pub const fn r2c_output_size(n: usize) -> usize {
    n / 2 + 1
}
/// A planned real-to-complex FFT for a fixed transform length.
///
/// Plans must:
/// - own any internal scratch buffers
/// - be reusable across many calls
/// - perform no heap allocation during `process`
pub trait R2cPlan<T = f64> {
    fn n_fft(&self) -> usize;
    fn output_len(&self) -> usize;

    /// Process real input to complex output.
    ///
    /// # Arguments
    ///
    /// * `input` - Real time domain input
    /// * `output` - Complex frequency domain output
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or SpectrogramError if dimensions are invalid.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
    fn process(&mut self, input: &[T], output: &mut [Complex<T>]) -> SpectrogramResult<()>;
}

/// Planner that can construct FFT plans.
pub trait R2cPlanner<T = f64> {
    type Plan: R2cPlan<T>;

    /// Plan a real-to-complex FFT.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size
    ///
    /// # Returns
    ///
    /// A plan that can perform the FFT.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError if planning fails.
    fn plan_r2c(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan>;
}

/// A planned complex-to-real inverse FFT for a fixed transform length.
pub trait C2rPlan<T = f64> {
    /// FFT size
    ///
    /// # Returns
    ///
    /// Length of the FFT.
    fn n_fft(&self) -> usize;

    /// Input length for complex-to-real inverse FFT is r2c_output_size(n_fft)
    ///
    /// # Returns
    ///
    /// Length of the complex input slice required for the inverse FFT.
    fn input_len(&self) -> usize;

    /// Process complex input to real output.
    ///
    /// # Arguments
    ///
    /// * `input` - Complex frequency domain input
    /// * `output` - Real time domain output
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or SpectrogramError if dimensions are invalid.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
    fn process(&mut self, input: &[Complex<T>], output: &mut [T]) -> SpectrogramResult<()>;
}

/// Planner that can construct inverse FFT plans.
pub trait C2rPlanner<T = f64> {
    type Plan: C2rPlan<T>;

    /// Plan a complex-to-real inverse FFT.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size
    ///
    /// # Returns
    ///
    /// A plan that can perform the inverse FFT.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError if planning fails.
    fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan>;
}

/// A planned complex-to-complex FFT for a fixed transform length.
///
/// The transform is **in-place**: input is overwritten with output.
/// Plans must own internal scratch buffers and perform no heap allocation during `forward`/`inverse`.
///
/// Normalization: neither `forward` nor `inverse` normalizes. The caller is responsible
/// for dividing by N after an inverse transform if a round-trip is desired.
pub trait C2cPlan<T = f64>: Send {
    fn n_fft(&self) -> usize;
    /// Forward DFT: `X[k] = Σ x[n] exp(−2πi·k·n/N)`
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length doesn't match the plan size.
    fn forward(&mut self, buf: &mut [Complex<T>]) -> SpectrogramResult<()>;
    /// Inverse DFT (unnormalized): `x[n] = Σ X[k] exp(+2πi·k·n/N)`
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length doesn't match the plan size.
    fn inverse(&mut self, buf: &mut [Complex<T>]) -> SpectrogramResult<()>;
}

// ============================================================================
// 2D FFT Traits
// ============================================================================

/// Output size for a 2D real-to-complex FFT of dimensions (nrows, ncols).
/// The output has shape (nrows, ncols/2 + 1) due to Hermitian symmetry.
///
/// # Arguments
///
/// * `nrows` - Number of rows in the input
/// * `ncols` - Number of columns in the input
///
/// # Returns
///
/// A tuple (out_rows, out_cols) representing the output dimensions.
#[inline]
#[must_use]
pub const fn r2c_output_size_2d(nrows: usize, ncols: usize) -> (usize, usize) {
    (nrows, ncols / 2 + 1)
}

/// A planned 2D real-to-complex FFT for fixed dimensions.
///
/// Plans must:
/// - own any internal scratch buffers
/// - be reusable across many calls
/// - perform no heap allocation during `process`
pub trait R2cPlan2d<T = f64> {
    /// Process 2D real input to complex output.
    ///
    /// # Arguments
    ///
    /// * `input` - nrows x ncols real values (row-major, flat slice)
    /// * `output` - nrows x (ncols/2 + 1) complex values (row-major, flat slice)
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or SpectrogramError if dimensions are invalid.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
    fn process(&mut self, input: &[T], output: &mut [Complex<T>]) -> SpectrogramResult<()>;
}

/// Planner that can construct 2D FFT plans.
pub trait R2cPlanner2d<T = f64> {
    type Plan: R2cPlan2d<T>;

    /// Plan a 2D real-to-complex FFT.
    ///
    ///  # Arguments
    ///
    /// * `nrows` - Number of rows in the input
    /// * `ncols` - Number of columns in the input
    ///
    /// # Returns
    ///
    /// A plan that can perform the 2D FFT.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError if planning fails.
    fn plan_r2c_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan>;
}

/// A planned 2D complex-to-real inverse FFT for fixed dimensions.
pub trait C2rPlan2d<T = f64> {
    /// Process 2D complex input to real output.
    ///
    /// # Arguments
    ///
    /// * `input` - nrows x (ncols/2 + 1) complex values (row-major, flat slice)
    /// * `output` - nrows x ncols real values (row-major, flat slice)
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or SpectrogramError if dimensions are invalid.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
    fn process(&mut self, input: &[Complex<T>], output: &mut [T]) -> SpectrogramResult<()>;
}

/// Planner that can construct 2D inverse FFT plans.
pub trait C2rPlanner2d<T = f64> {
    type Plan: C2rPlan2d<T>;

    /// Plan a 2D complex-to-real inverse FFT.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows in the output
    /// * `ncols` - Number of columns in the output
    ///
    /// # Returns
    ///
    /// A plan that can perform the 2D inverse FFT.
    ///
    /// # Errors
    ///
    /// Returns SpectrogramError if planning fails.
    fn plan_c2r_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan>;
}

/// Validate 1D FFT input/output dimensions.
///
/// # Arguments
///
/// * `n_fft` - FFT size
/// * `input` - Input slice of real values
/// * `output` - Output slice of complex values
///
/// # Returns
///
/// Ok(()) if dimensions are valid, or SpectrogramError if not.
///
/// # Errors
///
/// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
#[inline]
pub fn validate_fft_io<T>(
    n_fft: usize,
    input: &[T],
    output: &[Complex<T>],
) -> SpectrogramResult<()> {
    if input.len() != n_fft {
        return Err(SpectrogramError::dimension_mismatch(n_fft, input.len()));
    }

    let expected_out = r2c_output_size(n_fft);
    if output.len() != expected_out {
        return Err(SpectrogramError::dimension_mismatch(
            expected_out,
            output.len(),
        ));
    }

    Ok(())
}

/// Validate 2D FFT input/output dimensions.
///
/// # Arguments
///
/// * `nrows` - Number of rows in the input
/// * `ncols` - Number of columns in the input
/// * `input` - Input slice of real values
/// * `output` - Output slice of complex values
///
/// # Returns
///
/// Ok(()) if dimensions are valid, or SpectrogramError if not.
///
/// # Errors
///
/// Returns SpectrogramError::dimension_mismatch if input or output lengths do not match expected sizes.
#[inline]
pub fn validate_fft_io_2d<T>(
    nrows: usize,
    ncols: usize,
    input: &[T],
    output: &[Complex<T>],
) -> SpectrogramResult<()> {
    let input_len = nrows * ncols;
    if input.len() != input_len {
        return Err(SpectrogramError::dimension_mismatch(input_len, input.len()));
    }

    let (out_rows, out_cols) = r2c_output_size_2d(nrows, ncols);
    let output_len = out_rows * out_cols;
    if output.len() != output_len {
        return Err(SpectrogramError::dimension_mismatch(
            output_len,
            output.len(),
        ));
    }

    Ok(())
}

#[cfg(feature = "realfft")]
pub mod realfft_backend {
    use std::collections::HashMap;
    use std::sync::Arc;

    use num_complex::Complex;
    pub use realfft::{ComplexToReal, FftNum, RealFftPlanner as InnerPlanner, RealToComplex};

    use crate::fft_backend::{
        C2rPlan, C2rPlanner, R2cPlan, R2cPlanner, r2c_output_size, validate_fft_io,
    };
    use crate::{SpectrogramError, SpectrogramResult};

    /// RealFftPlanner
    ///
    /// Wraps the realfft::RealFftPlanner and adds caching for created plans.
    ///
    /// This planner maintains separate caches for forward (real-to-complex)
    /// and inverse (complex-to-real) FFT plans to avoid redundant plan creation.
    pub struct RealFftPlanner<T: FftNum = f64> {
        inner: InnerPlanner<T>,
        cache_r2c: HashMap<usize, Arc<dyn RealToComplex<T>>>,
        cache_c2r: HashMap<usize, Arc<dyn ComplexToReal<T>>>,
    }

    impl<T: FftNum> Default for RealFftPlanner<T> {
        #[inline]
        fn default() -> Self {
            Self {
                inner: InnerPlanner::<T>::new(),
                cache_r2c: HashMap::new(),
                cache_c2r: HashMap::new(),
            }
        }
    }

    impl<T: FftNum> RealFftPlanner<T> {
        /// Create a new RealFftPlanner with empty caches.
        ///
        /// # Returns
        ///
        /// A new instance of `RealFftPlanner` with empty caches.
        #[inline]
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        pub(crate) fn get_or_create(&mut self, n_fft: usize) -> Arc<dyn RealToComplex<T>> {
            if let Some(p) = self.cache_r2c.get(&n_fft) {
                return p.clone();
            }
            let p = self.inner.plan_fft_forward(n_fft);
            self.cache_r2c.insert(n_fft, p.clone());
            p
        }

        pub(crate) fn get_or_create_inverse(&mut self, n_fft: usize) -> Arc<dyn ComplexToReal<T>> {
            if let Some(p) = self.cache_c2r.get(&n_fft) {
                return p.clone();
            }
            let p = self.inner.plan_fft_inverse(n_fft);
            self.cache_c2r.insert(n_fft, p.clone());
            p
        }
    }

    /// Real-to-Complex FFT Plan
    ///
    /// Implements a plan for performing FFTs from real time domain to complex frequency domain.=
    #[derive(Clone)]
    pub struct RealFftPlan<T: FftNum = f64> {
        n_fft: usize,
        plan: Arc<dyn RealToComplex<T>>,
        scratch: Vec<T>,
    }

    impl<T: FftNum> RealFftPlan<T> {
        pub(crate) fn new(n_fft: usize, plan: Arc<dyn RealToComplex<T>>) -> Self {
            Self {
                n_fft,
                plan,
                scratch: vec![T::zero(); n_fft],
            }
        }
    }

    impl<T: FftNum> R2cPlan<T> for RealFftPlan<T> {
        #[inline]
        fn n_fft(&self) -> usize {
            self.n_fft
        }

        #[inline]
        fn output_len(&self) -> usize {
            r2c_output_size(self.n_fft)
        }

        #[inline]
        fn process(&mut self, input: &[T], output: &mut [Complex<T>]) -> SpectrogramResult<()> {
            validate_fft_io(self.n_fft, input, output)?;

            self.scratch.copy_from_slice(input);

            self.plan
                .process(&mut self.scratch, output)
                .map_err(|e| SpectrogramError::fft_backend("realfft", format!("{e:?}")))
        }
    }

    impl<T: FftNum> R2cPlanner<T> for RealFftPlanner<T> {
        type Plan = RealFftPlan<T>;

        #[inline]
        fn plan_r2c(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            let plan = self.get_or_create(n_fft);
            Ok(RealFftPlan::new(n_fft, plan))
        }
    }

    // ── Complex-to-Complex plans (f64 and f32) ────────────────────────────────

    /// Complex-to-complex FFT plan (f64) backed by rustfft.
    ///
    /// Both forward and inverse directions share a single scratch buffer sized to
    /// the larger of the two inplace scratch requirements.
    pub struct RealFftC2cPlan<T: rustfft::FftNum = f64> {
        n_fft: usize,
        fwd: Arc<dyn rustfft::Fft<T>>,
        inv: Arc<dyn rustfft::Fft<T>>,
        scratch: Vec<Complex<T>>,
    }

    impl<T: rustfft::FftNum> RealFftC2cPlan<T> {
        /// Create a C2c plan for transform size `n_fft`.
        #[must_use]
        pub fn new(n_fft: usize) -> Self {
            let mut p = RustFftPlanner::<T>::new();
            let fwd = p.plan_fft_forward(n_fft);
            let inv = p.plan_fft_inverse(n_fft);
            let scratch_len = fwd
                .get_inplace_scratch_len()
                .max(inv.get_inplace_scratch_len());
            Self {
                n_fft,
                fwd,
                inv,
                scratch: vec![Complex::new(T::zero(), T::zero()); scratch_len],
            }
        }
    }

    impl<T: rustfft::FftNum> crate::fft_backend::C2cPlan<T> for RealFftC2cPlan<T> {
        #[inline]
        fn n_fft(&self) -> usize {
            self.n_fft
        }

        #[inline]
        fn forward(&mut self, buf: &mut [Complex<T>]) -> SpectrogramResult<()> {
            if buf.len() != self.n_fft {
                return Err(SpectrogramError::dimension_mismatch(self.n_fft, buf.len()));
            }
            self.fwd.process_with_scratch(buf, &mut self.scratch);
            Ok(())
        }

        #[inline]
        fn inverse(&mut self, buf: &mut [Complex<T>]) -> SpectrogramResult<()> {
            if buf.len() != self.n_fft {
                return Err(SpectrogramError::dimension_mismatch(self.n_fft, buf.len()));
            }
            self.inv.process_with_scratch(buf, &mut self.scratch);
            Ok(())
        }
    }

    // SAFETY: rustfft::Fft<T> is Send; scratch is owned by the plan.
    unsafe impl<T: rustfft::FftNum> Send for RealFftC2cPlan<T> {}

    /// Complex-to-Real Inverse FFT Plan
    ///
    /// Implements a plan for performing inverse FFTs from complex frequency domain
    /// to real time domain.
    #[derive(Clone)]
    pub struct RealFftInversePlan<T: FftNum = f64> {
        n_fft: usize,
        plan: Arc<dyn ComplexToReal<T>>,
        scratch: Vec<Complex<T>>,
    }

    impl<T: FftNum> RealFftInversePlan<T> {
        pub(crate) fn new(n_fft: usize, plan: Arc<dyn ComplexToReal<T>>) -> Self {
            let scratch_len = r2c_output_size(n_fft);
            Self {
                n_fft,
                plan,
                scratch: vec![Complex::new(T::zero(), T::zero()); scratch_len],
            }
        }
    }

    impl<T: FftNum> C2rPlan<T> for RealFftInversePlan<T> {
        #[inline]
        fn n_fft(&self) -> usize {
            self.n_fft
        }

        #[inline]
        fn input_len(&self) -> usize {
            r2c_output_size(self.n_fft)
        }

        #[inline]
        fn process(&mut self, input: &[Complex<T>], output: &mut [T]) -> SpectrogramResult<()> {
            let expected_in = r2c_output_size(self.n_fft);
            if input.len() != expected_in {
                return Err(SpectrogramError::dimension_mismatch(
                    expected_in,
                    input.len(),
                ));
            }
            if output.len() != self.n_fft {
                return Err(SpectrogramError::dimension_mismatch(
                    self.n_fft,
                    output.len(),
                ));
            }

            self.scratch.copy_from_slice(input);

            self.plan
                .process(&mut self.scratch, output)
                .map_err(|e| SpectrogramError::fft_backend("realfft", format!("{e:?}")))?;

            // RealFFT inverse needs normalization
            let scale = T::one() / T::from_usize(self.n_fft).unwrap_or_else(T::one);
            for val in output.iter_mut() {
                *val = *val * scale;
            }

            Ok(())
        }
    }

    impl<T: FftNum> C2rPlanner<T> for RealFftPlanner<T> {
        type Plan = RealFftInversePlan<T>;

        #[inline]
        fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            let plan = self.get_or_create_inverse(n_fft);
            Ok(RealFftInversePlan::new(n_fft, plan))
        }
    }

    // ========================================================================
    // 2D FFT Implementation (Row-Column Decomposition)
    // ========================================================================

    #[cfg(feature = "realfft")]
    use rustfft::FftPlanner as RustFftPlanner;

    impl<T: FftNum> RealFftPlanner<T> {
        /// Get or create a complex-to-complex FFT plan for column transforms.
        pub(crate) fn get_or_create_complex(n: usize) -> Arc<dyn rustfft::Fft<T>> {
            // Use the inner RealFftPlanner's complex FFT planner
            // Note: realfft::RealFftPlanner wraps rustfft::FftPlanner
            // We need to create a separate planner for complex FFTs
            let mut complex_planner = RustFftPlanner::<T>::new();
            complex_planner.plan_fft_forward(n)
        }

        pub(crate) fn get_or_create_complex_inverse(n: usize) -> Arc<dyn rustfft::Fft<T>> {
            let mut complex_planner = RustFftPlanner::<T>::new();
            complex_planner.plan_fft_inverse(n)
        }
    }

    /// 2D Real-to-Complex FFT Plan
    ///
    /// Implements row-column decomposition:
    ///
    /// 1. Perform 1D real-to-complex FFTs on each row
    /// 2. Perform 1D complex-to-complex FFTs on each column of the intermediate result
    ///
    /// This plan owns scratch buffers for row and column transforms, as well as
    /// an intermediate buffer to hold the results after the row transforms.
    ///
    /// The process method performs the full 2D FFT in-place without additional allocations.
    #[derive(Clone)]
    pub struct RealFftPlan2d<T: FftNum = f64> {
        nrows: usize,
        ncols: usize,
        out_shape: (usize, usize),
        // Plans for row transforms (real -> complex, size ncols)
        row_plan: Arc<dyn RealToComplex<T>>,
        // Plans for column transforms (complex -> complex, size nrows)
        col_plan: Arc<dyn rustfft::Fft<T>>,
        // Scratch buffers
        scratch_row: Vec<T>,
        scratch_col: Vec<Complex<T>>,
        // Intermediate storage after row FFTs
        intermediate: Vec<Complex<T>>,
    }

    impl<T: FftNum> RealFftPlan2d<T> {
        pub(crate) fn new(
            nrows: usize,
            ncols: usize,
            row_plan: Arc<dyn RealToComplex<T>>,
            col_plan: Arc<dyn rustfft::Fft<T>>,
        ) -> Self {
            let out_shape = crate::fft_backend::r2c_output_size_2d(nrows, ncols);
            let intermediate_len = nrows * out_shape.1;

            Self {
                nrows,
                ncols,
                out_shape,
                row_plan,
                col_plan,
                scratch_row: vec![T::zero(); ncols],
                scratch_col: vec![Complex::new(T::zero(), T::zero()); nrows],
                intermediate: vec![Complex::new(T::zero(), T::zero()); intermediate_len],
            }
        }
    }

    impl<T: FftNum> crate::fft_backend::R2cPlan2d<T> for RealFftPlan2d<T> {
        fn process(&mut self, input: &[T], output: &mut [Complex<T>]) -> SpectrogramResult<()> {
            crate::fft_backend::validate_fft_io_2d(self.nrows, self.ncols, input, output)?;

            // Step 1: FFT each row (real -> complex)
            for row_idx in 0..self.nrows {
                let row_start = row_idx * self.ncols;
                let row_end = row_start + self.ncols;
                self.scratch_row.copy_from_slice(&input[row_start..row_end]);

                let out_start = row_idx * self.out_shape.1;
                let out_end = out_start + self.out_shape.1;

                self.row_plan
                    .process(
                        &mut self.scratch_row,
                        &mut self.intermediate[out_start..out_end],
                    )
                    .map_err(|e| SpectrogramError::fft_backend("realfft", format!("{e:?}")))?;
            }

            // Step 2: FFT each column (complex -> complex)
            for col_idx in 0..self.out_shape.1 {
                // Extract column from intermediate buffer
                for row_idx in 0..self.nrows {
                    self.scratch_col[row_idx] =
                        self.intermediate[row_idx * self.out_shape.1 + col_idx];
                }

                // FFT the column
                self.col_plan.process(&mut self.scratch_col);

                // Write column back to output
                for row_idx in 0..self.nrows {
                    output[row_idx * self.out_shape.1 + col_idx] = self.scratch_col[row_idx];
                }
            }

            Ok(())
        }
    }

    impl<T: FftNum> crate::fft_backend::R2cPlanner2d<T> for RealFftPlanner<T> {
        type Plan = RealFftPlan2d<T>;

        fn plan_r2c_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan> {
            let row_plan = self.get_or_create(ncols);
            let col_plan = Self::get_or_create_complex(nrows);
            Ok(RealFftPlan2d::new(nrows, ncols, row_plan, col_plan))
        }
    }

    #[derive(Clone)]
    pub struct RealFftInversePlan2d<T: FftNum = f64> {
        nrows: usize,
        ncols: usize,
        in_shape: (usize, usize),
        // Plans for column inverse transforms (complex -> complex, size nrows)
        col_plan: Arc<dyn rustfft::Fft<T>>,
        // Plans for row inverse transforms (complex -> real, size ncols)
        row_plan: Arc<dyn ComplexToReal<T>>,
        // Scratch buffers
        scratch_col: Vec<Complex<T>>,
        scratch_row: Vec<Complex<T>>,
        // Intermediate storage after column IFFTs
        intermediate: Vec<Complex<T>>,
    }

    impl<T: FftNum> RealFftInversePlan2d<T> {
        pub(crate) fn new(
            nrows: usize,
            ncols: usize,
            col_plan: Arc<dyn rustfft::Fft<T>>,
            row_plan: Arc<dyn ComplexToReal<T>>,
        ) -> Self {
            let in_shape = crate::fft_backend::r2c_output_size_2d(nrows, ncols);
            let intermediate_len = nrows * in_shape.1;

            Self {
                nrows,
                ncols,
                in_shape,
                col_plan,
                row_plan,
                scratch_col: vec![Complex::new(T::zero(), T::zero()); nrows],
                scratch_row: vec![Complex::new(T::zero(), T::zero()); in_shape.1],
                intermediate: vec![Complex::new(T::zero(), T::zero()); intermediate_len],
            }
        }
    }

    impl<T: FftNum> crate::fft_backend::C2rPlan2d<T> for RealFftInversePlan2d<T> {
        fn process(&mut self, input: &[Complex<T>], output: &mut [T]) -> SpectrogramResult<()> {
            let expected_in_len = self.in_shape.0 * self.in_shape.1;
            if input.len() != expected_in_len {
                return Err(SpectrogramError::dimension_mismatch(
                    expected_in_len,
                    input.len(),
                ));
            }

            let expected_out_len = self.nrows * self.ncols;
            if output.len() != expected_out_len {
                return Err(SpectrogramError::dimension_mismatch(
                    expected_out_len,
                    output.len(),
                ));
            }

            // Copy input to intermediate buffer
            self.intermediate.copy_from_slice(input);

            // Step 1: Inverse FFT each column (complex -> complex)
            for col_idx in 0..self.in_shape.1 {
                // Extract column
                for row_idx in 0..self.nrows {
                    self.scratch_col[row_idx] =
                        self.intermediate[row_idx * self.in_shape.1 + col_idx];
                }

                // IFFT the column
                self.col_plan.process(&mut self.scratch_col);

                // Write column back
                for row_idx in 0..self.nrows {
                    self.intermediate[row_idx * self.in_shape.1 + col_idx] =
                        self.scratch_col[row_idx];
                }
            }

            // Enforce Hermitian symmetry: DC (column 0) and Nyquist (last column if even)
            // must have purely real values for each row
            for row_idx in 0..self.nrows {
                // DC component (first column) must be real
                self.intermediate[row_idx * self.in_shape.1].im = T::zero();

                // Nyquist component (last column, if ncols is even) must be real
                if self.ncols.is_multiple_of(2) {
                    let nyquist_col = self.in_shape.1 - 1;
                    self.intermediate[row_idx * self.in_shape.1 + nyquist_col].im = T::zero();
                }
            }

            // Step 2: Inverse FFT each row (complex -> real)
            for row_idx in 0..self.nrows {
                let row_start = row_idx * self.in_shape.1;
                let row_end = row_start + self.in_shape.1;
                self.scratch_row
                    .copy_from_slice(&self.intermediate[row_start..row_end]);

                let out_start = row_idx * self.ncols;
                let out_end = out_start + self.ncols;

                self.row_plan
                    .process(&mut self.scratch_row, &mut output[out_start..out_end])
                    .map_err(|e| SpectrogramError::fft_backend("realfft", format!("{e:?}")))?;
            }

            // RealFFT inverse needs normalization
            let scale =
                T::one() / T::from_usize(self.nrows * self.ncols).unwrap_or_else(T::one);
            for val in output.iter_mut() {
                *val = *val * scale;
            }

            Ok(())
        }
    }

    impl<T: FftNum> crate::fft_backend::C2rPlanner2d<T> for RealFftPlanner<T> {
        type Plan = RealFftInversePlan2d<T>;

        fn plan_c2r_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan> {
            let col_plan = Self::get_or_create_complex_inverse(nrows);
            let row_plan = self.get_or_create_inverse(ncols);
            Ok(RealFftInversePlan2d::new(nrows, ncols, col_plan, row_plan))
        }
    }

    // ── Sample implementations (realfft backend) ──────────────────────────────
    //
    // Each `plan_*` constructor builds a fresh plan per call.
    // TODO: per-type plan cache to avoid re-planning on every construction.
    macro_rules! impl_sample_realfft {
        ($t:ty) => {
            impl crate::Sample for $t {
                type R2cPlan = RealFftPlan<$t>;
                type C2rPlan = RealFftInversePlan<$t>;
                type C2cPlan = RealFftC2cPlan<$t>;
                type R2cPlan2d = RealFftPlan2d<$t>;
                type C2rPlan2d = RealFftInversePlan2d<$t>;

                fn plan_r2c(n_fft: usize) -> SpectrogramResult<Self::R2cPlan> {
                    let mut planner = InnerPlanner::<$t>::new();
                    let plan = planner.plan_fft_forward(n_fft);
                    Ok(RealFftPlan::new(n_fft, plan))
                }

                fn plan_c2r(n_fft: usize) -> SpectrogramResult<Self::C2rPlan> {
                    let mut planner = InnerPlanner::<$t>::new();
                    let plan = planner.plan_fft_inverse(n_fft);
                    Ok(RealFftInversePlan::new(n_fft, plan))
                }

                fn plan_c2c(n_fft: usize) -> SpectrogramResult<Self::C2cPlan> {
                    Ok(RealFftC2cPlan::<$t>::new(n_fft))
                }

                fn plan_r2c_2d(
                    nrows: usize,
                    ncols: usize,
                ) -> SpectrogramResult<Self::R2cPlan2d> {
                    use crate::fft_backend::R2cPlanner2d;
                    let mut planner = RealFftPlanner::<$t>::new();
                    planner.plan_r2c_2d(nrows, ncols)
                }

                fn plan_c2r_2d(
                    nrows: usize,
                    ncols: usize,
                ) -> SpectrogramResult<Self::C2rPlan2d> {
                    use crate::fft_backend::C2rPlanner2d;
                    let mut planner = RealFftPlanner::<$t>::new();
                    planner.plan_c2r_2d(nrows, ncols)
                }

                #[inline]
                fn from_f64(x: f64) -> Self {
                    x as $t
                }

                #[inline]
                fn from_usize(n: usize) -> Self {
                    n as $t
                }
            }
        };
    }

    impl_sample_realfft!(f32);

    // f64 routes its one-shot R2C/C2R plans through the global plan cache, so
    // repeated `fft`/`irfft`/`fft_convolve` calls reuse precomputed twiddles and
    // `clear_plan_cache`/`cache_stats` stay meaningful. (f32 builds fresh above;
    // the f32 + plan-cache combination is uncommon.)
    impl crate::Sample for f64 {
        type R2cPlan = RealFftPlan<f64>;
        type C2rPlan = RealFftInversePlan<f64>;
        type C2cPlan = RealFftC2cPlan<f64>;
        type R2cPlan2d = RealFftPlan2d<f64>;
        type C2rPlan2d = RealFftInversePlan2d<f64>;

        fn plan_r2c(n_fft: usize) -> SpectrogramResult<Self::R2cPlan> {
            let plan = crate::fft_backend::get_or_create_r2c_plan(n_fft)?;
            Ok((*plan).clone())
        }

        fn plan_c2r(n_fft: usize) -> SpectrogramResult<Self::C2rPlan> {
            let plan = crate::fft_backend::get_or_create_c2r_plan(n_fft)?;
            Ok((*plan).clone())
        }

        fn plan_c2c(n_fft: usize) -> SpectrogramResult<Self::C2cPlan> {
            Ok(RealFftC2cPlan::<f64>::new(n_fft))
        }

        fn plan_r2c_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::R2cPlan2d> {
            use crate::fft_backend::R2cPlanner2d;
            let mut planner = RealFftPlanner::<f64>::new();
            planner.plan_r2c_2d(nrows, ncols)
        }

        fn plan_c2r_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::C2rPlan2d> {
            use crate::fft_backend::C2rPlanner2d;
            let mut planner = RealFftPlanner::<f64>::new();
            planner.plan_c2r_2d(nrows, ncols)
        }

        #[inline]
        fn from_f64(x: f64) -> Self {
            x
        }

        #[inline]
        fn from_usize(n: usize) -> Self {
            n as f64
        }
    }
}

// ============================================================================
// Global Plan Cache (for one-shot FFT functions)
// ============================================================================

#[cfg(feature = "realfft")]
pub mod plan_cache {
    use super::{C2rPlanner, R2cPlanner, SpectrogramError, SpectrogramResult};

    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use crate::fft_backend::realfft_backend::{RealFftInversePlan, RealFftPlan, RealFftPlanner};

    /// Global cache of forward FFT plans, keyed by n_fft size.
    /// Uses Arc for cheap cloning and Mutex for thread safety.
    static R2C_PLAN_CACHE: std::sync::LazyLock<Mutex<HashMap<usize, Arc<RealFftPlan>>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Global cache of inverse FFT plans, keyed by n_fft size.
    static C2R_PLAN_CACHE: std::sync::LazyLock<Mutex<HashMap<usize, Arc<RealFftInversePlan>>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Maximum number of cached plans to prevent unbounded memory growth.
    /// Plans are evicted FIFO when this limit is reached.
    const MAX_CACHED_PLANS: usize = 100;

    /// Get or create a forward (real-to-complex) FFT plan from the global cache.
    ///
    /// This function is thread-safe and will return a cached plan if one exists
    /// for the given size, or create a new one and cache it for future use.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size
    ///
    /// # Returns
    ///
    /// An Arc-wrapped plan that can be cloned cheaply for use across threads.
    pub fn get_or_create_r2c_plan(n_fft: usize) -> SpectrogramResult<Arc<RealFftPlan>> {
        let mut cache = R2C_PLAN_CACHE.lock().map_err(|e| {
            SpectrogramError::fft_backend("plan_cache", format!("mutex poisoned: {e}"))
        })?;

        // Fast path: plan already exists
        if let Some(plan) = cache.get(&n_fft) {
            return Ok(Arc::clone(plan));
        }

        // Slow path: create new plan
        let mut planner = RealFftPlanner::new();
        let plan = planner.plan_r2c(n_fft)?;
        let plan = Arc::new(plan);

        // Evict an arbitrary plan if cache is full (HashMap iteration order)
        if cache.len() >= MAX_CACHED_PLANS {
            if let Some(key) = cache.keys().next().copied() {
                cache.remove(&key);
            }
        }

        cache.insert(n_fft, Arc::clone(&plan));
        drop(cache);
        Ok(plan)
    }

    /// Get or create an inverse (complex-to-real) FFT plan from the global cache.
    ///
    /// This function is thread-safe and will return a cached plan if one exists
    /// for the given size, or create a new one and cache it for future use.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size
    ///
    /// # Returns
    ///
    /// An Arc-wrapped plan that can be cloned cheaply for use across threads.
    #[inline]
    pub fn get_or_create_c2r_plan(n_fft: usize) -> SpectrogramResult<Arc<RealFftInversePlan>> {
        let mut cache = C2R_PLAN_CACHE.lock().map_err(|e| {
            SpectrogramError::fft_backend("plan_cache", format!("mutex poisoned: {e}"))
        })?;

        // Fast path: plan already exists
        if let Some(plan) = cache.get(&n_fft) {
            return Ok(Arc::clone(plan));
        }

        // Slow path: create new plan
        let mut planner = RealFftPlanner::new();
        let plan = planner.plan_c2r(n_fft)?;
        let plan = Arc::new(plan);

        // Evict an arbitrary plan if cache is full (HashMap iteration order)
        if cache.len() >= MAX_CACHED_PLANS {
            if let Some(key) = cache.keys().next().copied() {
                cache.remove(&key);
            }
        }

        cache.insert(n_fft, Arc::clone(&plan));
        drop(cache); // Release lock before returning
        Ok(plan)
    }

    /// Clear all cached FFT plans.
    ///
    /// This is useful for:
    /// - Testing and benchmarking
    /// - Memory management in long-running applications
    /// - Forcing plan recreation with different settings
    ///
    /// Plans will be automatically recreated on next use.
    #[allow(unused)]
    #[inline]
    pub fn clear_plan_cache() {
        if let Ok(mut cache) = R2C_PLAN_CACHE.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = C2R_PLAN_CACHE.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics (for monitoring/debugging).
    ///
    /// Returns (forward_plans_cached, inverse_plans_cached).
    #[allow(unused)]
    #[inline]
    pub fn cache_stats() -> (usize, usize) {
        let r2c_count = R2C_PLAN_CACHE.lock().map(|c| c.len()).unwrap_or(0);
        let c2r_count = C2R_PLAN_CACHE.lock().map(|c| c.len()).unwrap_or(0);
        (r2c_count, c2r_count)
    }
}

#[cfg(feature = "realfft")]
pub use plan_cache::{get_or_create_c2r_plan, get_or_create_r2c_plan};

#[cfg(all(feature = "realfft", feature = "python"))]
pub use plan_cache::{cache_stats, clear_plan_cache};

#[cfg(feature = "fftw")]
pub mod fftw_backend {
    use std::collections::HashMap;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};

    use num_complex::Complex;

    use crate::fft_backend::{
        C2rPlan, C2rPlanner, R2cPlan, R2cPlanner, r2c_output_size, validate_fft_io,
    };
    use crate::{SpectrogramError, SpectrogramResult};

    // FFTW plan creation is not thread-safe, so we use a global mutex
    static FFTW_PLANNER_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Debug)]
    struct FftwBuffer<T> {
        ptr: NonNull<T>,
        _len: usize,
    }

    impl<T> FftwBuffer<T> {
        fn allocate(len: usize) -> SpectrogramResult<Self> {
            if len == 0 {
                return Err(SpectrogramError::invalid_input("buffer length must be > 0"));
            }

            let bytes = core::mem::size_of::<T>() * len;
            let raw = unsafe { fftw_sys::fftw_malloc(bytes) }.cast::<T>();

            let ptr = NonNull::new(raw).ok_or_else(|| {
                SpectrogramError::fft_backend("fftw", "fftw_malloc returned null")
            })?;

            Ok(Self { ptr, _len: len })
        }

        #[inline]
        const fn as_ptr(&self) -> *mut T {
            self.ptr.as_ptr()
        }
    }

    impl<T> Drop for FftwBuffer<T> {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_free(self.ptr.as_ptr().cast::<core::ffi::c_void>());
            }
        }
    }

    #[derive(Debug)]
    pub(crate) struct PlanInner {
        n_fft: usize,
        out_len: usize,
        plan: fftw_sys::fftw_plan,
        input: Arc<FftwBuffer<f64>>,
        output: Arc<FftwBuffer<fftw_sys::fftw_complex>>,
    }

    impl Drop for PlanInner {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_destroy_plan(self.plan);
            }
        }
    }

    #[derive(Debug)]
    pub(crate) struct InversePlanInner {
        n_fft: usize,
        in_len: usize,
        plan: fftw_sys::fftw_plan,
        input: Arc<FftwBuffer<fftw_sys::fftw_complex>>,
        output: Arc<FftwBuffer<f64>>,
    }

    impl Drop for InversePlanInner {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_destroy_plan(self.plan);
            }
        }
    }

    #[derive(Default)]
    pub struct FftwPlanner {
        cache_r2c: HashMap<usize, Arc<PlanInner>>,
        cache_c2r: HashMap<usize, Arc<InversePlanInner>>,
    }

    impl FftwPlanner {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        pub(crate) fn build_plan(n_fft: usize) -> SpectrogramResult<PlanInner> {
            let out_len = r2c_output_size(n_fft);

            let input = Arc::new(FftwBuffer::<f64>::allocate(n_fft)?);
            let output = Arc::new(FftwBuffer::<fftw_sys::fftw_complex>::allocate(out_len)?);

            let n_i32: i32 = n_fft
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("n_fft too large for FFTW"))?;

            // FFTW plan creation is not thread-safe - must be serialized
            let _lock = FFTW_PLANNER_LOCK.lock().map_err(|e| {
                SpectrogramError::fft_backend("fftw", format!("FFT planner mutex poisoned: {}", e))
            })?;

            let plan = unsafe {
                fftw_sys::fftw_plan_dft_r2c_1d(
                    n_i32,
                    input.as_ptr(),
                    output.as_ptr(),
                    fftw_sys::FFTW_ESTIMATE,
                )
            };

            if plan.is_null() {
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW plan",
                ));
            }

            Ok(PlanInner {
                n_fft,
                out_len,
                plan,
                input,
                output,
            })
        }

        pub(crate) fn get_or_create(&mut self, n_fft: usize) -> SpectrogramResult<Arc<PlanInner>> {
            if let Some(p) = self.cache_r2c.get(&n_fft) {
                return Ok(p.clone());
            }

            let p = Arc::new(Self::build_plan(n_fft)?);
            self.cache_r2c.insert(n_fft, p.clone());
            Ok(p)
        }

        pub(crate) fn build_inverse_plan(n_fft: usize) -> SpectrogramResult<InversePlanInner> {
            let in_len = r2c_output_size(n_fft);

            let input = Arc::new(FftwBuffer::<fftw_sys::fftw_complex>::allocate(in_len)?);
            let output = Arc::new(FftwBuffer::<f64>::allocate(n_fft)?);

            let n_i32: i32 = n_fft
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("n_fft too large for FFTW"))?;

            // FFTW plan creation is not thread-safe - must be serialized
            let _lock = FFTW_PLANNER_LOCK.lock().map_err(|e| {
                SpectrogramError::fft_backend("fftw", format!("FFT planner mutex poisoned: {}", e))
            })?;

            let plan = unsafe {
                fftw_sys::fftw_plan_dft_c2r_1d(
                    n_i32,
                    input.as_ptr(),
                    output.as_ptr(),
                    fftw_sys::FFTW_ESTIMATE,
                )
            };

            if plan.is_null() {
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW inverse plan",
                ));
            }

            Ok(InversePlanInner {
                n_fft,
                in_len,
                plan,
                input,
                output,
            })
        }

        pub(crate) fn get_or_create_inverse(
            &mut self,
            n_fft: usize,
        ) -> SpectrogramResult<Arc<InversePlanInner>> {
            if let Some(p) = self.cache_c2r.get(&n_fft) {
                return Ok(p.clone());
            }

            let p = Arc::new(Self::build_inverse_plan(n_fft)?);
            self.cache_c2r.insert(n_fft, p.clone());
            Ok(p)
        }
    }

    #[derive(Debug, Clone)]
    pub struct FftwPlan {
        inner: Arc<PlanInner>,
    }

    impl FftwPlan {
        pub(crate) const fn new(plan: Arc<PlanInner>) -> Self {
            Self { inner: plan }
        }
    }

    impl R2cPlan<f64> for FftwPlan {
        fn n_fft(&self) -> usize {
            self.inner.n_fft
        }

        fn output_len(&self) -> usize {
            self.inner.out_len
        }

        fn process(&mut self, input: &[f64], output: &mut [Complex<f64>]) -> SpectrogramResult<()> {
            validate_fft_io(self.inner.n_fft, input, output)?;

            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.inner.input.as_ptr(),
                    self.inner.n_fft,
                );

                fftw_sys::fftw_execute_dft_r2c(
                    self.inner.plan,
                    self.inner.input.as_ptr(),
                    self.inner.output.as_ptr(),
                );

                for i in 0..self.inner.out_len {
                    let c = self.inner.output.as_ptr().add(i) as *const f64;
                    let re = *c.add(0);
                    let im = *c.add(1);
                    output[i] = Complex::new(re, im);
                }
            }

            Ok(())
        }
    }

    impl R2cPlanner<f64> for FftwPlanner {
        type Plan = FftwPlan;

        fn plan_r2c(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            Ok(FftwPlan {
                inner: self.get_or_create(n_fft)?,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct FftwInversePlan {
        inner: Arc<InversePlanInner>,
    }

    impl C2rPlan<f64> for FftwInversePlan {
        fn n_fft(&self) -> usize {
            self.inner.n_fft
        }

        fn input_len(&self) -> usize {
            self.inner.in_len
        }

        fn process(&mut self, input: &[Complex<f64>], output: &mut [f64]) -> SpectrogramResult<()> {
            if input.len() != self.inner.in_len {
                return Err(SpectrogramError::dimension_mismatch(
                    self.inner.in_len,
                    input.len(),
                ));
            }
            if output.len() != self.inner.n_fft {
                return Err(SpectrogramError::dimension_mismatch(
                    self.inner.n_fft,
                    output.len(),
                ));
            }

            unsafe {
                // Copy input to FFTW buffer
                for i in 0..self.inner.in_len {
                    let ptr = self.inner.input.as_ptr().add(i).cast::<f64>();
                    *ptr.add(0) = input[i].re;
                    *ptr.add(1) = input[i].im;
                }

                // Execute inverse FFT
                fftw_sys::fftw_execute_dft_c2r(
                    self.inner.plan,
                    self.inner.input.as_ptr(),
                    self.inner.output.as_ptr(),
                );

                // Copy output and normalize
                let scale = 1.0 / self.inner.n_fft as f64;
                for i in 0..self.inner.n_fft {
                    output[i] = *self.inner.output.as_ptr().add(i) * scale;
                }
            }

            Ok(())
        }
    }

    impl C2rPlanner<f64> for FftwPlanner {
        type Plan = FftwInversePlan;

        fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            Ok(FftwInversePlan {
                inner: self.get_or_create_inverse(n_fft)?,
            })
        }
    }

    // ========================================================================
    // Complex-to-Complex FFT Implementation
    // ========================================================================

    /// Inner state for an FFTW complex-to-complex plan pair (forward + inverse).
    struct FftwC2cInner {
        n_fft: usize,
        fwd: fftw_sys::fftw_plan,
        inv: fftw_sys::fftw_plan,
        buf: Arc<FftwBuffer<fftw_sys::fftw_complex>>,
    }

    impl Drop for FftwC2cInner {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_destroy_plan(self.fwd);
                fftw_sys::fftw_destroy_plan(self.inv);
            }
        }
    }

    /// Complex-to-complex FFT plan (f64) backed by FFTW3.
    ///
    /// Both directions reuse the same FFTW-aligned buffer; data is copied in/out
    /// around each execute call (same pattern as `FftwPlan`).
    pub struct FftwC2cPlan {
        inner: Arc<FftwC2cInner>,
    }

    // SAFETY: FFTW plans are safe to use from one thread at a time; FftwC2cPlan is not Clone.
    unsafe impl Send for FftwC2cPlan {}

    impl FftwPlanner {
        /// Build a C2c plan pair for size `n_fft`.
        pub(crate) fn build_c2c_plan(n_fft: usize) -> SpectrogramResult<FftwC2cInner> {
            let buf = Arc::new(FftwBuffer::<fftw_sys::fftw_complex>::allocate(n_fft)?);
            let ptr = buf.as_ptr();

            let n_i32: i32 = n_fft
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("n_fft too large for FFTW"))?;

            let _lock = FFTW_PLANNER_LOCK.lock().map_err(|e| {
                SpectrogramError::fft_backend("fftw", format!("FFT planner mutex poisoned: {}", e))
            })?;

            let fwd = unsafe {
                fftw_sys::fftw_plan_dft_1d(
                    n_i32,
                    ptr,
                    ptr,
                    fftw_sys::FFTW_FORWARD as i32,
                    fftw_sys::FFTW_ESTIMATE,
                )
            };
            if fwd.is_null() {
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW C2c forward plan",
                ));
            }

            let inv = unsafe {
                fftw_sys::fftw_plan_dft_1d(
                    n_i32,
                    ptr,
                    ptr,
                    fftw_sys::FFTW_BACKWARD as i32,
                    fftw_sys::FFTW_ESTIMATE,
                )
            };
            if inv.is_null() {
                unsafe { fftw_sys::fftw_destroy_plan(fwd) };
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW C2c inverse plan",
                ));
            }

            Ok(FftwC2cInner {
                n_fft,
                fwd,
                inv,
                buf,
            })
        }

        /// Get or build a `FftwC2cPlan` (not cached — C2c plans are uncommon enough).
        pub fn plan_c2c(&mut self, n_fft: usize) -> SpectrogramResult<FftwC2cPlan> {
            Ok(FftwC2cPlan {
                inner: Arc::new(Self::build_c2c_plan(n_fft)?),
            })
        }
    }

    impl crate::fft_backend::C2cPlan<f64> for FftwC2cPlan {
        fn n_fft(&self) -> usize {
            self.inner.n_fft
        }

        fn forward(&mut self, buf: &mut [Complex<f64>]) -> SpectrogramResult<()> {
            let n = self.inner.n_fft;
            if buf.len() != n {
                return Err(SpectrogramError::dimension_mismatch(n, buf.len()));
            }
            unsafe {
                let ptr = self.inner.buf.as_ptr();
                for i in 0..n {
                    let dst = ptr.add(i).cast::<f64>();
                    *dst = buf[i].re;
                    *dst.add(1) = buf[i].im;
                }
                fftw_sys::fftw_execute_dft(self.inner.fwd, ptr, ptr);
                for i in 0..n {
                    let src = ptr.add(i).cast::<f64>();
                    buf[i] = Complex::new(*src, *src.add(1));
                }
            }
            Ok(())
        }

        fn inverse(&mut self, buf: &mut [Complex<f64>]) -> SpectrogramResult<()> {
            let n = self.inner.n_fft;
            if buf.len() != n {
                return Err(SpectrogramError::dimension_mismatch(n, buf.len()));
            }
            unsafe {
                let ptr = self.inner.buf.as_ptr();
                for i in 0..n {
                    let dst = ptr.add(i).cast::<f64>();
                    *dst = buf[i].re;
                    *dst.add(1) = buf[i].im;
                }
                fftw_sys::fftw_execute_dft(self.inner.inv, ptr, ptr);
                for i in 0..n {
                    let src = ptr.add(i).cast::<f64>();
                    buf[i] = Complex::new(*src, *src.add(1));
                }
            }
            Ok(())
        }
    }

    // ========================================================================
    // 2D FFT Implementation
    // ========================================================================

    #[derive(Debug)]
    pub(crate) struct PlanInner2d {
        nrows: usize,
        ncols: usize,
        out_shape: (usize, usize),
        plan: fftw_sys::fftw_plan,
        input: Arc<FftwBuffer<f64>>,
        output: Arc<FftwBuffer<fftw_sys::fftw_complex>>,
    }

    impl Drop for PlanInner2d {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_destroy_plan(self.plan);
            }
        }
    }

    #[derive(Debug)]
    pub(crate) struct InversePlanInner2d {
        nrows: usize,
        ncols: usize,
        in_shape: (usize, usize),
        plan: fftw_sys::fftw_plan,
        input: Arc<FftwBuffer<fftw_sys::fftw_complex>>,
        output: Arc<FftwBuffer<f64>>,
    }

    impl Drop for InversePlanInner2d {
        fn drop(&mut self) {
            unsafe {
                fftw_sys::fftw_destroy_plan(self.plan);
            }
        }
    }

    impl FftwPlanner {
        pub(crate) fn build_plan_2d(nrows: usize, ncols: usize) -> SpectrogramResult<PlanInner2d> {
            let out_shape = crate::fft_backend::r2c_output_size_2d(nrows, ncols);
            let input_len = nrows * ncols;
            let output_len = out_shape.0 * out_shape.1;

            let input = Arc::new(FftwBuffer::<f64>::allocate(input_len)?);
            let output = Arc::new(FftwBuffer::<fftw_sys::fftw_complex>::allocate(output_len)?);

            let n0: i32 = nrows
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("nrows too large for FFTW"))?;
            let n1: i32 = ncols
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("ncols too large for FFTW"))?;

            // FFTW plan creation is not thread-safe - must be serialized
            let _lock = FFTW_PLANNER_LOCK.lock().map_err(|e| {
                SpectrogramError::fft_backend("fftw", format!("FFT planner mutex poisoned: {}", e))
            })?;

            let plan = unsafe {
                fftw_sys::fftw_plan_dft_r2c_2d(
                    n0,
                    n1,
                    input.as_ptr(),
                    output.as_ptr(),
                    fftw_sys::FFTW_ESTIMATE,
                )
            };

            if plan.is_null() {
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW 2D plan",
                ));
            }

            Ok(PlanInner2d {
                nrows,
                ncols,
                out_shape,
                plan,
                input,
                output,
            })
        }

        pub(crate) fn get_or_create_2d(
            &mut self,
            nrows: usize,
            ncols: usize,
        ) -> SpectrogramResult<Arc<PlanInner2d>> {
            // For 2D plans, we need a different caching strategy
            // For now, we'll create a new plan each time
            // TODO: Add HashMap<(usize, usize), Arc<PlanInner2d>> cache
            let p = Arc::new(Self::build_plan_2d(nrows, ncols)?);
            Ok(p)
        }

        pub(crate) fn build_inverse_plan_2d(
            nrows: usize,
            ncols: usize,
        ) -> SpectrogramResult<InversePlanInner2d> {
            let in_shape = crate::fft_backend::r2c_output_size_2d(nrows, ncols);
            let input_len = in_shape.0 * in_shape.1;
            let output_len = nrows * ncols;

            let input = Arc::new(FftwBuffer::<fftw_sys::fftw_complex>::allocate(input_len)?);
            let output = Arc::new(FftwBuffer::<f64>::allocate(output_len)?);

            let n0: i32 = nrows
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("nrows too large for FFTW"))?;
            let n1: i32 = ncols
                .try_into()
                .map_err(|_| SpectrogramError::invalid_input("ncols too large for FFTW"))?;

            // FFTW plan creation is not thread-safe - must be serialized
            let _lock = FFTW_PLANNER_LOCK.lock().map_err(|e| {
                SpectrogramError::fft_backend("fftw", format!("FFT planner mutex poisoned: {}", e))
            })?;

            let plan = unsafe {
                fftw_sys::fftw_plan_dft_c2r_2d(
                    n0,
                    n1,
                    input.as_ptr(),
                    output.as_ptr(),
                    fftw_sys::FFTW_ESTIMATE,
                )
            };

            if plan.is_null() {
                return Err(SpectrogramError::fft_backend(
                    "fftw",
                    "failed to create FFTW 2D inverse plan",
                ));
            }

            Ok(InversePlanInner2d {
                nrows,
                ncols,
                in_shape,
                plan,
                input,
                output,
            })
        }

        pub(crate) fn get_or_create_inverse_2d(
            &mut self,
            nrows: usize,
            ncols: usize,
        ) -> SpectrogramResult<Arc<InversePlanInner2d>> {
            // For 2D plans, we need a different caching strategy
            // For now, we'll create a new plan each time
            // TODO: Add HashMap<(usize, usize), Arc<InversePlanInner2d>> cache
            let p = Arc::new(Self::build_inverse_plan_2d(nrows, ncols)?);
            Ok(p)
        }
    }

    #[derive(Debug, Clone)]
    pub struct FftwPlan2d {
        inner: Arc<PlanInner2d>,
    }

    impl crate::fft_backend::R2cPlan2d<f64> for FftwPlan2d {
        fn process(&mut self, input: &[f64], output: &mut [Complex<f64>]) -> SpectrogramResult<()> {
            crate::fft_backend::validate_fft_io_2d(
                self.inner.nrows,
                self.inner.ncols,
                input,
                output,
            )?;

            unsafe {
                // Copy input to FFTW buffer
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    self.inner.input.as_ptr(),
                    input.len(),
                );

                // Execute 2D FFT
                fftw_sys::fftw_execute_dft_r2c(
                    self.inner.plan,
                    self.inner.input.as_ptr(),
                    self.inner.output.as_ptr(),
                );

                // Copy output (convert from fftw_complex to Complex<f64>)
                let output_len = self.inner.out_shape.0 * self.inner.out_shape.1;
                for i in 0..output_len {
                    let c = self.inner.output.as_ptr().add(i) as *const f64;
                    let re = *c.add(0);
                    let im = *c.add(1);
                    output[i] = Complex::new(re, im);
                }
            }

            Ok(())
        }
    }

    impl crate::fft_backend::R2cPlanner2d<f64> for FftwPlanner {
        type Plan = FftwPlan2d;

        fn plan_r2c_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan> {
            Ok(FftwPlan2d {
                inner: self.get_or_create_2d(nrows, ncols)?,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct FftwInversePlan2d {
        inner: Arc<InversePlanInner2d>,
    }

    impl crate::fft_backend::C2rPlan2d<f64> for FftwInversePlan2d {
        fn process(&mut self, input: &[Complex<f64>], output: &mut [f64]) -> SpectrogramResult<()> {
            let expected_in_len = self.inner.in_shape.0 * self.inner.in_shape.1;
            if input.len() != expected_in_len {
                return Err(SpectrogramError::dimension_mismatch(
                    expected_in_len,
                    input.len(),
                ));
            }

            let expected_out_len = self.inner.nrows * self.inner.ncols;
            if output.len() != expected_out_len {
                return Err(SpectrogramError::dimension_mismatch(
                    expected_out_len,
                    output.len(),
                ));
            }

            unsafe {
                // Copy input to FFTW buffer
                for i in 0..input.len() {
                    let ptr = self.inner.input.as_ptr().add(i).cast::<f64>();
                    *ptr.add(0) = input[i].re;
                    *ptr.add(1) = input[i].im;
                }

                // Execute inverse 2D FFT
                fftw_sys::fftw_execute_dft_c2r(
                    self.inner.plan,
                    self.inner.input.as_ptr(),
                    self.inner.output.as_ptr(),
                );

                // Copy output and normalize
                let scale = 1.0 / (self.inner.nrows * self.inner.ncols) as f64;
                for i in 0..expected_out_len {
                    output[i] = *self.inner.output.as_ptr().add(i) * scale;
                }
            }

            Ok(())
        }
    }

    impl crate::fft_backend::C2rPlanner2d<f64> for FftwPlanner {
        type Plan = FftwInversePlan2d;

        fn plan_c2r_2d(&mut self, nrows: usize, ncols: usize) -> SpectrogramResult<Self::Plan> {
            Ok(FftwInversePlan2d {
                inner: self.get_or_create_inverse_2d(nrows, ncols)?,
            })
        }
    }

    // ── Sample implementation (fftw backend, f64 only) ────────────────────────
    //
    // FFTW here uses the double-precision `fftw_sys` API, so only `f64` is
    // supported. There is intentionally no `Sample for f32` under this backend.
    // Each `plan_*` constructor builds a fresh planner per call.
    // TODO: per-type plan cache to avoid re-planning on every construction.
    impl crate::Sample for f64 {
        type R2cPlan = FftwPlan;
        type C2rPlan = FftwInversePlan;
        type C2cPlan = FftwC2cPlan;
        type R2cPlan2d = FftwPlan2d;
        type C2rPlan2d = FftwInversePlan2d;

        fn plan_r2c(n_fft: usize) -> SpectrogramResult<Self::R2cPlan> {
            FftwPlanner::new().plan_r2c(n_fft)
        }

        fn plan_c2r(n_fft: usize) -> SpectrogramResult<Self::C2rPlan> {
            FftwPlanner::new().plan_c2r(n_fft)
        }

        fn plan_c2c(n_fft: usize) -> SpectrogramResult<Self::C2cPlan> {
            FftwPlanner::new().plan_c2c(n_fft)
        }

        fn plan_r2c_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::R2cPlan2d> {
            use crate::fft_backend::R2cPlanner2d;
            FftwPlanner::new().plan_r2c_2d(nrows, ncols)
        }

        fn plan_c2r_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::C2rPlan2d> {
            use crate::fft_backend::C2rPlanner2d;
            FftwPlanner::new().plan_c2r_2d(nrows, ncols)
        }

        #[inline]
        fn from_f64(x: f64) -> Self {
            x
        }

        #[inline]
        fn from_usize(n: usize) -> Self {
            n as f64
        }
    }
}

#[cfg(all(test, feature = "realfft"))]
mod tests_c2c {
    use super::realfft_backend::RealFftC2cPlan;
    use crate::fft_backend::C2cPlan;
    use num_complex::Complex;

    /// DFT of a pure DC signal (all ones): X[0] = N, X[k≠0] = 0.
    #[test]
    fn c2c_f64_dc_signal() {
        let n = 8;
        let mut plan = RealFftC2cPlan::new(n);
        let mut buf: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];
        plan.forward(&mut buf).unwrap();
        assert!((buf[0].re - n as f64).abs() < 1e-10, "DC bin should be N");
        assert!(buf[0].im.abs() < 1e-10);
        for k in 1..n {
            assert!(buf[k].norm() < 1e-10, "bin {k} should be zero");
        }
    }

    /// Round-trip: forward then inverse (divided by N) recovers the original signal.
    #[test]
    fn c2c_f64_round_trip() {
        let n = 16;
        let original: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        let mut buf = original.clone();
        let mut plan = RealFftC2cPlan::new(n);
        plan.forward(&mut buf).unwrap();
        plan.inverse(&mut buf).unwrap();
        let scale = 1.0 / n as f64;
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a.re * scale - b.re).abs() < 1e-10);
            assert!((a.im * scale - b.im).abs() < 1e-10);
        }
    }

    /// f32 round-trip.
    #[test]
    fn c2c_f32_round_trip() {
        let n = 16;
        let original: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect();
        let mut buf = original.clone();
        let mut plan = RealFftC2cPlan::<f32>::new(n);
        plan.forward(&mut buf).unwrap();
        plan.inverse(&mut buf).unwrap();
        let scale = 1.0f32 / n as f32;
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a.re * scale - b.re).abs() < 1e-5);
            assert!((a.im * scale - b.im).abs() < 1e-5);
        }
    }

    /// Dimension mismatch returns an error rather than panicking.
    #[test]
    fn c2c_f64_dimension_mismatch() {
        let mut plan = RealFftC2cPlan::new(8);
        let mut buf = vec![Complex::new(0.0, 0.0); 7]; // wrong size
        assert!(plan.forward(&mut buf).is_err());
        assert!(plan.inverse(&mut buf).is_err());
    }
}
