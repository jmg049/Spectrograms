use num_complex::Complex;

use crate::{SpectrogramError, SpectrogramResult};

/// Output size for a real-to-complex FFT of length `n`.
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
pub trait R2cPlan {
    fn n_fft(&self) -> usize;
    fn output_len(&self) -> usize;

    fn process(&mut self, input: &[f64], output: &mut [Complex<f64>]) -> SpectrogramResult<()>;
}

/// Planner that can construct FFT plans.
pub trait R2cPlanner {
    type Plan: R2cPlan;

    fn plan_r2c(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan>;
}

/// A planned complex-to-real inverse FFT for a fixed transform length.
pub trait C2rPlan {
    fn n_fft(&self) -> usize;
    fn input_len(&self) -> usize;

    fn process(&mut self, input: &[Complex<f64>], output: &mut [f64]) -> SpectrogramResult<()>;
}

/// Planner that can construct inverse FFT plans.
pub trait C2rPlanner {
    type Plan: C2rPlan;

    fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftBackendKind {
    RealFft,
    Fftw,
}

#[inline]
pub const fn validate_fft_io(
    n_fft: usize,
    input: &[f64],
    output: &[Complex<f64>],
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

/// Optional workload hint.
#[derive(Debug, Clone, Copy)]
pub struct BackendHint {
    pub expected_calls: usize,
}

impl Default for BackendHint {
    fn default() -> Self {
        Self { expected_calls: 1 }
    }
}

#[cfg(feature = "realfft")]
pub mod realfft_backend {
    use std::collections::HashMap;
    use std::sync::Arc;

    use num_complex::Complex;
    pub use realfft::{ComplexToReal, RealFftPlanner as InnerPlanner, RealToComplex};

    use crate::fft_backend::{
        C2rPlan, C2rPlanner, R2cPlan, R2cPlanner, r2c_output_size, validate_fft_io,
    };
    use crate::{SpectrogramError, SpectrogramResult};

    #[derive(Default)]
    pub struct RealFftPlanner {
        inner: InnerPlanner<f64>,
        cache_r2c: HashMap<usize, Arc<dyn RealToComplex<f64>>>,
        cache_c2r: HashMap<usize, Arc<dyn ComplexToReal<f64>>>,
    }

    impl RealFftPlanner {
        pub fn new() -> Self {
            Self::default()
        }

        pub(crate) fn get_or_create(&mut self, n_fft: usize) -> Arc<dyn RealToComplex<f64>> {
            if let Some(p) = self.cache_r2c.get(&n_fft) {
                return p.clone();
            }
            let p = self.inner.plan_fft_forward(n_fft);
            self.cache_r2c.insert(n_fft, p.clone());
            p
        }

        pub(crate) fn get_or_create_inverse(
            &mut self,
            n_fft: usize,
        ) -> Arc<dyn ComplexToReal<f64>> {
            if let Some(p) = self.cache_c2r.get(&n_fft) {
                return p.clone();
            }
            let p = self.inner.plan_fft_inverse(n_fft);
            self.cache_c2r.insert(n_fft, p.clone());
            p
        }
    }

    #[derive(Clone)]
    pub struct RealFftPlan {
        n_fft: usize,
        plan: Arc<dyn RealToComplex<f64>>,
        scratch: Vec<f64>,
    }

    impl RealFftPlan {
        pub(crate) fn new(n_fft: usize, plan: Arc<dyn RealToComplex<f64>>) -> Self {
            Self {
                n_fft,
                plan,
                scratch: vec![0.0; n_fft],
            }
        }
    }

    impl R2cPlan for RealFftPlan {
        fn n_fft(&self) -> usize {
            self.n_fft
        }

        fn output_len(&self) -> usize {
            r2c_output_size(self.n_fft)
        }

        fn process(&mut self, input: &[f64], output: &mut [Complex<f64>]) -> SpectrogramResult<()> {
            validate_fft_io(self.n_fft, input, output)?;

            self.scratch.copy_from_slice(input);

            self.plan
                .process(&mut self.scratch, output)
                .map_err(|e| SpectrogramError::fft_backend("realfft", format!("{e:?}")))
        }
    }

    impl R2cPlanner for RealFftPlanner {
        type Plan = RealFftPlan;

        fn plan_r2c(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            let plan = self.get_or_create(n_fft);
            Ok(RealFftPlan::new(n_fft, plan))
        }
    }

    #[derive(Clone)]
    pub struct RealFftInversePlan {
        n_fft: usize,
        plan: Arc<dyn ComplexToReal<f64>>,
        scratch: Vec<Complex<f64>>,
    }

    impl RealFftInversePlan {
        pub(crate) fn new(n_fft: usize, plan: Arc<dyn ComplexToReal<f64>>) -> Self {
            let scratch_len = r2c_output_size(n_fft);
            Self {
                n_fft,
                plan,
                scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            }
        }
    }

    impl C2rPlan for RealFftInversePlan {
        fn n_fft(&self) -> usize {
            self.n_fft
        }

        fn input_len(&self) -> usize {
            r2c_output_size(self.n_fft)
        }

        fn process(&mut self, input: &[Complex<f64>], output: &mut [f64]) -> SpectrogramResult<()> {
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
            let scale = 1.0 / self.n_fft as f64;
            for val in output.iter_mut() {
                *val *= scale;
            }

            Ok(())
        }
    }

    impl C2rPlanner for RealFftPlanner {
        type Plan = RealFftInversePlan;

        fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            let plan = self.get_or_create_inverse(n_fft);
            Ok(RealFftInversePlan::new(n_fft, plan))
        }
    }
}

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
            let _lock = FFTW_PLANNER_LOCK.lock().unwrap();

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
            let _lock = FFTW_PLANNER_LOCK.lock().unwrap();

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

    impl R2cPlan for FftwPlan {
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

    impl R2cPlanner for FftwPlanner {
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

    impl C2rPlan for FftwInversePlan {
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

    impl C2rPlanner for FftwPlanner {
        type Plan = FftwInversePlan;

        fn plan_c2r(&mut self, n_fft: usize) -> SpectrogramResult<Self::Plan> {
            Ok(FftwInversePlan {
                inner: self.get_or_create_inverse(n_fft)?,
            })
        }
    }
}
