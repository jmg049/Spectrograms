//! Sealed scalar trait abstracting the float precision used by the FFT backend.
//!
//! [`Sample`] is implemented for `f32` and `f64` only. It ties a concrete scalar
//! type to the associated FFT plan types and constructors provided by the active
//! backend (realfft or fftw), allowing the rest of the crate to be written
//! generically over precision.

use num_traits::{Float, FloatConst, NumAssign};

use crate::SpectrogramResult;
use crate::fft_backend::{C2cPlan, C2rPlan, C2rPlan2d, R2cPlan, R2cPlan2d};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Sealed trait describing a float scalar the FFT backend can operate on.
///
/// Implemented for `f32` and `f64`. Each implementor exposes the backend's
/// concrete plan types and constructors for those plans.
pub trait Sample:
    sealed::Sealed
    + Float
    + FloatConst
    + NumAssign
    + Copy
    + Send
    + Sync
    + 'static
    + core::fmt::Debug
    + Default
{
    /// Real-to-complex 1D plan type for this scalar.
    type R2cPlan: R2cPlan<Self>;
    /// Complex-to-real 1D plan type for this scalar.
    type C2rPlan: C2rPlan<Self>;
    /// Complex-to-complex 1D plan type for this scalar.
    type C2cPlan: C2cPlan<Self>;
    /// Real-to-complex 2D plan type for this scalar.
    type R2cPlan2d: R2cPlan2d<Self>;
    /// Complex-to-real 2D plan type for this scalar.
    type C2rPlan2d: C2rPlan2d<Self>;

    /// Build a real-to-complex 1D FFT plan of length `n_fft`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to construct the plan.
    fn plan_r2c(n_fft: usize) -> SpectrogramResult<Self::R2cPlan>;

    /// Build a complex-to-real 1D inverse FFT plan of length `n_fft`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to construct the plan.
    fn plan_c2r(n_fft: usize) -> SpectrogramResult<Self::C2rPlan>;

    /// Build a complex-to-complex 1D FFT plan of length `n_fft`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to construct the plan.
    fn plan_c2c(n_fft: usize) -> SpectrogramResult<Self::C2cPlan>;

    /// Build a real-to-complex 2D FFT plan of dimensions `nrows` x `ncols`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to construct the plan.
    fn plan_r2c_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::R2cPlan2d>;

    /// Build a complex-to-real 2D inverse FFT plan of dimensions `nrows` x `ncols`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to construct the plan.
    fn plan_c2r_2d(nrows: usize, ncols: usize) -> SpectrogramResult<Self::C2rPlan2d>;

    /// Convert an `f64` constant/literal to this scalar (for PI-derived consts, casts).
    fn from_f64(x: f64) -> Self;

    /// Convert a `usize` (e.g. an index or length) to this scalar.
    fn from_usize(n: usize) -> Self;
}
