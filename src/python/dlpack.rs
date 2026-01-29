//! DLPack protocol implementation for tensor interchange.
//!
//! # Safety
//!
//! This module uses extensive unsafe code to interface with C FFI and manage memory
//! across Python/Rust boundaries. Key safety considerations:
//!
//! - The deleter function must acquire the GIL before dropping Python objects
//! - Pointers are carefully managed through Box to ensure proper cleanup
//! - DLPackContext owns shape/strides arrays and the PyArray reference

use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::c_void;

/// Register DLPack constants in the parent module
pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add(
        "DLPACK_FLAG_BITMASK_READ_ONLY",
        DLPACK_FLAG_BITMASK_READ_ONLY,
    )?;
    parent.add(
        "DLPACK_FLAG_BITMASK_IS_COPIED",
        DLPACK_FLAG_BITMASK_IS_COPIED,
    )?;

    parent.add("KDLCPU", device_type::KDLCPU)?;
    parent.add("KDLCUDA", device_type::KDLCUDA)?;
    parent.add("KDLCUDA_HOST", device_type::KDLCUDA_HOST)?;
    parent.add("KDLOPENCL", device_type::KDLOPENCL)?;
    parent.add("KDLVULKAN", device_type::KDLVULKAN)?;
    parent.add("KDLMETAL", device_type::KDLMETAL)?;
    parent.add("KDLVPI", device_type::KDLVPI)?;
    parent.add("KDLROCM", device_type::KDLROCM)?;
    parent.add("KDLROCM_HOST", device_type::KDLROCM_HOST)?;
    parent.add("KDLEXT_DEV", device_type::KDLEXT_DEV)?;
    parent.add("KDLCUDA_MANAGED", device_type::KDLCUDA_MANAGED)?;
    parent.add("KDLONE_API", device_type::KDLONE_API)?;
    parent.add("KDLWEB_GPU", device_type::KDLWEB_GPU)?;
    parent.add("KDLHEXAGON", device_type::KDLHEXAGON)?;
    Ok(())
}

// DLPack flag constants
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

/// DLPack version structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// DLPack device type constants
pub mod device_type {
    pub const KDLCPU: i32 = 1;
    pub const KDLCUDA: i32 = 2;
    pub const KDLCUDA_HOST: i32 = 3;
    pub const KDLOPENCL: i32 = 4;
    pub const KDLVULKAN: i32 = 7;
    pub const KDLMETAL: i32 = 8;
    pub const KDLVPI: i32 = 9;
    pub const KDLROCM: i32 = 10;
    pub const KDLROCM_HOST: i32 = 11;
    pub const KDLEXT_DEV: i32 = 12;
    pub const KDLCUDA_MANAGED: i32 = 13;
    pub const KDLONE_API: i32 = 14;
    pub const KDLWEB_GPU: i32 = 15;
    pub const KDLHEXAGON: i32 = 16;
}

/// DLPack device structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32, // Use i32 directly to match C ABI exactly
    pub device_id: i32,
}

/// DLPack data type code enumeration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(clippy::enum_variant_names)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
}

/// DLPack data type structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    pub code: u8,   // DLDataTypeCode
    pub bits: u8,   // Number of bits
    pub lanes: u16, // Number of lanes (SIMD), typically 1
}

/// DLPack tensor structure (core data descriptor)
#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64, // Can be NULL for C-contiguous
    pub byte_offset: u64,
}

// Type alias for the deleter functions
type DeleterFn = unsafe extern "C" fn(*mut DLManagedTensor);

/// DLPack managed tensor (old format for maximum compatibility)
/// This is the widely-supported DLPack 0.x format
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: *const c_void, // Function pointer as raw pointer
}

/// Context for managing DLPack tensor lifetime
///
/// This structure owns:
/// - The shape array (always allocated on heap)
/// - The strides array (if needed; NULL for C-contiguous)
/// - A reference to the Python array to keep data alive
#[allow(unused)]
pub struct DLPackContext {
    shape: Box<[i64; 2]>,
    strides: Option<Box<[i64; 2]>>,
    data_owner: Py<PyAny>,
}

impl DLPackContext {
    /// Create a new DLPackContext from a PyArray2<f64>
    pub fn new(_py: Python<'_>, arr: &Bound<'_, PyArray2<f64>>) -> Self {
        let readonly = arr.readonly();
        let shape = readonly.shape();
        let shape_box = Box::new([shape[0] as i64, shape[1] as i64]);

        // Check if array is C-contiguous
        let strides = if readonly.is_c_contiguous() {
            // C-contiguous: strides can be NULL (framework will infer)
            None
        } else {
            // Non-contiguous: must provide explicit strides
            let strides_bytes = readonly.strides();
            let element_size = std::mem::size_of::<f64>() as isize;
            Some(Box::new([
                (strides_bytes[0] / element_size) as i64,
                (strides_bytes[1] / element_size) as i64,
            ]))
        };

        Self {
            shape: shape_box,
            strides,
            data_owner: arr.clone().into_any().unbind(),
        }
    }
}

/// Deleter function called when the DLPack tensor is consumed
///
/// # Safety
///
/// This function is called from C code (potentially from any thread) when the
/// consuming framework is done with the tensor. It MUST acquire the GIL before
/// touching any Python objects.
unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    // safety: We are called from C code when the consumer is done with the tensor. We must acquire the GIL before touching any Python objects. The managed pointer is owned by us and will be dropped exactly once, either here or in the pycapsule_destructor if not consumed.
    unsafe {
        if managed.is_null() {
            return;
        }
    }

    // safety: We own this pointer and it's being dropped exactly once. We must check if Python is still initialized before acquiring GIL to avoid panics during shutdown. If Python is not initialized, we leak the context to avoid undefined behavior, which is acceptable since the process is terminating.
    let ctx_ptr = unsafe { (*managed).manager_ctx.cast::<DLPackContext>() };
    if !ctx_ptr.is_null() {
        // CRITICAL: Check if Python is still running before acquiring GIL
        // During interpreter shutdown, Python::with_gil will panic
        // safety: We own this pointer and it's being dropped exactly once. We must check if Python is still initialized before acquiring GIL to avoid panics during shutdown. If Python is not initialized, we leak the context to avoid undefined behavior, which is acceptable since the process is terminating.
        unsafe {
            if ffi::Py_IsInitialized() != 0 {
                // SAFETY: We own this pointer and it's being dropped exactly once
                // Acquire GIL before dropping Python objects
                Python::attach(|_py| unsafe {
                    let _ctx = Box::from_raw(ctx_ptr);
                    // Drop happens here, releasing the Python array reference
                });
            } else {
                // Python is shutting down - leak the context to avoid undefined behavior
                // This is acceptable since the process is terminating anyway
            }
        }
    }

    // Free the managed tensor itself
    // SAFETY: We own this pointer and it's being dropped exactly once
    unsafe {
        let _managed = Box::from_raw(managed);
    }
}

/// PyCapsule destructor called when Python garbage collects the capsule
///
/// # Safety
///
/// This is called by Python's GC, so the GIL is already held.
/// According to DLPack protocol:
/// - If the capsule was consumed, the consumer calls the DLManagedTensor's deleter
/// - If the capsule was NOT consumed (e.g., error occurred), we must clean up here
/// - Consumed capsules have their name changed, so we check for the original name
unsafe extern "C" fn pycapsule_destructor(capsule: *mut ffi::PyObject) {
    // safety: We are called by Python's GC, so the GIL is already held. We must ensure that we only clean up if the capsule was not consumed. If the capsule was consumed, the consumer is responsible for cleanup via the deleter.
    unsafe {
        // safety: We are called by Python's GC, so the GIL is already held. We must ensure that we only clean up if the capsule was not consumed. If the capsule was consumed, the consumer is responsible for cleanup via the deleter.
        unsafe {
            if capsule.is_null() {
                return;
            }
        }

        // Check if capsule still has the original "dltensor" name
        // If the name is different, the capsule was consumed and cleanup is the consumer's responsibility
        let name = ffi::PyCapsule_GetName(capsule);
        if name.is_null() {
            // Name was cleared - capsule was consumed, consumer handles cleanup
            return;
        }

        // Check if it's still "dltensor"
        let name_str = std::ffi::CStr::from_ptr(name);
        if name_str.to_bytes() != b"dltensor" {
            // Name was changed - capsule was consumed
            return;
        }

        // Capsule was not consumed - we need to clean up
        let ptr = ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr());
        if !ptr.is_null() {
            let managed = ptr.cast::<DLManagedTensor>();
            // Call the deleter to clean up
            // safety: We own this pointer and it's being dropped exactly once. The deleter will handle GIL acquisition and cleanup.
            unsafe {
                if !(*managed).deleter.is_null() {
                    // safety: We own this pointer and it's being dropped exactly once. The deleter will handle GIL acquisition and cleanup.
                    let deleter: DeleterFn = unsafe { std::mem::transmute((*managed).deleter) };
                    // safety: We own this pointer and it's being dropped exactly once. The deleter will handle GIL acquisition and cleanup.
                    unsafe { deleter(managed) };
                }
            }
        }
    }
}

/// Create a DLPack PyCapsule from a PyArray2<f64>
///
/// # Arguments
///
/// * `py` - Python context
/// * `arr` - The NumPy array to wrap (must be f64)
/// * `flags` - DLPack flags (e.g., DLPACK_FLAG_BITMASK_IS_COPIED)
///
/// # Returns
///
/// A PyCapsule named "dltensor" containing a DLManagedTensorVersioned structure
///
/// # Safety
///
/// The capsule maintains ownership of the data through DLPackContext, which holds
/// a reference to the Python array. When the consuming framework calls the deleter,
/// the context is properly cleaned up with GIL safety.
#[allow(unused)]
pub fn create_dlpack_capsule<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyArray2<f64>>,
    flags: u64,
) -> PyResult<Bound<'py, PyCapsule>> {
    // Get raw data pointer directly from the PyArray using numpy's data() method
    // IMPORTANT: We must keep the array alive via DLPackContext
    let data_ptr = arr.data().cast::<c_void>();

    // Create context - this stores the array to keep data alive
    let mut ctx = Box::new(DLPackContext::new(py, arr));

    // Build DLTensor pointing to numpy array data
    let dl_tensor = DLTensor {
        data: data_ptr,
        device: DLDevice {
            device_type: device_type::KDLCPU,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLFloat as u8,
            bits: 64,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx
            .strides
            .as_mut()
            .map_or(std::ptr::null_mut(), |s| s.as_mut_ptr()),
        byte_offset: 0,
    };

    // Wrap in managed tensor (using old format for maximum compatibility)
    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(ctx).cast::<c_void>(),
        deleter: dlpack_deleter as *const c_void,
    });

    // Create PyCapsule with name "dltensor" using FFI
    let managed_ptr = Box::into_raw(managed).cast::<c_void>();
    let name_ptr = c"dltensor".as_ptr();

    // safety: We own the managed_ptr and are responsible for its cleanup. The capsule will call the deleter when consumed, or the pycapsule_destructor if not consumed. We must ensure that the capsule is created successfully to avoid leaks.
    unsafe {
        let capsule_ptr = ffi::PyCapsule_New(managed_ptr, name_ptr, Some(pycapsule_destructor));
        if capsule_ptr.is_null() {
            // Clean up on error
            let _ = Box::from_raw(managed_ptr.cast::<DLManagedTensor>());
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule_ptr).cast_into()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlpack_structs_layout() {
        // Verify struct sizes match DLPack spec expectations
        assert_eq!(std::mem::size_of::<DLDevice>(), 8);
        assert_eq!(std::mem::size_of::<DLDataType>(), 4);
        assert_eq!(std::mem::align_of::<DLTensor>(), 8);
    }

    #[test]
    fn test_device_type_values() {
        assert_eq!(device_type::KDLCPU, 1);
        assert_eq!(device_type::KDLCUDA, 2);
    }

    #[test]
    fn test_datatype_code_values() {
        assert_eq!(DLDataTypeCode::kDLFloat as u8, 2);
    }
}
