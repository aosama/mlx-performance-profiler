//! The C API of mlx-performance-profiler (src/mlx_perf_c_api.h), declared for Rust.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

use mlx_bridge::raw;

/// Appends a fresh, unevaluated graph's outputs; returns 0 on success.
pub type BuildGraphCallback =
    unsafe extern "C" fn(fresh_outputs: raw::mlx_vector_array, payload: *mut c_void) -> c_int;

/// Mirrors `mlx_perf_job_summary` field for field (`#[repr(C)]`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct JobSummary {
    pub job_wall_median_us: f64,
    pub job_wall_min_us: f64,
    pub job_wall_max_us: f64,
    pub timeline_wall_us: f64,
    pub sum_of_stages_us: f64,
    pub measured_gpu_execution_us: f64,
    pub cross_stream_stall_us: f64,
    pub executed_node_count: c_int,
    pub gpu_measured_node_count: c_int,
    pub stream_count: c_int,
}

unsafe extern "C" {
    fn mlx_perf_profile_to_report(
        build_graph: BuildGraphCallback,
        builder_payload: *mut c_void,
        variant_name: *const c_char,
        report_title: *const c_char,
        output_directory: *const c_char,
        report_file_base: *const c_char,
        summary: *mut JobSummary,
    ) -> c_int;
    fn mlx_perf_last_error() -> *const c_char;
}

/// What to measure and where the report goes.
pub struct ProfileRequest<'request> {
    pub variant_name: &'request str,
    pub report_title: &'request str,
    pub output_directory: &'request str,
    pub report_file_base: &'request str,
}

fn to_c_string(text: &str) -> Result<CString, String> {
    CString::new(text).map_err(|interior_nul| interior_nul.to_string())
}

/// Measures the job the callback builds and writes its HTML report.
///
/// # Safety
/// `builder_payload` must be whatever `build_graph` expects, and stay alive
/// and unaliased for the whole call (the callback runs synchronously).
pub unsafe fn profile_to_report(
    build_graph: BuildGraphCallback,
    builder_payload: *mut c_void,
    profile_request: &ProfileRequest<'_>,
) -> Result<JobSummary, String> {
    let variant_name = to_c_string(profile_request.variant_name)?;
    let report_title = to_c_string(profile_request.report_title)?;
    let output_directory = to_c_string(profile_request.output_directory)?;
    let report_file_base = to_c_string(profile_request.report_file_base)?;
    let mut job_summary = JobSummary::default();
    // SAFETY: Strings are NUL-terminated and outlive the call; the summary is
    // a valid #[repr(C)] out-parameter; payload validity is the caller's
    // contract above.
    let status = unsafe {
        mlx_perf_profile_to_report(
            build_graph,
            builder_payload,
            variant_name.as_ptr(),
            report_title.as_ptr(),
            output_directory.as_ptr(),
            report_file_base.as_ptr(),
            &mut job_summary,
        )
    };
    if status == 0 {
        return Ok(job_summary);
    }
    // SAFETY: The library returns a valid thread-local C string.
    let failure_text = unsafe { CStr::from_ptr(mlx_perf_last_error()) };
    Err(failure_text.to_string_lossy().into_owned())
}
