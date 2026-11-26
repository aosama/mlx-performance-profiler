//! The C API of mlx-performance-profiler (include/mlx_perf/mlx_perf_c_api.h), declared for Rust.

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

/// Mirrors `mlx_perf_variant`: one implementation in a comparison.
#[repr(C)]
#[allow(dead_code)] // declared for callers; this example profiles one job
pub struct Variant {
    pub variant_name: *const c_char,
    pub build_graph: Option<BuildGraphCallback>,
    pub builder_payload: *mut c_void,
}

/// Mirrors `mlx_perf_metric_comparison` field for field.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
#[allow(dead_code)] // declared for callers; this example profiles one job
pub struct MetricComparison {
    pub baseline_median_us: f64,
    pub baseline_min_us: f64,
    pub baseline_max_us: f64,
    pub baseline_lower_quartile_us: f64,
    pub baseline_upper_quartile_us: f64,
    pub candidate_median_us: f64,
    pub candidate_min_us: f64,
    pub candidate_max_us: f64,
    pub candidate_lower_quartile_us: f64,
    pub candidate_upper_quartile_us: f64,
    pub median_difference_us: f64,
    pub difference_interval_low_us: f64,
    pub difference_interval_high_us: f64,
    pub difference_interval_coverage: f64,
    pub sign_test_p_value: f64,
    pub relative_difference: f64,
    pub verdict: c_int,
    pub rounds_candidate_faster: c_int,
    pub round_count: c_int,
}

/// Mirrors `mlx_perf_comparison_summary` field for field.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
#[allow(dead_code)] // declared for callers; this example profiles one job
pub struct ComparisonSummary {
    pub baseline: JobSummary,
    pub candidate: JobSummary,
    pub job_wall: MetricComparison,
    pub gpu_span: MetricComparison,
    pub host_submission: MetricComparison,
    pub max_absolute_output_difference: f64,
}

unsafe extern "C" {
    #[allow(dead_code)] // declared for callers; this example writes the report too
    fn mlx_perf_profile_to_measurements(
        build_graph: BuildGraphCallback,
        builder_payload: *mut c_void,
        variant_name: *const c_char,
        output_directory: *const c_char,
        measurements_file_base: *const c_char,
        summary: *mut JobSummary,
    ) -> c_int;
    #[allow(dead_code)] // declared for callers; this example profiles one job
    fn mlx_perf_compare_to_report(
        baseline: *const Variant,
        candidate: *const Variant,
        report_title: *const c_char,
        output_directory: *const c_char,
        report_file_base: *const c_char,
        summary: *mut ComparisonSummary,
    ) -> c_int;
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

#[cfg(test)]
mod tests {
    use super::*;

    // The same sizes mlx_perf_c_api_test asserts with sizeof on the C side:
    // a field added on one side only changes one of the two and fails.
    #[test]
    fn should_match_the_c_struct_sizes() {
        assert_eq!(std::mem::size_of::<JobSummary>(), 72);
        assert_eq!(std::mem::size_of::<MetricComparison>(), 144);
        assert_eq!(std::mem::size_of::<ComparisonSummary>(), 584);
        assert_eq!(std::mem::size_of::<Variant>(), 24);
    }
}
