# Using it from Rust

Profile an MLX job built in Rust with [mlx-bridge](https://github.com/aosama/mlx-bridge), through the C API. The complete, working example is [`rust/mlx_perf_rust_example`](../rust/mlx_perf_rust_example/); this page walks through its files, each copied here verbatim.

## How the pieces fit

- **mlx-bridge** builds the MLX job in safe Rust and links MLX and MLX-C from its prebuilt prefix.
- **libmlx_perf.a**, built by CMake against **that same prefix**, does the measuring. It holds no MLX of its own, so the binary links exactly one MLX.
- A Rust `extern "C"` callback hands each fresh graph's output to the profiler, which writes the same HTML report and measurement file as from C.

Build the library first, pointing `MLX_PREFIX` at the prefix mlx-bridge built (by default under `~/.cache/mlx-bridge/native/<identity>/prefix`):

```sh
cmake -S . -B build -DMLX_PREFIX=$HOME/.cache/mlx-bridge/native/<identity>/prefix
cmake --build build
cd rust/mlx_perf_rust_example
cargo profile-job      # alias for: cargo run --release
```

It writes `build/rust_example.html` and `build/rust_example.json` in the repository.

## Link the static library: `build.rs`

<!-- example: rust/mlx_perf_rust_example/build.rs -->
```rust
//! Links mlx-performance-profiler's static C++ library. The path is derived from this
//! crate's location (the repository's CMake build directory) rather than an
//! environment variable, so the dependency is visible in one place.

use std::path::PathBuf;

fn main() {
    let manifest_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cmake_build_directory = manifest_directory.join("../../build");
    let static_library_path = cmake_build_directory.join("libmlx_perf.a");
    if !static_library_path.exists() {
        panic!(
            "{} is missing: build the C++ library first (see .cargo/config.toml)",
            static_library_path.display()
        );
    }
    println!("cargo:rerun-if-changed={}", static_library_path.display());
    println!(
        "cargo:rustc-link-search=native={}",
        cmake_build_directory.canonicalize().expect("build directory").display()
    );
    // Only this repository's objects: every MLX / MLX-C symbol it needs is
    // resolved from the libraries mlx-bridge's build script already links.
    println!("cargo:rustc-link-lib=static=mlx_perf");
}
```

In your own crate, point the search path at wherever your `libmlx_perf.a` lives: the CMake build directory, or `lib/` of an installed package. The example's `Cargo.toml` depends on mlx-bridge through a sibling checkout (`path = "../../../mlx-bridge"`). In your crate, depend on a published mlx-bridge release instead, and build `libmlx_perf.a` against the prefix that release builds, so both sides use one MLX.

## Declare the C API: `mlx_perf_ffi.rs`

`#[repr(C)]` mirrors of the C structs, the `extern "C"` declarations, and a safe wrapper around `mlx_perf_profile_to_report` that turns a failure into `Err` with the library's message. A unit test pins the struct sizes to the C side's, so a field added on one side only fails the build.

<!-- example: rust/mlx_perf_rust_example/src/mlx_perf_ffi.rs -->
```rust
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
    #[allow(dead_code)] // declared for callers; this example keeps output on
    fn mlx_perf_set_console_output(is_enabled: c_int);
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
```

## Build the job and the callback: `three_stream_job.rs`

The job runs on three GPU streams: two independent branches, each on its own stream, joined on the main stream. The callback builds a fresh graph on every call and must never let a panic unwind into C++, so it catches it and returns non-zero.

<!-- example: rust/mlx_perf_rust_example/src/three_stream_job.rs -->
```rust
//! The job this example profiles, built with mlx-bridge's safe API on three
//! GPU streams: two independent branches on their own streams, joined on the
//! main stream (the cross-stream "shuffle" edge the report shows as a stall).

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDeviceHandle, MlxDtype, MlxStream, raw};

const BATCH_ROWS: i32 = 256;
const FEATURE_WIDTH: i32 = 1024;
const REDUCTION_ROWS: i32 = 2048;

/// One context per stream: mlx-bridge submits every operation to its
/// context's stream, which is how the branches land on different streams.
pub struct ThreeStreamJob {
    main_context: MlxBindingsContext,
    projection_context: MlxBindingsContext,
    reduction_context: MlxBindingsContext,
}

impl ThreeStreamJob {
    /// The main stream is the default GPU stream; each branch gets a new one.
    pub fn new() -> Result<Self, MlxCError> {
        let gpu_device = MlxDeviceHandle::default_device()?;
        Ok(Self {
            main_context: MlxBindingsContext::on_stream(MlxStream::default_gpu()?)?,
            projection_context: MlxBindingsContext::on_stream(MlxStream::on_device(&gpu_device)?)?,
            reduction_context: MlxBindingsContext::on_stream(MlxStream::on_device(&gpu_device)?)?,
        })
    }

    /// A brand-new, unevaluated graph for the job's single output.
    pub fn build_fresh_output(&self) -> Result<MlxArray, MlxCError> {
        let projection = &self.projection_context;
        let batch = projection.ones(&[BATCH_ROWS, FEATURE_WIDTH], MlxDtype::Float32)?;
        let weights = projection.ones(&[FEATURE_WIDTH, FEATURE_WIDTH], MlxDtype::Float32)?;
        let projected = projection.multiply_scalar(&projection.matmul(&batch, &weights)?, 1e-3)?;
        let gated = projection.multiply(&projected, &projection.sigmoid(&projected)?)?;

        let reduction = &self.reduction_context;
        let samples = reduction.ones(&[REDUCTION_ROWS, FEATURE_WIDTH], MlxDtype::Float32)?;
        let mixed = reduction.add(&reduction.exp(&samples)?, &reduction.sin(&samples)?)?;
        let column_totals = reduction.sum_axis(&mixed, 0, false)?;

        let joined = self.main_context.add(&gated, &column_totals)?;
        self.main_context.tanh(&joined)
    }
}

/// The C callback handed to mlx-performance-profiler. A panic must never unwind into
/// C++ frames (undefined behavior), so it is caught and reported as failure;
/// catch_unwind's closure is the one closure this crate cannot avoid.
///
/// # Safety
/// `payload` must point at a live `ThreeStreamJob`.
pub unsafe extern "C" fn build_three_stream_graph(
    fresh_outputs: raw::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // SAFETY: The caller passes the ThreeStreamJob it registered as payload.
    let three_stream_job = unsafe { &*(payload as *const ThreeStreamJob) };
    let build_outcome = catch_unwind(AssertUnwindSafe(|| append_fresh_output(three_stream_job, fresh_outputs)));
    match build_outcome {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        Err(_) => 2,
    }
}

fn append_fresh_output(
    three_stream_job: &ThreeStreamJob,
    fresh_outputs: raw::mlx_vector_array,
) -> Result<(), MlxCError> {
    let job_output = three_stream_job.build_fresh_output()?;
    // SAFETY: Both handles are live; MLX-C copies the array handle on append,
    // so dropping `job_output` afterwards is fine.
    let append_status = unsafe { raw::mlx_vector_array_append_value(fresh_outputs, job_output.raw()) };
    if append_status == 0 {
        Ok(())
    } else {
        Err(MlxCError {
            operation: "append the job output",
            description: format!("mlx_vector_array_append_value returned {append_status}"),
        })
    }
}
```

## Call the profiler: `main.rs`

<!-- example: rust/mlx_perf_rust_example/src/main.rs -->
```rust
//! Profiles an MLX job built in Rust (mlx-bridge) with mlx-performance-profiler:
//! the same four passes, the same Spark-style HTML report the C++ demo
//! produces, and the per-node measurement file beside it. Orchestration only.

mod mlx_perf_ffi;
mod summary_format;
mod three_stream_job;

use std::ffi::c_void;
use std::process::ExitCode;

use mlx_perf_ffi::{ProfileRequest, profile_to_report};
use three_stream_job::{ThreeStreamJob, build_three_stream_graph};

fn main() -> ExitCode {
    let three_stream_job = match ThreeStreamJob::new() {
        Ok(job) => job,
        Err(error) => {
            eprintln!("error: cannot set up MLX streams: {error}");
            return ExitCode::FAILURE;
        }
    };
    let report_directory = concat!(env!("CARGO_MANIFEST_DIR"), "/../../build");
    let profile_request = ProfileRequest {
        variant_name: "rust",
        report_title: "Rust (mlx-bridge) three-stream job",
        output_directory: report_directory,
        report_file_base: "rust_example",
    };
    let job_payload = &three_stream_job as *const ThreeStreamJob as *mut c_void;
    // SAFETY: The payload is the ThreeStreamJob the callback expects; it lives
    // on this frame for the whole synchronous call.
    match unsafe { profile_to_report(build_three_stream_graph, job_payload, &profile_request) } {
        Ok(job_summary) => {
            println!("{}", summary_format::describe_summary(&job_summary));
            print_written_file("report", report_directory, "rust_example.html");
            print_written_file("measurements", report_directory, "rust_example.json");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("error: {failure}");
            ExitCode::FAILURE
        }
    }
}

/// Prints where the library wrote a file, as an absolute path when it exists.
fn print_written_file(file_role: &str, directory: &str, file_name: &str) {
    let written_path = std::path::Path::new(directory).join(file_name);
    let shown_path = written_path.canonicalize().unwrap_or(written_path);
    println!("{file_role}: {}", shown_path.display());
}
```

## Rules that matter from Rust

- **The callback is `unsafe extern "C"` and must not panic across the boundary.** Catch the panic and return non-zero; the profiler then reports `build_graph callback returned <value>`.
- **The payload is a raw pointer to your job.** It must stay alive, and must not be moved, for the whole synchronous call.
- **MLX is not `Send`:** profile on the thread that created the job's contexts and streams.
- **Read `mlx_perf_last_error()` right after a failed call,** before any other `mlx_perf_*` call on that thread.
- Run with `--release`, so building the Rust graph is not what gets measured. Graph-building time is reported separately in the JSON's `graph_build` record either way.
