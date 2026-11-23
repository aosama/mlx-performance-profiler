//! Profiles an MLX job built in Rust (mlx-c-rust) with mlx-visual-plan:
//! the same four passes and the same Spark-style HTML report the C++ demo
//! produces. Orchestration only.

mod mlx_plan_ffi;
mod summary_format;
mod two_stream_job;

use std::ffi::c_void;
use std::process::ExitCode;

use mlx_plan_ffi::{ProfileRequest, profile_to_report};
use two_stream_job::{TwoStreamJob, build_two_stream_graph};

fn main() -> ExitCode {
    let two_stream_job = match TwoStreamJob::new() {
        Ok(job) => job,
        Err(error) => {
            eprintln!("error: cannot set up MLX streams: {error}");
            return ExitCode::FAILURE;
        }
    };
    let report_directory = concat!(env!("CARGO_MANIFEST_DIR"), "/../../build");
    let profile_request = ProfileRequest {
        variant_name: "rust",
        report_title: "Rust (mlx-c-rust) two-stream job",
        output_directory: report_directory,
        report_file_base: "rust_example",
    };
    let job_payload = &two_stream_job as *const TwoStreamJob as *mut c_void;
    // SAFETY: The payload is the TwoStreamJob the callback expects; it lives
    // on this frame for the whole synchronous call.
    match unsafe { profile_to_report(build_two_stream_graph, job_payload, &profile_request) } {
        Ok(job_summary) => {
            println!("{}", summary_format::describe_summary(&job_summary));
            let report_path = std::path::Path::new(report_directory).join("rust_example.html");
            let shown_path = report_path.canonicalize().unwrap_or(report_path);
            println!("report: {}", shown_path.display());
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("error: {failure}");
            ExitCode::FAILURE
        }
    }
}
