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
