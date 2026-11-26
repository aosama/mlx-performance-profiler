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
    // resolved from the libraries mlx-c-rust's build script already links.
    println!("cargo:rustc-link-lib=static=mlx_perf");
}
