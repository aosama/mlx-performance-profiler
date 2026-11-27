# Building and linking

How to add mlx-performance-profiler to your own CMake project, either from source or as an installed package, and the rules about MLX that keep your binary correct.

## The one rule: one MLX, named by `MLX_PREFIX`

The library is built against a prebuilt MLX and MLX-C prefix (see [Getting started](getting-started.md#step-1-get-an-mlx-prefix)), named on the CMake command line:

```sh
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
```

- `MLX_PREFIX` is a CMake cache variable. It is never read from an environment variable, so the dependency stays visible in the build.
- `libmlx_perf.a` holds only this library's code. Every MLX and MLX-C symbol comes from your prefix, so your final binary links exactly one MLX. Linking a second MLX (for example a different release in another dependency) breaks it.
- The tested MLX releases are 0.32.4 and 0.32.5. A source build of an untested release warns at configure time and continues. An installed package refuses any MLX release other than the one it was built against.

## From source with FetchContent

The simplest way: your project fetches the library at a release tag and builds it with your MLX. This is [`examples/CMakeLists.txt`](examples/CMakeLists.txt), a complete project the test suite configures, builds and runs:

<!-- example: docs/examples/CMakeLists.txt -->
```cmake
# A project of your own that uses mlx-performance-profiler: it fetches the
# library at a release tag, builds it against your prebuilt MLX, and links
# five small C programs to it.
#
#   cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
#   cmake --build build
#   ./build/profile_one_job build
#
# MLX_PREFIX names a prebuilt MLX of a tested release (it holds include/ and
# lib/); it is a CMake cache variable, never an environment variable. CXX is
# enabled because the library is C++ and links with the C++ driver.
cmake_minimum_required(VERSION 3.24)
project(mlx_perf_examples LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(mlx_perf
  GIT_REPOSITORY https://github.com/aosama/mlx-performance-profiler.git
  GIT_TAG v0.7.0)
FetchContent_MakeAvailable(mlx_perf)

foreach(example_name profile_one_job compare_two_implementations compare_two_procedures measure_quietly
    measure_wired_memory)
  add_executable(${example_name} ${example_name}.c)
  target_link_libraries(${example_name} PRIVATE mlx_perf::mlx_perf)
endforeach()
```

Pulled in this way, only the library is built: no tools, no tests, no install rules. Only the C API's headers reach your include path. `add_subdirectory` on a checkout behaves the same.

## As an installed package

Build and install the library once:

```sh
git clone --branch v0.7.0 https://github.com/aosama/mlx-performance-profiler.git
cd mlx-performance-profiler
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix \
      -DMLX_PERF_BUILD_TOOLS=OFF -DMLX_PERF_BUILD_TESTS=OFF
cmake --build build
cmake --install build --prefix /path/to/install
```

Then find it from your project, with the same MLX prefix:

```cmake
# cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix -DCMAKE_PREFIX_PATH=/path/to/install
cmake_minimum_required(VERSION 3.24)
project(my_profiled_app LANGUAGES C CXX)
find_package(mlx_perf 0.7 REQUIRED)
add_executable(my_profiled_app main.c)
target_link_libraries(my_profiled_app PRIVATE mlx_perf::mlx_perf)
```

- `find_package(mlx_perf 0.7)` accepts any 0.7.x. Before 1.0, a different minor version is refused, because a minor release may change the C API.
- The installed files are `include/mlx_perf/mlx_perf_c_api.h`, `include/mlx_perf/mlx_perf_version.h`, `lib/libmlx_perf.a` and the CMake package.
- Without `REQUIRED`, a missing `MLX_PREFIX` makes the package report itself not found instead of stopping your configure, so profiling can be an optional feature.
- Enable `CXX` in your project even for a C program: the library is C++ and needs the C++ linker.

## Build options

| Option | Default | Builds |
|---|---|---|
| `MLX_PERF_BUILD_TOOLS` | on in this repository, off when included | `hello_world`, `mlx_perf_server`, `qwen35_profile` |
| `MLX_PERF_BUILD_TESTS` | the same | the test suite |
| `MLX_PERF_INSTALL` | the same | install rules and the CMake package |

The build type defaults to Release, matching the prebuilt MLX. The library is compiled optimised even when your project sets no build type, because its own host-side code is part of what it measures.

## Run the tests

```sh
cmake -S . -B build -DMLX_PREFIX=/path/to/mlx/prefix
cmake --build build
ctest --test-dir build --output-on-failure
```

The suite needs a GPU of the backend the prefix was built for: a Metal GPU on macOS, an NVIDIA GPU on Linux. It includes unit, component, integration and acceptance tests, packaging tests that build consumers against the installed package and through FetchContent, the examples on these pages (built and run), and the documentation checks.

## On Linux with an NVIDIA GPU

The library measures MLX's CUDA backend through CUPTI, NVIDIA's observation interface, which ships with the CUDA toolkit. Configure detects the backend from the prefix's MLX, so the commands are the same as on macOS. The prefix differs in three ways:

- **MLX built with `-DMLX_BUILD_CUDA=ON`** (CUDA 12 or later, compute capability 7.5 or later), with MLX-C built against it and installed into the same prefix. The library links MLX through MLX's own installed CMake package there, which carries its CUDA, cuDNN and BLAS dependencies.
- **MLX as a shared library** (`-DBUILD_SHARED_LIBS=ON`, giving `lib/libmlx.so`). MLX compiles some CUDA kernels at run time, such as its quantized matrix multiplies, and looks for their headers in the `include/` folder beside the library it is loaded from. Linked statically, MLX looks beside your executable instead and cannot compile them.
- **fmt headers in `include/fmt`**: MLX's CUDA headers include them, and MLX's install leaves them out. Copy the ones MLX was built with from its build tree's `_deps/fmt-src/include/fmt`.

Two things to know when measuring:

- **CUPTI keeps tracing.** Once the library starts observing, CUPTI records every kernel's start and end for the rest of the process, including the uninstrumented rounds of a comparison. Both sides of a comparison carry the same cost, so verdicts stay fair, but absolute durations include it.
- **One CUDA profiler at a time.** CUPTI accepts one subscriber per process, so the library cannot measure inside Nsight Systems or Nsight Compute, and they cannot profile a process that is using it.

Wired memory is a macOS concept: on Linux the wired memory sections read as unavailable.

## Related

- [Troubleshooting](troubleshooting.md#configure-and-link-problems) for configure and link errors.
- [Using it from Rust](using-from-rust.md) to link the library into a Cargo build.
