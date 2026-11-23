cmake_minimum_required(VERSION 3.24)
project(mlx_visual_plan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The prebuilt MLX 0.32.4 static library from the mlx-bridge native cache is
# the only native dependency. MLX_PREFIX is REQUIRED on the CMake command line
# and is never read from the environment: hard-coding a cached path would make
# the build silently bind to one machine's prebuilt MLX, and an environment
# variable would hide the dependency from `cmake --graphviz` and CI. Pass it
# explicitly every time:  cmake -DMLX_PREFIX=/path/to/prefix
set(MLX_PREFIX "" CACHE PATH "Prefix of a prebuilt MLX installation (containing include/ and lib/). Required: pass -DMLX_PREFIX=... on the CMake command line.")
if(NOT MLX_PREFIX)
  message(
    FATAL_ERROR
      "MLX_PREFIX is required but was not set. Re-run cmake with -DMLX_PREFIX=/path/to/prefix (a prefix containing include/ and lib/). It is intentionally not read from the environment."
  )
endif()
# A set-but-wrong prefix (typo, stale native-cache identity) would otherwise
# surface much later as a confusing "mlx/mlx.h not found" compile error;
# failing at configure time names the real cause.
if(NOT EXISTS "${MLX_PREFIX}/include/mlx/mlx.h")
  message(
    FATAL_ERROR
      "MLX_PREFIX='${MLX_PREFIX}' does not contain include/mlx/mlx.h. Point it at a prebuilt MLX prefix (containing include/ and lib/)."
  )
endif()

# The modules build once as a static library shared by the demo executable
# and the tests, so tests exercise exactly the code the demo ships.
add_library(mlx_plan STATIC
    src/graph_model.cpp
    src/telemetry.cpp
    src/latency_breakdown.cpp
    src/progress_reporter.cpp
    src/metal_command_buffer_probe.cpp
    src/gpu_timeline.cpp
    src/timeline_console.cpp
    src/dot_writer.cpp
    src/svg_renderer.cpp
    src/report_writer.cpp)

target_include_directories(mlx_plan PUBLIC src)

# SYSTEM keeps MLX's own headers from warning under our -Wall; they are a
# vendored surface, not code we maintain.
target_include_directories(mlx_plan SYSTEM PUBLIC
    "${MLX_PREFIX}/include"
    "${MLX_PREFIX}/include/metal_cpp")

# MLX's own consumer flags (mirrored from its MLXConfig): _METAL_ selects the
# Metal backend surface, ACCELERATE_NEW_LAPACK the Accelerate BLAS layout,
# MLX_STATIC the static-library linkage path inside the headers.
target_compile_definitions(mlx_plan PUBLIC
    MLX_STATIC _METAL_ ACCELERATE_NEW_LAPACK)

# Why not find_package(MLX): the install exports an `mlx` imported target whose
# INTERFACE_LINK_LIBRARIES references jaccl::jaccl and other targets it does
# not export, so generate fails. The archive list below is the link recipe
# proven working on this machine by ../mlx-bridge's build script: dependents
# first (mlx), then the libraries it references, then frameworks. libc++ is
# left to the C++ link driver; listing it again only draws a duplicate warning.
target_link_libraries(mlx_plan PUBLIC
    "${MLX_PREFIX}/lib/libmlx.a"
    "${MLX_PREFIX}/lib/libgguflib.a"
    "${MLX_PREFIX}/lib/libjaccl.a"
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Accelerate")

target_compile_options(mlx_plan PUBLIC -Wall -Wextra)

add_executable(hello_world src/hello_world.cpp)
target_link_libraries(hello_world PRIVATE mlx_plan)

# Tests: a header-only assert runner (tests/test_harness.h) instead of a
# framework, per the plan's "no heavy deps through Phase 2". Run with
#   ctest --test-dir build --output-on-failure
enable_testing()

# Unit: pure Spark arithmetic, no GPU needed.
add_executable(latency_breakdown_test tests/latency_breakdown_test.cpp)
target_link_libraries(latency_breakdown_test PRIVATE mlx_plan)
add_test(NAME latency_breakdown_test COMMAND latency_breakdown_test)

add_executable(progress_reporter_test tests/progress_reporter_test.cpp)
target_link_libraries(progress_reporter_test PRIVATE mlx_plan)
add_test(NAME progress_reporter_test COMMAND progress_reporter_test)

# Integration: drives real MLX streams on the Metal GPU.
add_executable(telemetry_test tests/telemetry_test.cpp)
target_link_libraries(telemetry_test PRIVATE mlx_plan)
add_test(NAME telemetry_test COMMAND telemetry_test)

add_executable(metal_command_buffer_probe_test
    tests/metal_command_buffer_probe_test.cpp)
target_link_libraries(metal_command_buffer_probe_test PRIVATE mlx_plan)
add_test(NAME metal_command_buffer_probe_test
    COMMAND metal_command_buffer_probe_test)

add_executable(gpu_timeline_test tests/gpu_timeline_test.cpp)
target_link_libraries(gpu_timeline_test PRIVATE mlx_plan)
add_test(NAME gpu_timeline_test COMMAND gpu_timeline_test)
