cmake_minimum_required(VERSION 3.24)
project(mlx_performance_profiler VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default: the prebuilt MLX is a Release build, and the host-side
# cost of this library's own code (issue, sweep, probe) is part of what it
# measures, so an unoptimized build would inflate the numbers it reports.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# Only a top-level build compiles the tools and tests by default, so a
# consumer pulling this in with FetchContent or add_subdirectory builds the
# library alone, in seconds.
option(MLX_PERF_BUILD_TOOLS
       "Build hello_world, mlx_perf_server and qwen35_profile" ${PROJECT_IS_TOP_LEVEL})
option(MLX_PERF_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(MLX_PERF_INSTALL "Generate install rules and the mlx_perf package" ${PROJECT_IS_TOP_LEVEL})

include(GNUInstallDirs)
include(cmake/mlx_perf_mlx_prefix.cmake)
include(cmake/mlx_perf_project_settings.cmake)

# The prebuilt MLX static library is the only native dependency. MLX_PREFIX
# is REQUIRED on the CMake command line and never read from the environment:
# a hard-coded cached path would silently bind the build to one machine's
# MLX, and an environment variable would hide the dependency from
# `cmake --graphviz` and CI.  cmake -DMLX_PREFIX=/path/to/prefix
set(MLX_PREFIX "" CACHE PATH "Prefix of a prebuilt MLX installation (containing include/ and lib/). Required: pass -DMLX_PREFIX=... on the CMake command line.")
mlx_perf_add_mlx_prefix_target(mlx_perf_mlx "${MLX_PREFIX}" GLOBAL)

configure_file(include/mlx_perf/mlx_perf_version.h.in
               "${PROJECT_BINARY_DIR}/generated/include/mlx_perf/mlx_perf_version.h"
               @ONLY)

# The profiler: capture, timelines, reports and the C API. The only target
# that is installed; it never depends on the demo, server or model code.
add_library(mlx_perf STATIC
    src/graph_model.cpp
    src/graph_build_timer.cpp
    src/telemetry.cpp
    src/latency_breakdown.cpp
    src/progress_reporter.cpp
    src/metal_command_buffer_probe.cpp
    src/gpu_timeline.cpp
    src/timeline_dependencies.cpp
    src/timed_units.cpp
    src/stage_plan.cpp
    src/stage_timeline.cpp
    src/timeline_console.cpp
    src/text_format.cpp
    src/latency_slices.cpp
    src/dot_writer.cpp
    src/svg_renderer.cpp
    src/report_assets.cpp
    src/report_components.cpp
    src/report_summary_section.cpp
    src/report_task_table_section.cpp
    src/report_stage_table_section.cpp
    src/report_gantt_section.cpp
    src/report_comparison_page.cpp
    src/report_writer.cpp
    src/report_bundle.cpp
    src/measurement_session.cpp
    src/json_value.cpp
    src/json_writer.cpp
    src/mlx_c_interop.cpp
    src/mlx_perf_c_api.cpp)
add_library(mlx_perf::mlx_perf ALIAS mlx_perf)
target_include_directories(mlx_perf PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    # Internal C++ headers: visible to this repository's tools and tests,
    # never installed. The installed surface is the C API alone.
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")
# MLX is attached here only for this build tree; the installed package
# attaches it from the consumer's own MLX_PREFIX (see cmake/), so the
# export never carries this machine's paths.
target_link_libraries(mlx_perf PUBLIC "$<BUILD_INTERFACE:mlx_perf_mlx>")
mlx_perf_apply_project_settings(mlx_perf)

# Support libraries for the tools and tests; never installed.
if(MLX_PERF_BUILD_TOOLS OR MLX_PERF_BUILD_TESTS)
  # The three-stream demo job (hello_world, the live server, tests).
  add_library(mlx_perf_demo_model STATIC src/demo_model.cpp)
  target_link_libraries(mlx_perf_demo_model PUBLIC mlx_perf)
  mlx_perf_apply_project_settings(mlx_perf_demo_model)

  # The live server: HTTP, the view store and the pages it serves.
  add_library(mlx_perf_live_view STATIC
      src/http_server.cpp
      src/live_view_store.cpp
      src/live_view_renderer.cpp
      src/live_view_page.cpp
      src/live_view_routes.cpp
      src/report_memory_section.cpp
      src/server_options.cpp)
  target_link_libraries(mlx_perf_live_view PUBLIC mlx_perf)
  mlx_perf_apply_project_settings(mlx_perf_live_view)

  # Qwen3.5 rebuilt in C++ from MLX kernels: model-specific, kept apart.
  add_library(mlx_perf_qwen35 STATIC
      src/qwen35/qwen35_config.cpp
      src/qwen35/qwen35_weights.cpp
      src/qwen35/qwen35_decode_step.cpp
      src/qwen35/qwen35_generation.cpp
      src/qwen35/token_text.cpp
      src/qwen35/qwen35_decode_job.cpp
      src/qwen35/qwen35_layer_stages.cpp)
  target_link_libraries(mlx_perf_qwen35 PUBLIC mlx_perf)
  mlx_perf_apply_project_settings(mlx_perf_qwen35)
endif()

if(MLX_PERF_BUILD_TOOLS)
  add_executable(hello_world src/hello_world.cpp)
  target_link_libraries(hello_world PRIVATE mlx_perf_demo_model)
  mlx_perf_apply_project_settings(hello_world)

  # A real model: ./build/qwen35_profile <dense Qwen3.5 MLX model directory>
  add_executable(qwen35_profile src/qwen35_profile.cpp)
  target_link_libraries(qwen35_profile PRIVATE mlx_perf_qwen35)
  mlx_perf_apply_project_settings(qwen35_profile)

  # The live view: ./build/mlx_perf_server [--port N], then open
  # http://127.0.0.1:N/ (loopback only).
  add_executable(mlx_perf_server src/mlx_perf_server.cpp)
  target_link_libraries(mlx_perf_server PRIVATE mlx_perf_live_view mlx_perf_demo_model)
  mlx_perf_apply_project_settings(mlx_perf_server)
endif()

# Install rules only for a top-level build by default: a project that pulls
# this repository in with FetchContent decides for itself what it installs.
if(MLX_PERF_INSTALL)
  include(cmake/mlx_perf_install.cmake)
endif()

if(MLX_PERF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
