# How mlx-performance-profiler links a prebuilt MLX. One recipe, used twice:
# by this repository's own build, and by the installed package config, so a
# consumer links MLX exactly the way the library was tested.
#
# Why not find_package(MLX): the MLX install exports an `mlx` imported target
# whose INTERFACE_LINK_LIBRARIES references jaccl::jaccl and other targets it
# does not export, so generation fails. The archive list below is the recipe
# proven by ../mlx-bridge's build script: dependents first (mlx), then the
# libraries it references, then frameworks. libc++ is left to the C++ link
# driver; listing it again only draws a duplicate warning.

# The MLX releases this library builds and passes its tests against. Others
# may work; configure warns so the gap is visible rather than silent.
set(MLX_PERF_TESTED_MLX_VERSIONS "0.32.4;0.32.5")

# A set-but-wrong prefix (typo, stale native-cache identity) would otherwise
# surface much later as a confusing "mlx/mlx.h not found" compile error;
# failing at configure time names the real cause.
function(mlx_perf_require_mlx_prefix mlx_prefix)
  if(NOT mlx_prefix)
    message(
      FATAL_ERROR
        "MLX_PREFIX is required but was not set. Re-run cmake with -DMLX_PREFIX=/path/to/prefix (a prebuilt MLX prefix containing include/ and lib/). It is intentionally not read from the environment."
    )
  endif()
  if(NOT EXISTS "${mlx_prefix}/include/mlx/mlx.h")
    message(
      FATAL_ERROR
        "MLX_PREFIX='${mlx_prefix}' does not contain include/mlx/mlx.h. Point it at a prebuilt MLX prefix (containing include/ and lib/)."
    )
  endif()
endfunction()

# Reads MAJOR.MINOR.PATCH from the prefix's mlx/version.h into `version_variable`.
function(mlx_perf_read_mlx_version mlx_prefix version_variable)
  file(READ "${mlx_prefix}/include/mlx/version.h" version_header_text)
  set(version_parts "")
  foreach(version_part MAJOR MINOR PATCH)
    if(NOT version_header_text MATCHES "#define MLX_VERSION_${version_part} ([0-9]+)")
      message(FATAL_ERROR "Cannot read MLX_VERSION_${version_part} from ${mlx_prefix}/include/mlx/version.h")
    endif()
    list(APPEND version_parts "${CMAKE_MATCH_1}")
  endforeach()
  list(JOIN version_parts "." mlx_version)
  set(${version_variable} "${mlx_version}" PARENT_SCOPE)
endfunction()

# Creates `target_name`, an interface target carrying MLX's include
# directories, compile definitions and link line from `mlx_prefix`.
# Pass GLOBAL when targets in other directories must see it (this
# repository's tests/ directory does).
function(mlx_perf_add_mlx_prefix_target target_name mlx_prefix)
  mlx_perf_require_mlx_prefix("${mlx_prefix}")
  mlx_perf_read_mlx_version("${mlx_prefix}" detected_mlx_version)
  if(NOT detected_mlx_version IN_LIST MLX_PERF_TESTED_MLX_VERSIONS)
    message(
      WARNING
        "MLX ${detected_mlx_version} at ${mlx_prefix} is not a tested release (tested: ${MLX_PERF_TESTED_MLX_VERSIONS}). It may still work; see the README's MLX compatibility section for what the library relies on."
    )
  else()
    message(STATUS "mlx_perf: MLX ${detected_mlx_version} at ${mlx_prefix}")
  endif()

  mlx_perf_detect_gpu_backend("${mlx_prefix}" detected_gpu_backend)
  set(MLX_PERF_GPU_BACKEND "${detected_gpu_backend}" PARENT_SCOPE)
  add_library(${target_name} INTERFACE IMPORTED ${ARGN})
  if(detected_gpu_backend STREQUAL "cuda")
    mlx_perf_attach_cuda_mlx(${target_name} "${mlx_prefix}")
  else()
    mlx_perf_attach_metal_mlx(${target_name} "${mlx_prefix}")
  endif()
endfunction()

# Which GPU backend the MLX at `mlx_prefix` was built with, into
# `backend_variable`: "metal" or "cuda". MLX installs every backend's headers
# whatever it was built with, so the headers cannot tell, and a shared MLX
# exports no link line either. MLX's own installed package can: its
# MLXConfig.cmake is configured from `if(@MLX_BUILD_CUDA@)` around
# find_dependency(CUDAToolkit), which reads `if(ON)` only in a CUDA build. An
# MLX built for its CPU backend alone has no GPU to measure and is refused
# with the reason.
function(mlx_perf_detect_gpu_backend mlx_prefix backend_variable)
  if(APPLE)
    set(${backend_variable} "metal" PARENT_SCOPE)
    return()
  endif()
  set(mlx_package_config_file "${mlx_prefix}/share/cmake/MLX/MLXConfig.cmake")
  set(mlx_package_config_text "")
  if(EXISTS "${mlx_package_config_file}")
    file(READ "${mlx_package_config_file}" mlx_package_config_text)
  endif()
  if(NOT mlx_package_config_text MATCHES "if\\(ON\\)[ \t\r\n]*find_dependency\\(CUDAToolkit")
    message(
      FATAL_ERROR
        "MLX_PREFIX='${mlx_prefix}' holds an MLX built without a GPU backend this library measures. Off macOS it needs MLX built with -DMLX_BUILD_CUDA=ON (its share/cmake/MLX/MLXConfig.cmake then finds CUDAToolkit)."
    )
  endif()
  set(${backend_variable} "cuda" PARENT_SCOPE)
endfunction()

# MLX's CUDA build, linked the way its own installed package describes: the
# exported `mlx` target carries its CUDA, cuDNN and BLAS dependencies, which
# a hand-written archive list would have to track per MLX release. CUPTI and
# the driver API are this library's own additions: CUPTI observes the GPU
# (cupti_session), the driver API names kernels and reads device facts.
function(mlx_perf_attach_cuda_mlx target_name mlx_prefix)
  # MLX's CUDA backend header (mlx/backend/cuda/device.h, read by
  # cuda_mlx_streams) includes <fmt/format.h>, which MLX's install leaves out.
  if(NOT EXISTS "${mlx_prefix}/include/fmt/format.h")
    message(
      FATAL_ERROR
        "MLX_PREFIX='${mlx_prefix}' has no include/fmt/format.h. MLX's CUDA headers include fmt, which MLX does not install: copy the fmt headers MLX was built with (its build tree's _deps/fmt-src/include/fmt) into ${mlx_prefix}/include/fmt."
    )
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_package(MLX CONFIG REQUIRED PATHS "${mlx_prefix}" NO_DEFAULT_PATH)
  set_target_properties(${target_name} PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${mlx_prefix}/include")
  target_include_directories(${target_name} INTERFACE "${mlx_prefix}/include")
  target_compile_definitions(${target_name} INTERFACE MLX_PERF_GPU_BACKEND_CUDA)
  target_link_libraries(${target_name} INTERFACE
    "${mlx_prefix}/lib/libmlxc.a"
    mlx
    CUDA::cupti
    CUDA::cuda_driver)
endfunction()

# MLX's Metal build: the archive list proven by ../mlx-bridge (see the top of
# this file for why not MLX's own package on macOS).
function(mlx_perf_attach_metal_mlx target_name mlx_prefix)
  # SYSTEM keeps MLX's own headers from warning under -Wall; they are a
  # vendored surface, not code this project maintains.
  set_target_properties(${target_name} PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
      "${mlx_prefix}/include;${mlx_prefix}/include/metal_cpp")
  target_include_directories(${target_name} INTERFACE
    "${mlx_prefix}/include"
    "${mlx_prefix}/include/metal_cpp")
  # MLX's own consumer flags (mirrored from its MLXConfig): _METAL_ selects
  # the Metal backend surface, ACCELERATE_NEW_LAPACK the Accelerate BLAS
  # layout, MLX_STATIC the static-library linkage path inside the headers.
  target_compile_definitions(${target_name} INTERFACE
    MLX_STATIC _METAL_ ACCELERATE_NEW_LAPACK)
  target_link_libraries(${target_name} INTERFACE
    "${mlx_prefix}/lib/libmlxc.a"
    "${mlx_prefix}/lib/libmlx.a"
    "${mlx_prefix}/lib/libgguflib.a"
    "${mlx_prefix}/lib/libjaccl.a"
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Accelerate"
    # MLX 0.32.5 reads the GPU core count through the IOKit registry
    # (metal::gpu_core_count); without it every executable fails to link.
    # A system framework, so listing it is harmless on 0.32.4.
    "-framework IOKit")
endfunction()
