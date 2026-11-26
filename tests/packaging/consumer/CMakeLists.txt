# Another repository's build, in miniature: finds the installed mlx_perf
# package and links a C program against it. CXX is enabled because the
# archive is C++ and its imported target asks for the C++ link driver.
cmake_minimum_required(VERSION 3.24)
project(mlx_perf_consumer LANGUAGES C CXX)

# The version a consumer pins; the packaging tests pass the current one, and
# the next minor to prove the package refuses it.
set(MLX_PERF_REQUESTED_VERSION "" CACHE STRING "mlx_perf version to request")
# A consumer for which profiling is optional asks without REQUIRED.
option(MLX_PERF_IS_OPTIONAL "Treat mlx_perf as optional" OFF)
if(MLX_PERF_IS_OPTIONAL)
  find_package(mlx_perf ${MLX_PERF_REQUESTED_VERSION} QUIET)
  message(STATUS "optional mlx_perf found: ${mlx_perf_FOUND}")
  if(NOT mlx_perf_FOUND)
    return()
  endif()
else()
  find_package(mlx_perf ${MLX_PERF_REQUESTED_VERSION} REQUIRED)
endif()

add_executable(mlx_perf_consumer consumer_main.c)
target_link_libraries(mlx_perf_consumer PRIVATE mlx_perf::mlx_perf)
