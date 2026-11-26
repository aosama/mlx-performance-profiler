# A project of your own that uses mlx-performance-profiler: it fetches the
# library at a release tag, builds it against your prebuilt MLX, and links
# four small C programs to it.
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
  GIT_TAG v0.4.0)
FetchContent_MakeAvailable(mlx_perf)

foreach(example_name profile_one_job compare_two_implementations compare_two_procedures measure_quietly)
  add_executable(${example_name} ${example_name}.c)
  target_link_libraries(${example_name} PRIVATE mlx_perf::mlx_perf)
endforeach()
