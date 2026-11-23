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

# Why not find_package(MLX): the install exports an `mlx` imported target whose
# INTERFACE_LINK_LIBRARIES references jaccl::jaccl and other targets it does
# not export, so generate fails. The archive list below is the link recipe
# proven working on this machine by ../mlx-bridge's build script: dependents
# first (mlx), then the libraries it references, then frameworks. libc++ is
# left to the C++ link driver; listing it again only draws a duplicate warning.
add_executable(hello_world
    src/graph_model.cpp
    src/telemetry.cpp
    src/dot_writer.cpp
    src/svg_renderer.cpp
    src/report_writer.cpp
    src/hello_world.cpp)

# SYSTEM keeps MLX's own headers from warning under our -Wall; they are a
# vendored surface, not code we maintain.
target_include_directories(hello_world SYSTEM PRIVATE
    "${MLX_PREFIX}/include"
    "${MLX_PREFIX}/include/metal_cpp")

# MLX's own consumer flags (mirrored from its MLXConfig): _METAL_ selects the
# Metal backend surface, ACCELERATE_NEW_LAPACK the Accelerate BLAS layout,
# MLX_STATIC the static-library linkage path inside the headers.
target_compile_definitions(hello_world PRIVATE
    MLX_STATIC _METAL_ ACCELERATE_NEW_LAPACK)

target_link_libraries(hello_world PRIVATE
    "${MLX_PREFIX}/lib/libmlx.a"
    "${MLX_PREFIX}/lib/libgguflib.a"
    "${MLX_PREFIX}/lib/libjaccl.a"
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Accelerate")

target_compile_options(hello_world PRIVATE -Wall -Wextra)
