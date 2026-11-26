# A consumer that builds mlx_perf from source with FetchContent, declared the
# way another repository would: this public repository at a release tag. The
# packaging tests set FETCHCONTENT_SOURCE_DIR_MLX_PERF to the checkout under
# test, so nothing is downloaded and the code being tested is what builds.
cmake_minimum_required(VERSION 3.24)
project(mlx_perf_source_consumer LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(mlx_perf
  GIT_REPOSITORY https://github.com/aosama/mlx-performance-profiler.git
  GIT_TAG v0.5.0)
FetchContent_MakeAvailable(mlx_perf)

# From source, the consumer gets the library alone ...
if(NOT TARGET mlx_perf::mlx_perf)
  message(FATAL_ERROR "mlx_perf::mlx_perf is missing from a source build")
endif()
foreach(unwanted_target hello_world mlx_perf_server qwen35_profile mlx_perf_internal
        mlx_perf_live_view mlx_perf_qwen35 mlx_perf_demo_model graph_model_test)
  if(TARGET ${unwanted_target})
    message(FATAL_ERROR "a source build created ${unwanted_target}")
  endif()
endforeach()
# ... with only the C API on its include path ...
get_target_property(mlx_perf_include_directories mlx_perf INTERFACE_INCLUDE_DIRECTORIES)
if(mlx_perf_include_directories MATCHES "/src[;>]|/src$")
  message(FATAL_ERROR "internal headers leak to consumers: ${mlx_perf_include_directories}")
endif()
# ... optimised even though this consumer chose no build type ...
get_target_property(mlx_perf_compile_options mlx_perf COMPILE_OPTIONS)
if(NOT mlx_perf_compile_options MATCHES "-O2")
  message(FATAL_ERROR "mlx_perf is not optimised without a build type: ${mlx_perf_compile_options}")
endif()
message(STATUS "source consumer: library only")

# ... and links a program against it through the in-tree MLX recipe.
add_executable(mlx_perf_source_consumer ../consumer/consumer_main.c)
target_link_libraries(mlx_perf_source_consumer PRIVATE mlx_perf::mlx_perf)
