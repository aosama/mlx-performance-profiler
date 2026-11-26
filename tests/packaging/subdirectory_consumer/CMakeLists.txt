# A consumer that pulls the repository in with add_subdirectory (the same
# path FetchContent takes). Configure-only: it checks that a non-top-level
# build gets the library and nothing else.
cmake_minimum_required(VERSION 3.24)
project(mlx_perf_subdirectory_consumer LANGUAGES C CXX)

add_subdirectory("${MLX_PERF_SOURCE_DIRECTORY}" mlx_perf_build)

if(NOT TARGET mlx_perf::mlx_perf)
  message(FATAL_ERROR "mlx_perf::mlx_perf is missing from a subdirectory build")
endif()
foreach(unwanted_target hello_world mlx_perf_server qwen35_profile
        mlx_perf_live_view mlx_perf_qwen35 mlx_perf_demo_model graph_model_test)
  if(TARGET ${unwanted_target})
    message(FATAL_ERROR "a subdirectory build created ${unwanted_target}")
  endif()
endforeach()
message(STATUS "subdirectory consumer: library only")
