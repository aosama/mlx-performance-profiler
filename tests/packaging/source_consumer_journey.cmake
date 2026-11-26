# cmake -DCONSUMER_SOURCE=... -DCONSUMER_BUILD=... -DMLX_PERF_SOURCE_DIRECTORY=...
#       -DMLX_PREFIX=... -DGENERATOR=... -P source_consumer_journey.cmake
#
# What another repository does to build mlx_perf from source: configure with
# FetchContent (pointed at the checkout under test), then build and link a
# program. The program is not run: the installed-package journey runs one.

file(REMOVE_RECURSE "${CONSUMER_BUILD}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${CONSUMER_BUILD}"
          -G "${GENERATOR}"
          "-DFETCHCONTENT_SOURCE_DIR_MLX_PERF=${MLX_PERF_SOURCE_DIRECTORY}"
          "-DMLX_PREFIX=${MLX_PREFIX}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD}" --target mlx_perf_source_consumer
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${CONSUMER_BUILD}/mlx_perf_source_consumer")
  message(FATAL_ERROR "the source consumer built but produced no program")
endif()
message(STATUS "source consumer: built and linked")
