# cmake -DCONSUMER_SOURCE=... -DCONSUMER_BUILD=... -DINSTALL_PREFIX=...
#       -DMLX_PREFIX=... -DGENERATOR=... -DREQUESTED_VERSION=... -P consumer_journey.cmake
#
# What another repository does with an installed mlx_perf: configure against
# the package and its own MLX prefix, build, and run a program that profiles
# a job through the C API.

file(REMOVE_RECURSE "${CONSUMER_BUILD}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${CONSUMER_BUILD}"
          -G "${GENERATOR}" -DCMAKE_BUILD_TYPE=Release
          "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}" "-DMLX_PREFIX=${MLX_PREFIX}"
          "-DMLX_PERF_REQUESTED_VERSION=${REQUESTED_VERSION}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${CONSUMER_BUILD}/mlx_perf_consumer" "${CONSUMER_BUILD}"
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${CONSUMER_BUILD}/consumer_report.html")
  message(FATAL_ERROR "the consumer ran but wrote no report")
endif()
