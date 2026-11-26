# cmake -DEXAMPLES_SOURCE=... -DEXAMPLES_BUILD=... -DMLX_PERF_SOURCE_DIRECTORY=...
#       -DMLX_PREFIX=... -DGENERATOR=... -DPYTHON=... -DDOCS_CHECK=... -P docs_examples_journey.cmake
#
# What a reader of the documentation does: configure docs/examples as their
# own project (FetchContent pointed at this checkout instead of the tag),
# build it, run every program, read the JSON with the documented reader, and
# confirm every file matches the published JSON Schemas. If an example in
# the docs stops compiling, running or describing real output, this fails.

file(REMOVE_RECURSE "${EXAMPLES_BUILD}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${EXAMPLES_SOURCE}" -B "${EXAMPLES_BUILD}" -G "${GENERATOR}"
          -DCMAKE_BUILD_TYPE=Release "-DMLX_PREFIX=${MLX_PREFIX}"
          "-DFETCHCONTENT_SOURCE_DIR_MLX_PERF=${MLX_PERF_SOURCE_DIRECTORY}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${EXAMPLES_BUILD}" COMMAND_ERROR_IS_FATAL ANY)

foreach(example_program profile_one_job compare_two_implementations compare_two_procedures measure_quietly
                        measure_wired_memory)
  execute_process(COMMAND "${EXAMPLES_BUILD}/${example_program}" "${EXAMPLES_BUILD}"
                  COMMAND_ERROR_IS_FATAL ANY)
endforeach()

# measure_quietly turns console output off: the library must print nothing,
# so its only line is the program's own.
execute_process(COMMAND "${EXAMPLES_BUILD}/measure_quietly" "${EXAMPLES_BUILD}"
                OUTPUT_VARIABLE quiet_standard_output ERROR_VARIABLE quiet_standard_error
                COMMAND_ERROR_IS_FATAL ANY)
string(STRIP "${quiet_standard_output}" quiet_standard_output)
if(quiet_standard_output MATCHES "\n" OR NOT quiet_standard_error STREQUAL "")
  message(FATAL_ERROR "measure_quietly printed more than its own line:\n${quiet_standard_output}\n${quiet_standard_error}")
endif()

execute_process(
  COMMAND "${PYTHON}" "${EXAMPLES_SOURCE}/read_measurements.py" "${EXAMPLES_BUILD}/softmax_matmul.json" 3
  COMMAND_ERROR_IS_FATAL ANY)

set(schema_directory "${MLX_PERF_SOURCE_DIRECTORY}/docs/schema")
execute_process(
  COMMAND "${PYTHON}" "${DOCS_CHECK}" schema "${schema_directory}/measurement-file.schema.json"
          "${EXAMPLES_BUILD}/softmax_matmul.json" "${EXAMPLES_BUILD}/layer_norm_job.json"
          "${EXAMPLES_BUILD}/softmax_mlx_softmax.json" "${EXAMPLES_BUILD}/softmax_written_out.json"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${PYTHON}" "${DOCS_CHECK}" schema "${schema_directory}/comparison-record.schema.json"
          "${EXAMPLES_BUILD}/softmax.json"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${PYTHON}" "${DOCS_CHECK}" schema "${schema_directory}/procedure-comparison-record.schema.json"
          "${EXAMPLES_BUILD}/token_loop.json"
  COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "docs examples: built, ran, read and matched the schemas")
