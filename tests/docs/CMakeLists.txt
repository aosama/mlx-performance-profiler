# Documentation tests: the pages agents and people read (docs/, published
# to Context7) must stay true to the code. Standard-library Python only.

find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS "Python 3 not found: the documentation tests will not run")
  return()
endif()
set(docs_check_script "${CMAKE_CURRENT_SOURCE_DIR}/docs_check.py")

# Unit: the checks themselves, on synthetic pages.
add_test(NAME docs_check_test
         COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p "*_test.py")

# Component: this repository's docs against its header, sources, CMake
# files, examples and context7.json. No build output needed.
add_test(NAME docs_repository_check
         COMMAND "${Python3_EXECUTABLE}" "${docs_check_script}" check "${PROJECT_SOURCE_DIR}")

# Acceptance, on the Metal GPU: the documented examples as a reader builds
# and runs them, and their output against the published JSON Schemas.
add_test(NAME docs_examples_journey
         COMMAND "${CMAKE_COMMAND}"
                 "-DEXAMPLES_SOURCE=${PROJECT_SOURCE_DIR}/docs/examples"
                 "-DEXAMPLES_BUILD=${CMAKE_CURRENT_BINARY_DIR}/examples_build"
                 "-DMLX_PERF_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
                 "-DMLX_PREFIX=${MLX_PREFIX}"
                 "-DGENERATOR=${CMAKE_GENERATOR}"
                 "-DPYTHON=${Python3_EXECUTABLE}"
                 "-DDOCS_CHECK=${docs_check_script}"
                 "-DGPU_BACKEND=${MLX_PERF_GPU_BACKEND}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/docs_examples_journey.cmake")
set_tests_properties(docs_examples_journey PROPERTIES
  PASS_REGULAR_EXPRESSION "docs examples: built, ran, read and matched the schemas"
  TIMEOUT 600)
