# Packaging tests: the library as another repository's build sees it. They
# install into a scratch prefix under the build directory, never elsewhere.

set(packaging_scratch_directory "${CMAKE_CURRENT_BINARY_DIR}/scratch")
set(packaging_install_prefix "${packaging_scratch_directory}/install_prefix")
set(consumer_source_directory "${CMAKE_CURRENT_SOURCE_DIR}/consumer")
# The version a consumer of this release pins (MAJOR.MINOR), and the next
# minor, which SameMinorVersion compatibility must refuse.
set(current_minor_version "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
math(EXPR next_minor_number "${PROJECT_VERSION_MINOR} + 1")
set(next_minor_version "${PROJECT_VERSION_MAJOR}.${next_minor_number}")

# A prefix that looks like MLX but reports another release, for the
# mismatch check. Written at configure time; it holds no library.
set(other_mlx_release_prefix "${packaging_scratch_directory}/other_mlx_release_prefix")
file(WRITE "${other_mlx_release_prefix}/include/mlx/mlx.h" "")
file(MAKE_DIRECTORY "${other_mlx_release_prefix}/include/metal_cpp")
file(WRITE "${other_mlx_release_prefix}/include/mlx/version.h"
     "#define MLX_VERSION_MAJOR 0\n#define MLX_VERSION_MINOR 0\n#define MLX_VERSION_PATCH 1\n")
# Off macOS the library links MLX through MLX's own installed package and
# reads the backend from it, so the stand-in borrows the real prefix's
# package, libraries and fmt headers; only its version differs.
if(MLX_PERF_GPU_BACKEND STREQUAL "cuda")
  foreach(borrowed_path share lib include/fmt)
    file(REMOVE "${other_mlx_release_prefix}/${borrowed_path}")
    file(CREATE_LINK "${MLX_PREFIX}/${borrowed_path}" "${other_mlx_release_prefix}/${borrowed_path}" SYMBOLIC)
  endforeach()
endif()

# --- positive ---

add_test(NAME should_install_exactly_the_public_surface
  COMMAND "${CMAKE_COMMAND}"
          "-DBUILD_DIRECTORY=${PROJECT_BINARY_DIR}"
          "-DINSTALL_PREFIX=${packaging_install_prefix}"
          "-DSOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
          "-DBUILT_AGAINST_MLX_PREFIX=${MLX_PREFIX}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/install_and_check.cmake")
set_tests_properties(should_install_exactly_the_public_surface PROPERTIES
  FIXTURES_SETUP mlx_perf_installed)

add_test(NAME should_build_and_run_a_consumer_of_the_installed_package
  COMMAND "${CMAKE_COMMAND}"
          "-DCONSUMER_SOURCE=${consumer_source_directory}"
          "-DCONSUMER_BUILD=${packaging_scratch_directory}/consumer_build"
          "-DINSTALL_PREFIX=${packaging_install_prefix}"
          "-DMLX_PREFIX=${MLX_PREFIX}"
          "-DGENERATOR=${CMAKE_GENERATOR}"
          "-DREQUESTED_VERSION=${current_minor_version}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/consumer_journey.cmake")
set_tests_properties(should_build_and_run_a_consumer_of_the_installed_package PROPERTIES
  FIXTURES_REQUIRED mlx_perf_installed)

# --- positive: built from source with FetchContent, the build is the library
# alone, with only the C API exported, and a program links against it ---

add_test(NAME should_build_and_link_a_consumer_that_fetches_the_source
  COMMAND "${CMAKE_COMMAND}"
          "-DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/source_consumer"
          "-DCONSUMER_BUILD=${packaging_scratch_directory}/source_consumer_build"
          "-DMLX_PERF_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
          "-DMLX_PREFIX=${MLX_PREFIX}"
          "-DGENERATOR=${CMAKE_GENERATOR}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/source_consumer_journey.cmake")
set_tests_properties(should_build_and_link_a_consumer_that_fetches_the_source PROPERTIES
  PASS_REGULAR_EXPRESSION "source consumer: library only.*source consumer: built and linked")

# --- negative: the package names the problem instead of failing late ---

add_test(NAME should_refuse_a_consumer_without_mlx_prefix
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${consumer_source_directory}"
          -B "${packaging_scratch_directory}/consumer_without_mlx_prefix"
          -G "${CMAKE_GENERATOR}"
          "-DCMAKE_PREFIX_PATH=${packaging_install_prefix}")
set_tests_properties(should_refuse_a_consumer_without_mlx_prefix PROPERTIES
  FIXTURES_REQUIRED mlx_perf_installed
  PASS_REGULAR_EXPRESSION "MLX_PREFIX is required")

# --- edge: an optional consumer without MLX goes on, told why ---

add_test(NAME should_let_an_optional_consumer_continue_without_mlx_prefix
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${consumer_source_directory}"
          -B "${packaging_scratch_directory}/optional_consumer_without_mlx_prefix"
          -G "${CMAKE_GENERATOR}"
          "-DCMAKE_PREFIX_PATH=${packaging_install_prefix}"
          -DMLX_PERF_IS_OPTIONAL=ON)
set_tests_properties(should_let_an_optional_consumer_continue_without_mlx_prefix PROPERTIES
  FIXTURES_REQUIRED mlx_perf_installed
  PASS_REGULAR_EXPRESSION "optional mlx_perf found: 0.*Generating done")

add_test(NAME should_refuse_an_mlx_release_other_than_the_one_built_against
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${consumer_source_directory}"
          -B "${packaging_scratch_directory}/consumer_with_other_mlx_release"
          -G "${CMAKE_GENERATOR}"
          "-DCMAKE_PREFIX_PATH=${packaging_install_prefix}"
          "-DMLX_PREFIX=${other_mlx_release_prefix}")
set_tests_properties(should_refuse_an_mlx_release_other_than_the_one_built_against PROPERTIES
  FIXTURES_REQUIRED mlx_perf_installed
  PASS_REGULAR_EXPRESSION "was built against MLX")

add_test(NAME should_refuse_a_consumer_asking_for_another_minor_version
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${consumer_source_directory}"
          -B "${packaging_scratch_directory}/consumer_wanting_next_minor_build"
          -G "${CMAKE_GENERATOR}"
          "-DCMAKE_PREFIX_PATH=${packaging_install_prefix}"
          "-DMLX_PREFIX=${MLX_PREFIX}"
          "-DMLX_PERF_REQUESTED_VERSION=${next_minor_version}")
set_tests_properties(should_refuse_a_consumer_asking_for_another_minor_version PROPERTIES
  FIXTURES_REQUIRED mlx_perf_installed
  PASS_REGULAR_EXPRESSION "compatible with requested version \"${next_minor_version}\"")

# A prefix that exists but holds no MLX at all.
set(prefix_without_mlx "${packaging_scratch_directory}/prefix_without_mlx")
file(MAKE_DIRECTORY "${prefix_without_mlx}")

add_test(NAME should_refuse_an_mlx_prefix_without_mlx
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${CMAKE_CURRENT_SOURCE_DIR}/source_consumer"
          -B "${packaging_scratch_directory}/source_consumer_without_mlx"
          -G "${CMAKE_GENERATOR}"
          "-DFETCHCONTENT_SOURCE_DIR_MLX_PERF=${PROJECT_SOURCE_DIR}"
          "-DMLX_PREFIX=${prefix_without_mlx}")
set_tests_properties(should_refuse_an_mlx_prefix_without_mlx PROPERTIES
  PASS_REGULAR_EXPRESSION "does not contain include/mlx/mlx.h")

# --- edge: an MLX release outside the tested set warns but does not stop a
# source build (a locally patched or newer MLX can still be tried). The
# warning must appear, then the configure and generate must both finish
# (CMake wraps long warnings, hence the whitespace class).

add_test(NAME should_warn_but_continue_on_an_untested_mlx_release
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${CMAKE_CURRENT_SOURCE_DIR}/source_consumer"
          -B "${packaging_scratch_directory}/source_consumer_untested_mlx"
          -G "${CMAKE_GENERATOR}"
          "-DFETCHCONTENT_SOURCE_DIR_MLX_PERF=${PROJECT_SOURCE_DIR}"
          "-DMLX_PREFIX=${other_mlx_release_prefix}")
set_tests_properties(should_warn_but_continue_on_an_untested_mlx_release PROPERTIES
  PASS_REGULAR_EXPRESSION "not a tested[ \n]+release.*source consumer: library only.*Generating done")
