# cmake -DBUILD_DIRECTORY=... -DINSTALL_PREFIX=... -DSOURCE_DIRECTORY=... -P install_and_check.cmake
#
# Installs this build into a fresh scratch prefix and checks the result is
# exactly the public surface: the C API headers, the archive and the package
# files, with no path of the machine that built it.

file(REMOVE_RECURSE "${INSTALL_PREFIX}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIRECTORY}" --prefix "${INSTALL_PREFIX}"
  COMMAND_ERROR_IS_FATAL ANY)

file(GLOB_RECURSE installed_files RELATIVE "${INSTALL_PREFIX}" "${INSTALL_PREFIX}/*")
# The per-configuration targets file is named after the build type.
list(TRANSFORM installed_files REPLACE "mlx_perfTargets-[a-z]+\\.cmake$"
     "mlx_perfTargets-<configuration>.cmake")
list(SORT installed_files)
set(expected_files
    include/mlx_perf/mlx_perf_c_api.h
    include/mlx_perf/mlx_perf_version.h
    lib/cmake/mlx_perf/mlx_perfConfig.cmake
    lib/cmake/mlx_perf/mlx_perfConfigVersion.cmake
    lib/cmake/mlx_perf/mlx_perfTargets-<configuration>.cmake
    lib/cmake/mlx_perf/mlx_perfTargets.cmake
    lib/cmake/mlx_perf/mlx_perf_mlx_link.cmake
    lib/libmlx_perf.a)
if(NOT installed_files STREQUAL expected_files)
  message(FATAL_ERROR "installed files differ from the public surface.\n"
                      "  installed: ${installed_files}\n  expected:  ${expected_files}")
endif()

# A package carrying the build machine's source or MLX paths would only work
# on that machine.
file(GLOB package_files "${INSTALL_PREFIX}/lib/cmake/mlx_perf/*.cmake")
foreach(package_file IN LISTS package_files)
  file(READ "${package_file}" package_text)
  foreach(machine_path "${SOURCE_DIRECTORY}" "${BUILD_DIRECTORY}" "${BUILT_AGAINST_MLX_PREFIX}")
    string(FIND "${package_text}" "${machine_path}" machine_path_position)
    if(NOT machine_path_position EQUAL -1)
      message(FATAL_ERROR "${package_file} contains the build machine path ${machine_path}")
    endif()
  endforeach()
endforeach()
message(STATUS "installed exactly the public surface into ${INSTALL_PREFIX}")
