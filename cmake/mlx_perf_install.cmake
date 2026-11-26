# Install rules and the CMake package for consumers:
#
#   <prefix>/include/mlx_perf/mlx_perf_c_api.h
#   <prefix>/include/mlx_perf/mlx_perf_version.h
#   <prefix>/lib/libmlx_perf.a
#   <prefix>/lib/cmake/mlx_perf/  (config, version, targets, MLX link recipe)
#
# A consumer then writes find_package(mlx_perf) with its own -DMLX_PREFIX and
# links mlx_perf::mlx_perf. Only the C API is installed: the C++ headers in
# src/ are internal and may change in any release.

include(CMakePackageConfigHelpers)

set(MLX_PERF_PACKAGE_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/cmake/mlx_perf")

# The archive is compiled against one MLX's C++ headers (the probe reads MLX
# backend objects), so the installed package records which one and refuses
# a different MLX at the consumer's configure time.
mlx_perf_read_mlx_version("${MLX_PREFIX}" MLX_PERF_BUILT_AGAINST_MLX_VERSION)

install(TARGETS mlx_perf EXPORT mlx_perf_targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES
          include/mlx_perf/mlx_perf_c_api.h
          "${PROJECT_BINARY_DIR}/generated/include/mlx_perf/mlx_perf_version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlx_perf")
install(EXPORT mlx_perf_targets
        NAMESPACE mlx_perf::
        FILE mlx_perfTargets.cmake
        DESTINATION "${MLX_PERF_PACKAGE_DIRECTORY}")

configure_package_config_file(
  cmake/mlx_perfConfig.cmake.in
  "${PROJECT_BINARY_DIR}/generated/mlx_perfConfig.cmake"
  INSTALL_DESTINATION "${MLX_PERF_PACKAGE_DIRECTORY}")
# Before 1.0 a minor release may break the C API (see VERSIONING.md), so a
# request for 0.1 accepts 0.1.x only.
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/generated/mlx_perfConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion)
install(FILES
          "${PROJECT_BINARY_DIR}/generated/mlx_perfConfig.cmake"
          "${PROJECT_BINARY_DIR}/generated/mlx_perfConfigVersion.cmake"
          cmake/mlx_perf_mlx_prefix.cmake
        DESTINATION "${MLX_PERF_PACKAGE_DIRECTORY}")
