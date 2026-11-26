# Compile settings every target of this repository shares. Applied PRIVATE,
# target by target, so a consumer that builds this library inside its own
# tree (FetchContent, add_subdirectory) never inherits them.
function(mlx_perf_apply_project_settings target_name)
  target_compile_options(${target_name} PRIVATE -Wall -Wextra)
  # MLX's headers change class layouts when NDEBUG is absent (debug-only
  # members in mlx/allocator.h and mlx/backend/metal/device.h), and the
  # prebuilt MLX is a Release build. Code that includes those headers must
  # agree with the library about NDEBUG, whatever build type is chosen here;
  # otherwise this code and MLX would compile those classes differently (a
  # one-definition-rule violation). Today the probe reads only fields that
  # precede the debug-only ones, so nothing breaks yet, but that is layout
  # luck rather than a guarantee. Nothing in this repository relies on
  # assert().
  target_compile_definitions(${target_name} PRIVATE NDEBUG)
endfunction()
