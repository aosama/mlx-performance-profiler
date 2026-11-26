# Tests: a header-only assert runner (test_harness.h) instead of a framework,
# per the plan's "no heavy deps". One executable per production module.
# Run with:  ctest --test-dir build --output-on-failure

# Adds `<test_name>` from `<test_name>.cpp`, linked to the support libraries
# named after it, or to the core library when none is named (each support
# library already carries the core; naming it twice makes the linker warn
# about a duplicate archive). Every test runs from the build directory, so a
# test that writes reports writes them there.
function(mlx_perf_add_test test_name)
  add_executable(${test_name} ${test_name}.cpp)
  if(ARGN)
    target_link_libraries(${test_name} PRIVATE ${ARGN})
  else()
    target_link_libraries(${test_name} PRIVATE mlx_perf)
  endif()
  mlx_perf_apply_project_settings(${test_name})
  add_test(NAME ${test_name} COMMAND ${test_name}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# Unit and component: pure arithmetic, text and HTML, no GPU needed.
mlx_perf_add_test(latency_breakdown_test)
mlx_perf_add_test(sample_statistics_test)
mlx_perf_add_test(text_format_test)
mlx_perf_add_test(latency_slices_test)
mlx_perf_add_test(report_components_test)
mlx_perf_add_test(report_writer_test)
mlx_perf_add_test(report_comparison_page_test)
mlx_perf_add_test(report_verdict_section_test)
mlx_perf_add_test(comparison_record_test)
mlx_perf_add_test(json_value_test)
mlx_perf_add_test(json_writer_test)
mlx_perf_add_test(graph_build_timer_test)
mlx_perf_add_test(measurement_document_test)
mlx_perf_add_test(report_bundle_test)
mlx_perf_add_test(progress_reporter_test)
mlx_perf_add_test(stage_plan_test)
mlx_perf_add_test(report_stage_table_section_test)
mlx_perf_add_test(dot_writer_test)
mlx_perf_add_test(timed_units_test)
mlx_perf_add_test(server_options_test mlx_perf_live_view)
mlx_perf_add_test(live_view_store_test mlx_perf_live_view)
mlx_perf_add_test(report_memory_section_test mlx_perf_live_view)
mlx_perf_add_test(http_server_test mlx_perf_live_view)
mlx_perf_add_test(live_view_routes_test mlx_perf_live_view)
mlx_perf_add_test(qwen35_config_test mlx_perf_qwen35)
mlx_perf_add_test(token_text_test mlx_perf_qwen35)

# Integration: drives real MLX streams on the Metal GPU.
mlx_perf_add_test(graph_model_test)
mlx_perf_add_test(telemetry_test)
mlx_perf_add_test(metal_command_buffer_probe_test)
mlx_perf_add_test(demo_model_test mlx_perf_demo_model)
mlx_perf_add_test(output_agreement_test)
mlx_perf_add_test(measurement_session_test)
mlx_perf_add_test(paired_measurement_test)
mlx_perf_add_test(measurement_file_test)
mlx_perf_add_test(mlx_perf_c_api_test)
mlx_perf_add_test(qwen35_decode_step_test mlx_perf_qwen35)
mlx_perf_add_test(gpu_timeline_test)
mlx_perf_add_test(stage_timeline_test)
mlx_perf_add_test(timeline_dependencies_test)
mlx_perf_add_test(qwen35_layer_stages_test mlx_perf_qwen35)

# Packaging: the installed library as another repository's build sees it.
# Needs the install rules, so it follows MLX_PERF_INSTALL.
if(MLX_PERF_INSTALL)
  add_subdirectory(packaging)
endif()
