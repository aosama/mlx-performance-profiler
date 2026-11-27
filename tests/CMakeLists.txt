# Tests: a header-only assert runner (test_harness.h) instead of a framework,
# per the plan's "no heavy deps". One executable per production module.
# Run with:  ctest --test-dir build --output-on-failure

# Adds `<test_name>` from `<test_name>.cpp`, linked to the support libraries
# named after it, or to the core library and its internal headers when none
# is named (each support library already carries both; naming the core twice
# makes the linker warn about a duplicate archive). Every test runs from the build directory, so a
# test that writes reports writes them there.
function(mlx_perf_add_test test_name)
  add_executable(${test_name} ${test_name}.cpp)
  if(ARGN)
    target_link_libraries(${test_name} PRIVATE ${ARGN})
  else()
    target_link_libraries(${test_name} PRIVATE mlx_perf_internal)
  endif()
  mlx_perf_apply_project_settings(${test_name})
  add_test(NAME ${test_name} COMMAND ${test_name}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# Unit and component: pure arithmetic, text and HTML, no GPU needed.
mlx_perf_add_test(latency_breakdown_test)
mlx_perf_add_test(dispatch_occupancy_test)
mlx_perf_add_test(gpu_busy_time_test)
mlx_perf_add_test(kernel_launch_summary_test)
mlx_perf_add_test(report_launch_comparison_section_test)
mlx_perf_add_test(sample_statistics_test)
mlx_perf_add_test(text_format_test)
mlx_perf_add_test(latency_slices_test)
mlx_perf_add_test(report_components_test)
mlx_perf_add_test(report_writer_test)
mlx_perf_add_test(report_comparison_page_test)
mlx_perf_add_test(report_verdict_section_test)
mlx_perf_add_test(comparison_record_test)
mlx_perf_add_test(procedure_comparison_record_test)
mlx_perf_add_test(report_procedure_page_test)
mlx_perf_add_test(procedure_report_files_test)
mlx_perf_add_test(c_api_support_test)
mlx_perf_add_test(recording_slots_test)
mlx_perf_add_test(wired_memory_sample_test)
mlx_perf_add_test(wired_memory_analysis_test)
mlx_perf_add_test(wired_memory_record_test)
mlx_perf_add_test(report_wired_memory_section_test)
mlx_perf_add_test(run_memory_record_test)
mlx_perf_add_test(report_memory_verdict_test)
mlx_perf_add_test(json_value_test)
mlx_perf_add_test(json_writer_test)
mlx_perf_add_test(console_output_test)
mlx_perf_add_test(graph_build_timer_test)
mlx_perf_add_test(measurement_document_test)
mlx_perf_add_test(report_bundle_test)
mlx_perf_add_test(progress_reporter_test)
mlx_perf_add_test(stage_plan_test)
mlx_perf_add_test(report_stage_table_section_test)
mlx_perf_add_test(dot_writer_test)
mlx_perf_add_test(graph_view_section_test)
mlx_perf_add_test(timed_units_test)
mlx_perf_add_test(server_options_test mlx_perf_live_view)
mlx_perf_add_test(live_view_store_test mlx_perf_live_view)
mlx_perf_add_test(report_memory_section_test mlx_perf_live_view)
mlx_perf_add_test(http_server_test mlx_perf_live_view)
mlx_perf_add_test(live_view_routes_test mlx_perf_live_view)
mlx_perf_add_test(live_view_renderer_test mlx_perf_live_view)
mlx_perf_add_test(qwen35_config_test mlx_perf_qwen35)
mlx_perf_add_test(token_text_test mlx_perf_qwen35)

# The graph view is JavaScript, so its model, style and panels are tested
# under Node, where they run the same as in a browser: one *_test.js per
# module in graph_view/. Skipped, with a note, where Node is absent.
find_program(MLX_PERF_NODE_EXECUTABLE node)
if(MLX_PERF_NODE_EXECUTABLE)
  add_test(NAME graph_view_scripts_test
           COMMAND "${MLX_PERF_NODE_EXECUTABLE}" --test "${CMAKE_CURRENT_SOURCE_DIR}/graph_view/")
else()
  message(STATUS "Node not found: graph_view_scripts_test (JavaScript) will not run")
endif()

# Integration: drives real MLX streams on the GPU.
mlx_perf_add_test(graph_model_test)
mlx_perf_add_test(telemetry_test)
mlx_perf_add_test(wired_memory_sampler_test)
mlx_perf_add_test(run_memory_meter_test)
mlx_perf_add_test(procedure_measurement_test)
mlx_perf_add_test(submission_recorder_test)
mlx_perf_add_test(demo_model_test mlx_perf_demo_model)
mlx_perf_add_test(output_agreement_test)
mlx_perf_add_test(measurement_session_test)
mlx_perf_add_test(paired_measurement_test)
mlx_perf_add_test(measurement_file_test)
mlx_perf_add_test(mlx_perf_c_api_test)
mlx_perf_add_test(mlx_perf_c_api_procedures_test)
mlx_perf_add_test(qwen35_decode_step_test mlx_perf_qwen35)
mlx_perf_add_test(gpu_timeline_test)
mlx_perf_add_test(stage_timeline_test)
mlx_perf_add_test(timeline_dependencies_test)
mlx_perf_add_test(qwen35_layer_stages_test mlx_perf_qwen35)

# The backend's own observers, against the GPU they observe: Metal's
# command-buffer probe, Objective-C hooks and macOS system counters, or
# CUDA's CUPTI session, probe and recorders and the Linux stand-ins.
if(MLX_PERF_GPU_BACKEND STREQUAL "cuda")
  mlx_perf_add_test(cupti_session_test)
  mlx_perf_add_test(cuda_execution_probe_test)
  mlx_perf_add_test(cuda_dispatch_recorder_test)
  mlx_perf_add_test(cuda_host_clock_test)
  mlx_perf_add_test(cuda_gpu_device_facts_test)
  mlx_perf_add_test(linux_system_wired_memory_test)
  mlx_perf_add_test(linux_system_memory_pressure_test)
  mlx_perf_add_test(cuda_host_hand_off_test)
else()
  mlx_perf_add_test(objc_method_hooks_test)
  mlx_perf_add_test(system_wired_memory_test)
  mlx_perf_add_test(system_memory_pressure_test)
  mlx_perf_add_test(metal_command_buffer_probe_test)
  mlx_perf_add_test(metal_dispatch_recorder_test)
  mlx_perf_add_test(metal_host_clock_test)
  mlx_perf_add_test(metal_gpu_device_facts_test)
  mlx_perf_add_test(metal_object_classes_test)
  # macOS's wired memory: the line the GPU's memory is wired under, which
  # Linux and a discrete GPU do not have.
  mlx_perf_add_test(wire_release_probe_test)
  mlx_perf_add_test(mlx_perf_c_api_wired_memory_test)
  mlx_perf_add_test(wired_memory_acceptance_test)
endif()

# Acceptance: a whole behaviour end to end, on the GPU.
mlx_perf_add_test(paired_comparison_acceptance_test)
mlx_perf_add_test(launch_occupancy_acceptance_test)
mlx_perf_add_test(procedure_handoffs_acceptance_test)
mlx_perf_add_test(memory_comparison_acceptance_test)

# Documentation: the published docs against the code, and their examples
# built and run as a reader would (see tests/docs/CMakeLists.txt).
add_subdirectory(docs)

# Packaging: the installed library as another repository's build sees it.
# Needs the install rules, so it follows MLX_PERF_INSTALL.
if(MLX_PERF_INSTALL)
  add_subdirectory(packaging)
endif()
